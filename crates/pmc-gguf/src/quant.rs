// SPDX-License-Identifier: GPL-3.0-only

//! Row quantizers for the fusion step.
//!
//! Ported from the reference implementations in `llama.cpp/ggml/src/
//! ggml-quants.c` (`quantize_row_q4_K_impl`, `quantize_row_q5_K_impl`,
//! `quantize_row_q6_K_ref`) including the helper routines `make_qkx3_quants`,
//! `make_qp_quants` and `make_qx_quants`. The writer side is intentionally
//! absent: these functions produce raw blocks in memory for training-time
//! fusion, they do not write GGUF files.
//!
//! All arithmetic is `f32`, matching the C reference. The produced blocks are
//! valid inputs for [`crate::dequant`].

use crate::dequant::{f16_to_f32, f32_to_f16, get_scale_min_k4, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES};
use crate::types::{GgufError, QK_K};

const GROUP_MAX_EPS: f32 = 1e-15;

fn put_f16(buf: &mut [u8], off: usize, value: f32) {
    buf[off..off + 2].copy_from_slice(&f32_to_f16(value).to_le_bytes());
}

/// Round to the nearest integer using the same bit trick as the C reference.
///
/// Valid for `|fval| <= 4194303`. The addition moves the value into the
/// integer range of binary32 mantissa bits, and the mask extracts the rounded
/// integer.
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4_194_303.0);
    let val = fval + 12_582_912.0f32;
    (val.to_bits() & 0x007f_ffff) as i32 - 0x0040_0000
}

/// `make_qx_quants` with `rmse_type = 1` and no external weights, as used by
/// Q6_K. Returns the block scale.
fn make_qx_quants_rmse1(n: usize, nmax: i32, x: &[f32], l: &mut [u8]) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &value in &x[..n] {
        let ax = value.abs();
        if ax > amax {
            amax = ax;
            max = value;
        }
    }
    if amax < GROUP_MAX_EPS {
        for slot in l[..n].iter_mut() {
            *slot = 0;
        }
        return 0.0;
    }

    let nmaxf = nmax as f32;
    let mut iscale = -nmaxf / max;
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let mut li = nearest_int(iscale * x[i]);
        li = li.clamp(-nmax, nmax - 1);
        l[i] = (li + nmax) as u8;
        let w = x[i] * x[i];
        sumlx += w * x[i] * li as f32;
        suml2 += w * (li * li) as f32;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;

    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmaxf + 0.1 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let mut li = nearest_int(iscale * x[i]);
            li = li.clamp(-nmax, nmax - 1);
            let w = x[i] * x[i];
            sumlx += w * x[i] * li as f32;
            suml2 += w * (li * li) as f32;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let li = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
                l[i] = (li + nmax) as u8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// `make_qkx3_quants`: scale/min search with asymmetric quantization.
#[allow(clippy::too_many_arguments)]
fn make_qkx3_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: Option<&[f32]>,
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights.map_or(x[0] * x[0], |w| w[0]);
    let mut sum_x = sum_w * x[0];

    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights.map_or(x[i] * x[i], |w| w[i]);
        sum_w += w;
        sum_x += w * x[i];
    }

    if min > 0.0 {
        min = 0.0;
    }
    if max <= min {
        for slot in l[..n].iter_mut() {
            *slot = 0;
        }
        *the_min = -min;
        return 0.0;
    }

    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = li as u8;
        let mut diff = scale * li as f32 + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        let w = weights.map_or(x[i] * x[i], |w| w[i]);
        best_mad += w * diff;
    }

    if nstep < 1 {
        *the_min = -min;
        return scale;
    }

    let nmaxf = nmax as f32;
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmaxf) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = li as u8;
            let w = weights.map_or(x[i] * x[i], |w| w[i]);
            sum_l += w * li as f32;
            sum_l2 += w * (li * li) as f32;
            sum_xl += w * li as f32 * x[i];
        }

        let det = sum_w * sum_l2 - sum_l * sum_l;
        if det > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0.0f32;
            for i in 0..n {
                let mut diff = this_scale * laux[i] as f32 + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                let w = weights.map_or(x[i] * x[i], |w| w[i]);
                mad += w * diff;
            }
            if mad < best_mad {
                l[..n].copy_from_slice(&laux[..n]);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }

    *the_min = -min;
    scale
}

/// `make_qp_quants`: optimal positive scale for the per-sub-block scales and
/// minima of Q4_K/Q5_K. Values in `x` are non-negative by construction.
fn make_qp_quants(n: usize, nmax: i32, x: &[f32], l: &mut [u8], quant_weights: &[f32]) -> f32 {
    let mut max = 0.0f32;
    for &value in &x[..n] {
        if value > max {
            max = value;
        }
    }
    if max < GROUP_MAX_EPS {
        for slot in l[..n].iter_mut() {
            *slot = 0;
        }
        return 0.0;
    }

    let nmaxf = nmax as f32;
    let mut iscale = nmaxf / max;
    for i in 0..n {
        l[i] = nearest_int(iscale * x[i]).clamp(0, nmax) as u8;
    }
    let scale = 1.0 / iscale;
    let mut best_mse = 0.0f32;
    for i in 0..n {
        let diff = x[i] - scale * l[i] as f32;
        let w = quant_weights[i];
        best_mse += w * diff * diff;
    }

    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmaxf) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0.0f32;
        for i in 0..n {
            let li = nearest_int(iscale_is * x[i]).clamp(0, nmax);
            let diff = x[i] - scale_is * li as f32;
            let w = quant_weights[i];
            mse += w * diff * diff;
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }

    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * x[i]).clamp(0, nmax);
        l[i] = li as u8;
        let w = quant_weights[i];
        sumlx += w * x[i] * li as f32;
        suml2 += w * (li * li) as f32;
    }

    for _ in 0..5 {
        let mut changed = false;
        for i in 0..n {
            let w = quant_weights[i];
            let li = l[i] as f32;
            let slx = sumlx - w * x[i] * li;
            let sl2 = suml2 - w * li * li;
            if slx > 0.0 && sl2 > 0.0 {
                let new_l = nearest_int(x[i] * sl2 / slx).clamp(0, nmax);
                if new_l as u8 != l[i] {
                    let new_slx = slx + w * x[i] * new_l as f32;
                    let new_sl2 = sl2 + w * (new_l * new_l) as f32;
                    if new_slx * new_slx * suml2 > sumlx * sumlx * new_sl2 {
                        l[i] = new_l as u8;
                        sumlx = new_slx;
                        suml2 = new_sl2;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

fn check_quant_input(x: &[f32], out: &[u8], block_bytes: usize) -> Result<usize, GgufError> {
    if x.len() % QK_K != 0 {
        return Err(GgufError::InvalidRowLength {
            name: "<quantize>".to_string(),
            row_len: x.len(),
            block_elems: QK_K,
        });
    }
    let blocks = x.len() / QK_K;
    let needed = blocks * block_bytes;
    if out.len() < needed {
        return Err(GgufError::BufferTooSmall {
            needed,
            actual: out.len(),
        });
    }
    Ok(blocks)
}

/// Shared body of the Q4_K/Q5_K quantizer: compute sub-block scales, minima and
/// the super-block scale/min, then re-quantize against the quantized values.
///
/// Returns the packed 12-byte scale table, the quantized `d`/`dmin` and the
/// per-element codes.
fn prepare_k_quants(
    x: &[f32],
    nmax: i32,
) -> ([u8; 12], f32, f32, [u8; QK_K]) {
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let mut ls = [0u8; 8];
    let mut lm = [0u8; 8];
    let mut weights = [0f32; 32];
    let mut sw = [0f32; 8];
    let mut mins = [0f32; 8];
    let mut scales = [0f32; 8];

    let mut sum_x2 = 0.0f32;
    for &value in x {
        sum_x2 += value * value;
    }
    let sigma2 = 2.0 * sum_x2 / QK_K as f32;
    let av_x = sigma2.sqrt();

    for j in 0..8 {
        for i in 0..32 {
            weights[i] = av_x + x[32 * j + i].abs();
        }
        let mut sumw = 0.0f32;
        for &w in &weights {
            sumw += w;
        }
        sw[j] = sumw;

        let mut min = 0.0f32;
        scales[j] = make_qkx3_quants(
            32,
            nmax,
            &x[32 * j..32 * j + 32],
            Some(&weights),
            &mut l[32 * j..32 * j + 32],
            &mut min,
            &mut laux,
            -0.9,
            0.05,
            36,
            false,
        );
        mins[j] = min;
    }

    let d_block = make_qp_quants(8, 63, &scales, &mut ls, &sw);
    let m_block = make_qp_quants(8, 63, &mins, &mut lm, &sw);

    // Same bit packing as ggml-quants.c: reproduce it byte for byte.
    let mut packed = [0u8; 12];
    for j in 0..8 {
        let s = ls[j].min(63);
        let m = lm[j].min(63);
        if j < 4 {
            packed[j] = s;
            packed[j + 4] = m;
        } else {
            packed[j + 4] = (s & 0x0f) | ((m & 0x0f) << 4);
            packed[j - 4] |= (s >> 4) << 6;
            packed[j] |= (m >> 4) << 6;
        }
    }

    // Recompute the codes against the fp16-rounded super-block values, exactly
    // as the reference does after writing y[i].d / y[i].dmin.
    let d_quant = f16_to_f32(f32_to_f16(d_block));
    let m_quant = f16_to_f32(f32_to_f16(m_block));
    for j in 0..8 {
        let (sc, m) = get_scale_min_k4(j, &packed);
        let d = d_quant * f32::from(sc);
        if d == 0.0 {
            continue;
        }
        let dm = m_quant * f32::from(m);
        for ii in 0..32 {
            let code = nearest_int((x[32 * j + ii] + dm) / d).clamp(0, nmax);
            l[32 * j + ii] = code as u8;
        }
    }

    (packed, d_quant, m_quant, l)
}

/// Quantize a slice of `f32` values (a multiple of 256 elements) to Q4_K
/// blocks. `out` must hold `x.len() / 256 * 144` bytes.
pub fn quantize_row_q4_k(x: &[f32], out: &mut [u8]) -> Result<(), GgufError> {
    let blocks = check_quant_input(x, out, Q4_K_BLOCK_BYTES)?;
    for b in 0..blocks {
        let xb = &x[b * QK_K..(b + 1) * QK_K];
        let yb = &mut out[b * Q4_K_BLOCK_BYTES..(b + 1) * Q4_K_BLOCK_BYTES];
        let (packed, d, dmin, l) = prepare_k_quants(xb, 15);

        put_f16(yb, 0, d);
        put_f16(yb, 2, dmin);
        yb[4..16].copy_from_slice(&packed);

        for chunk in 0..4 {
            for i in 0..32 {
                let lo = l[chunk * 64 + i] & 0x0f;
                let hi = l[chunk * 64 + i + 32] & 0x0f;
                yb[16 + chunk * 32 + i] = lo | (hi << 4);
            }
        }
    }
    Ok(())
}

/// Quantize a slice of `f32` values (a multiple of 256 elements) to Q5_K
/// blocks. `out` must hold `x.len() / 256 * 176` bytes.
pub fn quantize_row_q5_k(x: &[f32], out: &mut [u8]) -> Result<(), GgufError> {
    let blocks = check_quant_input(x, out, Q5_K_BLOCK_BYTES)?;
    for b in 0..blocks {
        let xb = &x[b * QK_K..(b + 1) * QK_K];
        let yb = &mut out[b * Q5_K_BLOCK_BYTES..(b + 1) * Q5_K_BLOCK_BYTES];
        let (packed, d, dmin, l) = prepare_k_quants(xb, 31);

        put_f16(yb, 0, d);
        put_f16(yb, 2, dmin);
        yb[4..16].copy_from_slice(&packed);
        yb[16..48].fill(0);

        for chunk in 0..4 {
            let m1 = 1u8 << (2 * chunk);
            let m2 = 2u8 << (2 * chunk);
            for i in 0..32 {
                let mut lo = l[chunk * 64 + i];
                let mut hi = l[chunk * 64 + i + 32];
                if lo > 15 {
                    lo -= 16;
                    yb[16 + i] |= m1;
                }
                if hi > 15 {
                    hi -= 16;
                    yb[16 + i] |= m2;
                }
                yb[48 + chunk * 32 + i] = lo | (hi << 4);
            }
        }
    }
    Ok(())
}

/// Quantize a slice of `f32` values (a multiple of 256 elements) to Q6_K
/// blocks. `out` must hold `x.len() / 256 * 210` bytes.
pub fn quantize_row_q6_k(x: &[f32], out: &mut [u8]) -> Result<(), GgufError> {
    let blocks = check_quant_input(x, out, Q6_K_BLOCK_BYTES)?;
    for b in 0..blocks {
        let xb = &x[b * QK_K..(b + 1) * QK_K];
        let yb = &mut out[b * Q6_K_BLOCK_BYTES..(b + 1) * Q6_K_BLOCK_BYTES];
        quantize_q6_k_block(xb, yb);
    }
    Ok(())
}

fn quantize_q6_k_block(x: &[f32], y: &mut [u8]) {
    let mut l = [0u8; QK_K];
    let mut scales = [0f32; 16];

    let mut max_scale = 0.0f32;
    let mut max_abs_scale = 0.0f32;
    for ib in 0..16 {
        let scale =
            make_qx_quants_rmse1(16, 32, &x[16 * ib..16 * ib + 16], &mut l[16 * ib..16 * ib + 16]);
        scales[ib] = scale;
        let abs_scale = scale.abs();
        if abs_scale > max_abs_scale {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }

    if max_abs_scale < GROUP_MAX_EPS {
        y.fill(0);
        return;
    }

    let iscale = -128.0 / max_scale;
    put_f16(y, 208, 1.0 / iscale);
    for ib in 0..16 {
        let code = nearest_int(iscale * scales[ib]).clamp(-128, 127);
        y[192 + ib] = code as i8 as u8;
    }

    let d = f16_to_f32(f32_to_f16(1.0 / iscale));
    for j in 0..16 {
        let dj = d * (y[192 + j] as i8) as f32;
        if dj == 0.0 {
            continue;
        }
        for ii in 0..16 {
            let code = nearest_int(x[16 * j + ii] / dj).clamp(-32, 31) + 32;
            l[16 * j + ii] = code as u8;
        }
    }

    // ql: low nibbles; qh: two high bits per code.
    for n in 0..2 {
        for ll in 0..32 {
            let q1 = l[n * 128 + ll] & 0x0f;
            let q2 = l[n * 128 + ll + 32] & 0x0f;
            let q3 = l[n * 128 + ll + 64] & 0x0f;
            let q4 = l[n * 128 + ll + 96] & 0x0f;
            y[n * 64 + ll] = q1 | (q3 << 4);
            y[n * 64 + 32 + ll] = q2 | (q4 << 4);
            y[128 + n * 32 + ll] = (l[n * 128 + ll] >> 4)
                | ((l[n * 128 + ll + 32] >> 4) << 2)
                | ((l[n * 128 + ll + 64] >> 4) << 4)
                | ((l[n * 128 + ll + 96] >> 4) << 6);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant::dequantize_row;
    use crate::types::{GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};

    fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
                unit * 2.0 - 1.0
            })
            .collect()
    }

    fn normalized_rms_error(original: &[f32], decoded: &[f32]) -> f32 {
        let mut err = 0.0f32;
        let mut sig = 0.0f32;
        for (&a, &b) in original.iter().zip(decoded) {
            let d = a - b;
            err += d * d;
            sig += a * a;
        }
        (err / sig.max(f32::MIN_POSITIVE)).sqrt()
    }

    #[test]
    fn nearest_int_matches_rounding() {
        for value in [-7.6f32, -2.4, -0.4, 0.0, 0.4, 0.6, 1.6, 9.1] {
            assert_eq!(nearest_int(value), value.round() as i32);
        }
        // Exact ties round to even, matching the C reference's magic-number
        // addition under round-to-nearest-even floating point.
        assert_eq!(nearest_int(0.5), 0);
        assert_eq!(nearest_int(1.5), 2);
        assert_eq!(nearest_int(2.5), 2);
        assert_eq!(nearest_int(-0.5), 0);
        assert_eq!(nearest_int(-1.5), -2);
        assert_eq!(nearest_int(-2.5), -2);
    }

    #[test]
    fn quant_roundtrip_q4_k() {
        let x = pseudo_random(512, 0x1234_5678);
        let mut raw = vec![0u8; 2 * Q4_K_BLOCK_BYTES];
        quantize_row_q4_k(&x, &mut raw).unwrap();
        let mut decoded = vec![0f32; x.len()];
        dequantize_row(GGML_TYPE_Q4_K, &raw, &mut decoded).unwrap();
        let error = normalized_rms_error(&x, &decoded);
        // 4.5 bits/element asymmetric quantization; on uniform noise the
        // normalized RMS error is expected around 3-4%. The bound is
        // deliberately loose while still catching packing mistakes.
        assert!(error < 0.06, "Q4_K normalized RMS error too high: {error}");
    }

    #[test]
    fn quant_roundtrip_q5_k() {
        let x = pseudo_random(256, 0x9e37_79b9);
        let mut raw = vec![0u8; Q5_K_BLOCK_BYTES];
        quantize_row_q5_k(&x, &mut raw).unwrap();
        let mut decoded = vec![0f32; x.len()];
        dequantize_row(GGML_TYPE_Q5_K, &raw, &mut decoded).unwrap();
        let error = normalized_rms_error(&x, &decoded);
        // 5.5 bits/element: documented acceptance threshold 3% normalized
        // RMS. The reference-style algorithm lands near 2.6% on this input
        // because a 32-element block spans most of the [-1, 1] range.
        assert!(error < 0.03, "Q5_K normalized RMS error too high: {error}");
    }

    #[test]
    fn quant_roundtrip_q6_k() {
        let x = pseudo_random(256, 0xdead_beef);
        let mut raw = vec![0u8; Q6_K_BLOCK_BYTES];
        quantize_row_q6_k(&x, &mut raw).unwrap();
        let mut decoded = vec![0f32; x.len()];
        dequantize_row(GGML_TYPE_Q6_K, &raw, &mut decoded).unwrap();
        let error = normalized_rms_error(&x, &decoded);
        // 6.56 bits/element: threshold 2% normalized RMS. Uniform noise in
        // [-1, 1] gives about 1.3% with the reference algorithm.
        assert!(error < 0.02, "Q6_K normalized RMS error too high: {error}");
    }

    #[test]
    fn quantizers_reject_bad_sizes() {
        let x = [0f32; 100];
        let mut raw = [0u8; 8];
        assert!(quantize_row_q4_k(&x, &mut raw).is_err());
        assert!(quantize_row_q5_k(&x, &mut raw).is_err());
        assert!(quantize_row_q6_k(&x, &mut raw).is_err());

        let x = [0f32; QK_K];
        assert!(quantize_row_q4_k(&x, &mut raw).is_err());
    }
}
