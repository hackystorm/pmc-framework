// SPDX-License-Identifier: GPL-3.0-only

//! Bit-exact dequantization of GGUF tensor rows.
//!
//! The block layouts and the arithmetic follow the C reference in
//! `llama.cpp/ggml/src/ggml-quants.c` (`dequantize_row_*`) and the struct
//! definitions in `ggml/src/ggml-common.h`. Block sizes agree with
//! `gguf-py/gguf/constants.py::GGML_QUANT_SIZES`.
//!
//! Supported types:
//!
//! | id | type | block elements | block bytes |
//! |----|------|----------------|-------------|
//! | 0  | F32  | 1              | 4           |
//! | 1  | F16  | 1              | 2           |
//! | 8  | Q8_0 | 32             | 34          |
//! | 12 | Q4_K | 256            | 144         |
//! | 13 | Q5_K | 256            | 176         |
//! | 14 | Q6_K | 256            | 210         |

use crate::types::{
    quant_block, GgufError, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, QK_K,
};

/// Bytes of a Q4_K block: half d, half dmin, 12 scale bytes, 128 nibbles.
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// Bytes of a Q5_K block: half d, half dmin, 12 scale bytes, 32 high bits, 128 nibbles.
pub const Q5_K_BLOCK_BYTES: usize = 176;
/// Bytes of a Q6_K block: 128 low nibbles, 64 high bits, 16 scales, half d.
pub const Q6_K_BLOCK_BYTES: usize = 210;
/// Bytes of a Q8_0 block: half d, 32 signed bytes.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Convert an IEEE 754 binary16 bit pattern to `f32`.
///
/// Handles zero, subnormals, infinities and NaNs without lookup tables.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x03ff);

    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal half. Normalize the mantissa for binary32.
            let mut m = mant;
            let mut e: i32 = 113; // 127 - 15 + 1
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            sign | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

/// Convert `f32` to IEEE 754 binary16 with round-to-nearest-even.
///
/// This mirrors `GGML_FP32_TO_FP16` for finite values, preserves infinities
/// and converts NaNs to quiet NaNs.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf or NaN; set the quiet bit for NaNs.
        let payload = if mant != 0 {
            ((mant >> 13) as u16) | 0x0200
        } else {
            0
        };
        return sign | 0x7c00 | payload;
    }

    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }

    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut half = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (half & 1) == 1) {
            half += 1;
        }
        return sign | half;
    }

    let mut half = u32::from(sign) | ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        half += 1;
    }
    half as u16
}

/// Dequantize `src` into `out`, using the GGML quantization type id `kind`.
///
/// `out.len()` must be a multiple of the type's block size. Any extra bytes in
/// `src` beyond the number of blocks are ignored.
pub fn dequantize_row(kind: u32, src: &[u8], out: &mut [f32]) -> Result<(), GgufError> {
    let (block_elems, block_bytes) = quant_block(kind).ok_or(GgufError::UnsupportedQuantType {
        name: "<row>".to_string(),
        kind,
    })?;

    if out.len() % block_elems != 0 {
        return Err(GgufError::InvalidRowLength {
            name: "<row>".to_string(),
            row_len: out.len(),
            block_elems,
        });
    }
    let needed = (out.len() / block_elems) * block_bytes;
    if src.len() < needed {
        return Err(GgufError::BufferTooSmall {
            needed,
            actual: src.len(),
        });
    }

    match kind {
        GGML_TYPE_F32 => dequantize_f32(src, out),
        GGML_TYPE_F16 => dequantize_f16(src, out),
        GGML_TYPE_Q8_0 => dequantize_q8_0(src, out),
        GGML_TYPE_Q4_K => dequantize_q4_k(src, out),
        GGML_TYPE_Q5_K => dequantize_q5_k(src, out),
        GGML_TYPE_Q6_K => dequantize_q6_k(src, out),
        _ => unreachable!("quant_block and the match arms must stay in sync"),
    }
    Ok(())
}

fn dequantize_f32(src: &[u8], out: &mut [f32]) {
    for (dst, chunk) in out.iter_mut().zip(src.chunks_exact(4)) {
        *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
}

fn dequantize_f16(src: &[u8], out: &mut [f32]) {
    for (dst, chunk) in out.iter_mut().zip(src.chunks_exact(2)) {
        *dst = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
}

fn half_at(src: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]))
}

/// Unpack 6-bit scale and minimum for sub-block `j` of a K-quant super-block.
///
/// This is `get_scale_min_k4` from `ggml-quants.c`. Scales and minima are
/// packed in 12 bytes as eight 6-bit values each.
pub(crate) fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn dequantize_q8_0(src: &[u8], out: &mut [f32]) {
    for (b, dst) in out.chunks_exact_mut(32).enumerate() {
        let base = b * Q8_0_BLOCK_BYTES;
        let d = half_at(src, base);
        for (l, value) in dst.iter_mut().enumerate() {
            let q = src[base + 2 + l] as i8;
            *value = d * f32::from(q);
        }
    }
}

fn dequantize_q4_k(src: &[u8], out: &mut [f32]) {
    for (b, dst) in out.chunks_exact_mut(QK_K).enumerate() {
        let base = b * Q4_K_BLOCK_BYTES;
        let d = half_at(src, base);
        let dmin = half_at(src, base + 2);
        let scales = &src[base + 4..base + 16];
        let qs = &src[base + 16..base + 144];

        for (chunk, piece) in dst.chunks_exact_mut(64).enumerate() {
            let is = 2 * chunk;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * f32::from(sc1);
            let m1f = dmin * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let m2f = dmin * f32::from(m2);
            let q = &qs[chunk * 32..chunk * 32 + 32];

            for l in 0..32 {
                piece[l] = d1 * f32::from(q[l] & 0x0f) - m1f;
                piece[l + 32] = d2 * f32::from(q[l] >> 4) - m2f;
            }
        }
    }
}

fn dequantize_q5_k(src: &[u8], out: &mut [f32]) {
    for (b, dst) in out.chunks_exact_mut(QK_K).enumerate() {
        let base = b * Q5_K_BLOCK_BYTES;
        let d = half_at(src, base);
        let dmin = half_at(src, base + 2);
        let scales = &src[base + 4..base + 16];
        let qh = &src[base + 16..base + 48];
        let qs = &src[base + 48..base + 176];

        for (chunk, piece) in dst.chunks_exact_mut(64).enumerate() {
            let is = 2 * chunk;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * f32::from(sc1);
            let m1f = dmin * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let m2f = dmin * f32::from(m2);
            let q = &qs[chunk * 32..chunk * 32 + 32];
            let h = qh;
            let u1 = 1u8 << (2 * chunk);
            let u2 = 2u8 << (2 * chunk);

            for l in 0..32 {
                let lo = (q[l] & 0x0f) + if h[l] & u1 != 0 { 16 } else { 0 };
                let hi = (q[l] >> 4) + if h[l] & u2 != 0 { 16 } else { 0 };
                piece[l] = d1 * f32::from(lo) - m1f;
                piece[l + 32] = d2 * f32::from(hi) - m2f;
            }
        }
    }
}

fn dequantize_q6_k(src: &[u8], out: &mut [f32]) {
    for (b, dst) in out.chunks_exact_mut(QK_K).enumerate() {
        let base = b * Q6_K_BLOCK_BYTES;
        let ql = &src[base..base + 128];
        let qh = &src[base + 128..base + 192];
        let scales = &src[base + 192..base + 208];
        let d = half_at(src, base + 208);

        for n in 0..2 {
            let ql = &ql[n * 64..n * 64 + 64];
            let qh = &qh[n * 32..n * 32 + 32];
            let scales = &scales[n * 8..n * 8 + 8];
            let dst = &mut dst[n * 128..n * 128 + 128];

            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0f) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;

                let s0 = f32::from(scales[is] as i8);
                let s1 = f32::from(scales[is + 2] as i8);
                let s2 = f32::from(scales[is + 4] as i8);
                let s3 = f32::from(scales[is + 6] as i8);

                dst[l] = d * s0 * q1 as f32;
                dst[l + 32] = d * s1 * q2 as f32;
                dst[l + 64] = d * s2 * q3 as f32;
                dst[l + 96] = d * s3 * q4 as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_f16(buf: &mut [u8], off: usize, value: f32) {
        buf[off..off + 2].copy_from_slice(&f32_to_f16(value).to_le_bytes());
    }

    #[test]
    fn f16_roundtrip_and_edges() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        // Smallest positive subnormal half = 2^-24.
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0xfc00).is_infinite() && f16_to_f32(0xfc00) < 0.0);
        assert!(f16_to_f32(0x7e00).is_nan());
        // Rounding: 1 + 2^-11 is exactly halfway between 1.0 and the next half;
        // ties-to-even keeps 1.0.
        let halfway = 1.0f32 + 2.0f32.powi(-11);
        assert_eq!(f32_to_f16(halfway), 0x3c00);
        let above = 1.0f32 + 2.0f32.powi(-11) + 2.0f32.powi(-23);
        assert_eq!(f32_to_f16(above), 0x3c01);
    }

    #[test]
    fn dequant_known_block_q4_k() {
        // Hand-built block: d = 0.5, dmin = 0.25, every sub-block scale = 2
        // and minimum = 1, so d1 = 1.0 and m1 = 0.25.
        let mut block = [0u8; Q4_K_BLOCK_BYTES];
        put_f16(&mut block, 0, 0.5);
        put_f16(&mut block, 2, 0.25);
        for j in 0..4 {
            block[4 + j] = 2; // sub-block scales 0..3
            block[8 + j] = 1; // sub-block minima 0..3
        }
        // Sub-blocks 4..7 take the scale from the low nibble and the minimum
        // from the high nibble of scales[8..12], so 0x12 means scale 2/min 1.
        for j in 0..4 {
            block[12 + j] = 0x12;
        }
        for chunk in 0..4 {
            for l in 0..32 {
                let lo = ((l + chunk) & 0x0f) as u8;
                let hi = 15 - lo;
                block[16 + chunk * 32 + l] = lo | (hi << 4);
            }
        }

        let mut out = [0f32; QK_K];
        dequantize_row(GGML_TYPE_Q4_K, &block, &mut out).unwrap();

        for chunk in 0..4 {
            for l in 0..32 {
                let lo = ((l + chunk) & 0x0f) as f32;
                let hi = 15.0 - lo;
                // y = d * sc * q - dmin * m
                assert_eq!(out[chunk * 64 + l], 1.0 * lo - 0.25, "chunk {chunk} lo {l}");
                assert_eq!(out[chunk * 64 + 32 + l], 1.0 * hi - 0.25, "chunk {chunk} hi {l}");
            }
        }
    }

    #[test]
    fn dequant_known_block_q5_k() {
        // Same super-block scales as Q4_K, plus the extra fifth bit. qh is
        // 0xAA: bits 1,3,5,7 are set, so every high nibble gets +16 while no
        // low nibble does.
        let mut block = [0u8; Q5_K_BLOCK_BYTES];
        put_f16(&mut block, 0, 0.5);
        put_f16(&mut block, 2, 0.25);
        for j in 0..4 {
            block[4 + j] = 2;
            block[8 + j] = 1;
        }
        for j in 0..4 {
            block[12 + j] = 0x12;
        }
        for l in 0..32 {
            block[16 + l] = 0xaa;
        }
        for chunk in 0..4 {
            for l in 0..32 {
                let lo = ((l + chunk) & 0x0f) as u8;
                let hi = 15 - lo;
                block[48 + chunk * 32 + l] = lo | (hi << 4);
            }
        }

        let mut out = [0f32; QK_K];
        dequantize_row(GGML_TYPE_Q5_K, &block, &mut out).unwrap();

        for chunk in 0..4 {
            for l in 0..32 {
                let lo = ((l + chunk) & 0x0f) as f32;
                let hi = 15.0 - lo;
                // Low nibble: q + 0. High nibble: q + 16.
                assert_eq!(out[chunk * 64 + l], lo - 0.25, "chunk {chunk} lo {l}");
                assert_eq!(out[chunk * 64 + 32 + l], hi + 16.0 - 0.25, "chunk {chunk} hi {l}");
            }
        }
    }

    #[test]
    fn dequant_known_block_q6_k() {
        // d = 0.5; first half of the scales are 2 (factor 1.0), second half
        // are 3 (factor 1.5). qh is zero, so codes come only from ql.
        let mut block = [0u8; Q6_K_BLOCK_BYTES];
        for l in 0..32 {
            block[l] = ((l % 16) as u8) | (3 << 4);
            block[32 + l] = 1 | (2 << 4);
            block[64 + l] = 4 | (5 << 4);
            block[96 + l] = 6 | (7 << 4);
        }
        for j in 0..8 {
            block[192 + j] = 2;
            block[200 + j] = 3;
        }
        put_f16(&mut block, 208, 0.5);

        let mut out = [0f32; QK_K];
        dequantize_row(GGML_TYPE_Q6_K, &block, &mut out).unwrap();

        for l in 0..32 {
            let q1 = (l % 16) as f32;
            let q2 = 1.0;
            let q3 = 3.0;
            let q4 = 2.0;
            assert_eq!(out[l], 0.5 * 2.0 * (q1 - 32.0));
            assert_eq!(out[32 + l], 0.5 * 2.0 * (q2 - 32.0));
            assert_eq!(out[64 + l], 0.5 * 2.0 * (q3 - 32.0));
            assert_eq!(out[96 + l], 0.5 * 2.0 * (q4 - 32.0));

            let q1 = 4.0;
            let q2 = 6.0;
            let q3 = 5.0;
            let q4 = 7.0;
            assert_eq!(out[128 + l], 0.5 * 3.0 * (q1 - 32.0));
            assert_eq!(out[160 + l], 0.5 * 3.0 * (q2 - 32.0));
            assert_eq!(out[192 + l], 0.5 * 3.0 * (q3 - 32.0));
            assert_eq!(out[224 + l], 0.5 * 3.0 * (q4 - 32.0));
        }
    }

    #[test]
    fn dequant_known_block_q8_0() {
        let mut block = [0u8; Q8_0_BLOCK_BYTES];
        put_f16(&mut block, 0, 0.5);
        for (l, value) in block[2..].iter_mut().enumerate() {
            *value = (l as i32 - 16) as i8 as u8;
        }
        let mut out = [0f32; 32];
        dequantize_row(GGML_TYPE_Q8_0, &block, &mut out).unwrap();
        for (l, value) in out.iter().enumerate() {
            assert_eq!(*value, 0.5 * (l as f32 - 16.0));
        }
    }

    #[test]
    fn dequant_rejects_short_source() {
        let block = [0u8; 10];
        let mut out = [0f32; QK_K];
        assert!(matches!(
            dequantize_row(GGML_TYPE_Q4_K, &block, &mut out),
            Err(GgufError::BufferTooSmall { .. })
        ));
    }
}
