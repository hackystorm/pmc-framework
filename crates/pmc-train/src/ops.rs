// SPDX-License-Identifier: GPL-3.0-only

//! Primitive forward and backward operations on row-major `f32` slices.
//!
//! The transformer is deliberately built from a small set of clearly
//! specified primitives rather than a general tensor framework. Every
//! backward routine lives next to the forward routine it differentiates, so
//! the two can be compared by eye and by finite differences.
//!
//! The matrix products are batched over the whole sequence and blocked for
//! cache locality. Nothing here loops over sequence positions one at a time:
//! a per-position GEMV against a 128k vocabulary would be memory bound and
//! would make training unusable.

use std::thread;

/// Side of the square cache tile used by the matrix products.
const TILE: usize = 32;

/// The trainer must leave headroom for the rest of the process.
const MAX_WORKERS: usize = 4;

/// Below this many multiply-accumulates, thread setup costs more than it
/// saves. Small gradient checks and unit tests stay single-threaded.
const MIN_PARALLEL_MACS: usize = 1 << 16;

fn worker_count(rows: usize, k: usize, n: usize) -> usize {
    let macs = rows.saturating_mul(k).saturating_mul(n);
    if macs < MIN_PARALLEL_MACS || rows < 2 {
        return 1;
    }
    let hardware = thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    hardware.min(MAX_WORKERS).min(rows).max(1)
}

fn min(a: usize, b: usize) -> usize {
    if a < b {
        a
    } else {
        b
    }
}

/// Dot product with four independent accumulators.
///
/// The fixed accumulation order keeps results reproducible: the same inputs
/// produce the same bits on every run and at every thread count.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let quads = a.len() / 4;
    let mut acc = [0.0f32; 4];
    for q in 0..quads {
        let p = q * 4;
        acc[0] += a[p] * b[p];
        acc[1] += a[p + 1] * b[p + 1];
        acc[2] += a[p + 2] * b[p + 2];
        acc[3] += a[p + 3] * b[p + 3];
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for p in (quads * 4)..a.len() {
        sum += a[p] * b[p];
    }
    sum
}

/// `C = A * B` for row-major matrices.
///
/// `A` is `[m, k]`, `B` is `[k, n]`, `C` is `[m, n]`. The output is tiled in
/// both the inner and outer dimensions. Threads own disjoint rows of `C`;
/// each output element is accumulated by one thread in a fixed order, so the
/// result is deterministic and needs no atomics.
pub fn matmul_tiled(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k, "left matrix does not match its dimensions");
    assert_eq!(b.len(), k * n, "right matrix does not match its dimensions");
    assert_eq!(c.len(), m * n, "output matrix does not match its dimensions");
    if m == 0 || n == 0 {
        return;
    }
    c.fill(0.0);
    if k == 0 {
        return;
    }

    let workers = worker_count(m, k, n);
    if workers == 1 {
        matmul_range(a, b, &mut c[..m * n], 0, k, n);
        return;
    }

    let rows_per_worker = (m + workers - 1) / workers;
    thread::scope(|scope| {
        for (block_index, block) in c[..m * n]
            .chunks_mut(rows_per_worker * n)
            .enumerate()
        {
            let row_start = block_index * rows_per_worker;
            scope.spawn(move || {
                matmul_range(a, b, block, row_start, k, n);
            });
        }
    });
}

fn matmul_range(a: &[f32], b: &[f32], c: &mut [f32], row_start: usize, k: usize, n: usize) {
    let rows = c.len() / n;
    // Block over the output columns and the reduction dimension outside the
    // row loop. This keeps a tile of B hot while every row in this worker
    // consumes it; putting the row on the outside would stream the whole
    // right-hand matrix once per row, which is fatal for the readout.
    for j0 in (0..n).step_by(TILE) {
        let j1 = min(j0 + TILE, n);
        for k0 in (0..k).step_by(TILE) {
            let k1 = min(k0 + TILE, k);
            for local_row in 0..rows {
                let row = row_start + local_row;
                let a_row = &a[row * k..(row + 1) * k];
                let c_row = &mut c[local_row * n..(local_row + 1) * n];
                for p in k0..k1 {
                    let left = a_row[p];
                    if left == 0.0 {
                        continue;
                    }
                    let b_row = &b[p * n + j0..p * n + j1];
                    for (target, right) in c_row[j0..j1].iter_mut().zip(b_row.iter()) {
                        *target += left * *right;
                    }
                }
            }
        }
    }
}

/// `C = A * B^T` for row-major matrices.
///
/// `A` is `[m, k]`, `B` is `[n, k]`, `C` is `[m, n]`. This is the form used
/// by attention scores and by every `dY * W^T` in the backward pass. The
/// readout itself uses a standard product because `V` is stored as
/// `[readout_rank, vocab]`.
pub fn matmul_tiled_bt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k, "left matrix does not match its dimensions");
    assert_eq!(b.len(), n * k, "transposed matrix does not match its dimensions");
    assert_eq!(c.len(), m * n, "output matrix does not match its dimensions");
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        c.fill(0.0);
        return;
    }

    let workers = worker_count(m, k, n);
    if workers == 1 {
        matmul_bt_range(a, b, &mut c[..m * n], 0, k, n);
        return;
    }

    let rows_per_worker = (m + workers - 1) / workers;
    thread::scope(|scope| {
        for (block_index, block) in c[..m * n]
            .chunks_mut(rows_per_worker * n)
            .enumerate()
        {
            let row_start = block_index * rows_per_worker;
            scope.spawn(move || {
                matmul_bt_range(a, b, block, row_start, k, n);
            });
        }
    });
}

fn matmul_bt_range(a: &[f32], b: &[f32], c: &mut [f32], row_start: usize, k: usize, n: usize) {
    let rows = c.len() / n;
    for j0 in (0..n).step_by(TILE) {
        let j1 = min(j0 + TILE, n);
        for local_row in 0..rows {
            let row = row_start + local_row;
            let a_row = &a[row * k..(row + 1) * k];
            let c_row = &mut c[local_row * n..(local_row + 1) * n];
            for j in j0..j1 {
                let b_row = &b[j * k..(j + 1) * k];
                c_row[j] = dot(a_row, b_row);
            }
        }
    }
}

/// `C = A^T * B` for row-major matrices.
///
/// `A` is `[k, m]`, `B` is `[k, n]`, `C` is `[m, n]`. This is the parameter
/// gradient form: `dW = X^T * dY`.
pub fn matmul_atb(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), k * m, "transposed left matrix does not match its dimensions");
    assert_eq!(b.len(), k * n, "right matrix does not match its dimensions");
    assert_eq!(c.len(), m * n, "output matrix does not match its dimensions");
    if m == 0 || n == 0 {
        return;
    }

    let workers = worker_count(m, k, n);
    if workers == 1 {
        matmul_atb_range(a, b, &mut c[..m * n], 0, k, n, m);
        return;
    }

    let rows_per_worker = (m + workers - 1) / workers;
    thread::scope(|scope| {
        for (block_index, block) in c[..m * n]
            .chunks_mut(rows_per_worker * n)
            .enumerate()
        {
            let row_start = block_index * rows_per_worker;
            scope.spawn(move || {
                matmul_atb_range(a, b, block, row_start, k, n, m);
            });
        }
    });
}

fn matmul_atb_range(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    row_start: usize,
    k: usize,
    n: usize,
    m: usize,
) {
    let rows = c.len() / n;
    for j0 in (0..n).step_by(TILE) {
        let j1 = min(j0 + TILE, n);
        for local_row in 0..rows {
            let c_row = &mut c[local_row * n..(local_row + 1) * n];
            for value in c_row[j0..j1].iter_mut() {
                *value = 0.0;
            }
        }
        for p in 0..k {
            let a_row = &a[p * m + row_start..p * m + row_start + rows];
            let b_row = &b[p * n + j0..p * n + j1];
            for local_row in 0..rows {
                let left = a_row[local_row];
                if left == 0.0 {
                    continue;
                }
                let c_row = &mut c[local_row * n..(local_row + 1) * n];
                for (target, right) in c_row[j0..j1].iter_mut().zip(b_row.iter()) {
                    *target += left * *right;
                }
            }
        }
    }
}

/// Root-mean-square normalisation over the last dimension.
///
/// `x` is `[rows, d]`, `w` is `[d]`, `out` is `[rows, d]`, `rstd` is
/// `[rows]`. The reciprocal standard deviation is kept because the backward
/// pass needs it.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32], rstd: &mut [f32]) {
    let d = w.len();
    assert!(d > 0, "rmsnorm needs a non-empty weight vector");
    assert_eq!(x.len() % d, 0, "rmsnorm input is not row aligned");
    assert_eq!(out.len(), x.len(), "rmsnorm output has the wrong size");
    let rows = x.len() / d;
    assert_eq!(rstd.len(), rows, "rmsnorm rstd has the wrong size");

    for row in 0..rows {
        let x_row = &x[row * d..(row + 1) * d];
        let mut sum_squares = 0.0f32;
        for value in x_row {
            sum_squares += *value * *value;
        }
        let inv = 1.0 / (sum_squares / d as f32 + eps).sqrt();
        rstd[row] = inv;
        let out_row = &mut out[row * d..(row + 1) * d];
        for j in 0..d {
            out_row[j] = x_row[j] * inv * w[j];
        }
    }
}

/// Backward pass of [`rmsnorm`].
///
/// `dw` is zeroed and then accumulated, so callers can pass a shared gradient
/// buffer without clearing it first.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_backward(
    dy: &[f32],
    x: &[f32],
    w: &[f32],
    rstd: &[f32],
    dx: &mut [f32],
    dw: &mut [f32],
) {
    let d = w.len();
    assert!(d > 0, "rmsnorm needs a non-empty weight vector");
    assert_eq!(x.len() % d, 0, "rmsnorm input is not row aligned");
    assert_eq!(dy.len(), x.len(), "rmsnorm gradient has the wrong size");
    assert_eq!(dx.len(), x.len(), "rmsnorm dx has the wrong size");
    assert_eq!(dw.len(), d, "rmsnorm dw has the wrong size");
    let rows = x.len() / d;
    assert_eq!(rstd.len(), rows, "rmsnorm rstd has the wrong size");

    dw.fill(0.0);
    for row in 0..rows {
        let x_row = &x[row * d..(row + 1) * d];
        let dy_row = &dy[row * d..(row + 1) * d];
        let dx_row = &mut dx[row * d..(row + 1) * d];
        let inv = rstd[row];

        let mut dot = 0.0f32;
        for j in 0..d {
            dot += dy_row[j] * w[j] * x_row[j];
        }
        let correction = inv * inv * inv * dot / d as f32;
        for j in 0..d {
            dx_row[j] = dy_row[j] * w[j] * inv - correction * x_row[j];
        }
        for j in 0..d {
            dw[j] += dy_row[j] * x_row[j] * inv;
        }
    }
}

/// Rotary position embedding, GPT-NeoX/llama.cpp half-split convention.
///
/// `x` is `[rows, n_head * head_dim]`. Within every head, coordinate `i` is
/// paired with coordinate `i + head_dim / 2`. Row `r` is at position
/// `pos_offset + r`.
pub fn rope(
    x: &[f32],
    n_head: usize,
    head_dim: usize,
    pos_offset: usize,
    theta: f32,
    out: &mut [f32],
) {
    assert!(n_head > 0, "rope needs at least one head");
    assert!(head_dim > 0 && head_dim % 2 == 0, "rope head dimension must be even");
    assert_eq!(x.len(), out.len(), "rope output has the wrong size");
    let width = n_head * head_dim;
    assert_eq!(x.len() % width, 0, "rope input is not head aligned");
    let rows = x.len() / width;
    let half = head_dim / 2;

    for row in 0..rows {
        let position = (pos_offset + row) as f32;
        for i in 0..half {
            let frequency = theta.powf(-((2 * i) as f32) / head_dim as f32);
            let (sin, cos) = (position * frequency).sin_cos();
            for head in 0..n_head {
                let base = row * width + head * head_dim;
                let first = x[base + i];
                let second = x[base + i + half];
                out[base + i] = first * cos - second * sin;
                out[base + i + half] = first * sin + second * cos;
            }
        }
    }
}

/// Backward pass of [`rope`], applying the inverse rotation.
pub fn rope_backward(
    dy: &[f32],
    n_head: usize,
    head_dim: usize,
    pos_offset: usize,
    theta: f32,
    dx: &mut [f32],
) {
    assert!(n_head > 0, "rope needs at least one head");
    assert!(head_dim > 0 && head_dim % 2 == 0, "rope head dimension must be even");
    assert_eq!(dy.len(), dx.len(), "rope dx has the wrong size");
    let width = n_head * head_dim;
    assert_eq!(dy.len() % width, 0, "rope gradient is not head aligned");
    let rows = dy.len() / width;
    let half = head_dim / 2;

    for row in 0..rows {
        let position = (pos_offset + row) as f32;
        for i in 0..half {
            let frequency = theta.powf(-((2 * i) as f32) / head_dim as f32);
            let (sin, cos) = (position * frequency).sin_cos();
            for head in 0..n_head {
                let base = row * width + head * head_dim;
                let first = dy[base + i];
                let second = dy[base + i + half];
                dx[base + i] = first * cos + second * sin;
                dx[base + i + half] = -first * sin + second * cos;
            }
        }
    }
}

/// Causal row-wise softmax over an `[n, n]` score matrix.
///
/// Position `j` is visible to position `i` only when `j <= i`. The maximum is
/// subtracted before exponentiation for numerical stability, and masked
/// entries are written as exact zeros.
pub fn softmax_causal(scores: &[f32], n: usize, out: &mut [f32]) {
    assert_eq!(scores.len(), n * n, "softmax score matrix has the wrong size");
    assert_eq!(out.len(), n * n, "softmax output has the wrong size");

    for row in 0..n {
        let scores_row = &scores[row * n..(row + 1) * n];
        let out_row = &mut out[row * n..(row + 1) * n];

        let mut max = f32::NEG_INFINITY;
        for j in 0..=row {
            if scores_row[j] > max {
                max = scores_row[j];
            }
        }
        let mut sum = 0.0f32;
        for j in 0..=row {
            let value = (scores_row[j] - max).exp();
            out_row[j] = value;
            sum += value;
        }
        for j in (row + 1)..n {
            out_row[j] = 0.0;
        }
        let inv = 1.0 / sum;
        for j in 0..=row {
            out_row[j] *= inv;
        }
    }
}

/// Backward pass of [`softmax_causal`].
///
/// `probs` is the forward output and `dy` is the gradient of the loss with
/// respect to that output. `dx` is overwritten.
pub fn softmax_causal_backward(probs: &[f32], dy: &[f32], n: usize, dx: &mut [f32]) {
    assert_eq!(probs.len(), n * n, "softmax probabilities have the wrong size");
    assert_eq!(dy.len(), n * n, "softmax gradient has the wrong size");
    assert_eq!(dx.len(), n * n, "softmax dx has the wrong size");

    for row in 0..n {
        let p_row = &probs[row * n..(row + 1) * n];
        let dy_row = &dy[row * n..(row + 1) * n];
        let dx_row = &mut dx[row * n..(row + 1) * n];

        let mut dot = 0.0f32;
        for j in 0..=row {
            dot += p_row[j] * dy_row[j];
        }
        for j in 0..=row {
            dx_row[j] = p_row[j] * (dy_row[j] - dot);
        }
        for j in (row + 1)..n {
            dx_row[j] = 0.0;
        }
    }
}

/// SwiGLU activation: `out = silu(gate) * up`.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_eq!(gate.len(), up.len(), "swiglu inputs have different sizes");
    assert_eq!(out.len(), gate.len(), "swiglu output has the wrong size");
    for i in 0..gate.len() {
        let g = gate[i];
        let sigmoid = 1.0 / (1.0 + (-g).exp());
        out[i] = g * sigmoid * up[i];
    }
}

/// Backward pass of [`swiglu`]. Both gradients are overwritten.
pub fn swiglu_backward(
    gate: &[f32],
    up: &[f32],
    dout: &[f32],
    dgate: &mut [f32],
    dup: &mut [f32],
) {
    assert_eq!(gate.len(), up.len(), "swiglu inputs have different sizes");
    assert_eq!(dout.len(), gate.len(), "swiglu gradient has the wrong size");
    assert_eq!(dgate.len(), gate.len(), "swiglu dgate has the wrong size");
    assert_eq!(dup.len(), gate.len(), "swiglu dup has the wrong size");
    for i in 0..gate.len() {
        let g = gate[i];
        let sigmoid = 1.0 / (1.0 + (-g).exp());
        let silu = g * sigmoid;
        // d/dg [g * sigmoid(g)] = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
        let d_silu = sigmoid * (1.0 + g * (1.0 - sigmoid));
        dgate[i] = dout[i] * up[i] * d_silu;
        dup[i] = dout[i] * silu;
    }
}

/// Stable masked cross entropy.
///
/// `logits` is `[rows, vocab]`, `targets` is `[rows]`, `mask` is `[rows]`.
/// Rows whose mask is false contribute neither to the loss nor to the
/// gradient. The returned loss is the mean over the masked rows, and the
/// returned gradient has the same shape as `logits`.
pub fn cross_entropy(logits: &[f32], targets: &[u32], mask: &[bool]) -> (f32, Vec<f32>) {
    let rows = targets.len();
    assert_eq!(mask.len(), rows, "cross entropy mask has the wrong size");
    if rows == 0 {
        assert!(logits.is_empty(), "cross entropy logits have the wrong size");
        return (0.0, Vec::new());
    }
    assert_eq!(logits.len() % rows, 0, "cross entropy logits are not row aligned");
    let vocab = logits.len() / rows;
    assert!(vocab > 0, "cross entropy needs a non-empty vocabulary");

    let mut dlogits = vec![0.0f32; logits.len()];
    let count = mask.iter().filter(|value| **value).count();
    if count == 0 {
        return (0.0, dlogits);
    }

    let inv_count = 1.0 / count as f32;
    let mut loss = 0.0f64;
    for row in 0..rows {
        if !mask[row] {
            continue;
        }
        let target = targets[row] as usize;
        assert!(target < vocab, "cross entropy target is outside the vocabulary");
        let logits_row = &logits[row * vocab..(row + 1) * vocab];

        let mut max = f32::NEG_INFINITY;
        for value in logits_row {
            if *value > max {
                max = *value;
            }
        }
        let mut sum = 0.0f32;
        for value in logits_row {
            sum += (*value - max).exp();
        }
        let log_sum = sum.ln();
        // log_softmax(target) = (target - max) - log(sum(exp(x - max)))
        loss += -((logits_row[target] - max) - log_sum) as f64;

        let inv_sum = 1.0 / sum;
        let grad_row = &mut dlogits[row * vocab..(row + 1) * vocab];
        for j in 0..vocab {
            let probability = (logits_row[j] - max).exp() * inv_sum;
            grad_row[j] = probability * inv_count;
        }
        grad_row[target] -= inv_count;
    }

    ((loss / count as f64) as f32, dlogits)
}

/// Clip a collection of gradient slices by their global L2 norm.
///
/// Returns the norm before clipping. A non-positive `max_norm` disables
/// clipping, which is useful when the caller wants only the measurement.
pub fn grad_clip(params_grads: &mut [&mut [f32]], max_norm: f32) -> f32 {
    let mut sum_squares = 0.0f64;
    for grad in params_grads.iter() {
        for value in grad.iter() {
            let item = *value as f64;
            sum_squares += item * item;
        }
    }
    let norm = sum_squares.sqrt() as f32;
    if max_norm > 0.0 && norm > max_norm {
        let scale = max_norm / norm;
        for grad in params_grads.iter_mut() {
            for value in grad.iter_mut() {
                *value *= scale;
            }
        }
    }
    norm
}

/// One Adam update for a single parameter slice.
///
/// `step` is one-based. The caller owns the moment buffers, which makes this
/// usable for large matrices without building a parallel tensor container.
#[allow(clippy::too_many_arguments)]
pub fn adam_step(
    param: &mut [f32],
    grad: &[f32],
    first_moment: &mut [f32],
    second_moment: &mut [f32],
    step: u32,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
) {
    assert_eq!(param.len(), grad.len(), "adam parameter and gradient differ in size");
    assert_eq!(param.len(), first_moment.len(), "adam first moment has the wrong size");
    assert_eq!(param.len(), second_moment.len(), "adam second moment has the wrong size");
    assert!(step >= 1, "adam step must be one-based");

    let bias_correction1 = 1.0 - beta1.powi(step as i32);
    let bias_correction2 = 1.0 - beta2.powi(step as i32);
    for i in 0..param.len() {
        let g = grad[i];
        first_moment[i] = beta1 * first_moment[i] + (1.0 - beta1) * g;
        second_moment[i] = beta2 * second_moment[i] + (1.0 - beta2) * g * g;
        let m_hat = first_moment[i] / bias_correction1;
        let v_hat = second_moment[i] / bias_correction2;
        param[i] -= learning_rate * m_hat / (v_hat.sqrt() + eps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmc_core::Rng;

    const EPS: f64 = 1e-5;

    fn random_vec(rng: &mut Rng, len: usize) -> Vec<f32> {
        (0..len).map(|_| rng.uniform(-1.0, 1.0)).collect()
    }

    fn to_f64(values: &[f32]) -> Vec<f64> {
        values.iter().map(|value| *value as f64).collect()
    }

    fn numeric_gradient<F>(params: &[f64], mut loss: F, eps: f64) -> Vec<f64>
    where
        F: FnMut(&[f64]) -> f64,
    {
        let mut point = params.to_vec();
        let mut gradient = vec![0.0f64; params.len()];
        for i in 0..params.len() {
            let original = point[i];
            point[i] = original + eps;
            let plus = loss(&point);
            point[i] = original - eps;
            let minus = loss(&point);
            point[i] = original;
            gradient[i] = (plus - minus) / (2.0 * eps);
        }
        gradient
    }

    fn assert_relative(analytic: &[f32], numeric: &[f64], label: &str) {
        assert_eq!(analytic.len(), numeric.len(), "{} length mismatch", label);
        let mut worst = 0.0f64;
        let mut worst_index = 0usize;
        for i in 0..analytic.len() {
            let a = analytic[i] as f64;
            let n = numeric[i];
            let denominator = a.abs().max(n.abs()).max(1e-4);
            let error = (a - n).abs() / denominator;
            if error > worst {
                worst = error;
                worst_index = i;
            }
        }
        assert!(
            worst < 1e-2,
            "{}: relative error {:.3e} at index {} (analytic {}, numeric {})",
            label,
            worst,
            worst_index,
            analytic[worst_index],
            numeric[worst_index],
        );
    }

    fn matmul_reference(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; m * n];
        for i in 0..m {
            for p in 0..k {
                let left = a[i * k + p];
                for j in 0..n {
                    out[i * n + j] += left * b[p * n + j];
                }
            }
        }
        out
    }

    #[test]
    fn matrix_products_match_a_naive_reference() {
        let mut rng = Rng::new(11);
        for &(m, k, n) in &[(1usize, 1usize, 1usize), (3, 5, 7), (40, 33, 71)] {
            let a = random_vec(&mut rng, m * k);
            let b = random_vec(&mut rng, k * n);
            let bt = random_vec(&mut rng, n * k);
            let a64 = to_f64(&a);
            let b64 = to_f64(&b);
            let bt64 = to_f64(&bt);

            let mut c = vec![0.0f32; m * n];
            matmul_tiled(&a, &b, &mut c, m, k, n);
            let expected = matmul_reference(&a64, &b64, m, k, n);
            assert_relative(&c, &expected, "matmul_tiled");

            let mut c_bt = vec![0.0f32; m * n];
            matmul_tiled_bt(&a, &bt, &mut c_bt, m, k, n);
            let mut bt_t = vec![0.0f64; k * n];
            for i in 0..n {
                for p in 0..k {
                    bt_t[p * n + i] = bt64[i * k + p];
                }
            }
            let expected_bt = matmul_reference(&a64, &bt_t, m, k, n);
            assert_relative(&c_bt, &expected_bt, "matmul_tiled_bt");

            let a_atb = random_vec(&mut rng, k * m);
            let a_atb64 = to_f64(&a_atb);
            let mut c_atb = vec![0.0f32; m * n];
            matmul_atb(&a_atb, &b, &mut c_atb, m, k, n);
            let mut explicit = vec![0.0f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for p in 0..k {
                        sum += a_atb64[p * m + i] * b64[p * n + j];
                    }
                    explicit[i * n + j] = sum;
                }
            }
            assert_relative(&c_atb, &explicit, "matmul_atb");
        }
    }

    #[test]
    fn gradient_rmsnorm_matches_finite_differences() {
        let mut rng = Rng::new(21);
        let rows = 3usize;
        let d = 6usize;
        let x = random_vec(&mut rng, rows * d);
        let w = random_vec(&mut rng, d).iter().map(|v| v * 0.5 + 0.75).collect::<Vec<_>>();
        let dy = random_vec(&mut rng, rows * d);

        let mut out = vec![0.0f32; rows * d];
        let mut rstd = vec![0.0f32; rows];
        rmsnorm(&x, &w, EPS as f32, &mut out, &mut rstd);
        let mut dx = vec![0.0f32; rows * d];
        let mut dw = vec![0.0f32; d];
        rmsnorm_backward(&dy, &x, &w, &rstd, &mut dx, &mut dw);

        let x64 = to_f64(&x);
        let w64 = to_f64(&w);
        let dy64 = to_f64(&dy);
        let mut params = x64.clone();
        params.extend_from_slice(&w64);
        let reference = |point: &[f64]| -> f64 {
            let mut out = vec![0.0f64; rows * d];
            for row in 0..rows {
                let mut sum = 0.0f64;
                for j in 0..d {
                    sum += point[row * d + j] * point[row * d + j];
                }
                let inv = 1.0 / (sum / d as f64 + EPS).sqrt();
                for j in 0..d {
                    out[row * d + j] = point[row * d + j] * inv * point[rows * d + j];
                }
            }
            out.iter().zip(dy64.iter()).map(|(a, b)| a * b).sum()
        };
        let numeric = numeric_gradient(&params, reference, 1e-3);

        let mut analytic = dx.clone();
        analytic.extend_from_slice(&dw);
        assert_relative(&analytic, &numeric, "rmsnorm");
    }

    fn rope_reference(x: &[f64], n_head: usize, head_dim: usize, pos_offset: usize, theta: f64) -> Vec<f64> {
        let width = n_head * head_dim;
        let half = head_dim / 2;
        let rows = x.len() / width;
        let mut out = vec![0.0f64; x.len()];
        for row in 0..rows {
            let position = (pos_offset + row) as f64;
            for head in 0..n_head {
                let base = row * width + head * head_dim;
                for i in 0..half {
                    let frequency = theta.powf(-((2 * i) as f64) / head_dim as f64);
                    let angle = position * frequency;
                    let (sin, cos) = angle.sin_cos();
                    let first = x[base + i];
                    let second = x[base + i + half];
                    out[base + i] = first * cos - second * sin;
                    out[base + i + half] = first * sin + second * cos;
                }
            }
        }
        out
    }

    #[test]
    fn gradient_rope_matches_finite_differences() {
        let mut rng = Rng::new(22);
        let n_head = 2usize;
        let head_dim = 4usize;
        let rows = 3usize;
        let width = n_head * head_dim;
        let theta = 10_000.0f32;
        let x = random_vec(&mut rng, rows * width);
        let dy = random_vec(&mut rng, rows * width);

        let mut out = vec![0.0f32; rows * width];
        rope(&x, n_head, head_dim, 2, theta, &mut out);
        let mut dx = vec![0.0f32; rows * width];
        rope_backward(&dy, n_head, head_dim, 2, theta, &mut dx);

        let x64 = to_f64(&x);
        let dy64 = to_f64(&dy);
        let reference = |point: &[f64]| -> f64 {
            let out = rope_reference(point, n_head, head_dim, 2, theta as f64);
            out.iter().zip(dy64.iter()).map(|(a, b)| a * b).sum()
        };
        let numeric = numeric_gradient(&x64, reference, 1e-3);
        assert_relative(&dx, &numeric, "rope");
    }

    fn attention_reference(
        q: &[f64],
        k: &[f64],
        v: &[f64],
        seq: usize,
        head_dim: usize,
    ) -> Vec<f64> {
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut out = vec![0.0f64; seq * head_dim];
        for i in 0..seq {
            let mut scores = vec![f64::NEG_INFINITY; seq];
            let mut max = f64::NEG_INFINITY;
            for j in 0..=i {
                let mut dot = 0.0f64;
                for p in 0..head_dim {
                    dot += q[i * head_dim + p] * k[j * head_dim + p];
                }
                scores[j] = dot * scale;
                if scores[j] > max {
                    max = scores[j];
                }
            }
            let mut sum = 0.0f64;
            for j in 0..=i {
                scores[j] = (scores[j] - max).exp();
                sum += scores[j];
            }
            for j in 0..=i {
                scores[j] /= sum;
            }
            for p in 0..head_dim {
                let mut value = 0.0f64;
                for j in 0..=i {
                    value += scores[j] * v[j * head_dim + p];
                }
                out[i * head_dim + p] = value;
            }
        }
        out
    }

    #[test]
    fn gradient_causal_attention_matches_finite_differences() {
        let mut rng = Rng::new(23);
        let seq = 4usize;
        let head_dim = 3usize;
        let q = random_vec(&mut rng, seq * head_dim);
        let k = random_vec(&mut rng, seq * head_dim);
        let v = random_vec(&mut rng, seq * head_dim);
        let dy = random_vec(&mut rng, seq * head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut scores = vec![0.0f32; seq * seq];
        matmul_tiled_bt(&q, &k, &mut scores, seq, head_dim, seq);
        for value in scores.iter_mut() {
            *value *= scale;
        }
        let mut probs = vec![0.0f32; seq * seq];
        softmax_causal(&scores, seq, &mut probs);
        let mut out = vec![0.0f32; seq * head_dim];
        matmul_tiled(&probs, &v, &mut out, seq, seq, head_dim);

        let mut d_probs = vec![0.0f32; seq * seq];
        matmul_tiled_bt(&dy, &v, &mut d_probs, seq, head_dim, seq);
        let mut d_v = vec![0.0f32; seq * head_dim];
        matmul_atb(&probs, &dy, &mut d_v, seq, seq, head_dim);
        let mut d_scores = vec![0.0f32; seq * seq];
        softmax_causal_backward(&probs, &d_probs, seq, &mut d_scores);
        let mut d_q = vec![0.0f32; seq * head_dim];
        matmul_tiled(&d_scores, &k, &mut d_q, seq, seq, head_dim);
        let mut d_k = vec![0.0f32; seq * head_dim];
        matmul_atb(&d_scores, &q, &mut d_k, seq, seq, head_dim);
        for value in d_q.iter_mut() {
            *value *= scale;
        }
        for value in d_k.iter_mut() {
            *value *= scale;
        }

        let mut params = to_f64(&q);
        params.extend_from_slice(&to_f64(&k));
        params.extend_from_slice(&to_f64(&v));
        let dy64 = to_f64(&dy);
        let reference = |point: &[f64]| -> f64 {
            let q = &point[0..seq * head_dim];
            let k = &point[seq * head_dim..2 * seq * head_dim];
            let v = &point[2 * seq * head_dim..3 * seq * head_dim];
            let out = attention_reference(q, k, v, seq, head_dim);
            out.iter().zip(dy64.iter()).map(|(a, b)| a * b).sum()
        };
        let numeric = numeric_gradient(&params, reference, 1e-3);

        let mut analytic = d_q;
        analytic.extend_from_slice(&d_k);
        analytic.extend_from_slice(&d_v);
        assert_relative(&analytic, &numeric, "causal attention");
    }

    fn silu(value: f64) -> f64 {
        value / (1.0 + (-value).exp())
    }

    #[test]
    fn gradient_swiglu_matches_finite_differences() {
        let mut rng = Rng::new(24);
        let len = 7usize;
        let gate = random_vec(&mut rng, len);
        let up = random_vec(&mut rng, len);
        let dy = random_vec(&mut rng, len);

        let mut out = vec![0.0f32; len];
        swiglu(&gate, &up, &mut out);
        let mut dgate = vec![0.0f32; len];
        let mut dup = vec![0.0f32; len];
        swiglu_backward(&gate, &up, &dy, &mut dgate, &mut dup);

        let mut params = to_f64(&gate);
        params.extend_from_slice(&to_f64(&up));
        let dy64 = to_f64(&dy);
        let reference = |point: &[f64]| -> f64 {
            let mut loss = 0.0f64;
            for i in 0..len {
                let gate = point[i];
                let up = point[len + i];
                loss += dy64[i] * silu(gate) * up;
            }
            loss
        };
        let numeric = numeric_gradient(&params, reference, 1e-3);

        let mut analytic = dgate;
        analytic.extend_from_slice(&dup);
        assert_relative(&analytic, &numeric, "swiglu");
    }

    fn cross_entropy_reference(
        logits: &[f64],
        targets: &[u32],
        mask: &[bool],
        vocab: usize,
    ) -> f64 {
        let count = mask.iter().filter(|value| **value).count();
        if count == 0 {
            return 0.0;
        }
        let mut loss = 0.0f64;
        for row in 0..targets.len() {
            if !mask[row] {
                continue;
            }
            let mut max = f64::NEG_INFINITY;
            for j in 0..vocab {
                max = max.max(logits[row * vocab + j]);
            }
            let mut sum = 0.0f64;
            for j in 0..vocab {
                sum += (logits[row * vocab + j] - max).exp();
            }
            loss += -((logits[row * vocab + targets[row] as usize] - max) - sum.ln());
        }
        loss / count as f64
    }

    #[test]
    fn gradient_cross_entropy_matches_finite_differences() {
        let mut rng = Rng::new(25);
        let rows = 4usize;
        let vocab = 7usize;
        let logits = random_vec(&mut rng, rows * vocab)
            .iter()
            .map(|value| value * 2.0)
            .collect::<Vec<_>>();
        let targets = vec![2u32, 0, 6, 3];
        let mask = vec![true, false, true, true];

        let (loss, dlogits) = cross_entropy(&logits, &targets, &mask);
        let logits64 = to_f64(&logits);
        let expected_loss = cross_entropy_reference(&logits64, &targets, &mask, vocab);
        assert!(
            (loss as f64 - expected_loss).abs() < 1e-5,
            "cross entropy loss {} != {}",
            loss,
            expected_loss
        );

        let reference = |point: &[f64]| cross_entropy_reference(point, &targets, &mask, vocab);
        let numeric = numeric_gradient(&logits64, reference, 1e-3);
        assert_relative(&dlogits, &numeric, "cross entropy");
    }

    #[test]
    fn causal_softmax_never_attends_to_the_future() {
        let scores = vec![
            1.0f32, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ];
        let mut probs = vec![0.0f32; 9];
        softmax_causal(&scores, 3, &mut probs);
        for row in 0..3 {
            for j in 0..3 {
                if j > row {
                    assert_eq!(probs[row * 3 + j], 0.0);
                }
            }
            let sum: f32 = probs[row * 3..(row + 1) * 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn grad_clip_scales_by_the_global_norm() {
        let mut first = vec![3.0f32, 0.0];
        let mut second = vec![0.0f32, 4.0];
        let mut grads: Vec<&mut [f32]> = vec![&mut first, &mut second];
        let norm = grad_clip(&mut grads, 1.0);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((first[0] - 0.6).abs() < 1e-6);
        assert!((second[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    #[ignore = "prints a throughput number on demand"]
    fn matmul_throughput() {
        use std::time::Instant;

        let m = 256usize;
        let k = 1024usize;
        let n = 1024usize;
        let mut rng = Rng::new(99);
        let a = random_vec(&mut rng, m * k);
        let b = random_vec(&mut rng, k * n);
        let mut c = vec![0.0f32; m * n];

        // Warm the caches and the allocator, then time a single product.
        matmul_tiled(&a, &b, &mut c, m, k, n);
        let started = Instant::now();
        matmul_tiled(&a, &b, &mut c, m, k, n);
        let elapsed = started.elapsed().as_secs_f64();
        let macs = (m as f64) * (k as f64) * (n as f64);
        let gmacs = macs / elapsed / 1e9;
        println!("matmul_tiled {}x{}x{}: {:.3} GMAC/s", m, k, n, gmacs);
    }
}
