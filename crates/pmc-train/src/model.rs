// SPDX-License-Identifier: GPL-3.0-only

//! The trainable model: a two-layer perceptron with explicit gradients.
//!
//! Explicit gradients rather than an autodiff tape, on purpose. The
//! architecture is fixed and small, so a tape would add machinery without
//! adding flexibility, and hand-written gradients are easier to audit
//! against the Python reference implementation that the parity test uses.

use pmc_core::{Rng, Tensor};

pub struct Mlp {
    pub w1: Tensor,
    pub b1: Tensor,
    pub w2: Tensor,
    pub b2: Tensor,
    pub d_in: usize,
    pub d_hidden: usize,
    pub d_out: usize,
}

pub struct ForwardCache {
    pub x: Tensor,
    pub z1: Tensor,
    pub h: Tensor,
    pub probs: Tensor,
}

pub struct Grads {
    pub dw1: Tensor,
    pub db1: Tensor,
    pub dw2: Tensor,
    pub db2: Tensor,
}

impl Mlp {
    /// He initialisation: variance scaled by the fan-in.
    pub fn new(d_in: usize, d_hidden: usize, d_out: usize, rng: &mut Rng) -> Mlp {
        let scale1 = (2.0 / d_in as f32).sqrt();
        let scale2 = (2.0 / d_hidden as f32).sqrt();

        let mut w1 = Tensor::zeros(&[d_in, d_hidden]);
        for value in w1.data.iter_mut() {
            *value = rng.normal() * scale1;
        }
        let mut w2 = Tensor::zeros(&[d_hidden, d_out]);
        for value in w2.data.iter_mut() {
            *value = rng.normal() * scale2;
        }

        Mlp {
            w1,
            b1: Tensor::zeros(&[1, d_hidden]),
            w2,
            b2: Tensor::zeros(&[1, d_out]),
            d_in,
            d_hidden,
            d_out,
        }
    }

    pub fn forward(&self, x: &Tensor) -> ForwardCache {
        let z1 = x.matmul(&self.w1).add_bias(&self.b1);
        let h = z1.relu();
        let logits = h.matmul(&self.w2).add_bias(&self.b2);
        let probs = logits.softmax_rows();
        ForwardCache { x: x.clone(), z1, h, probs }
    }

    /// Cross-entropy gradients against one-hot targets.
    pub fn backward(&self, cache: &ForwardCache, targets: &Tensor) -> Grads {
        let rows = cache.probs.rows();
        let d_out = cache.probs.cols();

        // d_logits = probs - one_hot(target), scaled by 1/rows
        let mut d_logits = cache.probs.clone();
        for i in 0..rows {
            let target = crate::target_index(targets, i);
            if target < d_out {
                let current = d_logits.get(i, target);
                d_logits.set(i, target, current - 1.0);
            }
        }
        let scale = 1.0 / rows as f32;
        d_logits = d_logits.mul_scalar(scale);

        let dw2 = cache.h.transpose().matmul(&d_logits);
        let mut db2 = Tensor::zeros(&[1, d_out]);
        for i in 0..rows {
            for j in 0..d_out {
                db2.data[j] += d_logits.get(i, j);
            }
        }

        let d_h = d_logits.matmul(&self.w2.transpose());
        // ReLU derivative: the gradient passes through unchanged where the
        // pre-activation was positive, and is zero elsewhere.
        let mut d_z1 = d_h.clone();
        for i in 0..cache.z1.rows() {
            for j in 0..cache.z1.cols() {
                if cache.z1.get(i, j) <= 0.0 {
                    d_z1.set(i, j, 0.0);
                }
            }
        }

        let dw1 = cache.x.transpose().matmul(&d_z1);
        let mut db1 = Tensor::zeros(&[1, self.d_hidden]);
        for i in 0..rows {
            for j in 0..self.d_hidden {
                db1.data[j] += d_z1.get(i, j);
            }
        }

        Grads { dw1, db1, dw2, db2 }
    }

    pub fn param_shapes(&self) -> Vec<Vec<usize>> {
        vec![
            self.w1.shape.clone(),
            self.b1.shape.clone(),
            self.w2.shape.clone(),
            self.b2.shape.clone(),
        ]
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.w1.to_bytes());
        out.extend_from_slice(&self.b1.to_bytes());
        out.extend_from_slice(&self.w2.to_bytes());
        out.extend_from_slice(&self.b2.to_bytes());
        out
    }
}
