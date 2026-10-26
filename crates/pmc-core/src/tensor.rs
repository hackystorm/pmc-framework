// SPDX-License-Identifier: GPL-3.0-only

//! Dense row-major tensors and the handful of operations the trainer needs.
//!
//! Deliberately minimal. The PMC core is a small, fixed architecture, so a
//! general-purpose autodiff framework would cost build time, disk and
//! compile cycles without returning anything the trainer actually uses.

use std::ops::{Add, Mul, Sub};

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape: shape.to_vec(),
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        assert_eq!(size, data.len(), "tensor shape does not match its data");
        Tensor { data, shape: shape.to_vec() }
    }

    pub fn rows(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape[0]
        }
    }

    pub fn cols(&self) -> usize {
        if self.shape.len() >= 2 {
            self.shape[1]
        } else {
            self.data.len()
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols() + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        let cols = self.cols();
        self.data[row * cols + col] = value;
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shape mismatch in add");
        Tensor {
            data: self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn add_scalar(&self, value: f32) -> Tensor {
        Tensor {
            data: self.data.iter().map(|x| x + value).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shape mismatch in sub");
        Tensor {
            data: self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shape mismatch in mul");
        Tensor {
            data: self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn mul_scalar(&self, value: f32) -> Tensor {
        Tensor {
            data: self.data.iter().map(|x| x * value).collect(),
            shape: self.shape.clone(),
        }
    }

    /// Row-major matrix product: `(m, k) * (k, n) -> (m, n)`.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let m = self.rows();
        let k = self.cols();
        assert_eq!(other.rows(), k, "inner dimension mismatch in matmul");
        let n = other.cols();
        let mut out = Tensor::zeros(&[m, n]);
        for i in 0..m {
            for p in 0..k {
                let left = self.get(i, p);
                if left == 0.0 {
                    continue;
                }
                for j in 0..n {
                    let value = out.get(i, j) + left * other.get(p, j);
                    out.set(i, j, value);
                }
            }
        }
        out
    }

    pub fn relu(&self) -> Tensor {
        Tensor {
            data: self.data.iter().map(|x| if *x > 0.0 { *x } else { 0.0 }).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn tanh(&self) -> Tensor {
        Tensor {
            data: self.data.iter().map(|x| x.tanh()).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        Tensor {
            data: self.data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect(),
            shape: self.shape.clone(),
        }
    }

    /// Row-wise softmax. Rows with no finite maximum fall back to uniform.
    pub fn softmax_rows(&self) -> Tensor {
        let rows = self.rows();
        let cols = self.cols();
        let mut out = Tensor::zeros(&[rows, cols]);
        for i in 0..rows {
            let mut max = f32::NEG_INFINITY;
            for j in 0..cols {
                max = max.max(self.get(i, j));
            }
            if !max.is_finite() {
                for j in 0..cols {
                    out.set(i, j, 1.0 / cols as f32);
                }
                continue;
            }
            let mut sum = 0.0f32;
            for j in 0..cols {
                let value = (self.get(i, j) - max).exp();
                out.set(i, j, value);
                sum += value;
            }
            for j in 0..cols {
                let value = out.get(i, j) / sum;
                out.set(i, j, value);
            }
        }
        out
    }

    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    pub fn l2_norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Add a bias row, broadcasting a `[1, n]` (or `[n]`) bias across rows.
    ///
    /// Plain `add` requires identical shapes, which does not work for a
    /// bias that has to apply to every row of a batch.
    pub fn add_bias(&self, bias: &Tensor) -> Tensor {
        assert_eq!(
            self.cols(),
            bias.cols(),
            "bias width does not match the tensor"
        );
        let mut out = self.clone();
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                let value = out.get(i, j) + bias.get(0, j);
                out.set(i, j, value);
            }
        }
        out
    }

    /// Transpose a 2-D tensor.
    pub fn transpose(&self) -> Tensor {
        let rows = self.rows();
        let cols = self.cols();
        let mut out = Tensor::zeros(&[cols, rows]);
        for i in 0..rows {
            for j in 0..cols {
                out.set(j, i, self.get(i, j));
            }
        }
        out
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len() * 4);
        for value in &self.data {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }
}

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        self.add(rhs)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        self.mul(rhs)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        self.sub(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_multiplies_correctly() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.get(0, 0), 1.0);
        assert_eq!(c.get(0, 1), 2.0);
        assert_eq!(c.get(1, 0), 3.0);
        assert_eq!(c.get(1, 1), 4.0);
    }

    #[test]
    fn softmax_rows_sums_to_one() {
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let probs = logits.softmax_rows();
        assert!((probs.data.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_is_stable_for_large_logits() {
        let logits = Tensor::from_vec(vec![1000.0, 1000.0], vec![1, 2]);
        let probs = logits.softmax_rows();
        assert!((probs.data.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}
