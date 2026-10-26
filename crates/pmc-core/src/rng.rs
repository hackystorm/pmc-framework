// SPDX-License-Identifier: GPL-3.0-only

//! A small, reproducible pseudo-random generator.
//!
//! Reproducibility is a hard requirement of this project: a result that
//! cannot be regenerated from a seed is not a result. The generator is
//! xorshift64*, chosen because it is tiny, fast, and identical across
//! platforms, so a seed produces the same weights on every machine.

use std::f32::consts::TAU;

#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    /// Standard normal via Box-Muller.
    pub fn normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_sequence() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn uniforms_stay_in_range() {
        let mut rng = Rng::new(7);
        for _ in 0..1000 {
            let value = rng.next_f32();
            assert!((0.0..1.0).contains(&value));
        }
    }
}
