// SPDX-License-Identifier: GPL-3.0-only

//! The synthetic memory task used to exercise the trainer.
//!
//! The task is deliberately the smallest thing that looks like memory: given
//! a key, produce the value that was associated with it. Nothing more clever
//! is needed to check that a trainer can learn, that two implementations
//! agree, and that recall can be measured.

use pmc_core::{Rng, Tensor};

pub struct MemoryTask {
    pub keys: usize,
    /// `mapping[key]` is the value associated with that key.
    pub mapping: Vec<usize>,
}

impl MemoryTask {
    /// A deterministic key-to-value association, shuffled with the seed.
    pub fn new(keys: usize, seed: u64) -> MemoryTask {
        let mut mapping: Vec<usize> = (0..keys).collect();
        let mut rng = Rng::new(seed ^ 0x5DEE_CE66_D5A4_1B0D);
        // Fisher-Yates, descending.
        for i in (1..keys).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            mapping.swap(i, j);
        }
        MemoryTask { keys, mapping }
    }

    pub fn input(&self, key: usize) -> Tensor {
        let mut row = Tensor::zeros(&[1, self.keys]);
        row.set(0, key % self.keys, 1.0);
        row
    }

    pub fn target(&self, key: usize) -> Tensor {
        let mut row = Tensor::zeros(&[1, self.keys]);
        row.set(0, self.mapping[key % self.keys], 1.0);
        row
    }

    /// A batch of one-hot inputs and their one-hot targets.
    pub fn batch(&self, keys: &[usize]) -> (Tensor, Tensor) {
        let mut inputs = Tensor::zeros(&[keys.len(), self.keys]);
        let mut targets = Tensor::zeros(&[keys.len(), self.keys]);
        for (index, key) in keys.iter().enumerate() {
            inputs.set(index, key % self.keys, 1.0);
            targets.set(index, self.mapping[key % self.keys], 1.0);
        }
        (inputs, targets)
    }
}
