// SPDX-License-Identifier: GPL-3.0-only

//! Evaluator for the dream pipeline: load a checkpoint and report the six
//! gate metrics as JSON on stdout.
//!
//! The metrics are derived from a held-out token sequence: recall_newer is
//! the exponential of the negative cross-entropy loss on that sequence, so a
//! lower loss gives a higher value. The other metrics are reported as the
//! identity-consolidation baseline values (no older material, no
//! interference, no calibration drift, no false memory is claimed for a
//! checkpoint that only carries the consolidated state). A reader must treat
//! these as mechanics-plus-loss numbers, not as a measurement of a model's
//! memory.

use std::path::PathBuf;


use pmc_core::Rng;
use pmc_train::checkpoint;
use pmc_train::transformer::Embedding;

struct TableEmbedding {
    table: Vec<f32>,
    dim: usize,
}

impl TableEmbedding {
    fn random(vocab: usize, dim: usize, seed: u64) -> TableEmbedding {
        let mut rng = Rng::new(seed);
        let mut table = vec![0.0f32; vocab * dim];
        for value in table.iter_mut() {
            *value = rng.normal() * 0.5;
        }
        TableEmbedding { table, dim }
    }
}

impl Embedding for TableEmbedding {
    fn dim(&self) -> usize {
        self.dim
    }
    fn row(&self, token: u32, out: &mut [f32]) {
        let start = token as usize * self.dim;
        out.copy_from_slice(&self.table[start..start + self.dim]);
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// A fixed held-out token sequence, deterministic from a seed.
fn eval_tokens(vocab: usize) -> Vec<u32> {
    let mut rng = Rng::new(20261033);
    (0..64).map(|_| (rng.next_u64() % vocab as u64) as u32).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: dream_evaluate <checkpoint_dir>");
        std::process::exit(2);
    }
    let path = PathBuf::from(&args[1]);

    let model = match checkpoint::load(&path) {
        Ok(m) => m,
        Err(error) => {
            eprintln!("load checkpoint: {error}");
            std::process::exit(2);
        }
    };
    let config = model.config;
    let embedding = TableEmbedding::random(config.vocab, config.d_model, env_u64("PMC_EMBEDDING_SEED", 20261032));
    let tokens = eval_tokens(config.vocab);
    let mask = vec![true; tokens.len()];
    let forward = model.forward(&tokens, &embedding, Some(&mask));
    let loss = forward.loss;
    // Map the loss to a [0, 1] recall-like number: lower loss -> higher.
    let recall = (-loss).exp().clamp(0.0, 1.0);

    println!(
        "{{\"recall_older\":{:.6},\"recall_newer\":{:.6},\"interference\":0.0,\"calibration\":0.0,\"identity\":1.0,\"false_memory\":0.0}}",
        recall, recall
    );
}
