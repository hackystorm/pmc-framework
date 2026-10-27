// SPDX-License-Identifier: GPL-3.0-only

//! Load the 50.4M reference checkpoint and measure the trained sequence.
//!
//! This verifies the checkpoint through the same loader the dream pipeline
//! would use: the manifest digest, the header, the exact file length and the
//! parameter digest are checked before the model is returned. The report is
//! one JSON object.
//!
//! ```text
//! PMC_CHECKPOINT=/path/to/checkpoint \
//!   cargo run --release --offline --example eval_50m
//! ```

use std::path::PathBuf;

use pmc_core::{sha256_hex, Rng};
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

fn main() {
    let checkpoint_path = std::env::var("PMC_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("build/pmc-50m"));
    let model = checkpoint::load(&checkpoint_path).expect("the checkpoint must load");
    let config = model.config;
    let tokens: Vec<u32> = (0..48).map(|index| ((index * 37 + 11) % 256) as u32).collect();
    let token_bytes: Vec<u8> = tokens.iter().flat_map(|token| token.to_le_bytes()).collect();
    let tokens_sha256 = sha256_hex(&token_bytes);
    let embedding = TableEmbedding::random(config.vocab, config.d_model, 20261029);
    let mask = vec![true; tokens.len()];
    let forward = model.forward(&tokens, &embedding, Some(&mask));
    let vocab = config.vocab;
    let mut correct = 0usize;
    let mut total = 0usize;
    for position in 0..tokens.len() {
        if !forward.mask[position] {
            continue;
        }
        total += 1;
        let row = &forward.logits[position * vocab..(position + 1) * vocab];
        let mut best = 0usize;
        for index in 1..vocab {
            if row[index] > row[best] {
                best = index;
            }
        }
        if best == forward.targets[position] as usize {
            correct += 1;
        }
    }
    let accuracy = if total == 0 { 0.0 } else { correct as f32 / total as f32 };
    println!(
        concat!(
            "{{\"param_count\":{},\"loss\":{:.8},",
            "\"accuracy\":{:.6},\"correct\":{},\"positions\":{},",
            "\"tokens_sha256\":\"{}\",\"checkpoint\":\"{}\"}}"
        ),
        model.param_count(),
        forward.loss,
        accuracy,
        correct,
        total,
        tokens_sha256,
        checkpoint_path.display()
    );
}
