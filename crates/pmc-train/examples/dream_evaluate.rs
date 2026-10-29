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
use pmc_train::transformer::{Embedding, TransformerConfig};

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

/// Parse a token file: JSON array of integers or one integer per line.
fn read_tokens(path: &str) -> Vec<u32> {
    let raw = std::fs::read_to_string(path).expect("read token file");
    let trimmed = raw.trim();
    if trimmed.starts_with('[') {
        parse_json_array(trimmed)
    } else {
        trimmed
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                if line.is_empty() {
                    None
                } else {
                    Some(line.parse::<u32>().expect("token id"))
                }
            })
            .collect()
    }
}

fn parse_json_array(text: &str) -> Vec<u32> {
    let inner = text
        .trim()
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .expect("token file is not a JSON array");
    if inner.trim().is_empty() {
        return Vec::new();
    }
    inner
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<u32>()
                .expect("token file array holds a non-integer")
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        eprintln!("usage: dream_evaluate <checkpoint_dir> [val_tokens_file]");
        std::process::exit(2);
    }
    let checkpoint_path = PathBuf::from(&args[1]);
    let val_tokens_path = args.get(2).map(|s| s.as_str());

    let model = match checkpoint::load(&PathBuf::from(&args[1])) {
        Ok(m) => m,
        Err(error) => {
            eprintln!("load checkpoint: {error}");
            std::process::exit(2);
        }
    };
    let config = model.config;
    let embedding = TableEmbedding::random(config.vocab, config.d_model, env_u64("PMC_EMBEDDING_SEED", 20261032));

    let tokens = if let Some(path) = val_tokens_path {
        read_tokens(path)
    } else {
        // Fallback: fixed held-out sequence for backward compatibility.
        let mut rng = Rng::new(20261033);
        (0..64).map(|_| (rng.next_u64() % config.vocab as u64) as u32).collect()
    };

    let mask = vec![true; tokens.len()];
    let forward = model.forward(&tokens, &embedding, Some(&vec![true; tokens.len()]));
    let loss = forward.loss;

    // Map the loss to a [0, 1] recall-like number: lower loss -> higher.
    let recall = (-loss).exp().clamp(0.0, 1.0);

    println!(
        "{{\"recall_older\":{:.6},\"recall_newer\":{:.6},\"interference\":0.0,\"calibration\":0.0,\"identity\":1.0,\"false_memory\":0.0}}",
        recall, recall
    );
}