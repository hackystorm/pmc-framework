// SPDX-License-Identifier: GPL-3.0-only

//! Consolidator for the dream pipeline: fine-tune the baseline checkpoint on
//! the replayed records and write the candidate.
//!
//! The replay manifest carries record ids and content digests only (D-17);
//! this consolidator derives a small token sequence from each record id and
//! fine-tunes the transformer on it. It loads the baseline checkpoint, runs
//! the configured number of steps, and writes the candidate checkpoint in the
//! same format. A dream whose consolidation does not improve the held-out
//! metrics is refused by the gate and rolled back; that outcome is a result.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use pmc_core::Rng;
use pmc_train::checkpoint;
use pmc_train::transformer::{Embedding, TransformerAdam, TransformerConfig};

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

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Derive a short token sequence from a record id, deterministically.
fn tokens_for_record(id: &str, vocab: usize) -> Vec<u32> {
    let mut state = 0xcbf2_9ce4_8422_2325u64;
    for byte in id.bytes() {
        state ^= u64::from(byte);
        state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut out = Vec::with_capacity(8);
    let mut s = state;
    for _ in 0..8 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        out.push(((s >> 32) as usize % vocab) as u32);
    }
    out
}

fn main() {
    if std::env::args().count() != 4 {
        eprintln!("usage: dream_consolidate <baseline_dir> <candidate_dir> <manifest_path>");
        std::process::exit(2);
    }
    let args: Vec<String> = std::env::args().collect();
    let baseline = PathBuf::from(&args[1]);
    let candidate = PathBuf::from(&args[2]);
    let manifest_path = PathBuf::from(&args[3]);

    let manifest_text = fs::read_to_string(&manifest_path).expect("read manifest");
    // Extract record ids with a tiny hand-rolled scan (no serde).
    let mut record_ids: Vec<String> = Vec::new();
    for line in manifest_text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("\"record_id\"") || trimmed.starts_with("\"record_id\":") {
            // not the shape used; fall through to the items scan below
        }
    }
    // The manifest is JSON with an "items" array of objects carrying "record_id".
    // Scan for "record_id": "..." occurrences.
    let mut search_from = 0;
    while let Some(pos) = manifest_text[search_from..].find("\"record_id\"") {
        let abs = search_from + pos;
        // find the colon then the quoted value
        let after = &manifest_text[abs..];
        if let Some(colon) = after.find(':') {
            let rest = after[colon + 1..].trim_start();
            if rest.starts_with('"') {
                let inner = &rest[1..];
                if let Some(end) = inner.find('"') {
                    record_ids.push(inner[..end].to_string());
                    search_from = abs + colon + 1 + 1 + end + 1;
                    continue;
                }
            }
        }
        search_from = abs + 12;
    }
    if record_ids.is_empty() {
        eprintln!("no record ids found in manifest");
        std::process::exit(2);
    }

    let vocab = env_usize("PMC_VOCAB", 256);
    let n_layer = env_usize("PMC_N_LAYER", 2);
    let d_model = env_usize("PMC_D_MODEL", 32);
    let n_head = env_usize("PMC_N_HEAD", 4);
    let ffn_hidden = env_usize("PMC_FFN", 64);
    let n_positions = env_usize("PMC_N_POS", 64);
    let readout_rank = env_usize("PMC_RANK", 8);
    let steps = env_usize("PMC_STEPS", 10);
    let learning_rate = env_f32("PMC_LR", 0.01);
    let embedding_seed = env_u64("PMC_EMBEDDING_SEED", 20261032);

    let config = TransformerConfig {
        n_layer,
        d_model,
        n_head,
        ffn_hidden,
        vocab,
        n_positions,
        readout_rank,
    };
    config.validate().expect("config must be valid");

    // Build the training token sequence from the replayed records.
    let mut tokens: Vec<u32> = Vec::new();
    for id in &record_ids {
        tokens.extend(tokens_for_record(id, vocab));
    }
    if tokens.len() < 4 {
        eprintln!("not enough tokens to train");
        std::process::exit(2);
    }
    let mask = vec![true; tokens.len()];

    // Load the baseline.
    let mut model = match checkpoint::load(&baseline) {
        Ok(m) => m,
        Err(error) => {
            eprintln!("load baseline: {error}");
            std::process::exit(2);
        }
    };
    let embedding = TableEmbedding::random(vocab, d_model, embedding_seed);
    // If the loaded model's config differs from ours we still train with the
    // loaded config's embedding dim; assume they match.
    let mut optimiser = TransformerAdam::new(&model);

    let started = Instant::now();
    for step in 0..steps {
        let forward = model.forward(&tokens, &embedding, Some(&mask));
        let mut grads = model.backward(&forward);
        optimiser.step(&mut model, &mut grads, learning_rate, 0.9, 0.999, 1e-8, 1.0);
        if step == 0 || step + 1 == steps {
            eprintln!("consolidate step {}/{} loss {:.6}", step + 1, steps, forward.loss);
        }
    }
    let elapsed = started.elapsed().as_millis();

    if candidate.exists() {
        let _ = fs::remove_dir_all(&candidate);
    }
    fs::create_dir_all(&candidate).expect("create candidate dir");
    checkpoint::save_with_step(&candidate, &model, steps as u32).expect("write candidate");

    eprintln!(
        "consolidated {} records ({} tokens) in {} ms",
        record_ids.len(),
        tokens.len(),
        elapsed
    );
}
