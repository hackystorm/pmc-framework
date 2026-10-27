// SPDX-License-Identifier: GPL-3.0-only

//! Train a 50.4M-parameter reference transformer on one fixed token sequence.
//!
//! This is the PMC trainer's first 50-100M checkpoint, not the PMC memory
//! itself. The model is a decoder-only transformer with a frozen random
//! embedding table; the task is next-token prediction on a fixed 48-token
//! sequence. Memorising the sequence is the point: it gives the dream
//! pipeline a checkpoint of the required size with a real parameter count,
//! a real manifest and a checkpoint digest.
//!
//! The example writes `checkpoint.bin` and `manifest.json` in the checkpoint
//! directory (the format from `pmc_train::checkpoint`) and prints one JSON
//! report to stdout. The frozen embedding table is reproducible from
//! `embedding_seed`; it is not part of the checkpoint.
//!
//! Run it with:
//!
//! ```text
//! PMC_CHECKPOINT=/path/to/checkpoint \
//!   cargo run --release --offline --example train_50m
//! ```

use std::path::PathBuf;
use std::time::Instant;

use pmc_core::{sha256_hex, Rng};
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

fn main() {
    let config = TransformerConfig {
        n_layer: 12,
        d_model: 512,
        n_head: 8,
        ffn_hidden: 2048,
        vocab: 256,
        n_positions: 64,
        readout_rank: 64,
    };
    config.validate().expect("the 50M configuration must be valid");

    let steps: usize = std::env::var("PMC_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(200);
    let learning_rate: f32 = std::env::var("PMC_LR")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.003);
    let model_seed: u64 = 20261030;
    let embedding_seed: u64 = 20261029;
    let checkpoint_path = std::env::var("PMC_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("build/pmc-50m"));

    let tokens: Vec<u32> = (0..48).map(|index| ((index * 37 + 11) % 256) as u32).collect();
    let token_bytes: Vec<u8> = tokens.iter().flat_map(|token| token.to_le_bytes()).collect();
    let tokens_sha256 = sha256_hex(&token_bytes);
    let embedding = TableEmbedding::random(config.vocab, config.d_model, embedding_seed);

    let mut model = config.init(model_seed);
    let mut optimiser = TransformerAdam::new(&model);
    let mask = vec![true; tokens.len()];
    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;
    let started = Instant::now();

    for step in 0..steps {
        let forward = model.forward(&tokens, &embedding, Some(&mask));
        if step == 0 {
            initial_loss = forward.loss;
        }
        final_loss = forward.loss;
        let mut grads = model.backward(&forward);
        optimiser.step(&mut model, &mut grads, learning_rate, 0.9, 0.999, 1e-8, 1.0);
        if step % 20 == 0 {
            eprintln!(
                "step {:>3}/{} loss {:.6}",
                step, steps, forward.loss
            );
        }
    }

    let elapsed_millis = started.elapsed().as_millis();
    let param_count = model.param_count();
    checkpoint::save_with_step(&checkpoint_path, &model, steps as u32)
        .expect("the checkpoint must be written");

    let tokens_json = tokens
        .iter()
        .map(|token| token.to_string())
        .collect::<Vec<String>>()
        .join(",");
    println!(
        concat!(
            "{{\"config\":{{\"n_layer\":{},\"d_model\":{},\"n_head\":{},",
            "\"ffn_hidden\":{},\"vocab\":{},\"n_positions\":{},",
            "\"readout_rank\":{}}},\"param_count\":{},",
            "\"embedding_param_count\":{},\"steps\":{},",
            "\"learning_rate\":{:.6},\"model_seed\":{},",
            "\"embedding_seed\":{},\"initial_loss\":{:.6},",
            "\"final_loss\":{:.6},\"elapsed_millis\":{},",
            "\"tokens\":[{}],\"tokens_sha256\":\"{}\",",
            "\"checkpoint\":\"{}\"}}"
        ),
        config.n_layer,
        config.d_model,
        config.n_head,
        config.ffn_hidden,
        config.vocab,
        config.n_positions,
        config.readout_rank,
        param_count,
        config.vocab * config.d_model,
        steps,
        learning_rate,
        model_seed,
        embedding_seed,
        initial_loss,
        final_loss,
        elapsed_millis,
        tokens_json,
        tokens_sha256,
        checkpoint_path.display()
    );
}
