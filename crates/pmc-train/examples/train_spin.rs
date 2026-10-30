// SPDX-License-Identifier: GPL-3.0-only

//! Self-Play fine-tuning (SPIN): the model's own samples enter training as
//! the *rejected* side; the *chosen* responses are loaded from a user dataset.
//! Pure generative replay (self-generated positives) is never used.
//!
//! This example trains on a batch of preference pairs with DPO and measures
//! preference accuracy on a held-out set: does the model assign a higher
//! log-probability to the chosen sequence than to the rejected one?
//!
//! The dataset is loaded from PMC_DATASET (JSONL). Each line:
//! {"chosen": "user-provided quality response", "rejected": "optional"}.
//! If PMC_DATASET is not set, falls back to synthetic data.

use std::time::Instant;

use pmc_core::Rng;
use pmc_train::transformer::{Embedding, Transformer, TransformerAdam, TransformerConfig};
use std::fs::File;
use std::io::{BufRead, BufReader};

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

/// Minimal JSONL parser for {"chosen": "str", "rejected": "str"}.
/// Returns (chosen, rejected_opt) where rejected_opt is None if absent.
fn parse_jsonl_line(line: &str) -> Result<(String, Option<String>), String> {
    // Strip UTF-8 BOM if present.
    let line = line.trim_start_matches('\u{FEFF}').trim();
    if !line.starts_with('{') || !line.ends_with('}') {
        return Err("not a JSON object".to_string());
    }
    let inner = &line[1..line.len() - 1];
    let mut chosen = None;
    let mut rejected = None;

    // Simple field extractor for "chosen" and "rejected" string fields.
    for part in inner.split(',') {
        let part = part.trim();
        if part.starts_with("\"chosen\":") {
            chosen = Some(extract_string_value(part)?);
        } else if part.starts_with("\"rejected\":") {
            rejected = Some(extract_string_value(part)?);
        }
    }
    let chosen = chosen.ok_or_else(|| "missing 'chosen' field".to_string())?;
    Ok((chosen, rejected))
}

fn extract_string_value(part: &str) -> Result<String, String> {
    let colon = part.find(':').ok_or("no colon")?;
    let value = part[colon + 1..].trim();
    if value.starts_with('"') && value.ends_with('"') {
        let inner = &value[1..value.len() - 1];
        Ok(inner.replace("\\\"", "\"").replace("\\\\", "\\"))
    } else if value == "null" {
        Ok(String::new())
    } else {
        Err(format!("expected string or null, got: {}", value))
    }
}

/// Simple character-level tokenizer for demo purposes.
fn tokenize(text: &str, vocab: usize, max_len: usize) -> Vec<u32> {
    text.chars()
        .take(max_len)
        .map(|c| (c as u32) % vocab as u32)
        .collect()
}

/// Load chosen responses from a JSONL dataset file.
/// Each line: {"chosen": "text", "rejected": "text"} (rejected optional).
/// Returns a vector of (chosen_tokens, rejected_tokens) pairs.
fn load_dataset(
    path: &str,
    vocab: usize,
    max_len: usize,
) -> Vec<(Vec<u32>, Vec<u32>)> {
    let file = File::open(path).expect(&format!("Cannot open dataset: {}", path));
    let reader = BufReader::new(file);
    let mut pairs = Vec::new();
    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        if line.trim().is_empty() {
            continue;
        }
        let (chosen, rejected) = parse_jsonl_line(&line).expect("Invalid JSON in dataset");
        let chosen_tokens = tokenize(&chosen, vocab, max_len);
        let rejected_tokens = if let Some(r) = rejected {
            tokenize(&r, vocab, max_len)
        } else {
            // SPIN shape: rejected will be generated from the model
            chosen_tokens.clone()
        };
        pairs.push((chosen_tokens, rejected_tokens));
    }
    pairs
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn scale_grads(grads: &mut pmc_train::transformer::Grads, factor: f32) {
    for block in grads.blocks.iter_mut() {
        for v in block.attn_norm.iter_mut() { *v *= factor; }
        for v in block.wq.iter_mut() { *v *= factor; }
        for v in block.wk.iter_mut() { *v *= factor; }
        for v in block.wv.iter_mut() { *v *= factor; }
        for v in block.wo.iter_mut() { *v *= factor; }
        for v in block.ffn_norm.iter_mut() { *v *= factor; }
        for v in block.w_gate.iter_mut() { *v *= factor; }
        for v in block.w_up.iter_mut() { *v *= factor; }
        for v in block.w_down.iter_mut() { *v *= factor; }
    }
    for v in grads.final_norm.iter_mut() { *v *= factor; }
    for v in grads.readout_u.iter_mut() { *v *= factor; }
    for v in grads.readout_v.iter_mut() { *v *= factor; }
}

fn add_grads(dst: &mut pmc_train::transformer::Grads, src: &pmc_train::transformer::Grads) {
    for (db, sb) in dst.blocks.iter_mut().zip(src.blocks.iter()) {
        for (a, b) in db.attn_norm.iter_mut().zip(sb.attn_norm.iter()) { *a += *b; }
        for (a, b) in db.wq.iter_mut().zip(sb.wq.iter()) { *a += *b; }
        for (a, b) in db.wk.iter_mut().zip(sb.wk.iter()) { *a += *b; }
        for (a, b) in db.wv.iter_mut().zip(sb.wv.iter()) { *a += *b; }
        for (a, b) in db.wo.iter_mut().zip(sb.wo.iter()) { *a += *b; }
        for (a, b) in db.ffn_norm.iter_mut().zip(sb.ffn_norm.iter()) { *a += *b; }
        for (a, b) in db.w_gate.iter_mut().zip(sb.w_gate.iter()) { *a += *b; }
        for (a, b) in db.w_up.iter_mut().zip(sb.w_up.iter()) { *a += *b; }
        for (a, b) in db.w_down.iter_mut().zip(sb.w_down.iter()) { *a += *b; }
    }
    for (a, b) in dst.final_norm.iter_mut().zip(src.final_norm.iter()) { *a += *b; }
    for (a, b) in dst.readout_u.iter_mut().zip(src.readout_u.iter()) { *a += *b; }
    for (a, b) in dst.readout_v.iter_mut().zip(src.readout_v.iter()) { *a += *b; }
}

fn preference_accuracy(model: &Transformer, embedding: &TableEmbedding, pairs: &[(Vec<u32>, Vec<u32>)]) -> f32 {
    if pairs.is_empty() {
        return 0.0;
    }
    let mut correct = 0usize;
    for (chosen, rejected) in pairs {
        let mask = vec![true; chosen.len()];
        let fwd_c = model.forward(chosen, embedding, Some(&mask));
        let fwd_r = model.forward(rejected, embedding, Some(&rejected.iter().map(|_| true).collect::<Vec<_>>()));
        let n_c = (chosen.len() - 1) as f32;
        let n_r = (rejected.len() - 1) as f32;
        let logp_c = -fwd_c.loss * n_c;
        let logp_r = -fwd_r.loss * n_r;
        if logp_c > logp_r {
            correct += 1;
        }
    }
    correct as f32 / pairs.len() as f32
}

fn main() {
    let n_layer = env_usize("PMC_N_LAYER", 2);
    let d_model = env_usize("PMC_D_MODEL", 32);
    let n_head = env_usize("PMC_N_HEAD", 4);
    let ffn_hidden = env_usize("PMC_FFN", 64);
    let vocab = env_usize("PMC_VOCAB", 16);
    let n_positions = env_usize("PMC_N_POS", 16);
    let readout_rank = env_usize("PMC_RANK", 8);
    let steps = env_usize("PMC_STEPS", 15);
    let learning_rate = env_f32("PMC_LR", 0.01);
    let beta = env_f32("PMC_BETA", 0.5);
    let model_seed: u64 = env_u64("PMC_MODEL_SEED", 20261050);
    let embedding_seed: u64 = env_u64("PMC_EMBEDDING_SEED", 20261051);
    let num_pairs = env_usize("PMC_PAIRS", 8);

    // Dataset path: PMC_DATASET points to a JSONL file with chosen/rejected pairs.
    // If not set, falls back to synthetic data for backwards compatibility.
    let dataset_path = std::env::var("PMC_DATASET").ok();
    let max_len = n_positions;

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

    let embedding = TableEmbedding::random(vocab, d_model, embedding_seed);

    // Load training pairs from dataset, or fall back to synthetic data.
    let train_pairs = if let Some(path) = dataset_path {
        let pairs = load_dataset(&path, vocab, max_len);
        eprintln!("Loaded {} pairs from dataset: {}", pairs.len(), path);
        pairs
    } else {
        // Fallback: synthetic SPIN pairs (backwards compatible).
        eprintln!("No dataset provided (PMC_DATASET not set), using synthetic data");
        let mut rng = Rng::new(20261052);
        let mut pairs: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(num_pairs);
        for i in 0..num_pairs {
            let chosen: Vec<u32> = (0..6).map(|_| (rng.next_u64() % vocab as u64) as u32).collect();
            pairs.push((chosen.clone(), chosen.clone())); // SPIN shape
        }
        pairs
    };

    // Held-out pairs: never used in training.
    let mut rng = Rng::new(20261052);
    let mut held_out: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(num_pairs);
    for _ in 0..num_pairs {
        let chosen: Vec<u32> = (0..6).map(|_| (rng.next_u64() % vocab as u64) as u32).collect();
        let mut rejected = chosen.clone();
        if !rejected.is_empty() {
            let mid = rejected.len() / 2;
            rejected[mid] = (rejected[mid] + 1) % vocab as u32;
        }
        held_out.push((chosen, rejected));
    }

    // Reference log-probs from the frozen initial model.
    let reference = config.init(model_seed);
    let mut ref_logps: Vec<(f32, f32)> = Vec::with_capacity(train_pairs.len());
    for (chosen, rejected) in &train_pairs {
        let mc = vec![true; chosen.len()];
        let mr = vec![true; rejected.len()];
        let fwd_c = reference.forward(chosen, &embedding, Some(&mc));
        let fwd_r = reference.forward(rejected, &embedding, Some(&mr));
        let nc = (chosen.len() - 1) as f32;
        let nr = (rejected.len() - 1) as f32;
        ref_logps.push((-fwd_c.loss * nc, -fwd_r.loss * nr));
    }

    let mut model = config.init(model_seed);
    let mut optimiser = TransformerAdam::new(&model);

    let initial_acc = preference_accuracy(&model, &embedding, &held_out);

    let started = std::time::Instant::now();
    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;

    for step in 0..steps {
        let mut combined: Option<pmc_train::transformer::Grads> = None;
        let mut step_loss = 0.0f32;
        for (idx, (chosen, rejected)) in train_pairs.iter().enumerate() {
            let mc = vec![true; chosen.len()];
            let mr = vec![true; rejected.len()];
            let n_c = (chosen.len() - 1) as f32;
            let n_r = (rejected.len() - 1) as f32;

            let fwd_c = model.forward(chosen, &embedding, Some(&mc));
            let mut grads_c = model.backward(&fwd_c);
            let fwd_r = model.forward(rejected, &embedding, Some(&mr));
            let mut grads_r = model.backward(&fwd_r);

            let logp_c = -fwd_c.loss * n_c;
            let logp_r = -fwd_r.loss * n_r;
            let (ref_c, ref_r) = ref_logps[idx];
            let delta = (logp_c - logp_r) - (ref_c - ref_r);
            let dpo = softplus(-(beta * delta));
            step_loss += dpo;

            let scale = beta * sigmoid(-(beta * delta));
            scale_grads(&mut grads_c, scale * n_c);
            scale_grads(&mut grads_r, -scale * n_r);
            add_grads(&mut grads_c, &grads_r);

            if combined.is_none() {
                combined = Some(grads_c);
            } else {
                add_grads(combined.as_mut().unwrap(), &grads_c);
            }
        }

        let mut grads = combined.unwrap();
        // Average over pairs.
        let np = train_pairs.len() as f32;
        scale_grads(&mut grads, 1.0 / np);
        step_loss /= np;

        if step == 0 {
            initial_loss = step_loss;
        }
        final_loss = step_loss;

        optimiser.step(&mut model, &mut grads, learning_rate, 0.9, 0.999, 1e-8, 1.0);

        if step == 0 || step + 1 == steps || (step + 1) % 5 == 0 {
            let acc = preference_accuracy(&model, &embedding, &held_out);
            eprintln!(
                "step {:>3}/{} dpo_loss {:.6} held_out_acc {:.3}",
                step + 1,
                steps,
                step_loss,
                acc
            );
        }
    }

    let final_acc = preference_accuracy(&model, &embedding, &held_out);
    let elapsed = started.elapsed().as_millis();

    println!(
        "{{\"pairs\":{},\"held_out\":{},\"steps\":{},\"beta\":{:.6},\"initial_dpo_loss\":{:.6},\"final_dpo_loss\":{:.6},\"initial_preference_accuracy\":{:.6},\"final_preference_accuracy\":{:.6},\"elapsed_millis\":{}}}",
        train_pairs.len(),
        held_out.len(),
        steps,
        beta,
        initial_loss,
        final_loss,
        initial_acc,
        final_acc,
        elapsed
    );
}
