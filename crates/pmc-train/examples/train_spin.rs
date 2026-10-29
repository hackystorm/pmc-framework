// SPDX-License-Identifier: GPL-3.0-only

//! Self-Play fine-tuning (SPIN): the model's own samples enter training as
//! the *rejected* side; a verified or previous-iteration sample is the *chosen*
//! side. Pure generative replay (self-generated positives) is never used.
//!
//! This example trains on a batch of preference pairs with DPO and measures
//! preference accuracy on a held-out set: does the model assign a higher
//! log-probability to the chosen sequence than to the rejected one?

use std::time::Instant;

use pmc_core::Rng;
use pmc_train::transformer::{Embedding, Transformer, TransformerAdam, TransformerConfig};

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

    // Build preference pairs: chosen = a clean sequence, rejected = chosen
    // with one token changed. The model's own sample is the rejected side
    // (SPIN shape); here we synthesise it deterministically.
    let mut rng = Rng::new(20261052);
    let mut train_pairs: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(num_pairs);
    let mut held_out: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(num_pairs);
    for i in 0..(num_pairs * 2) {
        let mut chosen: Vec<u32> = (0..6).map(|_| (rng.next_u64() % vocab as u64) as u32).collect();
        let mut rejected = chosen.clone();
        if !rejected.is_empty() {
            let mid = rejected.len() / 2;
            rejected[mid] = (rejected[mid] + 1) % vocab as u32;
        }
        if i < num_pairs {
            train_pairs.push((chosen, rejected));
        } else {
            held_out.push((chosen, rejected));
        }
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
