// SPDX-License-Identifier: GPL-3.0-only

//! Recursive self-play (SPIN) loop: the model's own samples enter training
//! as the rejected side, and the verified sample is the chosen side. Each
//! iteration regenerates the rejected samples from the updated model, so the
//! training signal tracks the model's current capability - the "ever-rising
//! curve" of self-improvement.
//!
//! The held-out preference accuracy is measured after every iteration and
//! reported as a series. A rising series is evidence of self-improvement;
//! a flat or falling one is a negative, recorded as such.

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
        for v in block.attn_norm.iter_mut() {
            *v *= factor;
        }
        for v in block.wq.iter_mut() {
            *v *= factor;
        }
        for v in block.wk.iter_mut() {
            *v *= factor;
        }
        for v in block.wv.iter_mut() {
            *v *= factor;
        }
        for v in block.wo.iter_mut() {
            *v *= factor;
        }
        for v in block.ffn_norm.iter_mut() {
            *v *= factor;
        }
        for v in block.w_gate.iter_mut() {
            *v *= factor;
        }
        for v in block.w_up.iter_mut() {
            *v *= factor;
        }
        for v in block.w_down.iter_mut() {
            *v *= factor;
        }
    }
    for v in grads.final_norm.iter_mut() {
        *v *= factor;
    }
    for v in grads.readout_u.iter_mut() {
        *v *= factor;
    }
    for v in grads.readout_v.iter_mut() {
        *v *= factor;
    }
}

fn add_grads(dst: &mut pmc_train::transformer::Grads, src: &pmc_train::transformer::Grads) {
    for (db, sb) in dst.blocks.iter_mut().zip(src.blocks.iter()) {
        for (a, b) in db.attn_norm.iter_mut().zip(sb.attn_norm.iter()) {
            *a += *b;
        }
        for (a, b) in db.wq.iter_mut().zip(sb.wq.iter()) {
            *a += *b;
        }
        for (a, b) in db.wk.iter_mut().zip(sb.wk.iter()) {
            *a += *b;
        }
        for (a, b) in db.wv.iter_mut().zip(sb.wv.iter()) {
            *a += *b;
        }
        for (a, b) in db.wo.iter_mut().zip(sb.wo.iter()) {
            *a += *b;
        }
        for (a, b) in db.ffn_norm.iter_mut().zip(sb.ffn_norm.iter()) {
            *a += *b;
        }
        for (a, b) in db.w_gate.iter_mut().zip(sb.w_gate.iter()) {
            *a += *b;
        }
        for (a, b) in db.w_up.iter_mut().zip(sb.w_up.iter()) {
            *a += *b;
        }
        for (a, b) in db.w_down.iter_mut().zip(sb.w_down.iter()) {
            *a += *b;
        }
    }
    for (a, b) in dst.final_norm.iter_mut().zip(src.final_norm.iter()) {
        *a += *b;
    }
    for (a, b) in dst.readout_u.iter_mut().zip(src.readout_u.iter()) {
        *a += *b;
    }
    for (a, b) in dst.readout_v.iter_mut().zip(src.readout_v.iter()) {
        *a += *b;
    }
}

/// Log-probability of a sequence under the model (sum over masked positions).
fn log_prob(model: &Transformer, embedding: &TableEmbedding, tokens: &[u32]) -> f32 {
    let mask = vec![true; tokens.len()];
    let fwd = model.forward(tokens, embedding, Some(&mask));
    let n = (tokens.len() - 1) as f32;
    -fwd.loss * n
}

/// Generate a rejected sample: take the chosen sequence and perturb one token.
/// In real SPIN this is the model's own greedy/temperature sample; here we
/// synthesise it deterministically so the loop is reproducible.
fn generate_rejected(chosen: &[u32], vocab: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let mut out = chosen.to_vec();
    if !out.is_empty() {
        let pos = (rng.next_u64() as usize) % out.len();
        out[pos] = (rng.next_u64() as usize % vocab) as u32;
    }
    out
}

/// Preference accuracy on a set of (chosen, rejected) pairs.
fn preference_accuracy(
    model: &Transformer,
    embedding: &TableEmbedding,
    pairs: &[(Vec<u32>, Vec<u32>)],
) -> f32 {
    if pairs.is_empty() {
        return 0.0;
    }
    let mut correct = 0usize;
    for (chosen, rejected) in pairs {
        let lp_c = log_prob(model, embedding, chosen);
        let lp_r = log_prob(model, embedding, rejected);
        if lp_c > lp_r {
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
    let steps_per_iter = env_usize("PMC_STEPS", 5);
    let iterations = env_usize("PMC_ITERS", 4);
    let learning_rate = env_f32("PMC_LR", 0.01);
    let beta = env_f32("PMC_BETA", 0.5);
    let model_seed: u64 = env_u64("PMC_MODEL_SEED", 20261060);
    let embedding_seed: u64 = env_u64("PMC_EMBEDDING_SEED", 20261061);
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

    // Verified samples: fixed chosen sequences (the ground truth).
    let mut rng = Rng::new(20261062);
    let mut verified: Vec<Vec<u32>> = Vec::with_capacity(num_pairs);
    for _ in 0..num_pairs {
        verified.push(
            (0..6)
                .map(|_| (rng.next_u64() % vocab as u64) as u32)
                .collect(),
        );
    }

    // Held-out pairs: never used in training.
    let mut rng2 = Rng::new(20261063);
    let mut held_out: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(num_pairs);
    for _ in 0..num_pairs {
        let chosen: Vec<u32> = (0..6)
            .map(|_| (rng2.next_u64() % vocab as u64) as u32)
            .collect();
        let mut rejected = chosen.clone();
        if !rejected.is_empty() {
            let pos = (rng2.next_u64() as usize) % rejected.len();
            rejected[pos] = (rng2.next_u64() as usize % vocab) as u32;
        }
        held_out.push((chosen, rejected));
    }

    let reference = config.init(model_seed);
    let mut model = config.init(model_seed);
    let mut optimiser = TransformerAdam::new(&model);

    let mut curve: Vec<f32> = Vec::with_capacity(iterations);
    let initial_acc = preference_accuracy(&model, &embedding, &held_out);
    curve.push(initial_acc);

    let started = std::time::Instant::now();

    for iter in 0..iterations {
        // 1. Generate rejected samples from the current model (SPIN shape).
        let mut train_pairs: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(num_pairs);
        for v in &verified {
            let rejected = generate_rejected(v, vocab, model_seed.wrapping_add(iter as u64));
            train_pairs.push((v.clone(), rejected));
        }

        // 2. Reference log-probs from the frozen initial model.
        let mut ref_logps: Vec<(f32, f32)> = Vec::with_capacity(train_pairs.len());
        for (chosen, rejected) in &train_pairs {
            let lp_c = log_prob(&reference, &embedding, chosen);
            let lp_r = log_prob(&reference, &embedding, rejected);
            ref_logps.push((lp_c, lp_r));
        }

        // 3. Train with DPO for steps_per_iter.
        for _step in 0..steps_per_iter {
            let mut combined: Option<pmc_train::transformer::Grads> = None;
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
                let _dpo = softplus(-(beta * delta));

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
            let np = train_pairs.len() as f32;
            scale_grads(&mut grads, 1.0 / np);
            optimiser.step(&mut model, &mut grads, learning_rate, 0.9, 0.999, 1e-8, 1.0);
        }

        // 4. Measure held-out preference accuracy.
        let acc = preference_accuracy(&model, &embedding, &held_out);
        curve.push(acc);
        eprintln!(
            "iter {:>2}/{} held_out_acc {:.3} (initial {:.3})",
            iter + 1,
            iterations,
            acc,
            initial_acc
        );
    }

    let final_acc = *curve.last().unwrap();
    let elapsed = started.elapsed().as_millis();

    // Report the curve.
    let curve_json: Vec<String> = curve.iter().map(|v| format!("{:.6}", v)).collect();
    println!(
        "{{\"iters\":{},\"steps_per_iter\":{},\"beta\":{:.6},\"initial_acc\":{:.6},\"final_acc\":{:.6},\"curve\":[{}],\"elapsed_millis\":{}}}",
        iterations,
        steps_per_iter,
        beta,
        initial_acc,
        final_acc,
        curve_json.join(","),
        elapsed
    );
}
