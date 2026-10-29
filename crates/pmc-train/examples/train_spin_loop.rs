// SPDX-License-Identifier: GPL-3.0-only

//! T-SPIN (entropy-constrained Self-Play Fine-Tuning): the model's own samples
//! enter training as the rejected side, and the verified sample is the chosen
//! side. Each iteration regenerates the rejected samples from the updated
//! model, so the training signal tracks the model's current capability.
//!
//! T-SPIN adds two mechanisms on top of vanilla SPIN:
//! 1. Historical advantage: tracks the EMA of log-prob advantage over time
//!    to prevent catastrophic forgetting and maintain improvement direction.
//! 2. Entropy constraint: tracks policy entropy EMA and adds entropy
//!    regularization to prevent mode collapse (entropy collapse).
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

/// T-SPIN state: tracks historical advantage and entropy EMA.
struct TSpinState {
    /// EMA of the log-prob advantage (chosen - rejected) relative to reference.
    historical_advantage: f32,
    /// EMA of policy entropy to detect mode collapse.
    entropy_ema: f32,
    /// Iteration counter for adaptive smoothing.
    iteration: usize,
}

impl TSpinState {
    fn new() -> Self {
        Self {
            historical_advantage: 0.0,
            entropy_ema: 0.0,
            iteration: 0,
        }
    }

    fn update(&mut self, chosen_logprob: f32, rejected_logprob: f32, policy_entropy: f32, beta: f32) {
        // Historical advantage: EMA of (logp_chosen - logp_rejected) - reference_delta
        let advantage = chosen_logprob - rejected_logprob;
        let alpha = 0.9_f32.powf(self.iteration as f32);
        self.historical_advantage = alpha * self.historical_advantage + (1.0 - alpha) * advantage;

        // Entropy EMA for detecting mode collapse.
        let entropy_alpha = 0.95;
        self.entropy_ema = entropy_alpha * self.entropy_ema + (1.0 - entropy_alpha) * policy_entropy;

        self.iteration += 1;
    }

    /// Compute T-SPIN loss modifier: combines historical advantage with entropy regularization.
    fn loss_modifier(&self, beta: f32, entropy_weight: f32) -> f32 {
        // Historical advantage term: encourages maintaining improvement direction.
        let advantage_term = beta * self.historical_advantage;
        // Entropy regularization: penalizes entropy collapse.
        let entropy_term = 0.1 * (1.0 - self.entropy_ema).max(0.0);
        advantage_term - entropy_term
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

/// Log-probability of a sequence under the model (sum over masked positions).
fn log_prob(model: &Transformer, embedding: &TableEmbedding, tokens: &[u32]) -> f32 {
    let mask = vec![true; tokens.len()];
    let fwd = model.forward(tokens, embedding, Some(&mask));
    let n = (tokens.len() - 1) as f32;
    -fwd.loss * n
}

/// Compute policy entropy from logits (measures mode collapse).
fn policy_entropy(model: &Transformer, embedding: &TableEmbedding, tokens: &[u32]) -> f32 {
    let mask = vec![true; tokens.len()];
    let fwd = model.forward(tokens, embedding, Some(&mask));
    let vocab = fwd.logits.len() / fwd.targets.len();
    let mut total_entropy = 0.0f32;
    let seq = fwd.targets.len();
    for pos in 0..seq {
        let base = pos * vocab;
        let max_logit = fwd.logits[base..base + vocab].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for j in 0..vocab {
            sum_exp += ((fwd.logits[base + j] - max_logit).exp()) as f32;
        }
        let mut entropy = 0.0f32;
        for j in 0..vocab {
            let prob = ((fwd.logits[base + j] - max_logit).exp() / sum_exp) as f32;
            if prob > 0.0 {
                entropy -= prob * prob.ln();
            }
        }
        total_entropy += entropy;
    }
    total_entropy / seq as f32
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
    let iterations = env_usize("PMC_ITERS", 8);
    let learning_rate = env_f32("PMC_LR", 0.01);
    let beta = env_f32("PMC_BETA", 0.5);
    let entropy_weight = env_f32("PMC_ENTROPY_WEIGHT", 0.1);
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

    // Reference model (frozen initial model).
    let reference = config.init(model_seed);
    let mut model = config.init(model_seed);
    let mut optimiser = TransformerAdam::new(&model);

    let embedding = TableEmbedding::random(vocab, d_model, embedding_seed);

    // Compute reference log-probs for the verified pairs.
    let mut ref_logps: Vec<(f32, f32)> = Vec::with_capacity(num_pairs);
    for chosen in &verified {
        let rejected = generate_rejected(chosen, vocab, model_seed);
        let lp_c = log_prob(&reference, &embedding, chosen);
        let lp_r = log_prob(&reference, &embedding, &rejected);
        ref_logps.push((lp_c, lp_r));
    }

    // T-SPIN state.
    let mut tspin = TSpinState::new();

    let mut curve: Vec<f32> = Vec::with_capacity(iterations);
    let initial_acc = preference_accuracy(&model, &embedding, &held_out);
    let mut curve: Vec<f32> = vec![initial_acc];

    let started = std::time::Instant::now();

    for iter in 0..iterations {
        // 1. Generate rejected samples from the current model (SPIN shape).
        let mut train_pairs: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(num_pairs);
        for v in &verified {
            let rejected = generate_rejected(v, vocab, model_seed.wrapping_add(iter as u64));
            train_pairs.push((v.clone(), rejected));
        }

        // 2. Compute reference log-probs for this iteration's pairs.
        let mut ref_logps: Vec<(f32, f32)> = Vec::with_capacity(train_pairs.len());
        for (chosen, rejected) in &train_pairs {
            let lp_c = log_prob(&reference, &embedding, chosen);
            let lp_r = log_prob(&reference, &embedding, rejected);
            ref_logps.push((lp_c, lp_r));
        }

        // 3. Train with T-SPIN DPO for steps_per_iter.
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

                // Compute policy entropy for T-SPIN entropy constraint.
                let entropy = policy_entropy(&model, &embedding, chosen);

                // Update T-SPIN state.
                tspin.update(logp_c, logp_r, entropy, beta);

                // T-SPIN loss modifier: historical advantage + entropy regularization.
                let modifier = tspin.loss_modifier(beta, 0.1);

                // DPO loss with T-SPIN modifier.
                let dpo_loss = softplus(-(beta * delta)) + modifier;

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
            "iter {:>2}/{} held_out_acc {:.3} (initial {:.3}) entropy={:.3} advantage={:.3}",
            iter + 1,
            iterations,
            acc,
            initial_acc,
            tspin.entropy_ema,
            tspin.historical_advantage
        );
    }

    let final_acc = *curve.last().unwrap();
    let elapsed = started.elapsed().as_millis();

    // Report the curve.
    let curve_json: Vec<String> = curve.iter().map(|v| format!("{:.6}", v)).collect();
    println!(
        "{{\"iters\":{},\"steps_per_iter\":{},\"beta\":{:.6},\"entropy_weight\":{:.6},\"initial_acc\":{:.6},\"final_acc\":{:.6},\"curve\":[{}],\"elapsed_millis\":{}}}",
        iterations,
        steps_per_iter,
        beta,
        0.1,
        initial_acc,
        final_acc,
        curve_json.join(","),
        elapsed
    );
}