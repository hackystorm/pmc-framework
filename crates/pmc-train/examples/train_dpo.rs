// SPDX-License-Identifier: GPL-3.0-only

//! Direct Preference Optimisation on a tiny transformer.
//!
//! SPIN-style self-improvement: the model's own samples enter training as
//! the *rejected* side, and a verified or previous-iteration sample is the
//! *chosen* side. Pure generative replay (self-generated positives) is not
//! used; see the prime directives. The loss is DPO:
//!
//!     L = -log sigma(beta * (log pi(y_w|x) - log pi(y_l|x)
//!                       - (log pi_ref(y_w|x) - log pi_ref(y_l|x))))
//!
//! The gradient is derived analytically from the existing backward pass:
//! dL/dtheta = beta * sigma(-beta*delta) * (n_w * grad_w - n_l * grad_l),
//! where grad_w and grad_l are the cross-entropy gradients of the chosen
//! and rejected sequences and n_w, n_l their masked position counts.

use pmc_core::Rng;
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn main() {
    let n_layer = env_usize("PMC_N_LAYER", 2);
    let d_model = env_usize("PMC_D_MODEL", 32);
    let n_head = env_usize("PMC_N_HEAD", 4);
    let ffn_hidden = env_usize("PMC_FFN", 64);
    let vocab = env_usize("PMC_VOCAB", 32);
    let n_positions = env_usize("PMC_N_POS", 16);
    let readout_rank = env_usize("PMC_RANK", 8);
    let steps = env_usize("PMC_STEPS", 20);
    let learning_rate = env_f32("PMC_LR", 0.01);
    let beta = env_f32("PMC_BETA", 0.1);
    let model_seed: u64 = env_u64("PMC_MODEL_SEED", 20261040);
    let embedding_seed: u64 = env_u64("PMC_EMBEDDING_SEED", 20261041);

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

    // Preference pairs: chosen is a fixed sequence; rejected is the same
    // sequence with one token changed. This is the SPIN shape - the model's
    // own sample is the rejected side.
    let chosen: Vec<u32> = (0..8).map(|i| (i * 3 + 1) as u32 % vocab as u32).collect();
    let mut rejected = chosen.clone();
    if rejected.len() > 2 {
        rejected[2] = (rejected[2] + 7) % vocab as u32;
    }
    let chosen_mask = vec![true; chosen.len()];
    let rejected_mask = vec![true; rejected.len()];
    let n_w = (chosen.len() - 1) as f32;
    let n_l = (rejected.len() - 1) as f32;

    // Reference log-probs from the frozen initial model.
    let reference = config.init(model_seed);
    let ref_fwd_w = reference.forward(&chosen, &embedding, Some(&chosen_mask));
    let ref_fwd_l = reference.forward(&rejected, &embedding, Some(&rejected_mask));
    let logp_ref_w = -ref_fwd_w.loss * n_w;
    let logp_ref_l = -ref_fwd_l.loss * n_l;

    // Trainable model starts from the same init.
    let mut model = config.init(model_seed);
    let mut optimiser = TransformerAdam::new(&model);

    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;
    let started = std::time::Instant::now();

    for step in 0..steps {
        // Chosen.
        let fwd_w = model.forward(&chosen, &embedding, Some(&chosen_mask));
        let mut grads_w = model.backward(&fwd_w);
        // Rejected.
        let fwd_l = model.forward(&rejected, &embedding, Some(&rejected_mask));
        let mut grads_l = model.backward(&fwd_l);

        let logp_w = -fwd_w.loss * n_w;
        let logp_l = -fwd_l.loss * n_l;
        let delta = (logp_w - logp_l) - (logp_ref_w - logp_ref_l);
        // -log sigma(x) = softplus(-x), computed stably.
        let stable_loss = softplus(-(beta * delta));

        if step == 0 {
            initial_loss = stable_loss;
        }
        final_loss = stable_loss;

        let scale = beta * sigmoid(-(beta * delta));
        // combined grads = scale * (n_w * grads_w - n_l * grads_l)
        // Scale each gradient slice.
        scale_grads(&mut grads_w, scale * n_w);
        scale_grads(&mut grads_l, -scale * n_l);
        // Add grads_l into grads_w.
        add_grads(&mut grads_w, &grads_l);

        optimiser.step(&mut model, &mut grads_w, learning_rate, 0.9, 0.999, 1e-8, 1.0);

        if step == 0 || step + 1 == steps || (step + 1) % 5 == 0 {
            eprintln!(
                "step {:>3}/{} dpo_loss {:.6} delta {:.6}",
                step + 1,
                steps,
                stable_loss,
                delta
            );
        }
    }

    let elapsed = started.elapsed().as_millis();
    println!(
        "{{\"steps\":{},\"beta\":{:.6},\"initial_dpo_loss\":{:.6},\"final_dpo_loss\":{:.6},\"elapsed_millis\":{}}}",
        steps, beta, initial_loss, final_loss, elapsed
    );
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
