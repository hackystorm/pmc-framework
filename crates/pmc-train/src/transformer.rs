// SPDX-License-Identifier: GPL-3.0-only

//! The trainable PMC body: a small decoder-only transformer.
//!
//! The embeddings are frozen and supplied by an [`Embedding`] implementation;
//! everything above them is trained. The architecture is a standard
//! pre-norm decoder block:
//!
//! ```text
//! x = x + Attention(RMSNorm(x))
//! x = x + SwiGLU(RMSNorm(x))
//! h = RMSNorm(x)
//! z = h * U                    (d -> readout_rank)
//! logits = z * V               (readout_rank -> vocab)
//! ```
//!
//! The low-rank readout keeps the vocabulary projection small enough to train
//! on a CPU. Every projection, including the readout, is one batched matrix
//! product over all sequence positions. The backward pass is written out
//! explicitly, operation by operation, and uses the primitives in
//! [`crate::ops`].
//!
//! The loss is next-token prediction: position `i` is trained to predict
//! `tokens[i + 1]`. The final position has no target. Callers may pass a
//! mask for the other positions.

use pmc_core::Rng;

use crate::ops::{
    adam_step, cross_entropy, grad_clip, matmul_atb, matmul_tiled, matmul_tiled_bt, rmsnorm,
    rmsnorm_backward, rope, rope_backward, softmax_causal, softmax_causal_backward, swiglu,
    swiglu_backward,
};

const RMS_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10_000.0;

/// Architecture description. `d_model` must be divisible by `n_head` and the
/// resulting head dimension must be even, because RoPE rotates coordinate
/// pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransformerConfig {
    pub n_layer: usize,
    pub d_model: usize,
    pub n_head: usize,
    pub ffn_hidden: usize,
    pub vocab: usize,
    pub n_positions: usize,
    pub readout_rank: usize,
}

impl TransformerConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_head
    }

    /// Number of trainable parameters implied by the configuration.
    pub fn param_count(&self) -> usize {
        let d = self.d_model;
        let f = self.ffn_hidden;
        let per_block = 2 * d + 4 * d * d + 3 * d * f;
        self.n_layer * per_block + d + d * self.readout_rank + self.readout_rank * self.vocab
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_layer == 0 {
            return Err("n_layer must be positive".to_string());
        }
        if self.d_model == 0 {
            return Err("d_model must be positive".to_string());
        }
        if self.n_head == 0 {
            return Err("n_head must be positive".to_string());
        }
        if self.d_model % self.n_head != 0 {
            return Err("d_model must be divisible by n_head".to_string());
        }
        if self.head_dim() % 2 != 0 {
            return Err("head_dim must be even for RoPE".to_string());
        }
        if self.ffn_hidden == 0 {
            return Err("ffn_hidden must be positive".to_string());
        }
        if self.vocab == 0 {
            return Err("vocab must be positive".to_string());
        }
        if self.n_positions == 0 {
            return Err("n_positions must be positive".to_string());
        }
        if self.readout_rank == 0 {
            return Err("readout_rank must be positive".to_string());
        }
        Ok(())
    }

    /// Build a randomly initialised transformer from this configuration.
    pub fn init(&self, seed: u64) -> Transformer {
        Transformer::new(*self, seed)
    }
}

/// A frozen token embedding table.
///
/// The real implementation will read rows from the frozen LLM's GGUF file.
/// Tests use a seeded random table. A row is copied into `out`, whose length
/// is always [`Embedding::dim`].
pub trait Embedding {
    fn dim(&self) -> usize;
    fn row(&self, token: u32, out: &mut [f32]);
}

/// Trainable parameters of one pre-norm decoder block.
#[derive(Clone, Debug)]
pub struct Block {
    pub attn_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

/// Everything the backward pass needs from one block.
#[derive(Clone, Debug)]
pub struct BlockCache {
    pub input: Vec<f32>,
    pub attn_norm_out: Vec<f32>,
    pub attn_rstd: Vec<f32>,
    /// Query after RoPE, before the attention scale.
    pub q: Vec<f32>,
    /// Key after RoPE.
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// Causal attention probabilities, laid out as `[n_head, seq, seq]`.
    pub probs: Vec<f32>,
    pub attn_concat: Vec<f32>,
    pub x_after_attn: Vec<f32>,
    pub ffn_norm_out: Vec<f32>,
    pub ffn_rstd: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub swiglu: Vec<f32>,
}

/// Forward activations needed for the backward pass, plus the loss.
#[derive(Clone, Debug)]
pub struct ForwardOutput {
    pub seq: usize,
    /// Mean cross entropy over the masked positions. Zero if no position is
    /// masked in.
    pub loss: f32,
    /// Logits, laid out as `[seq, vocab]`.
    pub logits: Vec<f32>,
    /// Next-token targets. The final position is unused.
    pub targets: Vec<u32>,
    /// Positions that contribute to the loss. The final position is false.
    pub mask: Vec<bool>,
    pub blocks: Vec<BlockCache>,
    pub final_input: Vec<f32>,
    pub final_norm_out: Vec<f32>,
    pub final_rstd: Vec<f32>,
    /// `h * U`, laid out as `[seq, readout_rank]`.
    pub readout_hidden: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct BlockGrads {
    pub attn_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

/// Gradients for every trainable parameter.
///
/// The field order matches [`Transformer::params`] and [`Transformer::params_mut`],
/// which is what lets the optimiser and the checkpoint writer treat the model
/// as a list of slices.
#[derive(Clone, Debug)]
pub struct Grads {
    pub blocks: Vec<BlockGrads>,
    pub final_norm: Vec<f32>,
    pub readout_u: Vec<f32>,
    pub readout_v: Vec<f32>,
}

impl Grads {
    pub fn new(config: &TransformerConfig) -> Grads {
        let d = config.d_model;
        let f = config.ffn_hidden;
        let r = config.readout_rank;
        let mut blocks = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            blocks.push(BlockGrads {
                attn_norm: vec![0.0; d],
                wq: vec![0.0; d * d],
                wk: vec![0.0; d * d],
                wv: vec![0.0; d * d],
                wo: vec![0.0; d * d],
                ffn_norm: vec![0.0; d],
                w_gate: vec![0.0; d * f],
                w_up: vec![0.0; d * f],
                w_down: vec![0.0; f * d],
            });
        }
        Grads {
            blocks,
            final_norm: vec![0.0; d],
            readout_u: vec![0.0; d * r],
            readout_v: vec![0.0; r * config.vocab],
        }
    }

    pub fn slices(&self) -> Vec<&[f32]> {
        let mut out = Vec::new();
        for block in &self.blocks {
            out.push(&block.attn_norm[..]);
            out.push(&block.wq[..]);
            out.push(&block.wk[..]);
            out.push(&block.wv[..]);
            out.push(&block.wo[..]);
            out.push(&block.ffn_norm[..]);
            out.push(&block.w_gate[..]);
            out.push(&block.w_up[..]);
            out.push(&block.w_down[..]);
        }
        out.push(&self.final_norm[..]);
        out.push(&self.readout_u[..]);
        out.push(&self.readout_v[..]);
        out
    }

    pub fn slices_mut(&mut self) -> Vec<&mut [f32]> {
        let mut out = Vec::new();
        for block in &mut self.blocks {
            out.push(&mut block.attn_norm[..]);
            out.push(&mut block.wq[..]);
            out.push(&mut block.wk[..]);
            out.push(&mut block.wv[..]);
            out.push(&mut block.wo[..]);
            out.push(&mut block.ffn_norm[..]);
            out.push(&mut block.w_gate[..]);
            out.push(&mut block.w_up[..]);
            out.push(&mut block.w_down[..]);
        }
        out.push(&mut self.final_norm[..]);
        out.push(&mut self.readout_u[..]);
        out.push(&mut self.readout_v[..]);
        out
    }
}

/// A decoder-only transformer with a frozen token embedding.
#[derive(Clone, Debug)]
pub struct Transformer {
    pub config: TransformerConfig,
    pub blocks: Vec<Block>,
    pub final_norm: Vec<f32>,
    /// Readout up-projection, `[d_model, readout_rank]`.
    pub readout_u: Vec<f32>,
    /// Readout down-projection, `[readout_rank, vocab]`.
    pub readout_v: Vec<f32>,
}

fn he_fill(rng: &mut Rng, values: &mut [f32], fan_in: usize) {
    let scale = (2.0 / fan_in as f32).sqrt();
    for value in values.iter_mut() {
        *value = rng.normal() * scale;
    }
}

fn xavier_fill(rng: &mut Rng, values: &mut [f32], fan_in: usize, fan_out: usize) {
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    for value in values.iter_mut() {
        *value = rng.uniform(-limit, limit);
    }
}

impl Transformer {
    /// Alias for [`Transformer::new`], matching the configuration-level
    /// `init(seed)` constructor.
    pub fn init(config: TransformerConfig, seed: u64) -> Transformer {
        Transformer::new(config, seed)
    }

    /// Randomly initialise the model with He projections and Xavier readout.
    ///
    /// The same seed always produces the same parameters, because the
    /// generator is the deterministic [`pmc_core::Rng`].
    pub fn new(config: TransformerConfig, seed: u64) -> Transformer {
        if let Err(message) = config.validate() {
            panic!("invalid transformer configuration: {}", message);
        }
        let d = config.d_model;
        let f = config.ffn_hidden;
        let r = config.readout_rank;
        let mut rng = Rng::new(seed);

        let mut blocks = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            let mut wq = vec![0.0; d * d];
            let mut wk = vec![0.0; d * d];
            let mut wv = vec![0.0; d * d];
            let mut wo = vec![0.0; d * d];
            let mut w_gate = vec![0.0; d * f];
            let mut w_up = vec![0.0; d * f];
            let mut w_down = vec![0.0; f * d];
            he_fill(&mut rng, &mut wq, d);
            he_fill(&mut rng, &mut wk, d);
            he_fill(&mut rng, &mut wv, d);
            he_fill(&mut rng, &mut wo, d);
            he_fill(&mut rng, &mut w_gate, d);
            he_fill(&mut rng, &mut w_up, d);
            he_fill(&mut rng, &mut w_down, f);
            blocks.push(Block {
                attn_norm: vec![1.0; d],
                wq,
                wk,
                wv,
                wo,
                ffn_norm: vec![1.0; d],
                w_gate,
                w_up,
                w_down,
            });
        }

        let mut readout_u = vec![0.0; d * r];
        let mut readout_v = vec![0.0; r * config.vocab];
        xavier_fill(&mut rng, &mut readout_u, d, r);
        xavier_fill(&mut rng, &mut readout_v, r, config.vocab);

        Transformer {
            config,
            blocks,
            final_norm: vec![1.0; d],
            readout_u,
            readout_v,
        }
    }

    /// Allocate a model with zeroed parameters. Used by checkpoint loading,
    /// which fills the values from disk.
    pub fn zeros(config: TransformerConfig) -> Transformer {
        if let Err(message) = config.validate() {
            panic!("invalid transformer configuration: {}", message);
        }
        let d = config.d_model;
        let f = config.ffn_hidden;
        let r = config.readout_rank;
        let mut blocks = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            blocks.push(Block {
                attn_norm: vec![0.0; d],
                wq: vec![0.0; d * d],
                wk: vec![0.0; d * d],
                wv: vec![0.0; d * d],
                wo: vec![0.0; d * d],
                ffn_norm: vec![0.0; d],
                w_gate: vec![0.0; d * f],
                w_up: vec![0.0; d * f],
                w_down: vec![0.0; f * d],
            });
        }
        Transformer {
            config,
            blocks,
            final_norm: vec![0.0; d],
            readout_u: vec![0.0; d * r],
            readout_v: vec![0.0; r * config.vocab],
        }
    }

    pub fn param_count(&self) -> usize {
        self.params().iter().map(|param| param.len()).sum()
    }

    /// Parameters in the order expected by the optimiser and the checkpoint.
    pub fn params(&self) -> Vec<&[f32]> {
        let mut out = Vec::new();
        for block in &self.blocks {
            out.push(&block.attn_norm[..]);
            out.push(&block.wq[..]);
            out.push(&block.wk[..]);
            out.push(&block.wv[..]);
            out.push(&block.wo[..]);
            out.push(&block.ffn_norm[..]);
            out.push(&block.w_gate[..]);
            out.push(&block.w_up[..]);
            out.push(&block.w_down[..]);
        }
        out.push(&self.final_norm[..]);
        out.push(&self.readout_u[..]);
        out.push(&self.readout_v[..]);
        out
    }

    pub fn params_mut(&mut self) -> Vec<&mut [f32]> {
        let mut out = Vec::new();
        for block in &mut self.blocks {
            out.push(&mut block.attn_norm[..]);
            out.push(&mut block.wq[..]);
            out.push(&mut block.wk[..]);
            out.push(&mut block.wv[..]);
            out.push(&mut block.wo[..]);
            out.push(&mut block.ffn_norm[..]);
            out.push(&mut block.w_gate[..]);
            out.push(&mut block.w_up[..]);
            out.push(&mut block.w_down[..]);
        }
        out.push(&mut self.final_norm[..]);
        out.push(&mut self.readout_u[..]);
        out.push(&mut self.readout_v[..]);
        out
    }

    /// Concatenated parameter bytes, little-endian, in [`Transformer::params`]
    /// order.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.param_count() * 4);
        for param in self.params() {
            for value in param {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        out
    }

    /// Run the model and compute next-token cross entropy.
    ///
    /// `loss_mask`, when present, must have one entry per token. Position `i`
    /// contributes when `loss_mask[i]` is true and `i + 1 < tokens.len()`.
    /// Passing `None` trains on every position that has a next token.
    pub fn forward(
        &self,
        tokens: &[u32],
        embed: &dyn Embedding,
        loss_mask: Option<&[bool]>,
    ) -> ForwardOutput {
        let seq = tokens.len();
        assert!(seq > 0, "the transformer needs at least one token");
        assert!(
            seq <= self.config.n_positions,
            "sequence length {} exceeds n_positions {}",
            seq,
            self.config.n_positions
        );
        let d = self.config.d_model;
        let f = self.config.ffn_hidden;
        let n_head = self.config.n_head;
        let head_dim = self.config.head_dim();
        let vocab = self.config.vocab;
        let rank = self.config.readout_rank;
        assert_eq!(
            embed.dim(),
            d,
            "embedding dimension {} does not match d_model {}",
            embed.dim(),
            d
        );
        if let Some(mask) = loss_mask {
            assert_eq!(mask.len(), seq, "loss mask has the wrong length");
        }

        let mut x = vec![0.0f32; seq * d];
        for position in 0..seq {
            embed.row(tokens[position], &mut x[position * d..(position + 1) * d]);
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut blocks = Vec::with_capacity(self.config.n_layer);

        for block in &self.blocks {
            let input = x.clone();
            let mut attn_norm_out = vec![0.0f32; seq * d];
            let mut attn_rstd = vec![0.0f32; seq];
            rmsnorm(
                &input,
                &block.attn_norm,
                RMS_EPS,
                &mut attn_norm_out,
                &mut attn_rstd,
            );

            let mut q = vec![0.0f32; seq * d];
            let mut k = vec![0.0f32; seq * d];
            let mut v = vec![0.0f32; seq * d];
            matmul_tiled(&attn_norm_out, &block.wq, &mut q, seq, d, d);
            matmul_tiled(&attn_norm_out, &block.wk, &mut k, seq, d, d);
            matmul_tiled(&attn_norm_out, &block.wv, &mut v, seq, d, d);

            let mut q_rope = vec![0.0f32; seq * d];
            let mut k_rope = vec![0.0f32; seq * d];
            rope(&q, n_head, head_dim, 0, ROPE_THETA, &mut q_rope);
            rope(&k, n_head, head_dim, 0, ROPE_THETA, &mut k_rope);

            let mut probs = vec![0.0f32; n_head * seq * seq];
            let mut attn_concat = vec![0.0f32; seq * d];
            let mut q_head = vec![0.0f32; seq * head_dim];
            let mut k_head = vec![0.0f32; seq * head_dim];
            let mut v_head = vec![0.0f32; seq * head_dim];
            let mut scores = vec![0.0f32; seq * seq];
            let mut attn_head = vec![0.0f32; seq * head_dim];

            for head in 0..n_head {
                for position in 0..seq {
                    let source = position * d + head * head_dim;
                    let target = position * head_dim;
                    let q_source = &q_rope[source..source + head_dim];
                    let k_source = &k_rope[source..source + head_dim];
                    let v_source = &v[source..source + head_dim];
                    q_head[target..target + head_dim].copy_from_slice(q_source);
                    k_head[target..target + head_dim].copy_from_slice(k_source);
                    v_head[target..target + head_dim].copy_from_slice(v_source);
                }

                matmul_tiled_bt(&q_head, &k_head, &mut scores, seq, head_dim, seq);
                for score in scores.iter_mut() {
                    *score *= scale;
                }
                let probs_head = &mut probs[head * seq * seq..(head + 1) * seq * seq];
                softmax_causal(&scores, seq, probs_head);
                matmul_tiled(probs_head, &v_head, &mut attn_head, seq, seq, head_dim);

                for position in 0..seq {
                    let target = position * d + head * head_dim;
                    let source = position * head_dim;
                    attn_concat[target..target + head_dim]
                        .copy_from_slice(&attn_head[source..source + head_dim]);
                }
            }

            let mut attn_proj = vec![0.0f32; seq * d];
            matmul_tiled(&attn_concat, &block.wo, &mut attn_proj, seq, d, d);
            let mut x_after_attn = vec![0.0f32; seq * d];
            for i in 0..seq * d {
                x_after_attn[i] = input[i] + attn_proj[i];
            }

            let mut ffn_norm_out = vec![0.0f32; seq * d];
            let mut ffn_rstd = vec![0.0f32; seq];
            rmsnorm(
                &x_after_attn,
                &block.ffn_norm,
                RMS_EPS,
                &mut ffn_norm_out,
                &mut ffn_rstd,
            );
            let mut gate = vec![0.0f32; seq * f];
            let mut up = vec![0.0f32; seq * f];
            matmul_tiled(&ffn_norm_out, &block.w_gate, &mut gate, seq, d, f);
            matmul_tiled(&ffn_norm_out, &block.w_up, &mut up, seq, d, f);
            let mut swiglu_out = vec![0.0f32; seq * f];
            swiglu(&gate, &up, &mut swiglu_out);
            let mut down = vec![0.0f32; seq * d];
            matmul_tiled(&swiglu_out, &block.w_down, &mut down, seq, f, d);

            let mut output = vec![0.0f32; seq * d];
            for i in 0..seq * d {
                output[i] = x_after_attn[i] + down[i];
            }
            x = output;

            blocks.push(BlockCache {
                input,
                attn_norm_out,
                attn_rstd,
                q: q_rope,
                k: k_rope,
                v,
                probs,
                attn_concat,
                x_after_attn,
                ffn_norm_out,
                ffn_rstd,
                gate,
                up,
                swiglu: swiglu_out,
            });
        }

        let final_input = x.clone();
        let mut final_norm_out = vec![0.0f32; seq * d];
        let mut final_rstd = vec![0.0f32; seq];
        rmsnorm(
            &final_input,
            &self.final_norm,
            RMS_EPS,
            &mut final_norm_out,
            &mut final_rstd,
        );
        let mut readout_hidden = vec![0.0f32; seq * rank];
        matmul_tiled(&final_norm_out, &self.readout_u, &mut readout_hidden, seq, d, rank);
        let mut logits = vec![0.0f32; seq * vocab];
        // V is [rank, vocab], so logits = z * V, one batched product over all
        // positions. Using a per-position GEMV here would be memory bound.
        matmul_tiled(&readout_hidden, &self.readout_v, &mut logits, seq, rank, vocab);

        let mut targets = vec![0u32; seq];
        let mut mask = vec![false; seq];
        for position in 0..seq.saturating_sub(1) {
            targets[position] = tokens[position + 1];
            mask[position] = match loss_mask {
                Some(values) => values[position],
                None => true,
            };
        }
        let (loss, _) = cross_entropy(&logits, &targets, &mask);

        ForwardOutput {
            seq,
            loss,
            logits,
            targets,
            mask,
            blocks,
            final_input,
            final_norm_out,
            final_rstd,
            readout_hidden,
        }
    }

    /// Backward pass with next-token cross entropy as the objective.
    pub fn backward(&mut self, forward: &ForwardOutput) -> Grads {
        let (_, dlogits) = cross_entropy(&forward.logits, &forward.targets, &forward.mask);
        self.backward_with_dlogits(forward, &dlogits)
    }

    /// Backward pass with a caller-supplied gradient on the logits.
    ///
    /// This is exposed because it makes the whole model differentiable with
    /// respect to an arbitrary scalar objective, which is what the
    /// finite-difference tests use. Training normally calls [`Self::backward`].
    pub fn backward_with_dlogits(&mut self, forward: &ForwardOutput, dlogits: &[f32]) -> Grads {
        let seq = forward.seq;
        let d = self.config.d_model;
        let f = self.config.ffn_hidden;
        let n_head = self.config.n_head;
        let head_dim = self.config.head_dim();
        let vocab = self.config.vocab;
        let rank = self.config.readout_rank;
        assert_eq!(
            forward.blocks.len(),
            self.blocks.len(),
            "forward cache does not match the model depth"
        );
        assert_eq!(
            dlogits.len(),
            seq * vocab,
            "logit gradient has the wrong size"
        );

        let mut grads = Grads::new(&self.config);

        // Readout: logits = z * V, z = h * U.
        matmul_atb(
            &forward.readout_hidden,
            dlogits,
            &mut grads.readout_v,
            rank,
            seq,
            vocab,
        );
        let mut d_readout_hidden = vec![0.0f32; seq * rank];
        matmul_tiled_bt(
            dlogits,
            &self.readout_v,
            &mut d_readout_hidden,
            seq,
            vocab,
            rank,
        );
        matmul_atb(
            &forward.final_norm_out,
            &d_readout_hidden,
            &mut grads.readout_u,
            d,
            seq,
            rank,
        );
        let mut d_final_norm = vec![0.0f32; seq * d];
        matmul_tiled_bt(
            &d_readout_hidden,
            &self.readout_u,
            &mut d_final_norm,
            seq,
            rank,
            d,
        );
        let mut dx = vec![0.0f32; seq * d];
        rmsnorm_backward(
            &d_final_norm,
            &forward.final_input,
            &self.final_norm,
            &forward.final_rstd,
            &mut dx,
            &mut grads.final_norm,
        );

        let scale = 1.0 / (head_dim as f32).sqrt();
        for layer in (0..self.blocks.len()).rev() {
            let block = &self.blocks[layer];
            let cache = &forward.blocks[layer];
            let block_grads = &mut grads.blocks[layer];

            // FFN: x = x1 + swiglu(x1 * W_gate, x1 * W_up) * W_down.
            let mut d_swiglu = vec![0.0f32; seq * f];
            matmul_tiled_bt(&dx, &block.w_down, &mut d_swiglu, seq, d, f);
            matmul_atb(
                &cache.swiglu,
                &dx,
                &mut block_grads.w_down,
                f,
                seq,
                d,
            );

            let mut d_gate = vec![0.0f32; seq * f];
            let mut d_up = vec![0.0f32; seq * f];
            swiglu_backward(
                &cache.gate,
                &cache.up,
                &d_swiglu,
                &mut d_gate,
                &mut d_up,
            );
            matmul_atb(
                &cache.ffn_norm_out,
                &d_gate,
                &mut block_grads.w_gate,
                d,
                seq,
                f,
            );
            matmul_atb(
                &cache.ffn_norm_out,
                &d_up,
                &mut block_grads.w_up,
                d,
                seq,
                f,
            );

            let mut d_ffn_norm = vec![0.0f32; seq * d];
            let mut temp = vec![0.0f32; seq * d];
            matmul_tiled_bt(&d_gate, &block.w_gate, &mut d_ffn_norm, seq, f, d);
            matmul_tiled_bt(&d_up, &block.w_up, &mut temp, seq, f, d);
            for i in 0..seq * d {
                d_ffn_norm[i] += temp[i];
            }
            let mut d_after_attn = vec![0.0f32; seq * d];
            rmsnorm_backward(
                &d_ffn_norm,
                &cache.x_after_attn,
                &block.ffn_norm,
                &cache.ffn_rstd,
                &mut d_after_attn,
                &mut block_grads.ffn_norm,
            );

            // Residual from the FFN branch plus the residual stream itself.
            let mut d_x1 = vec![0.0f32; seq * d];
            for i in 0..seq * d {
                d_x1[i] = dx[i] + d_after_attn[i];
            }

            // Attention output projection.
            matmul_atb(
                &cache.attn_concat,
                &d_x1,
                &mut block_grads.wo,
                d,
                seq,
                d,
            );
            let mut d_attn_concat = vec![0.0f32; seq * d];
            matmul_tiled_bt(&d_x1, &block.wo, &mut d_attn_concat, seq, d, d);

            // Per-head attention. The projections above are batched; the
            // attention itself is naturally a batch of small matrices.
            let mut d_q = vec![0.0f32; seq * d];
            let mut d_k = vec![0.0f32; seq * d];
            let mut d_v = vec![0.0f32; seq * d];
            let mut q_head = vec![0.0f32; seq * head_dim];
            let mut k_head = vec![0.0f32; seq * head_dim];
            let mut v_head = vec![0.0f32; seq * head_dim];
            let mut d_attn_head = vec![0.0f32; seq * head_dim];
            let mut d_probs = vec![0.0f32; seq * seq];
            let mut d_scores = vec![0.0f32; seq * seq];
            let mut d_q_head = vec![0.0f32; seq * head_dim];
            let mut d_k_head = vec![0.0f32; seq * head_dim];
            let mut d_v_head = vec![0.0f32; seq * head_dim];

            for head in 0..n_head {
                for position in 0..seq {
                    let target = position * head_dim;
                    let source = position * d + head * head_dim;
                    let q_source = &cache.q[source..source + head_dim];
                    let k_source = &cache.k[source..source + head_dim];
                    let v_source = &cache.v[source..source + head_dim];
                    let dy_source = &d_attn_concat[source..source + head_dim];
                    q_head[target..target + head_dim].copy_from_slice(q_source);
                    k_head[target..target + head_dim].copy_from_slice(k_source);
                    v_head[target..target + head_dim].copy_from_slice(v_source);
                    d_attn_head[target..target + head_dim].copy_from_slice(dy_source);
                }

                let probs_head = &cache.probs[head * seq * seq..(head + 1) * seq * seq];
                matmul_tiled_bt(
                    &d_attn_head,
                    &v_head,
                    &mut d_probs,
                    seq,
                    head_dim,
                    seq,
                );
                matmul_atb(
                    probs_head,
                    &d_attn_head,
                    &mut d_v_head,
                    seq,
                    seq,
                    head_dim,
                );
                softmax_causal_backward(probs_head, &d_probs, seq, &mut d_scores);
                matmul_tiled(&d_scores, &k_head, &mut d_q_head, seq, seq, head_dim);
                matmul_atb(&d_scores, &q_head, &mut d_k_head, seq, seq, head_dim);

                for position in 0..seq {
                    let target = position * d + head * head_dim;
                    let source = position * head_dim;
                    for j in 0..head_dim {
                        d_q[target + j] += d_q_head[source + j] * scale;
                        d_k[target + j] += d_k_head[source + j] * scale;
                        d_v[target + j] += d_v_head[source + j];
                    }
                }
            }

            let mut d_q_pre_rope = vec![0.0f32; seq * d];
            let mut d_k_pre_rope = vec![0.0f32; seq * d];
            rope_backward(&d_q, n_head, head_dim, 0, ROPE_THETA, &mut d_q_pre_rope);
            rope_backward(&d_k, n_head, head_dim, 0, ROPE_THETA, &mut d_k_pre_rope);

            matmul_atb(
                &cache.attn_norm_out,
                &d_q_pre_rope,
                &mut block_grads.wq,
                d,
                seq,
                d,
            );
            matmul_atb(
                &cache.attn_norm_out,
                &d_k_pre_rope,
                &mut block_grads.wk,
                d,
                seq,
                d,
            );
            matmul_atb(
                &cache.attn_norm_out,
                &d_v,
                &mut block_grads.wv,
                d,
                seq,
                d,
            );

            let mut d_attn_norm = vec![0.0f32; seq * d];
            matmul_tiled_bt(&d_q_pre_rope, &block.wq, &mut d_attn_norm, seq, d, d);
            matmul_tiled_bt(&d_k_pre_rope, &block.wk, &mut temp, seq, d, d);
            for i in 0..seq * d {
                d_attn_norm[i] += temp[i];
            }
            matmul_tiled_bt(&d_v, &block.wv, &mut temp, seq, d, d);
            for i in 0..seq * d {
                d_attn_norm[i] += temp[i];
            }

            let mut d_block_input = vec![0.0f32; seq * d];
            rmsnorm_backward(
                &d_attn_norm,
                &cache.input,
                &block.attn_norm,
                &cache.attn_rstd,
                &mut d_block_input,
                &mut block_grads.attn_norm,
            );
            for i in 0..seq * d {
                d_block_input[i] += d_x1[i];
            }
            dx = d_block_input;
        }

        // `dx` is the gradient with respect to the frozen embedding output.
        // There is nothing to apply it to, so the backward pass stops here.
        grads
    }
}

/// Adam state for a whole transformer.
///
/// The per-slice update is in [`crate::ops::adam_step`]. This wrapper keeps
/// the moment buffers in the same order as [`Transformer::params`] and
/// applies global norm clipping before each update.
pub struct TransformerAdam {
    first_moment: Vec<Vec<f32>>,
    second_moment: Vec<Vec<f32>>,
    step: u32,
}

impl TransformerAdam {
    pub fn new(model: &Transformer) -> TransformerAdam {
        let params = model.params();
        TransformerAdam {
            first_moment: params.iter().map(|param| vec![0.0; param.len()]).collect(),
            second_moment: params.iter().map(|param| vec![0.0; param.len()]).collect(),
            step: 0,
        }
    }

    pub fn step_count(&self) -> u32 {
        self.step
    }

    /// Apply one Adam update. Returns the gradient norm before clipping.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        model: &mut Transformer,
        grads: &mut Grads,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        max_norm: f32,
    ) -> f32 {
        let mut grad_slices = grads.slices_mut();
        let norm = grad_clip(&mut grad_slices, max_norm);
        self.step = self.step.saturating_add(1);

        let mut params = model.params_mut();
        assert_eq!(
            params.len(),
            grad_slices.len(),
            "parameter and gradient counts differ"
        );
        assert_eq!(
            params.len(),
            self.first_moment.len(),
            "optimiser state does not match the model"
        );

        for i in 0..params.len() {
            let param = &mut *params[i];
            let grad = &*grad_slices[i];
            adam_step(
                param,
                grad,
                &mut self.first_moment[i],
                &mut self.second_moment[i],
                self.step,
                learning_rate,
                beta1,
                beta2,
                eps,
            );
        }
        norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::matmul_tiled;
    use pmc_core::Rng;

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

    fn to_f64(values: &[f32]) -> Vec<f64> {
        values.iter().map(|value| *value as f64).collect()
    }

    fn numeric_gradient<F>(params: &[f64], mut loss: F, eps: f64) -> Vec<f64>
    where
        F: FnMut(&[f64]) -> f64,
    {
        let mut point = params.to_vec();
        let mut gradient = vec![0.0f64; params.len()];
        for i in 0..params.len() {
            let original = point[i];
            point[i] = original + eps;
            let plus = loss(&point);
            point[i] = original - eps;
            let minus = loss(&point);
            point[i] = original;
            gradient[i] = (plus - minus) / (2.0 * eps);
        }
        gradient
    }

    fn assert_relative(analytic: &[f32], numeric: &[f64], label: &str) {
        assert_eq!(analytic.len(), numeric.len(), "{} length mismatch", label);
        let mut worst = 0.0f64;
        let mut worst_index = 0usize;
        for i in 0..analytic.len() {
            let a = analytic[i] as f64;
            let n = numeric[i];
            let denominator = a.abs().max(n.abs()).max(1e-4);
            let error = (a - n).abs() / denominator;
            if error > worst {
                worst = error;
                worst_index = i;
            }
        }
        assert!(
            worst < 1e-2,
            "{}: relative error {:.3e} at index {} (analytic {}, numeric {})",
            label,
            worst,
            worst_index,
            analytic[worst_index],
            numeric[worst_index],
        );
    }

    #[test]
    fn param_count_matches_config() {
        let config = TransformerConfig {
            n_layer: 2,
            d_model: 8,
            n_head: 2,
            ffn_hidden: 16,
            vocab: 32,
            n_positions: 64,
            readout_rank: 4,
        };
        let mut model = config.init(7);
        let per_block = 2 * 8 + 4 * 8 * 8 + 3 * 8 * 16;
        let expected = 2 * per_block + 8 + 8 * 4 + 4 * 32;
        assert_eq!(model.param_count(), expected);
        assert_eq!(config.param_count(), expected);
        assert_eq!(model.params().len(), 2 * 9 + 3);
        assert_eq!(model.params_mut().len(), 2 * 9 + 3);
        assert_eq!(model.to_bytes().len(), expected * 4);
    }

    #[test]
    fn gradient_low_rank_readout_matches_finite_differences() {
        let mut rng = Rng::new(31);
        let seq = 3usize;
        let d = 5usize;
        let rank = 2usize;
        let vocab = 6usize;
        let hidden: Vec<f32> = (0..seq * d).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let u: Vec<f32> = (0..d * rank).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let v: Vec<f32> = (0..rank * vocab).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let weight: Vec<f32> = (0..seq * vocab).map(|_| rng.uniform(-1.0, 1.0)).collect();

        let mut z = vec![0.0f32; seq * rank];
        matmul_tiled(&hidden, &u, &mut z, seq, d, rank);
        let mut logits = vec![0.0f32; seq * vocab];
        matmul_tiled_bt(&z, &v, &mut logits, seq, rank, vocab);

        let mut dz = vec![0.0f32; seq * rank];
        matmul_tiled_bt(&weight, &v, &mut dz, seq, vocab, rank);
        let mut du = vec![0.0f32; d * rank];
        matmul_atb(&hidden, &dz, &mut du, d, seq, rank);
        let mut dv = vec![0.0f32; rank * vocab];
        matmul_atb(&z, &weight, &mut dv, rank, seq, vocab);
        let mut dh = vec![0.0f32; seq * d];
        matmul_tiled_bt(&dz, &u, &mut dh, seq, rank, d);

        let mut params = to_f64(&hidden);
        params.extend_from_slice(&to_f64(&u));
        params.extend_from_slice(&to_f64(&v));
        let weight64 = to_f64(&weight);
        let reference = |point: &[f64]| -> f64 {
            let hidden = &point[0..seq * d];
            let u = &point[seq * d..seq * d + d * rank];
            let v = &point[seq * d + d * rank..seq * d + d * rank + rank * vocab];
            let mut loss = 0.0f64;
            for i in 0..seq {
                for r in 0..rank {
                    let mut z_value = 0.0f64;
                    for p in 0..d {
                        z_value += hidden[i * d + p] * u[p * rank + r];
                    }
                    for j in 0..vocab {
                        loss += weight64[i * vocab + j] * z_value * v[r * vocab + j];
                    }
                }
            }
            loss
        };
        let numeric = numeric_gradient(&params, reference, 1e-3);

        let mut analytic = dh;
        analytic.extend_from_slice(&du);
        analytic.extend_from_slice(&dv);
        assert_relative(&analytic, &numeric, "low-rank readout");
    }

    #[test]
    fn overfit_tiny_sequence() {
        let config = TransformerConfig {
            n_layer: 2,
            d_model: 16,
            n_head: 2,
            ffn_hidden: 32,
            vocab: 16,
            n_positions: 64,
            readout_rank: 16,
        };
        let embedding = TableEmbedding::random(config.vocab, config.d_model, 123);
        let mut rng = Rng::new(456);
        let tokens: Vec<u32> = (0..32)
            .map(|_| (rng.next_u64() % config.vocab as u64) as u32)
            .collect();

        let mut model = config.init(789);
        let mut optimiser = TransformerAdam::new(&model);
        let mask = vec![true; tokens.len()];
        let mut initial_loss = 0.0f32;
        let mut final_loss = 0.0f32;

        for step in 0..300 {
            let forward = model.forward(&tokens, &embedding, Some(&mask));
            if step == 0 {
                initial_loss = forward.loss;
            }
            final_loss = forward.loss;
            let mut grads = model.backward(&forward);
            optimiser.step(
                &mut model,
                &mut grads,
                0.01,
                0.9,
                0.999,
                1e-8,
                1.0,
            );
        }

        assert!(
            initial_loss > final_loss,
            "training did not reduce the loss: {} -> {}",
            initial_loss,
            final_loss
        );
        println!(
            "overfit tiny sequence: initial loss {:.6}, final loss {:.6}",
            initial_loss, final_loss
        );
        assert!(
            final_loss < 0.05,
            "final loss {} is not below 0.05 (initial {})",
            final_loss,
            initial_loss
        );
    }

    fn reference_matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; m * n];
        for i in 0..m {
            for p in 0..k {
                let left = a[i * k + p];
                for j in 0..n {
                    out[i * n + j] += left * b[p * n + j];
                }
            }
        }
        out
    }

    fn reference_matmul_bt(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    sum += a[i * k + p] * b[j * k + p];
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    fn reference_rmsnorm(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
        let d = w.len();
        let rows = x.len() / d;
        let mut out = vec![0.0f64; x.len()];
        for row in 0..rows {
            let mut sum = 0.0f64;
            for j in 0..d {
                sum += x[row * d + j] * x[row * d + j];
            }
            let inv = 1.0 / (sum / d as f64 + eps).sqrt();
            for j in 0..d {
                out[row * d + j] = x[row * d + j] * inv * w[j];
            }
        }
        out
    }

    fn reference_rope(x: &[f64], n_head: usize, head_dim: usize, theta: f64) -> Vec<f64> {
        let width = n_head * head_dim;
        let half = head_dim / 2;
        let rows = x.len() / width;
        let mut out = vec![0.0f64; x.len()];
        for row in 0..rows {
            let position = row as f64;
            for head in 0..n_head {
                let base = row * width + head * head_dim;
                for i in 0..half {
                    let frequency = theta.powf(-((2 * i) as f64) / head_dim as f64);
                    let (sin, cos) = (position * frequency).sin_cos();
                    let first = x[base + i];
                    let second = x[base + i + half];
                    out[base + i] = first * cos - second * sin;
                    out[base + i + half] = first * sin + second * cos;
                }
            }
        }
        out
    }

    fn reference_softmax_causal(scores: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; n * n];
        for row in 0..n {
            let mut max = f64::NEG_INFINITY;
            for j in 0..=row {
                max = max.max(scores[row * n + j]);
            }
            let mut sum = 0.0f64;
            for j in 0..=row {
                out[row * n + j] = (scores[row * n + j] - max).exp();
                sum += out[row * n + j];
            }
            for j in 0..=row {
                out[row * n + j] /= sum;
            }
        }
        out
    }

    fn reference_swiglu(gate: &[f64], up: &[f64]) -> Vec<f64> {
        gate.iter()
            .zip(up.iter())
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect()
    }

    /// Independent f64 implementation of the forward pass, used only by the
    /// gradient test. Parameter order is the same as `Transformer::params`.
    fn reference_logits(
        point: &[f64],
        config: &TransformerConfig,
        embedding: &[f64],
        tokens: &[u32],
    ) -> Vec<f64> {
        let seq = tokens.len();
        let d = config.d_model;
        let f = config.ffn_hidden;
        let n_head = config.n_head;
        let head_dim = config.head_dim();
        let vocab = config.vocab;
        let rank = config.readout_rank;
        let eps = 1e-5f64;
        let theta = 10_000.0f64;

        let mut x = vec![0.0f64; seq * d];
        for (position, token) in tokens.iter().enumerate() {
            let start = *token as usize * d;
            x[position * d..(position + 1) * d]
                .copy_from_slice(&embedding[start..start + d]);
        }

        let mut offset = 0usize;
        for _ in 0..config.n_layer {
            let attn_norm = &point[offset..offset + d];
            offset += d;
            let wq = &point[offset..offset + d * d];
            offset += d * d;
            let wk = &point[offset..offset + d * d];
            offset += d * d;
            let wv = &point[offset..offset + d * d];
            offset += d * d;
            let wo = &point[offset..offset + d * d];
            offset += d * d;
            let ffn_norm = &point[offset..offset + d];
            offset += d;
            let w_gate = &point[offset..offset + d * f];
            offset += d * f;
            let w_up = &point[offset..offset + d * f];
            offset += d * f;
            let w_down = &point[offset..offset + f * d];
            offset += f * d;

            let normalized = reference_rmsnorm(&x, attn_norm, eps);
            let q = reference_rope(&reference_matmul(&normalized, wq, seq, d, d), n_head, head_dim, theta);
            let k = reference_rope(&reference_matmul(&normalized, wk, seq, d, d), n_head, head_dim, theta);
            let v = reference_matmul(&normalized, wv, seq, d, d);

            let scale = 1.0 / (head_dim as f64).sqrt();
            let mut concat = vec![0.0f64; seq * d];
            for head in 0..n_head {
                let mut q_head = vec![0.0f64; seq * head_dim];
                let mut k_head = vec![0.0f64; seq * head_dim];
                let mut v_head = vec![0.0f64; seq * head_dim];
                for position in 0..seq {
                    let source = position * d + head * head_dim;
                    let target = position * head_dim;
                    q_head[target..target + head_dim]
                        .copy_from_slice(&q[source..source + head_dim]);
                    k_head[target..target + head_dim]
                        .copy_from_slice(&k[source..source + head_dim]);
                    v_head[target..target + head_dim]
                        .copy_from_slice(&v[source..source + head_dim]);
                }
                let mut scores = reference_matmul_bt(&q_head, &k_head, seq, head_dim, seq);
                for score in scores.iter_mut() {
                    *score *= scale;
                }
                let probs = reference_softmax_causal(&scores, seq);
                let attention = reference_matmul(&probs, &v_head, seq, seq, head_dim);
                for position in 0..seq {
                    let target = position * d + head * head_dim;
                    let source = position * head_dim;
                    concat[target..target + head_dim]
                        .copy_from_slice(&attention[source..source + head_dim]);
                }
            }

            let attention_projection = reference_matmul(&concat, wo, seq, d, d);
            for i in 0..seq * d {
                x[i] += attention_projection[i];
            }

            let normalized = reference_rmsnorm(&x, ffn_norm, eps);
            let gate = reference_matmul(&normalized, w_gate, seq, d, f);
            let up = reference_matmul(&normalized, w_up, seq, d, f);
            let activated = reference_swiglu(&gate, &up);
            let down = reference_matmul(&activated, w_down, seq, f, d);
            for i in 0..seq * d {
                x[i] += down[i];
            }
        }

        let final_norm = &point[offset..offset + d];
        offset += d;
        let u = &point[offset..offset + d * rank];
        offset += d * rank;
        let v = &point[offset..offset + rank * vocab];
        offset += rank * vocab;
        assert_eq!(offset, point.len(), "reference consumed the wrong parameter count");

        let normalized = reference_rmsnorm(&x, final_norm, eps);
        let hidden = reference_matmul(&normalized, u, seq, d, rank);
        reference_matmul(&hidden, v, seq, rank, vocab)
    }

    #[test]
    fn full_model_gradient_matches_finite_differences() {
        // The analytic gradients come from the f32 model. The numerical
        // gradients come from an independent f64 implementation of the same
        // forward pass, so this checks the complete backward composition
        // without f32 finite-difference noise.
        let config = TransformerConfig {
            n_layer: 1,
            d_model: 8,
            n_head: 2,
            ffn_hidden: 16,
            vocab: 16,
            n_positions: 16,
            readout_rank: 4,
        };
        let embedding = TableEmbedding::random(config.vocab, config.d_model, 77);
        let embedding64: Vec<f64> = embedding.table.iter().map(|value| *value as f64).collect();
        let tokens = [1u32, 5, 2, 9];
        let mut model = config.init(88);
        let forward = model.forward(&tokens, &embedding, None);
        let mut rng = Rng::new(99);
        let dlogits32: Vec<f32> = (0..forward.logits.len())
            .map(|_| rng.uniform(-1.0, 1.0))
            .collect();
        let dlogits64: Vec<f64> = dlogits32.iter().map(|value| *value as f64).collect();
        let grads = model.backward_with_dlogits(&forward, &dlogits32);
        let analytic: Vec<f32> = grads
            .slices()
            .iter()
            .flat_map(|slice| slice.iter().copied())
            .collect();
        let parameters: Vec<f64> = model
            .params()
            .iter()
            .flat_map(|slice| slice.iter().map(|value| *value as f64))
            .collect();
        assert_eq!(analytic.len(), parameters.len());

        let reference = |point: &[f64]| -> f64 {
            let logits = reference_logits(point, &config, &embedding64, &tokens);
            logits
                .iter()
                .zip(dlogits64.iter())
                .map(|(a, b)| a * b)
                .sum()
        };
        let numeric = numeric_gradient(&parameters, reference, 1e-3);
        assert_relative(&analytic, &numeric, "full transformer");
    }

    #[test]
    fn forward_is_deterministic() {
        let config = TransformerConfig {
            n_layer: 1,
            d_model: 8,
            n_head: 2,
            ffn_hidden: 16,
            vocab: 16,
            n_positions: 32,
            readout_rank: 4,
        };
        let embedding = TableEmbedding::random(config.vocab, config.d_model, 1);
        let model = config.init(2);
        let tokens = [1u32, 4, 2, 9];
        let first = model.forward(&tokens, &embedding, None);
        let second = model.forward(&tokens, &embedding, None);
        assert_eq!(first.loss.to_bits(), second.loss.to_bits());
        assert_eq!(first.logits, second.logits);
    }
}
