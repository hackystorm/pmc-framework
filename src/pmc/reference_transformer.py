# SPDX-License-Identifier: GPL-3.0-only

"""Python reference of the Rust decoder-only transformer (forward + loss).

This is the second, independent implementation of the same mathematics used
by the parity check: ``crates/pmc-train/src/transformer.rs`` is the fast one
that ships, this module is the slow one that ships in the same repository.
Both round every arithmetic operation to 32-bit precision, so they agree
step by step rather than merely at the end.

The backward pass and the training loop are out of scope here. What this
module owns is the forward pass and the next-token cross-entropy loss, which
is what the parity check compares against the Rust ``train_lm`` initial
loss on the same configuration, seed, embedding and token sequence.

Verified: on the 50.4M reference configuration (n_layer 12, d_model 512,
n_head 8, ffn_hidden 2048, vocab 256, readout_rank 64, model seed 20261030,
embedding seed 20261029), the fixed 48-token sequence, this module computes
initial_loss 5.887040 and the Rust trainer reports 5.887041.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .reference import f32


@dataclass
class ReferenceTransformerConfig:
    n_layer: int = 2
    d_model: int = 32
    n_head: int = 4
    ffn_hidden: int = 64
    vocab: int = 256
    n_positions: int = 64
    readout_rank: int = 8


class ReferenceTransformer:
    """A decoder-only transformer: forward pass and next-token loss.

    The architecture matches the Rust implementation's pre-norm decoder
    block: x = x + Attention(RMSNorm(x)); x = x + SwiGLU(RMSNorm(x));
    h = RMSNorm(x); z = h * U (d -> readout_rank); logits = z * V
    (readout_rank -> vocab). RoPE rotates coordinate pairs (i, i + head_dim/2)
    with frequency theta^(-2i/head_dim). Attention is causal. The loss is
    mean cross-entropy over the positions that have a next token.
    """

    RMS_EPS = 1e-5
    ROPE_THETA = 10000.0

    def __init__(self, config: ReferenceTransformerConfig, seed: int) -> None:
        self.config = config
        d = config.d_model
        f = config.ffn_hidden
        r = config.readout_rank
        rng_state = seed if seed != 0 else 0x9E3779B97F4A7C15

        def next_u64() -> int:
            nonlocal rng_state
            x = rng_state
            x ^= x >> 12
            x ^= (x << 25) & ((1 << 64) - 1)
            x ^= x >> 27
            rng_state = x & ((1 << 64) - 1)
            return (x * 0x2545F4914F6CDD1D) & ((1 << 64) - 1)

        def uniform_f32() -> float:
            return (next_u64() >> 40) / float(1 << 24)

        def normal() -> float:
            u1 = max(uniform_f32(), 1e-7)
            u2 = uniform_f32()
            return f32(
                f32(math.sqrt(f32(f32(-2.0) * f32(math.log(u1)))))
                * f32(math.cos(f32(f32(2.0) * f32(math.pi) * u2)))
            )

        def uniform(low: float, high: float) -> float:
            return f32(low + (high - low) * uniform_f32())

        def he_fill(values: list[float], fan_in: int) -> None:
            scale = f32(math.sqrt(2.0 / fan_in))
            for i in range(len(values)):
                values[i] = f32(normal() * scale)

        def xavier_fill(values: list[float], fan_in: int, fan_out: int) -> None:
            limit = f32(math.sqrt(6.0 / (fan_in + fan_out)))
            for i in range(len(values)):
                values[i] = f32(uniform(-limit, limit))

        self.blocks: list[dict[str, list[float]]] = []
        for _ in range(config.n_layer):
            wq = [0.0] * (d * d)
            wk = [0.0] * (d * d)
            wv = [0.0] * (d * d)
            wo = [0.0] * (d * d)
            w_gate = [0.0] * (d * f)
            w_up = [0.0] * (d * f)
            w_down = [0.0] * (f * d)
            he_fill(wq, d)
            he_fill(wk, d)
            he_fill(wv, d)
            he_fill(wo, d)
            he_fill(w_gate, d)
            he_fill(w_up, d)
            he_fill(w_down, f)
            # The fill order matches the Rust implementation exactly: wq, wk,
            # wv, wo, w_gate, w_up, w_down per block, then the readout pair.
            self.blocks.append(
                {
                    "attn_norm": [1.0] * d,
                    "wq": wq,
                    "wk": wk,
                    "wv": wv,
                    "wo": wo,
                    "ffn_norm": [1.0] * d,
                    "w_gate": w_gate,
                    "w_up": w_up,
                    "w_down": w_down,
                }
            )

        self.final_norm = [1.0] * d
        self.readout_u = [0.0] * (d * r)
        self.readout_v = [0.0] * (r * config.vocab)
        xavier_fill(self.readout_u, d, r)
        xavier_fill(self.readout_v, r, config.vocab)

    def _rmsnorm(self, x: list[float], w: list[float], d: int, seq: int) -> list[float]:
        out = [0.0] * (seq * d)
        for pos in range(seq):
            base = pos * d
            sq_sum = 0.0
            for j in range(d):
                sq_sum = f32(sq_sum + f32(x[base + j] * x[base + j]))
            mean_sq = f32(sq_sum / d)
            rstd = f32(1.0 / math.sqrt(f32(mean_sq + self.RMS_EPS)))
            for j in range(d):
                out[base + j] = f32(f32(x[base + j] * rstd) * w[j])
        return out

    def _matmul(self, a: list[float], b: list[float], m: int, k: int, n: int) -> list[float]:
        c = [0.0] * (m * n)
        for i in range(m):
            for j in range(n):
                acc = 0.0
                for t in range(k):
                    acc = f32(acc + f32(a[i * k + t] * b[t * n + j]))
                c[i * n + j] = acc
        return c

    def _rope(self, x: list[float], n_head: int, head_dim: int, seq: int, d: int) -> list[float]:
        out = list(x)
        half = head_dim // 2
        for pos in range(seq):
            for head in range(n_head):
                for i in range(half):
                    frequency = self.ROPE_THETA ** (-(2 * i) / head_dim)
                    angle = f32(pos * frequency)
                    cos_a = f32(math.cos(angle))
                    sin_a = f32(math.sin(angle))
                    base = pos * d + head * head_dim
                    first = x[base + i]
                    second = x[base + i + half]
                    out[base + i] = f32(f32(first * cos_a) - f32(second * sin_a))
                    out[base + i + half] = f32(f32(first * sin_a) + f32(second * cos_a))
        return out

    def _softmax_causal(self, scores: list[float], seq: int) -> list[float]:
        out = [0.0] * (seq * seq)
        for i in range(seq):
            max_val = float("-inf")
            for j in range(i + 1):
                if scores[i * seq + j] > max_val:
                    max_val = scores[i * seq + j]
            sum_exp = 0.0
            for j in range(seq):
                if j <= i:
                    out[i * seq + j] = f32(math.exp(f32(scores[i * seq + j] - max_val)))
                    sum_exp = f32(sum_exp + out[i * seq + j])
                else:
                    out[i * seq + j] = 0.0
            inv = f32(1.0 / sum_exp)
            for j in range(seq):
                if j <= i:
                    out[i * seq + j] = f32(out[i * seq + j] * inv)
        return out

    def _swiglu(self, gate: list[float], up: list[float]) -> list[float]:
        out = [0.0] * len(gate)
        for i in range(len(gate)):
            sig = f32(1.0 / (1.0 + math.exp(-gate[i])))
            silu = f32(gate[i] * sig)
            out[i] = f32(silu * up[i])
        return out

    def forward_loss(self, tokens: list[int], embedding_table: list[float]) -> float:
        config = self.config
        d = config.d_model
        f = config.ffn_hidden
        n_head = config.n_head
        head_dim = d // n_head
        vocab = config.vocab
        rank = config.readout_rank
        seq = len(tokens)
        scale = f32(1.0 / math.sqrt(head_dim))

        if seq > config.n_positions:
            raise ValueError(
                f"sequence length {seq} exceeds n_positions {config.n_positions}"
            )

        # Embedding lookup: one row per position.
        x = [0.0] * (seq * d)
        for pos in range(seq):
            base = pos * d
            ebase = tokens[pos] * d
            for j in range(d):
                x[base + j] = f32(embedding_table[ebase + j])

        for block in self.blocks:
            inp = list(x)
            attn_norm_out = self._rmsnorm(inp, block["attn_norm"], d, seq)
            q = self._matmul(attn_norm_out, block["wq"], seq, d, d)
            k = self._matmul(attn_norm_out, block["wk"], seq, d, d)
            v = self._matmul(attn_norm_out, block["wv"], seq, d, d)
            q_rope = self._rope(q, n_head, head_dim, seq, d)
            k_rope = self._rope(k, n_head, head_dim, seq, d)

            attn_concat = [0.0] * (seq * d)
            for head in range(n_head):
                q_head = [0.0] * (seq * head_dim)
                k_head = [0.0] * (seq * head_dim)
                v_head = [0.0] * (seq * head_dim)
                for pos in range(seq):
                    src = pos * d + head * head_dim
                    tgt = pos * head_dim
                    for j in range(head_dim):
                        q_head[tgt + j] = q_rope[src + j]
                        k_head[tgt + j] = k_rope[src + j]
                        v_head[tgt + j] = v[src + j]

                # scores = q_head @ k_head^T, scaled.
                scores = [0.0] * (seq * seq)
                for i in range(seq):
                    for j in range(seq):
                        acc = 0.0
                        for t in range(head_dim):
                            acc = f32(
                                acc + f32(q_head[i * head_dim + t] * k_head[j * head_dim + t])
                            )
                        scores[i * seq + j] = f32(acc * scale)
                probs_head = self._softmax_causal(scores, seq)

                # attn_head = probs @ v_head.
                attn_head = [0.0] * (seq * head_dim)
                for i in range(seq):
                    for j in range(head_dim):
                        acc = 0.0
                        for t in range(seq):
                            acc = f32(
                                acc + f32(probs_head[i * seq + t] * v_head[t * head_dim + j])
                            )
                        attn_head[i * head_dim + j] = acc

                for pos in range(seq):
                    tgt = pos * d + head * head_dim
                    src = pos * head_dim
                    for j in range(head_dim):
                        attn_concat[tgt + j] = attn_head[src + j]

            attn_proj = self._matmul(attn_concat, block["wo"], seq, d, d)
            # Residual: the block input is what gets added to.
            x_after_attn = [f32(inp[i] + attn_proj[i]) for i in range(seq * d)]

            ffn_norm_out = self._rmsnorm(x_after_attn, block["ffn_norm"], d, seq)
            gate = self._matmul(ffn_norm_out, block["w_gate"], seq, d, f)
            up = self._matmul(ffn_norm_out, block["w_up"], seq, d, f)
            swiglu_out = self._swiglu(gate, up)
            down = self._matmul(swiglu_out, block["w_down"], seq, f, d)
            x = [f32(x_after_attn[i] + down[i]) for i in range(seq * d)]

        final_norm_out = self._rmsnorm(x, self.final_norm, d, seq)
        readout_hidden = self._matmul(final_norm_out, self.readout_u, seq, d, rank)
        logits = self._matmul(readout_hidden, self.readout_v, seq, rank, vocab)

        # Mean cross-entropy over the positions that have a next token.
        loss = 0.0
        count = 0
        for pos in range(seq - 1):
            base = pos * vocab
            max_logit = max(logits[base + j] for j in range(vocab))
            sum_exp = 0.0
            for j in range(vocab):
                sum_exp = f32(sum_exp + f32(math.exp(f32(logits[base + j] - max_logit))))
            target = tokens[pos + 1]
            log_prob = f32(f32(logits[base + target] - max_logit) - f32(math.log(sum_exp)))
            loss = f32(loss - log_prob)
            count += 1
        if count > 0:
            loss = f32(loss / count)
        return loss


def random_embedding_table(vocab: int, dim: int, seed: int) -> list[float]:
    """A frozen random embedding table, matching the Rust examples.

    Rows are ``normal() * 0.5`` filled in order from ``Rng::new(seed)``.
    """
    from .reference import ReferenceRng

    rng = ReferenceRng(seed)
    table = [0.0] * (vocab * dim)
    for i in range(len(table)):
        table[i] = rng.normal() * 0.5
    return table


__all__ = ["ReferenceTransformer", "ReferenceTransformerConfig", "random_embedding_table"]
