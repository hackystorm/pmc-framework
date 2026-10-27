# SPDX-License-Identifier: GPL-3.0-only

"""WP7a feasibility spike: hidden state, fast-weight row, and logit bias.

This is a spike, not the live runtime. It loads one model in one process and
proves as much of the WP7 loop as this llama.cpp build allows:

* tokenize a prompt and decode it once,
* read the final-token hidden state ``h`` and the logits for that token,
* update one fast-weight row for a chosen output token,
* apply the row readout or a large bias through ``llama_sampler_init_logit_bias``
  in a sampler chain before ``llama_sampler_init_greedy``,
* show the sampled token before and after the bias.

The trainer never appears here. Only token ids are produced; the prompt text is
used for tokenization and is never decoded back or printed.

A hard finding of this build is recorded rather than hidden: the LFM2 graph in
``src/models/lfm2.cpp`` builds the output projection only when
``cparams.embeddings`` is false. With embeddings enabled,
``llama_get_logits_ith`` returns a non-NULL buffer of zeros. The strict
``--mode single`` gate therefore fails on this model and exits 2. The
``--mode paired`` fallback decodes the same token sequence twice in two
contexts in the same process; it is labelled as a fallback, never as a
single-decode success.

The context and model parameter structs are created by a small C shim compiled
against the exact llama.h of the build in use. Mirroring those structs in
ctypes is fragile: fields have been added over time and the trailing booleans
make a silent off-by-one layout error very likely.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import struct
import sys
import time
from array import array
from collections.abc import Mapping, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Pure helpers. These are unit-tested without loading a model.
# ---------------------------------------------------------------------------

REQUIRED_SUMMARY_KEYS = (
    "h_norm",
    "h_sha256",
    "logits_top1_before",
    "logits_top1_after",
    "biased_token",
    "finite_h",
    "finite_logits",
    "decode_ms",
    "extract_ms",
)

DEFAULT_PROMPT = "Hello, my name is"
DEFAULT_BIAS = 1000.0
DEFAULT_LR = 1e-4


def token_argmax(values: Sequence[float]) -> int:
    """Return the index of the largest value; ties go to the lowest index."""
    if not values:
        raise ValueError("token_argmax needs at least one value")
    best_index = 0
    best_value = values[0]
    for index in range(1, len(values)):
        value = values[index]
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def runner_up_token(values: Sequence[float], top1: int) -> int:
    """Return the best token other than ``top1``."""
    if len(values) < 2:
        raise ValueError("runner_up_token needs at least two values")
    if top1 < 0 or top1 >= len(values):
        raise ValueError("top1 is out of range")
    best_index = -1
    best_value = -math.inf
    for index, value in enumerate(values):
        if index == top1:
            continue
        if best_index < 0 or value > best_value:
            best_index = index
            best_value = value
    return best_index


def argmax_with_bias(values: Sequence[float], token: int, bias: float) -> int:
    """Return the argmax after adding ``bias`` to one token's logit."""
    if not values:
        raise ValueError("argmax_with_bias needs at least one value")
    if token < 0 or token >= len(values):
        raise ValueError("biased token is out of range")
    best_index = 0
    best_value = values[0] + (bias if token == 0 else 0.0)
    for index in range(1, len(values)):
        value = values[index] + (bias if token == index else 0.0)
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def finite_vector(values: Sequence[float], expected_len: int) -> bool:
    """True when the vector has the expected length and all values are finite."""
    if len(values) != expected_len:
        return False
    return all(math.isfinite(value) for value in values)


def count_nonzero(values: Sequence[float]) -> int:
    """Count values that are not exactly 0.0.

    This is a cheap informativeness check for the zero-filled logits buffer
    returned by an embeddings-only decode. Zero logits are finite, so
    ``finite_vector`` alone cannot catch that failure.
    """
    return sum(1 for value in values if value != 0.0)


def is_informative(values: Sequence[float]) -> bool:
    """True when a logits vector has variation and at least one non-zero value."""
    if len(values) < 2:
        return False
    if not all(math.isfinite(value) for value in values):
        return False
    if not any(value != 0.0 for value in values):
        return False
    return max(values) > min(values)


def vector_norm(values: Sequence[float]) -> float:
    """Euclidean norm, using fsum for a stable accumulation."""
    return math.sqrt(math.fsum(value * value for value in values))


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Stable dot product of two equal-length vectors."""
    if len(left) != len(right):
        raise ValueError("dot needs vectors of equal length")
    return math.fsum(a * b for a, b in zip(left, right))


def sha256_prefix(values: Sequence[float], length: int = 12) -> str:
    """SHA-256 of the canonical little-endian float32 encoding, truncated."""
    if length < 1 or length > 64:
        raise ValueError("hash prefix length must be in [1, 64]")
    raw = struct.pack(f"<{len(values)}f", *values)
    return hashlib.sha256(raw).hexdigest()[:length]


def softmax_probability(values: Sequence[float], index: int) -> float:
    """Stable softmax probability of one index."""
    if not values:
        raise ValueError("softmax_probability needs at least one value")
    if index < 0 or index >= len(values):
        raise ValueError("softmax index is out of range")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("softmax_probability needs finite logits")
    high = max(values)
    total = math.fsum(math.exp(value - high) for value in values)
    if total <= 0.0:
        raise ValueError("softmax normalization is not positive")
    return math.exp(values[index] - high) / total


def fast_weight_update(
    row: Sequence[float],
    hidden: Sequence[float],
    probability: float,
    learning_rate: float,
) -> list[float]:
    """One SGD step on ``-log p_target`` for a single output row.

    For a target token ``t`` the gradient of ``-log p_t`` with respect to the
    output row is ``(p_t - 1) * h``. Gradient descent therefore adds
    ``learning_rate * (1 - p_t) * h`` to the row. The returned list is the new
    row; the input is not mutated.
    """
    if len(row) != len(hidden):
        raise ValueError("row and hidden state must have the same length")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if not math.isfinite(learning_rate) or learning_rate < 0.0:
        raise ValueError("learning_rate must be finite and non-negative")
    step = learning_rate * (1.0 - probability)
    return [weight + step * value for weight, value in zip(row, hidden)]


def build_summary(values: Mapping[str, Any]) -> dict[str, Any]:
    """Build the JSON summary and normalize the required fields.

    Extra keys are preserved. The required keys are exactly the ones requested
    by the WP7a deliverable; ``-1`` in a top1 field means the gate failed before
    a valid token could be sampled.
    """
    missing = [key for key in REQUIRED_SUMMARY_KEYS if key not in values]
    if missing:
        raise KeyError(f"summary is missing required keys: {missing}")
    summary = dict(values)
    summary["h_sha256"] = str(summary["h_sha256"])[:12]
    summary["h_norm"] = round(float(summary["h_norm"]), 6)
    summary["decode_ms"] = round(float(summary["decode_ms"]), 6)
    summary["extract_ms"] = round(float(summary["extract_ms"]), 6)
    for key in ("logits_top1_before", "logits_top1_after", "biased_token"):
        summary[key] = int(summary[key])
    for key in ("finite_h", "finite_logits"):
        summary[key] = bool(summary[key])
    return summary


class SpikeError(RuntimeError):
    """A hard error in loading, decoding, or extracting the spike tensors."""


class GateFailure(SpikeError):
    """The requested gate failed; carries the JSON evidence to print."""

    def __init__(self, message: str, summary: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.summary = dict(summary) if summary is not None else None


# ---------------------------------------------------------------------------
# ctypes ABI. Only simple, stable types are mirrored here. The parameter
# structs stay in the C shim.
# ---------------------------------------------------------------------------


class LlamaBatch(ctypes.Structure):
    """Mirror of ``struct llama_batch`` (fixed field types, no parameters)."""

    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(ctypes.c_int32)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(ctypes.c_int32)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


class LlamaLogitBias(ctypes.Structure):
    """Mirror of ``struct llama_logit_bias``."""

    _fields_ = [
        ("token", ctypes.c_int32),
        ("bias", ctypes.c_float),
    ]


class LlamaChatMessage(ctypes.Structure):
    """Mirror of ``struct llama_chat_message``."""

    _fields_ = [
        ("role", ctypes.c_char_p),
        ("content", ctypes.c_char_p),
    ]


class LlamaSamplerChainParams(ctypes.Structure):
    """Mirror of ``struct llama_sampler_chain_params``."""

    _fields_ = [("no_perf", ctypes.c_bool)]


def _read_float_vector(pointer: ctypes.POINTER(ctypes.c_float), length: int, label: str) -> array:
    """Copy a C float buffer into a Python array of native floats."""
    if not pointer:
        raise SpikeError(f"{label} pointer is NULL")
    if length <= 0:
        raise SpikeError(f"{label} length must be positive")
    raw = ctypes.string_at(pointer, length * ctypes.sizeof(ctypes.c_float))
    values = array("f")
    values.frombytes(raw)
    if values.itemsize != ctypes.sizeof(ctypes.c_float):
        raise SpikeError(f"{label}: Python array float size does not match C float")
    if len(values) != length:
        raise SpikeError(f"{label}: copied {len(values)} floats, expected {length}")
    return values


class LlamaSpike:
    """One process, one model, one or two contexts for the spike."""

    def __init__(self, lib_path: str, shim_path: str, model_path: str) -> None:
        self.lib_path = lib_path
        self.shim_path = shim_path
        self.model_path = model_path
        self.lib: ctypes.CDLL | None = None
        self.shim: ctypes.CDLL | None = None
        self.model: int | None = None
        self.vocab: int | None = None
        self.n_vocab = 0
        self.n_embd = 0
        self.arch = ""
        self._contexts: list[int] = []

        self._load_libraries()
        self._configure_shim()
        try:
            self._load_model()
        except Exception:
            self.close()
            raise

    # -- setup ------------------------------------------------------------

    def _load_libraries(self) -> None:
        for label, path in (
            ("libllama", self.lib_path),
            ("shim", self.shim_path),
            ("model", self.model_path),
        ):
            if not os.path.isfile(path):
                raise SpikeError(f"{label} not found: {path}")
        try:
            # RTLD_GLOBAL lets the shim's undefined llama_* symbols resolve to
            # this exact library when both are loaded in one process.
            self.lib = ctypes.CDLL(self.lib_path, mode=ctypes.RTLD_GLOBAL)
            self.shim = ctypes.CDLL(self.shim_path)
        except OSError as error:
            raise SpikeError(
                f"could not load libllama or the shim: {error}; "
                "build the shim against the same llama.h and libllama"
            ) from error

    def _configure_shim(self) -> None:
        assert self.shim is not None
        self.shim.pmc_model_load.restype = ctypes.c_void_p
        self.shim.pmc_model_load.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        self.shim.pmc_ctx_new.restype = ctypes.c_void_p
        self.shim.pmc_ctx_new.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        self.shim.pmc_ctx_free.argtypes = [ctypes.c_void_p]
        self.shim.pmc_ctx_free.restype = None
        self.shim.pmc_model_free.argtypes = [ctypes.c_void_p]
        self.shim.pmc_model_free.restype = None

    def _configure_lib(self) -> None:
        assert self.lib is not None
        lib = self.lib
        lib.llama_model_get_vocab.restype = ctypes.c_void_p
        lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
        lib.llama_vocab_n_tokens.restype = ctypes.c_int32
        lib.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
        lib.llama_model_n_embd.restype = ctypes.c_int32
        lib.llama_model_n_embd.argtypes = [ctypes.c_void_p]
        lib.llama_model_meta_val_str.restype = ctypes.c_int32
        lib.llama_model_meta_val_str.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.llama_tokenize.restype = ctypes.c_int32
        lib.llama_tokenize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_bool,
            ctypes.c_bool,
        ]
        lib.llama_batch_init.restype = LlamaBatch
        lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        lib.llama_batch_free.argtypes = [LlamaBatch]
        lib.llama_batch_free.restype = None
        lib.llama_decode.restype = ctypes.c_int32
        lib.llama_decode.argtypes = [ctypes.c_void_p, LlamaBatch]
        lib.llama_get_embeddings_ith.restype = ctypes.POINTER(ctypes.c_float)
        lib.llama_get_embeddings_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)
        lib.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.llama_state_get_size.restype = ctypes.c_size_t
        lib.llama_state_get_size.argtypes = [ctypes.c_void_p]
        lib.llama_state_get_data.restype = ctypes.c_size_t
        lib.llama_state_get_data.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        lib.llama_state_set_data.restype = ctypes.c_size_t
        lib.llama_state_set_data.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        lib.llama_sampler_chain_default_params.restype = LlamaSamplerChainParams
        lib.llama_sampler_chain_init.restype = ctypes.c_void_p
        lib.llama_sampler_chain_init.argtypes = [LlamaSamplerChainParams]
        lib.llama_sampler_chain_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.llama_sampler_chain_add.restype = None
        lib.llama_sampler_init_greedy.restype = ctypes.c_void_p
        lib.llama_sampler_init_logit_bias.restype = ctypes.c_void_p
        lib.llama_sampler_init_logit_bias.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(LlamaLogitBias),
        ]
        lib.llama_sampler_sample.restype = ctypes.c_int32
        lib.llama_sampler_sample.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
        ]
        lib.llama_sampler_free.argtypes = [ctypes.c_void_p]
        lib.llama_sampler_free.restype = None
        lib.llama_get_memory.restype = ctypes.c_void_p
        lib.llama_get_memory.argtypes = [ctypes.c_void_p]
        lib.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.llama_memory_clear.restype = None
        lib.llama_vocab_is_eog.restype = ctypes.c_bool
        lib.llama_vocab_is_eog.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.llama_token_to_piece.restype = ctypes.c_int32
        lib.llama_token_to_piece.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_bool,
        ]
        lib.llama_detokenize.restype = ctypes.c_int32
        lib.llama_detokenize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_bool,
            ctypes.c_bool,
        ]
        lib.llama_model_chat_template.restype = ctypes.c_char_p
        lib.llama_model_chat_template.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.llama_chat_apply_template.restype = ctypes.c_int32
        lib.llama_chat_apply_template.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(LlamaChatMessage),
            ctypes.c_size_t,
            ctypes.c_bool,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]

    def _load_model(self) -> None:
        assert self.lib is not None and self.shim is not None
        self._configure_lib()
        error = ctypes.create_string_buffer(512)
        self.model = self.shim.pmc_model_load(
            os.fsencode(self.model_path), error, len(error)
        )
        if not self.model:
            detail = error.value.decode("utf-8", errors="replace")
            raise SpikeError(f"model load failed: {detail}")
        self.vocab = self.lib.llama_model_get_vocab(self.model)
        if not self.vocab:
            raise SpikeError("llama_model_get_vocab returned NULL")
        self.n_vocab = int(self.lib.llama_vocab_n_tokens(self.vocab))
        self.n_embd = int(self.lib.llama_model_n_embd(self.model))
        if self.n_vocab <= 0:
            raise SpikeError(f"invalid vocabulary size: {self.n_vocab}")
        if self.n_embd <= 0:
            raise SpikeError(f"invalid embedding size: {self.n_embd}")
        self.arch = self._meta_str("general.architecture")

    def _meta_str(self, key: str) -> str:
        assert self.lib is not None and self.model is not None
        buffer = ctypes.create_string_buffer(256)
        result = self.lib.llama_model_meta_val_str(
            self.model, key.encode("utf-8"), buffer, len(buffer)
        )
        if result < 0:
            return ""
        return buffer.value.decode("utf-8", errors="replace")

    # -- lifecycle --------------------------------------------------------

    def new_context(
        self,
        *,
        n_ctx: int,
        n_batch: int,
        n_threads: int,
        embeddings: bool,
    ) -> int:
        assert self.shim is not None and self.model is not None
        error = ctypes.create_string_buffer(512)
        context = self.shim.pmc_ctx_new(
            self.model,
            n_ctx,
            n_batch,
            n_threads,
            1 if embeddings else 0,
            1,  # no_perf
            error,
            len(error),
        )
        if not context:
            detail = error.value.decode("utf-8", errors="replace")
            raise SpikeError(f"context creation failed: {detail}")
        self._contexts.append(context)
        return context

    def close(self) -> None:
        if self.shim is not None:
            for context in reversed(self._contexts):
                self.shim.pmc_ctx_free(context)
            self._contexts.clear()
            if self.model is not None:
                self.shim.pmc_model_free(self.model)
                self.model = None

    def __enter__(self) -> "LlamaSpike":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- inference --------------------------------------------------------

    def tokenize(self, text: str, *, parse_special: bool = False) -> list[int]:
        """Tokenize only; the text is never decoded back or logged.

        ``parse_special`` must be set for a templated chat prompt: the chat
        template writes control tokens such as ``<|im_start|>`` as text, and
        they only become single vocabulary tokens when the tokenizer is asked
        to parse them. With the default the markers are split into characters,
        which silently changes the prompt the model sees.
        """
        assert self.lib is not None and self.vocab is not None
        raw = text.encode("utf-8")
        capacity = max(8, len(raw) + 8)
        while True:
            buffer = (ctypes.c_int32 * capacity)()
            count = self.lib.llama_tokenize(
                self.vocab,
                raw,
                len(raw),
                buffer,
                capacity,
                True,
                parse_special,
            )
            if count == -(2**31):
                raise SpikeError("llama_tokenize overflowed int32")
            if count < 0:
                capacity = -count
                continue
            if count > capacity:
                raise SpikeError(
                    f"llama_tokenize returned {count} tokens for capacity {capacity}"
                )
            return [int(buffer[index]) for index in range(count)]

    def decode(
        self,
        context: int,
        tokens: Sequence[int],
        *,
        all_outputs: bool,
        start_pos: int = 0,
    ) -> float:
        """Decode tokens and return the wall time of the llama_decode call."""
        assert self.lib is not None
        if not tokens:
            raise SpikeError("cannot decode an empty token sequence")
        if start_pos < 0:
            raise ValueError("start_pos must be non-negative")
        batch = self.lib.llama_batch_init(len(tokens), 0, 1)
        try:
            if not (
                batch.token
                and batch.pos
                and batch.n_seq_id
                and batch.seq_id
                and batch.logits
            ):
                raise SpikeError("llama_batch_init returned an incomplete batch")
            batch.n_tokens = len(tokens)
            last = len(tokens) - 1
            for index, token in enumerate(tokens):
                batch.token[index] = int(token)
                batch.pos[index] = start_pos + index
                batch.n_seq_id[index] = 1
                if not batch.seq_id[index]:
                    raise SpikeError("llama_batch_init left a seq_id row unallocated")
                batch.seq_id[index][0] = 0
                batch.logits[index] = 1 if (all_outputs or index == last) else 0
            started = time.perf_counter()
            result = self.lib.llama_decode(context, batch)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        finally:
            self.lib.llama_batch_free(batch)
        if result != 0:
            raise SpikeError(
                f"llama_decode returned {result}; check n_batch/n_ctx and batch arrays"
            )
        return elapsed_ms

    def read_embeddings(self, context: int, index: int = -1) -> array:
        assert self.lib is not None
        pointer = self.lib.llama_get_embeddings_ith(context, index)
        return _read_float_vector(pointer, self.n_embd, "llama_get_embeddings_ith")

    def read_logits(self, context: int, index: int = -1) -> array:
        assert self.lib is not None
        pointer = self.lib.llama_get_logits_ith(context, index)
        return _read_float_vector(pointer, self.n_vocab, "llama_get_logits_ith")

    def state_size(self, context: int) -> int:
        """Size in bytes of this context's saved state."""
        assert self.lib is not None
        size = int(self.lib.llama_state_get_size(context))
        if size <= 0:
            raise SpikeError(f"llama_state_get_size returned {size}")
        return size

    def state_save(self, context: int) -> bytes:
        """Copy the whole context state (KV cache and position) into bytes."""
        assert self.lib is not None
        size = self.state_size(context)
        buffer = (ctypes.c_uint8 * size)()
        written = int(self.lib.llama_state_get_data(context, buffer, size))
        if written != size:
            raise SpikeError(
                f"llama_state_get_data wrote {written} bytes, expected {size}"
            )
        return bytes(buffer)

    def state_load(self, context: int, data: bytes) -> None:
        """Restore a state saved by :meth:`state_save` into a cleared context.

        A fresh context reports only the size of its empty state (a few hundred
        bytes), so ``state_size`` is not the required input size here. The saved
        state carries its own memory-module layout; the C side reads that and
        writes it into the context's allocated memory.
        """
        assert self.lib is not None
        if not data:
            raise SpikeError("cannot restore an empty state")
        size = len(data)
        buffer = (ctypes.c_uint8 * size).from_buffer_copy(data)
        read = int(self.lib.llama_state_set_data(context, buffer, size))
        if read != size:
            raise SpikeError(
                f"llama_state_set_data read {read} bytes of {size}; "
                "the saved state does not fit this context"
            )

    def sample_biased(
        self, context: int, biases: Sequence[tuple[int, float]] = ()
    ) -> int:
        """Greedy sample after applying zero or more logit biases.

        The chain is always ``[logit_bias] -> greedy``. The bias sampler copies
        its entries, and the chain owns every sampler added to it.
        """
        assert self.lib is not None
        entries = (LlamaLogitBias * len(biases))()
        for index, (token, bias) in enumerate(biases):
            token = int(token)
            bias = float(bias)
            if token < 0 or token >= self.n_vocab:
                raise ValueError(f"bias token {token} is outside the vocabulary")
            if not math.isfinite(bias):
                raise ValueError("bias must be finite")
            entries[index] = LlamaLogitBias(token, bias)
        params = self.lib.llama_sampler_chain_default_params()
        chain = self.lib.llama_sampler_chain_init(params)
        if not chain:
            raise SpikeError("llama_sampler_chain_init returned NULL")
        try:
            if biases:
                bias_sampler = self.lib.llama_sampler_init_logit_bias(
                    self.n_vocab, len(biases), entries
                )
                if not bias_sampler:
                    raise SpikeError("llama_sampler_init_logit_bias returned NULL")
                self.lib.llama_sampler_chain_add(chain, bias_sampler)
            greedy = self.lib.llama_sampler_init_greedy()
            if not greedy:
                raise SpikeError("llama_sampler_init_greedy returned NULL")
            self.lib.llama_sampler_chain_add(chain, greedy)
            token = int(self.lib.llama_sampler_sample(chain, context, -1))
        finally:
            self.lib.llama_sampler_free(chain)
        if token < 0 or token >= self.n_vocab:
            raise SpikeError(f"sampler returned out-of-range token {token}")
        return token

    def sample(self, context: int, biased_token: int | None = None, bias: float = 0.0) -> int:
        """Backward-compatible single-bias wrapper around ``sample_biased``."""
        if biased_token is None:
            if bias != 0.0:
                raise ValueError("bias was given without a token")
            return self.sample_biased(context)
        return self.sample_biased(context, [(biased_token, bias)])

    def clear_context(self, context: int) -> None:
        """Drop every token and KV entry from ``context``."""
        assert self.lib is not None
        memory = self.lib.llama_get_memory(context)
        if not memory:
            raise SpikeError("llama_get_memory returned NULL")
        self.lib.llama_memory_clear(memory, True)

    def token_is_eog(self, token: int) -> bool:
        assert self.lib is not None and self.vocab is not None
        return bool(self.lib.llama_vocab_is_eog(self.vocab, int(token)))

    def token_piece(self, token: int) -> str:
        """One token rendered as text, without special tokens."""
        assert self.lib is not None and self.vocab is not None
        capacity = 32
        while True:
            buffer = ctypes.create_string_buffer(capacity)
            needed = int(
                self.lib.llama_token_to_piece(
                    self.vocab, int(token), buffer, capacity, 0, False
                )
            )
            if needed < 0:
                capacity = -needed + 1
                continue
            if needed > capacity:
                capacity = needed + 1
                continue
            return buffer.raw[:needed].decode("utf-8", errors="replace")

    def detokenize(self, tokens: Sequence[int]) -> str:
        """Render a token sequence as text, dropping BOS/EOS tokens."""
        assert self.lib is not None and self.vocab is not None
        if not tokens:
            return ""
        buffer_tokens = (ctypes.c_int32 * len(tokens))(*[int(t) for t in tokens])
        capacity = max(32, len(tokens) * 8)
        while True:
            buffer = ctypes.create_string_buffer(capacity)
            needed = int(
                self.lib.llama_detokenize(
                    self.vocab,
                    buffer_tokens,
                    len(tokens),
                    buffer,
                    capacity,
                    True,
                    False,
                )
            )
            if needed < 0:
                capacity = -needed + 1
                continue
            if needed > capacity:
                capacity = needed + 1
                continue
            return buffer.raw[:needed].decode("utf-8", errors="replace")

    def model_chat_template(self) -> str:
        """The model's default chat template, or an empty string."""
        assert self.lib is not None and self.model is not None
        pointer = self.lib.llama_model_chat_template(self.model, None)
        if not pointer:
            return ""
        return pointer.decode("utf-8", errors="replace")

    def chat_prompt(
        self,
        messages: Sequence[tuple[str, str]],
        *,
        add_assistant: bool = True,
    ) -> str:
        """Apply the model's chat template to ``(role, content)`` pairs."""
        assert self.lib is not None
        if not messages:
            raise ValueError("chat_prompt needs at least one message")
        template = self.model_chat_template()
        if not template:
            raise SpikeError(
                "the model has no chat template; refusing to guess a prompt format"
            )
        encoded: list[tuple[bytes, bytes]] = []
        chat = (LlamaChatMessage * len(messages))()
        for index, (role, content) in enumerate(messages):
            if not isinstance(role, str) or not role:
                raise ValueError(f"message {index}: role must be a non-empty string")
            if not isinstance(content, str):
                raise ValueError(f"message {index}: content must be a string")
            role_bytes = role.encode("utf-8")
            content_bytes = content.encode("utf-8")
            encoded.append((role_bytes, content_bytes))
            chat[index].role = role_bytes
            chat[index].content = content_bytes
        capacity = max(
            256, 2 * sum(len(role) + len(content) + 8 for role, content in encoded)
        )
        while True:
            buffer = ctypes.create_string_buffer(capacity)
            needed = int(
                self.lib.llama_chat_apply_template(
                    template.encode("utf-8"),
                    chat,
                    len(messages),
                    add_assistant,
                    buffer,
                    capacity,
                )
            )
            if needed < 0:
                raise SpikeError(
                    "llama_chat_apply_template rejected the model's chat template"
                )
            if needed > capacity:
                capacity = needed + 1
                continue
            return buffer.raw[:needed].decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Spike flow
# ---------------------------------------------------------------------------


def _coexistence_reason(
    *,
    arch: str,
    n_embd: int,
    n_vocab: int,
    h_length: int,
    logits_length: int,
    emb_nonzero: int,
    h_finite: bool,
) -> str:
    """Exact diagnostic for a non-informative embeddings-mode logits buffer."""
    zero_note = (
        " Zero logits are finite, so finite_logits alone is true; the buffer is "
        "not a computed distribution."
        if emb_nonzero == 0
        else ""
    )
    detail = (
        f"llama_get_embeddings_ith returned {h_length} floats (expected {n_embd}, "
        f"finite={h_finite}), but llama_get_logits_ith returned a non-NULL "
        f"{logits_length}-float buffer (expected {n_vocab}) with {emb_nonzero} "
        f"non-zero values." + zero_note
    )
    if arch == "lfm2":
        source = (
            " Architecture 'lfm2' builds the output projection only when "
            "cparams.embeddings is false (src/models/lfm2.cpp:606 in this "
            "llama.cpp tree), so h and logits cannot coexist in one decode."
        )
    else:
        source = (
            f" Architecture '{arch or 'unknown'}' did not emit logits while "
            "embeddings were enabled in this build."
        )
    return (
        "single-decode coexistence gate failed: " + detail + source
    )


def _finish(
    spike: LlamaSpike,
    context_logits: int,
    hidden: array,
    logits: array,
    *,
    h_norm: float,
    h_sha256: str,
    decode_ms: float,
    decode_h_ms: float,
    extract_ms: float,
    mode: str,
    gate_ok: bool,
    same_decode: bool,
    fallback_used: bool,
    emb_ctx_logits_nonzero: int,
    gate_reason: str,
    prompt_tokens: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run the sampling and fast-weight stages on a valid h/logits pair."""
    if not finite_vector(hidden, spike.n_embd):
        raise SpikeError(
            f"hidden state is not a finite vector of length {spike.n_embd}"
        )
    if not finite_vector(logits, spike.n_vocab):
        raise SpikeError(f"logits are not a finite vector of length {spike.n_vocab}")
    if not is_informative(logits):
        raise SpikeError("logits are not informative; cannot sample or update a row")

    top1 = token_argmax(logits)
    biased = runner_up_token(logits, top1)
    top1_logit = float(logits[top1])
    biased_logit = float(logits[biased])
    gap = top1_logit - biased_logit

    probability = softmax_probability(logits, biased)
    row = fast_weight_update(
        [0.0] * spike.n_embd, hidden, probability, args.lr
    )
    row_norm = vector_norm(row)
    readout = dot(row, hidden)

    sampled_before = spike.sample(context_logits)
    top1_after = argmax_with_bias(logits, biased, args.bias)
    sampled_after = spike.sample(context_logits, biased, args.bias)
    sampled_with_row = spike.sample(context_logits, biased, readout)

    if sampled_before != top1:
        raise SpikeError(
            f"greedy sampler returned {sampled_before}, raw argmax is {top1}"
        )
    if sampled_after != top1_after:
        raise SpikeError(
            f"biased sampler returned {sampled_after}, biased argmax is {top1_after}"
        )
    if sampled_after == sampled_before:
        raise SpikeError(
            "the logit bias did not change the sampled token; "
            f"increase --bias (gap={gap:.6f}, bias={args.bias})"
        )

    logits_nonzero = count_nonzero(logits)
    summary_values: dict[str, Any] = {
        "h_norm": h_norm,
        "h_sha256": h_sha256,
        "logits_top1_before": top1,
        "logits_top1_after": top1_after,
        "biased_token": biased,
        "finite_h": True,
        "finite_logits": True,
        "decode_ms": decode_ms,
        "extract_ms": extract_ms,
        "ok": gate_ok,
        "gate": "single_decode_coexistence",
        "gate_ok": gate_ok,
        "same_decode": same_decode,
        "mode": mode,
        "fallback_used": fallback_used,
        "fallback_ok": mode == "paired",
        "gate_reason": gate_reason,
        "reason": (
            "h and logits came from one llama_decode call"
            if same_decode
            else "paired fallback: h and logits came from two deterministic "
            "decodes in two contexts of the same process; this does not satisfy "
            "the same-decode gate"
        ),
        "arch": spike.arch,
        "n_embd": spike.n_embd,
        "n_vocab": spike.n_vocab,
        "prompt_tokens": prompt_tokens,
        "decode_h_ms": round(decode_h_ms, 6),
        "decode_total_ms": round(
            decode_h_ms if same_decode else decode_h_ms + decode_ms, 6
        ),
        "logits_nonzero": logits_nonzero,
        "emb_ctx_logits_nonzero": emb_ctx_logits_nonzero,
        "logits_informative": True,
        "h_length": len(hidden),
        "logits_length": len(logits),
        "top1_logit_before": round(top1_logit, 6),
        "biased_logit_before": round(biased_logit, 6),
        "logit_gap": round(gap, 6),
        "top1_logit_after": round(
            float(logits[top1_after])
            + (args.bias if top1_after == biased else 0.0),
            6,
        ),
        "sampler_bias": round(float(args.bias), 6),
        "sampled_before": sampled_before,
        "sampled_after": sampled_after,
        "fw_prob_target": round(probability, 9),
        "fw_lr": float(args.lr),
        "fw_row_norm": round(row_norm, 9),
        "fw_readout": round(readout, 9),
        "fw_bias_token": biased,
        "fw_sampled": sampled_with_row,
        "fw_caused_flip": sampled_with_row != sampled_before,
    }
    return build_summary(summary_values)


def run(spike: LlamaSpike, args: argparse.Namespace) -> dict[str, Any]:
    """Run the strict gate and, when explicitly asked, the paired fallback."""
    tokens = spike.tokenize(args.prompt)
    if not tokens:
        raise SpikeError("tokenizer produced no tokens")
    if len(tokens) > args.ctx:
        raise SpikeError(
            f"prompt has {len(tokens)} tokens, which exceeds --ctx {args.ctx}"
        )
    if len(tokens) > args.batch:
        raise SpikeError(
            f"prompt has {len(tokens)} tokens, which exceeds --batch {args.batch}"
        )

    context_h = spike.new_context(
        n_ctx=args.ctx,
        n_batch=args.batch,
        n_threads=args.threads,
        embeddings=True,
    )
    decode_h_ms = spike.decode(context_h, tokens, all_outputs=True)

    extract_started = time.perf_counter()
    hidden = spike.read_embeddings(context_h, -1)
    emb_logits = spike.read_logits(context_h, -1)
    h_len_ok = len(hidden) == spike.n_embd
    h_finite = finite_vector(hidden, spike.n_embd)
    logits_len_ok = len(emb_logits) == spike.n_vocab
    emb_logits_finite = finite_vector(emb_logits, spike.n_vocab)
    emb_nonzero = count_nonzero(emb_logits)
    h_norm = vector_norm(hidden) if h_finite else 0.0
    h_sha256 = sha256_prefix(hidden) if h_finite else ""
    extract_h_ms = (time.perf_counter() - extract_started) * 1000.0

    coexist = (
        h_len_ok
        and h_finite
        and logits_len_ok
        and emb_logits_finite
        and is_informative(emb_logits)
    )

    if not coexist and args.mode == "single":
        reason = _coexistence_reason(
            arch=spike.arch,
            n_embd=spike.n_embd,
            n_vocab=spike.n_vocab,
            h_length=len(hidden),
            logits_length=len(emb_logits),
            emb_nonzero=emb_nonzero,
            h_finite=h_finite,
        )
        summary = build_summary(
            {
                "h_norm": h_norm,
                "h_sha256": h_sha256,
                "logits_top1_before": -1,
                "logits_top1_after": -1,
                "biased_token": -1,
                "finite_h": h_finite,
                "finite_logits": emb_logits_finite,
                "decode_ms": decode_h_ms,
                "extract_ms": extract_h_ms,
                "ok": False,
                "gate": "single_decode_coexistence",
                "gate_ok": False,
                "same_decode": False,
                "mode": "single",
                "fallback_used": False,
                "fallback_ok": False,
                "gate_reason": reason,
                "reason": reason,
                "arch": spike.arch,
                "n_embd": spike.n_embd,
                "n_vocab": spike.n_vocab,
                "prompt_tokens": len(tokens),
                "decode_h_ms": round(decode_h_ms, 6),
                "decode_total_ms": round(decode_h_ms, 6),
                "logits_nonzero": emb_nonzero,
                "emb_ctx_logits_nonzero": emb_nonzero,
                "logits_informative": False,
                "logits_all_zero": emb_nonzero == 0,
                "h_length": len(hidden),
                "logits_length": len(emb_logits),
                "sampled_before": -1,
                "sampled_after": -1,
            }
        )
        raise GateFailure(reason, summary)

    context_logits = context_h
    logits = emb_logits
    decode_ms = decode_h_ms
    extract_ms = extract_h_ms
    same_decode = True
    gate_ok = True
    fallback_used = False
    gate_reason = ""

    if not coexist:
        # Only --mode paired can reach this branch. The h context proved that
        # the same decode cannot emit logits; a second context in the same
        # process supplies them. This is deterministic, but it is not the gate.
        context_logits = spike.new_context(
            n_ctx=args.ctx,
            n_batch=args.batch,
            n_threads=args.threads,
            embeddings=False,
        )
        decode_ms = spike.decode(context_logits, tokens, all_outputs=False)
        extract_started = time.perf_counter()
        logits = spike.read_logits(context_logits, -1)
        logits_len_ok = len(logits) == spike.n_vocab
        logits_finite = finite_vector(logits, spike.n_vocab)
        logits_nonzero = count_nonzero(logits)
        extract_ms += (time.perf_counter() - extract_started) * 1000.0
        if not (logits_len_ok and logits_finite and is_informative(logits)):
            raise SpikeError(
                "paired fallback could not obtain informative logits either: "
                f"length_ok={logits_len_ok}, finite={logits_finite}, "
                f"nonzero={logits_nonzero}"
            )
        same_decode = False
        gate_ok = False
        fallback_used = True
        gate_reason = _coexistence_reason(
            arch=spike.arch,
            n_embd=spike.n_embd,
            n_vocab=spike.n_vocab,
            h_length=len(hidden),
            logits_length=len(emb_logits),
            emb_nonzero=emb_nonzero,
            h_finite=h_finite,
        )

    return _finish(
        spike,
        context_logits,
        hidden,
        logits,
        h_norm=h_norm,
        h_sha256=h_sha256,
        decode_ms=decode_ms,
        decode_h_ms=decode_h_ms,
        extract_ms=extract_ms,
        mode=args.mode,
        gate_ok=gate_ok,
        same_decode=same_decode,
        fallback_used=fallback_used,
        emb_ctx_logits_nonzero=emb_nonzero,
        gate_reason=gate_reason,
        prompt_tokens=len(tokens),
        args=args,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "WP7a spike: hidden state + logits + a fast-weight row + a sampler "
            "logit bias, in one process. No HTTP and no server."
        )
    )
    parser.add_argument(
        "--lib",
        required=True,
        metavar="PATH",
        help="path to libllama.so from the build under test",
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="path to the GGUF model to load",
    )
    parser.add_argument(
        "--shim",
        required=True,
        metavar="PATH",
        help=(
            "path to libpmc_spike.so; it must be compiled against the same "
            "llama.h and libllama as --lib"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("single", "paired"),
        default="single",
        help=(
            "single = strict same-decode gate, fails if the model cannot emit "
            "h and logits together; paired = same process, two contexts, "
            "explicitly labelled fallback (default: single)"
        ),
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="prompt text; only its token ids are used and never printed",
    )
    parser.add_argument("--ctx", type=int, default=128, help="n_ctx (default: 128)")
    parser.add_argument("--batch", type=int, default=32, help="n_batch (default: 32)")
    parser.add_argument("--threads", type=int, default=2, help="n_threads (default: 2)")
    parser.add_argument(
        "--bias",
        type=float,
        default=DEFAULT_BIAS,
        help=f"logit bias applied to the chosen token (default: {DEFAULT_BIAS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"fast-weight row learning rate (default: {DEFAULT_LR})",
    )
    args = parser.parse_args(argv)

    for name in ("ctx", "batch", "threads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if not math.isfinite(args.bias):
        parser.error("--bias must be finite")
    if not math.isfinite(args.lr) or args.lr < 0.0:
        parser.error("--lr must be finite and non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spike: LlamaSpike | None = None
    try:
        spike = LlamaSpike(args.lib, args.shim, args.model)
        summary = run(spike, args)
        print(json.dumps(summary, allow_nan=False, sort_keys=True))
        return 0 if (summary.get("gate_ok") or summary.get("fallback_ok")) else 2
    except GateFailure as failure:
        if failure.summary is not None:
            print(json.dumps(failure.summary, allow_nan=False, sort_keys=True))
        print(f"live_spike: {failure}", file=sys.stderr)
        return 2
    except SpikeError as error:
        print(f"live_spike: {error}", file=sys.stderr)
        return 1
    finally:
        if spike is not None:
            spike.close()


if __name__ == "__main__":
    raise SystemExit(main())
