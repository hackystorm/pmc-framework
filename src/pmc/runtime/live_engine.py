# SPDX-License-Identifier: GPL-3.0-only
"""The live runtime: one model, one process, mutable memory next to it.

The frozen mode runs the artifact on any stock provider and hands the instance
its own text back. The live mode is different: the patched ``libllama`` is
loaded in this process, the hidden state and the logits come from the same
decode, and a small set of output-row deltas (the *fast weights*) is updated
from the tokens the instance produces. Those deltas and the last hidden state
form the live delta, which is persisted atomically next to the model.

Three boundaries are deliberate:

* **No second server.** One process owns the model; the OpenAI-compatible
  HTTP layer in ``live_server.py`` talks to this engine in-process.
* **No inner-state labels.** The engine stores the hidden vector and its digest.
  It never names, classifies or gates it (D-17); the state-bank names in the
  file are reported as they are, without interpretation.
* **No silent fallback.** Sampling is greedy. A request that asks for a
  temperature above zero is rejected instead of being quietly turned into
  greedy sampling, and a delta written for a different embedding size or model
  digest is rejected instead of being reused.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .live_spike import LlamaSpike, SpikeError, softmax_probability
from .live_state import DEFAULT_LEARNING_RATE, DeltaError, LiveDelta
from .state_banks import StateBank, read_state_banks
from .state_memory import StateMemory, StateMemoryError

DEFAULT_MAX_TOKENS = 64
"""Default completion budget when a request does not set one."""

MAX_REQUEST_TOKENS = 4096
"""Upper bound accepted from an API request before the context check."""


class LiveEngineError(RuntimeError):
    """The live engine cannot satisfy a request as specified."""


class KvStateError(ValueError):
    """A saved context state is malformed, truncated or from another model."""


_KV_STATE_MAGIC = b"PMCKV1\0"
_KV_STATE_FORMAT = "pmc-kv-state/1"
_KV_STATE_HEADER_LIMIT = 1 << 20
"""The JSON header is small; a larger length is a malformed or hostile file."""


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` with a same-directory rename."""
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class KvState:
    """A saved context state and the exact token prefix it covers.

    The bytes are the llama.cpp context state (KV cache and positions). The
    token prefix is kept outside the state so a later call can verify that the
    new prompt really starts with the tokens that produced this state. A state
    is bound to one model and one context geometry. The engine rejects a state
    whose model digest, embedding size or context size differs before it calls
    into the C API, and rejects a prompt whose token prefix is different.
    """

    data: bytes = field(repr=False)
    tokens: tuple[int, ...]
    sha256: str
    model_digest: str = ""
    n_embd: int = 0
    n_ctx: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("KV state data must be non-empty bytes")
        if not isinstance(self.tokens, tuple) or not self.tokens:
            raise ValueError("KV state must cover a non-empty token prefix")
        for token in self.tokens:
            if not isinstance(token, int) or isinstance(token, bool) or token < 0:
                raise ValueError("KV state tokens must be non-negative integers")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64:
            raise ValueError("KV state sha256 must be 64 hexadecimal characters")
        if any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("KV state sha256 must be lowercase hexadecimal")
        if hashlib.sha256(self.data).hexdigest() != self.sha256:
            raise ValueError("KV state sha256 does not match its bytes")
        if not isinstance(self.model_digest, str):
            raise ValueError("KV state model_digest must be a string")
        if self.model_digest and (
            len(self.model_digest) != 64
            or any(char not in "0123456789abcdef" for char in self.model_digest)
        ):
            raise ValueError(
                "KV state model_digest must be 64 lowercase hexadecimal characters"
            )
        for label, value in (("n_embd", self.n_embd), ("n_ctx", self.n_ctx)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"KV state {label} must be a non-negative integer")
        if (self.n_embd == 0) != (self.n_ctx == 0):
            raise ValueError(
                "KV state geometry must be either fully unknown or fully specified"
            )

    def matches(self, model_digest: str, n_embd: int, n_ctx: int) -> bool:
        """True when this state may be restored into this model and context.

        A known mismatch always rejects. An unknown value is accepted only
        because older or hand-built states predate the identity fields; every
        state written by this version carries the model digest and both geometry
        values. The engine still checks the token prefix before restoring.
        """
        if self.model_digest and model_digest and self.model_digest != model_digest:
            return False
        if self.n_embd and self.n_embd != n_embd:
            return False
        if self.n_ctx and self.n_ctx != n_ctx:
            return False
        return True

    def summary(self) -> dict[str, Any]:
        """Metadata only: no state bytes, no hidden values."""
        return {
            "bytes": len(self.data),
            "tokens": len(self.tokens),
            "sha256": self.sha256,
            "model_digest": self.model_digest,
            "n_embd": self.n_embd,
            "n_ctx": self.n_ctx,
        }

    def save(self, path: str | Path) -> None:
        """Write the state to one self-describing file, atomically.

        Layout: the magic ``PMCKV1\\0``, a big-endian ``u32`` header length, a
        compact JSON header, then the raw state bytes. One file means a crash
        cannot leave a header that describes a different payload; the digest
        check catches any other corruption.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        header = json.dumps(
            {
                "format": _KV_STATE_FORMAT,
                "bytes": len(self.data),
                "tokens": list(self.tokens),
                "sha256": self.sha256,
                "model_digest": self.model_digest,
                "n_embd": self.n_embd,
                "n_ctx": self.n_ctx,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(header) > _KV_STATE_HEADER_LIMIT:
            raise KvStateError("KV state header is too large")
        payload = (
            _KV_STATE_MAGIC
            + len(header).to_bytes(4, "big")
            + header
            + self.data
        )
        _atomic_write(target, payload)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_model_digest: str | None = None,
        expected_n_embd: int | None = None,
        expected_n_ctx: int | None = None,
    ) -> "KvState":
        """Read a state written by :meth:`save` and verify its identity."""
        raw = Path(path).read_bytes()
        prefix = len(_KV_STATE_MAGIC) + 4
        if len(raw) < prefix:
            raise KvStateError("KV state file is truncated")
        if raw[: len(_KV_STATE_MAGIC)] != _KV_STATE_MAGIC:
            raise KvStateError("KV state file has the wrong magic")
        header_length = int.from_bytes(
            raw[len(_KV_STATE_MAGIC) : prefix], "big"
        )
        if header_length == 0 or header_length > _KV_STATE_HEADER_LIMIT:
            raise KvStateError("KV state header length is invalid")
        if len(raw) < prefix + header_length:
            raise KvStateError("KV state header is truncated")
        header_bytes = raw[prefix : prefix + header_length]
        data = raw[prefix + header_length :]
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise KvStateError(f"KV state header is not valid JSON: {error}") from error
        if not isinstance(header, dict):
            raise KvStateError("KV state header must be an object")
        if header.get("format") != _KV_STATE_FORMAT:
            raise KvStateError(
                f"KV state format {header.get('format')!r} is not supported"
            )
        if header.get("bytes") != len(data):
            raise KvStateError("KV state payload length does not match its header")
        raw_tokens = header.get("tokens")
        if not isinstance(raw_tokens, list):
            raise KvStateError("KV state tokens must be an array")
        try:
            state = cls(
                data=data,
                tokens=tuple(raw_tokens),
                sha256=header.get("sha256", ""),
                model_digest=header.get("model_digest", ""),
                n_embd=header.get("n_embd", 0),
                n_ctx=header.get("n_ctx", 0),
            )
        except ValueError as error:
            raise KvStateError(str(error)) from error
        if expected_model_digest and state.model_digest != expected_model_digest:
            raise KvStateError(
                "KV state was saved for a different model file"
            )
        if expected_n_embd is not None and state.n_embd != expected_n_embd:
            raise KvStateError(
                f"KV state embedding size {state.n_embd} does not match {expected_n_embd}"
            )
        if expected_n_ctx is not None and state.n_ctx != expected_n_ctx:
            raise KvStateError(
                f"KV state context size {state.n_ctx} does not match {expected_n_ctx}"
            )
        return state


@dataclass(frozen=True)
class ChatResult:
    """One live completion and the substrate facts that produced it."""

    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    fast_weight_rows: int
    fast_weight_digest: str
    state_digest: str
    kv_state: KvState | None = None


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Content digest of a file, read in bounded chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class LiveEngine:
    """One loaded model with in-process fast weights and a persistent delta."""

    def __init__(
        self,
        model_path: str,
        lib_path: str,
        shim_path: str,
        *,
        n_ctx: int = 2048,
        n_batch: int = 256,
        n_threads: int = 2,
        delta_path: str | Path | None = None,
        system_prompt: str = "",
        model_name: str = "pmc-live",
        learning_rate: float = DEFAULT_LEARNING_RATE,
        learn_from_prompt: bool = False,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        compute_model_digest: bool = True,
    ) -> None:
        if n_ctx <= 0:
            raise LiveEngineError("n_ctx must be positive")
        if n_batch <= 0:
            raise LiveEngineError("n_batch must be positive")
        if n_threads <= 0:
            raise LiveEngineError("n_threads must be positive")
        if learning_rate < 0.0 or learning_rate != learning_rate:
            raise LiveEngineError("learning_rate must be finite and non-negative")
        if default_max_tokens <= 0:
            raise LiveEngineError("default_max_tokens must be positive")
        if not isinstance(system_prompt, str):
            raise LiveEngineError("system_prompt must be a string")

        self.model_path = str(model_path)
        self.model_name = model_name
        self.n_ctx = int(n_ctx)
        self.n_batch = min(int(n_batch), int(n_ctx))
        self.n_threads = int(n_threads)
        self.learning_rate = float(learning_rate)
        self.learn_from_prompt = bool(learn_from_prompt)
        self.default_max_tokens = int(default_max_tokens)
        self.system_prompt = system_prompt
        self.delta_path = Path(delta_path) if delta_path else None
        self._closed = False
        self._lock = threading.RLock()

        self.spike = LlamaSpike(lib_path, shim_path, self.model_path)
        try:
            self.n_embd = int(self.spike.n_embd)
            self.n_vocab = int(self.spike.n_vocab)
            self.model_digest = (
                file_sha256(self.model_path) if compute_model_digest else ""
            )
            self.state_banks: dict[str, StateBank] = read_state_banks(self.model_path)
            try:
                self.state_memory = StateMemory.from_banks(
                    self.state_banks,
                    n_embd=self.n_embd,
                    n_vocab=self.n_vocab,
                )
            except StateMemoryError as error:
                raise LiveEngineError(f"cannot read the state memory: {error}") from error
            self.context = self.spike.new_context(
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_threads=self.n_threads,
                embeddings=True,
            )
            self.delta = self._load_or_create_delta()
        except Exception:
            self.spike.close()
            raise

    # -- lifecycle --------------------------------------------------------

    def _load_or_create_delta(self) -> LiveDelta:
        if self.delta_path is not None and self.delta_path.exists():
            try:
                delta = LiveDelta.load(self.delta_path)
            except (DeltaError, OSError) as error:
                raise LiveEngineError(f"cannot load delta: {error}") from error
            if delta.dim != self.n_embd:
                raise LiveEngineError(
                    f"delta dimension {delta.dim} does not match model {self.n_embd}"
                )
            if (
                delta.model_digest
                and self.model_digest
                and delta.model_digest != self.model_digest
            ):
                raise LiveEngineError(
                    "delta was written for a different model file"
                )
            if not delta.model_digest:
                delta.model_digest = self.model_digest
            return delta
        return LiveDelta.empty(self.n_embd, model_digest=self.model_digest)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if self.delta_path is not None:
                    self.delta.save(self.delta_path)
            finally:
                self.spike.close()
                self._closed = True

    def __enter__(self) -> "LiveEngine":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- request validation ----------------------------------------------

    def _pairs(self, messages: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise LiveEngineError("messages must be an array")
        pairs: list[tuple[str, str]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise LiveEngineError(f"messages[{index}] must be an object")
            role = message.get("role")
            content = message.get("content")
            if role not in ("system", "user", "assistant"):
                raise LiveEngineError(
                    f"messages[{index}].role must be system, user or assistant"
                )
            if not isinstance(content, str) or not content:
                raise LiveEngineError(
                    f"messages[{index}].content must be a non-empty string"
                )
            pairs.append((str(role), content))
        if not pairs:
            raise LiveEngineError("messages must not be empty")
        if self.system_prompt and not any(role == "system" for role, _ in pairs):
            pairs.insert(0, ("system", self.system_prompt))
        return pairs

    @staticmethod
    def _last_user(pairs: Sequence[tuple[str, str]]) -> str:
        for role, content in reversed(pairs):
            if role == "user":
                return content
        return ""

    @staticmethod
    def _step_size(hidden: Sequence[float], learning_rate: float) -> float:
        """Scale a learning rate by ``1 / ||h||^2``.

        The raw SGD row update is ``row += lr * (1 - p) * h`` and the readout is
        ``dot(row, h)``. With the raw hidden, the first bias is
        ``lr * ||h||^2``, which for a real hidden state is thousands of logits
        and collapses the sampler onto one token. Dividing the step by
        ``||h||^2`` makes the configured learning rate a step in *logit space*:
        the first bias is exactly ``learning_rate * (1 - p)``.
        """
        if not math.isfinite(learning_rate) or learning_rate < 0.0:
            raise LiveEngineError("learning_rate must be finite and non-negative")
        norm_sq = math.fsum(value * value for value in hidden)
        if not math.isfinite(norm_sq) or norm_sq <= 0.0:
            raise LiveEngineError("hidden state has no usable norm")
        return learning_rate / norm_sq

    # -- inference --------------------------------------------------------

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        restore: KvState | None = None,
        save_state: bool = False,
        learn: bool = True,
    ) -> ChatResult:
        """Generate one assistant turn, learn from it, and persist the delta.

        ``restore`` continues from a saved context state: the prompt must start
        with exactly the tokens that state covers, the state is loaded into a
        cleared context, and only the new suffix is decoded. ``save_state``
        captures the state after this turn so a later call can restore it.
        ``learn=False`` keeps the fast weights unchanged; it is the control that
        separates the base model from the online update.
        """
        if temperature != 0.0:
            raise LiveEngineError(
                "the live runtime samples greedily; temperature must be 0"
            )
        if max_tokens is None:
            budget_request = self.default_max_tokens
        else:
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
                raise LiveEngineError("max_tokens must be an integer")
            if max_tokens <= 0 or max_tokens > MAX_REQUEST_TOKENS:
                raise LiveEngineError(
                    f"max_tokens must be in [1, {MAX_REQUEST_TOKENS}]"
                )
            budget_request = max_tokens
        if restore is not None and not isinstance(restore, KvState):
            raise LiveEngineError("restore must be a KvState")
        if not isinstance(save_state, bool):
            raise LiveEngineError("save_state must be a boolean")
        if not isinstance(learn, bool):
            raise LiveEngineError("learn must be a boolean")
        if restore is not None and not restore.matches(
            self.model_digest, self.n_embd, self.n_ctx
        ):
            raise LiveEngineError(
                "the restored state was saved for a different model or context geometry"
            )

        pairs = self._pairs(messages)

        with self._lock:
            if self._closed:
                raise LiveEngineError("the engine is closed")
            try:
                prompt = self.spike.chat_prompt(pairs, add_assistant=True)
            except (SpikeError, ValueError) as error:
                raise LiveEngineError(f"cannot build the chat prompt: {error}") from error
            tokens = self.spike.tokenize(prompt, parse_special=True)
            if not tokens:
                raise LiveEngineError("the tokenizer produced no tokens")
            if len(tokens) >= self.n_ctx:
                raise LiveEngineError(
                    f"prompt has {len(tokens)} tokens, which fills n_ctx {self.n_ctx}"
                )
            budget = min(budget_request, self.n_ctx - len(tokens))
            if budget < 1:
                raise LiveEngineError("no context budget is left for a completion")

            prefix = restore.tokens if restore is not None else ()
            if prefix:
                if len(prefix) >= len(tokens):
                    raise LiveEngineError(
                        "the restored state already covers the whole prompt; "
                        "append at least one new token"
                    )
                if tuple(tokens[: len(prefix)]) != prefix:
                    raise LiveEngineError(
                        "the restored state does not match the prompt prefix"
                    )
                self.spike.clear_context(self.context)
                try:
                    self.spike.state_load(self.context, restore.data)
                except SpikeError as error:
                    raise LiveEngineError(
                        f"cannot restore the context state: {error}"
                    ) from error
                prompt_start = len(prefix)
            else:
                self.spike.clear_context(self.context)
                prompt_start = 0

            hidden = None
            logits = None
            for start in range(prompt_start, len(tokens), self.n_batch):
                chunk = tokens[start : start + self.n_batch]
                last_chunk = start + len(chunk) == len(tokens)
                self.spike.decode(
                    self.context,
                    chunk,
                    all_outputs=self.learn_from_prompt or last_chunk,
                    start_pos=start,
                )
                if self.learn_from_prompt and learn:
                    for local in range(len(chunk)):
                        global_index = start + local
                        if global_index + 1 >= len(tokens):
                            continue
                        target = tokens[global_index + 1]
                        hidden_i = self.spike.read_embeddings(self.context, local)
                        logits_i = self.spike.read_logits(self.context, local)
                        probability = softmax_probability(logits_i, target)
                        self.delta.fast_weights.update(
                            target,
                            hidden_i,
                            probability,
                            self._step_size(hidden_i, self.learning_rate),
                        )
                if last_chunk:
                    hidden = self.spike.read_embeddings(self.context, -1)
                    logits = self.spike.read_logits(self.context, -1)
            if hidden is None or logits is None:
                raise LiveEngineError("the prompt decode produced no final state")
            self.delta.set_state(hidden)
            state_memory_readout: tuple[int, float] | None = None
            if self.state_memory is not None:
                try:
                    state_memory_readout = self.state_memory.readout(hidden)
                except StateMemoryError as error:
                    raise LiveEngineError(f"cannot read the state memory: {error}") from error

            generated: list[int] = []
            finish_reason = "length"
            position = len(tokens)
            for _ in range(budget):
                biases = [
                    (token, self.delta.fast_weights.bias(token, hidden))
                    for token in sorted(self.delta.fast_weights.rows)
                ]
                if not generated and state_memory_readout is not None:
                    biases.append(state_memory_readout)
                token = self.spike.sample_biased(self.context, biases)
                if self.spike.token_is_eog(token):
                    finish_reason = "stop"
                    break
                if learn:
                    probability = softmax_probability(logits, token)
                    self.delta.fast_weights.update(
                        token,
                        hidden,
                        probability,
                        self._step_size(hidden, self.learning_rate),
                    )
                generated.append(token)
                if len(generated) >= budget:
                    break
                self.spike.decode(
                    self.context,
                    [token],
                    all_outputs=False,
                    start_pos=position,
                )
                position += 1
                hidden = self.spike.read_embeddings(self.context, -1)
                logits = self.spike.read_logits(self.context, -1)

            if save_state and position < len(tokens) + len(generated):
                # The budget ended before the last sampled token was decoded.
                # Decode it so the saved state covers the whole reply.
                self.spike.decode(
                    self.context,
                    [generated[-1]],
                    all_outputs=False,
                    start_pos=position,
                )
                position += 1
                hidden = self.spike.read_embeddings(self.context, -1)
                logits = self.spike.read_logits(self.context, -1)

            kv_state = None
            if save_state:
                try:
                    state_data = self.spike.state_save(self.context)
                except SpikeError as error:
                    raise LiveEngineError(
                        f"cannot save the context state: {error}"
                    ) from error
                kv_state = KvState(
                    data=state_data,
                    tokens=tuple(tokens) + tuple(generated),
                    sha256=hashlib.sha256(state_data).hexdigest(),
                    model_digest=self.model_digest,
                    n_embd=self.n_embd,
                    n_ctx=self.n_ctx,
                )

            text = self.spike.detokenize(generated)
            self.delta.set_state(hidden)
            if learn:
                self.delta.remember_turn(self._last_user(pairs), text)
            if self.delta_path is not None:
                self.delta.save(self.delta_path)
            return ChatResult(
                text=text,
                finish_reason=finish_reason,
                prompt_tokens=len(tokens),
                completion_tokens=len(generated),
                fast_weight_rows=len(self.delta.fast_weights.rows),
                fast_weight_digest=self.delta.fast_weights.digest(),
                state_digest=self.delta.state_digest,
                kv_state=kv_state,
            )

    # -- inspection -------------------------------------------------------

    def state_summary(self) -> dict[str, Any]:
        """Metadata-only view of the substrate. No bank data, no hidden values."""
        return {
            "model": self.model_name,
            "model_digest": self.model_digest,
            "embedding_dim": self.n_embd,
            "vocab_size": self.n_vocab,
            "fast_weight_rows": len(self.delta.fast_weights.rows),
            "fast_weight_digest": self.delta.fast_weights.digest(),
            "state_digest": self.delta.state_digest,
            "turns": len(self.delta.turns),
            "state_memory": (
                self.state_memory.summary() if self.state_memory is not None else None
            ),
            "state_banks": [
                {
                    "name": bank.name,
                    "dims": list(bank.dims),
                    "elements": bank.elements,
                }
                for bank in self.state_banks.values()
            ],
        }


__all__ = [
    "ChatResult",
    "DEFAULT_MAX_TOKENS",
    "KvState",
    "KvStateError",
    "LiveEngine",
    "LiveEngineError",
    "MAX_REQUEST_TOKENS",
    "file_sha256",
]
