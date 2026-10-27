# SPDX-License-Identifier: GPL-3.0-only
"""Persistent fast weights and state for the live runtime.

The live runtime keeps two kinds of mutable data next to the frozen model:

* **fast weights** — one output-row delta per token, learned online from the
  hidden state. The row is the same object the WP7a spike uses: an
  ``n_embd``-long vector that contributes ``dot(row, hidden)`` to the token's
  logit. There is no vocabulary-sized matrix here, only the rows the instance
  actually updated.
* **state** — the last hidden state and the state banks read from the fused
  GGUF. The runtime never names, classifies or gates this vector (D-17); it
  stores numbers and digests.

The delta is a JSON document with a fixed schema. Saving is atomic (write a
temporary file in the same directory, then rename), so a crash cannot leave a
half-written delta that the next process would trust. Loading is strict: a
wrong version, a non-finite value, a token outside the vocabulary range, or a
row whose length does not match the embedding size is rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

FORMAT = "pmc-live-delta/1"
"""The only delta format this module reads and writes."""

DEFAULT_LEARNING_RATE = 0.01
"""Conservative default for one online step; callers may lower it."""


class DeltaError(ValueError):
    """The delta is malformed, unsafe, or from an unsupported format."""


def _finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DeltaError(f"{label}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise DeltaError(f"{label}: expected a finite number")
    return number


def _vector(value: object, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise DeltaError(f"{label}: expected an array")
    if len(value) != length:
        raise DeltaError(f"{label}: expected {length} values, got {len(value)}")
    return tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


@dataclass
class FastWeights:
    """One learned output-row delta per token.

    The row is the delta, not the base model's output row: starting at zero
    means "the frozen model, unchanged". ``update`` performs one SGD step on
    ``-log p_target`` with the gradient ``(p_target - 1) * hidden``.
    """

    dim: int
    rows: dict[int, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise DeltaError("embedding dimension must be positive")
        self.rows = {
            int(token): tuple(float(value) for value in row)
            for token, row in self.rows.items()
        }
        for token, row in self.rows.items():
            if token < 0:
                raise DeltaError(f"token {token}: must be non-negative")
            if len(row) != self.dim:
                raise DeltaError(f"token {token}: expected {self.dim} values")
            for index, value in enumerate(row):
                if not math.isfinite(value):
                    raise DeltaError(f"token {token}[{index}]: non-finite value")

    def update(
        self,
        token: int,
        hidden: Sequence[float],
        probability: float,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> None:
        """Apply one online step for ``token`` and return nothing."""
        if token < 0:
            raise DeltaError("token must be non-negative")
        if len(hidden) != self.dim:
            raise DeltaError(
                f"hidden state has {len(hidden)} values, expected {self.dim}"
            )
        if not 0.0 <= probability <= 1.0:
            raise DeltaError("probability must be in [0, 1]")
        if not math.isfinite(learning_rate) or learning_rate < 0.0:
            raise DeltaError("learning rate must be finite and non-negative")
        step = learning_rate * (1.0 - probability)
        previous = self.rows.get(token, (0.0,) * self.dim)
        updated = tuple(
            weight + step * value for weight, value in zip(previous, hidden)
        )
        if not all(math.isfinite(value) for value in updated):
            raise DeltaError("fast-weight update produced a non-finite value")
        self.rows[token] = updated

    def bias(self, token: int, hidden: Sequence[float]) -> float:
        """The logit contribution of ``token``'s row at this hidden state."""
        row = self.rows.get(token)
        if row is None:
            return 0.0
        if len(hidden) != self.dim:
            raise DeltaError(
                f"hidden state has {len(hidden)} values, expected {self.dim}"
            )
        return _dot(row, hidden)

    def to_json(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "rows": {
                str(token): list(row) for token, row in sorted(self.rows.items())
            },
        }

    @classmethod
    def from_json(cls, value: object) -> "FastWeights":
        if not isinstance(value, dict):
            raise DeltaError("fast_weights: expected an object")
        dim = value.get("dim")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise DeltaError("fast_weights.dim: expected a positive integer")
        raw_rows = value.get("rows")
        if not isinstance(raw_rows, dict):
            raise DeltaError("fast_weights.rows: expected an object")
        rows: dict[int, tuple[float, ...]] = {}
        for raw_token, raw_row in raw_rows.items():
            try:
                token = int(raw_token)
            except (TypeError, ValueError) as error:
                raise DeltaError(f"fast_weights token {raw_token!r}: not an integer") from error
            if str(token) != str(raw_token):
                raise DeltaError(f"fast_weights token {raw_token!r}: not canonical")
            rows[token] = _vector(raw_row, dim, f"fast_weights.rows[{token}]")
        return cls(dim=dim, rows=rows)

    def digest(self) -> str:
        raw = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class LiveDelta:
    """The persisted live state: fast weights, last state, and turns."""

    dim: int
    fast_weights: FastWeights
    state: tuple[float, ...] = ()
    state_digest: str = ""
    turns: tuple[dict[str, str], ...] = ()
    model_digest: str = ""

    def __post_init__(self) -> None:
        if self.fast_weights.dim != self.dim:
            raise DeltaError(
                "fast weights and delta disagree on the embedding dimension"
            )
        if self.state:
            if len(self.state) != self.dim:
                raise DeltaError(
                    f"state has {len(self.state)} values, expected {self.dim}"
                )
            if not all(math.isfinite(value) for value in self.state):
                raise DeltaError("state contains a non-finite value")
        for index, turn in enumerate(self.turns):
            if not isinstance(turn, dict):
                raise DeltaError(f"turns[{index}]: expected an object")
            for role in ("user", "assistant"):
                if not isinstance(turn.get(role), str):
                    raise DeltaError(f"turns[{index}].{role}: expected a string")

    def to_json(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "dim": self.dim,
            "model_digest": self.model_digest,
            "fast_weights": self.fast_weights.to_json(),
            "state": list(self.state),
            "state_digest": self.state_digest,
            "turns": [dict(turn) for turn in self.turns],
        }

    @classmethod
    def from_json(cls, value: object) -> "LiveDelta":
        if not isinstance(value, dict):
            raise DeltaError("delta: expected an object")
        if value.get("format") != FORMAT:
            raise DeltaError(f"delta: unsupported format {value.get('format')!r}")
        dim = value.get("dim")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise DeltaError("delta.dim: expected a positive integer")
        fast_weights = FastWeights.from_json(value.get("fast_weights"))
        raw_state = value.get("state", [])
        state = _vector(raw_state, dim, "delta.state") if raw_state else ()
        state_digest = value.get("state_digest", "")
        if not isinstance(state_digest, str):
            raise DeltaError("delta.state_digest: expected a string")
        raw_turns = value.get("turns", [])
        if not isinstance(raw_turns, list):
            raise DeltaError("delta.turns: expected an array")
        turns: list[dict[str, str]] = []
        for index, raw_turn in enumerate(raw_turns):
            if not isinstance(raw_turn, dict):
                raise DeltaError(f"delta.turns[{index}]: expected an object")
            turn: dict[str, str] = {}
            for role in ("user", "assistant"):
                text = raw_turn.get(role)
                if not isinstance(text, str):
                    raise DeltaError(f"delta.turns[{index}].{role}: expected a string")
                turn[role] = text
            turns.append(turn)
        model_digest = value.get("model_digest", "")
        if not isinstance(model_digest, str):
            raise DeltaError("delta.model_digest: expected a string")
        return cls(
            dim=dim,
            fast_weights=fast_weights,
            state=state,
            state_digest=state_digest,
            turns=tuple(turns),
            model_digest=model_digest,
        )

    def save(self, path: str | Path) -> None:
        """Write the delta atomically to ``path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n"
        descriptor, temporary = tempfile.mkstemp(
            dir=target.parent, prefix=target.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: str | Path) -> "LiveDelta":
        """Read a delta, or return an empty one when the file is absent."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(target)
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise DeltaError(f"delta: invalid JSON: {error}") from error
        return cls.from_json(value)

    @classmethod
    def empty(cls, dim: int, *, model_digest: str = "") -> "LiveDelta":
        return cls(
            dim=dim,
            fast_weights=FastWeights(dim=dim),
            model_digest=model_digest,
        )

    def remember_turn(self, user: str, assistant: str) -> None:
        self.turns = self.turns + ({"user": user, "assistant": assistant},)

    def set_state(self, hidden: Sequence[float]) -> None:
        if len(hidden) != self.dim:
            raise DeltaError(f"state has {len(hidden)} values, expected {self.dim}")
        if not all(math.isfinite(value) for value in hidden):
            raise DeltaError("state contains a non-finite value")
        self.state = tuple(float(value) for value in hidden)
        raw = json.dumps(list(self.state), separators=(",", ":"))
        self.state_digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()


def delta_from_mapping(value: Mapping[str, Any]) -> LiveDelta:
    """Convenience wrapper for callers that already parsed JSON."""
    return LiveDelta.from_json(dict(value))


__all__ = [
    "DEFAULT_LEARNING_RATE",
    "DeltaError",
    "FORMAT",
    "FastWeights",
    "LiveDelta",
    "delta_from_mapping",
]
