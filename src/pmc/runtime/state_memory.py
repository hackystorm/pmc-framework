# SPDX-License-Identifier: GPL-3.0-only
"""A self-contained memory bank consumed by the live runtime.

The state-bank reader in :mod:`pmc.runtime.state_banks` is deliberately
generic: it returns every ``pmc.state.*`` bank as numbers.  This module gives
one bank schema a meaning.  A fused model may carry:

    pmc.state.memory.keys             [n, d]      unit-length keys
    pmc.state.memory.values           [n]         token ids
    pmc.state.memory.projection       [d, n_embd] hidden -> key space
    pmc.state.memory.projection_bias  [d]
    pmc.state.memory.logit_bias       [1]         positive readout bias
    pmc.state.memory.min_similarity   [1]         optional cosine floor

At the first generated token the runtime projects the current hidden state,
normalises it, finds the nearest key, and adds ``logit_bias`` to the stored
value token if the cosine is at least ``min_similarity``.  The bank is read
from GGUF metadata only; the tensor data is never touched.  The engine does
not name, classify or gate the hidden state (D-17): the bank supplies a
numeric readout, and the runtime uses it.

A ``pmc.state.swap.*`` bank from the PMC-mini benchmark is intentionally not
this schema.  It needs the benchmark's host projection, which is part of the
bench spec rather than the model file, so the benchmark reads it with its own
harness.  The live engine reports it as metadata but does not consume it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .state_banks import StateBank

PREFIX = "memory."
"""The bank namespace this module consumes."""

_REQUIRED = frozenset(
    {
        "keys",
        "values",
        "projection",
        "projection_bias",
        "logit_bias",
    }
)
_OPTIONAL = frozenset({"min_similarity"})
_NORM_TOLERANCE = 1e-4


class StateMemoryError(ValueError):
    """A memory bank is present but malformed."""


@dataclass(frozen=True)
class StateMemory:
    """A validated key/value memory read from ``pmc.state.memory.*``."""

    keys: tuple[tuple[float, ...], ...]
    values: tuple[int, ...]
    projection: tuple[tuple[float, ...], ...]
    projection_bias: tuple[float, ...]
    logit_bias: float
    min_similarity: float
    input_dim: int

    @classmethod
    def from_banks(
        cls,
        banks: Mapping[str, StateBank],
        *,
        n_embd: int,
        n_vocab: int,
    ) -> "StateMemory | None":
        """Return the memory in ``banks``, or ``None`` when no memory exists.

        A partial or malformed ``memory.*`` namespace is an error, never a
        silent fallback to the base model.
        """
        if n_embd <= 0 or n_vocab <= 0:
            raise StateMemoryError("n_embd and n_vocab must be positive")
        present = {
            name[len(PREFIX) :]: bank
            for name, bank in banks.items()
            if name.startswith(PREFIX)
        }
        if not present:
            return None
        unknown = set(present) - _REQUIRED - _OPTIONAL
        missing = _REQUIRED - set(present)
        if unknown or missing:
            raise StateMemoryError(
                f"memory banks are incomplete or unknown: missing {sorted(missing)}, "
                f"unknown {sorted(unknown)}"
            )
        for name, bank in present.items():
            if not all(math.isfinite(float(value)) for value in bank.data):
                raise StateMemoryError(f"{name} contains a non-finite value")

        keys_bank = present["keys"]
        values_bank = present["values"]
        projection_bank = present["projection"]
        bias_bank = present["projection_bias"]
        logit_bank = present["logit_bias"]

        if len(keys_bank.dims) != 2:
            raise StateMemoryError("memory.keys must have two dimensions")
        count, key_dim = keys_bank.dims
        if count <= 0 or key_dim <= 0:
            raise StateMemoryError("memory.keys dimensions must be positive")
        if len(keys_bank.data) != count * key_dim:
            raise StateMemoryError("memory.keys data does not match its dimensions")
        if len(values_bank.data) != count:
            raise StateMemoryError("memory.values must have one value per key")
        if len(projection_bank.data) != key_dim * n_embd:
            raise StateMemoryError("memory.projection data does not match its dimensions")
        if len(bias_bank.data) != key_dim:
            raise StateMemoryError("memory.projection_bias must have [key_dim] values")
        if len(logit_bank.data) != 1:
            raise StateMemoryError("memory.logit_bias must have one value")
        if values_bank.dims != (count,):
            raise StateMemoryError("memory.values must have one value per key")
        if projection_bank.dims != (key_dim, n_embd):
            raise StateMemoryError(
                "memory.projection must be [key_dim, n_embd]"
            )
        if bias_bank.dims != (key_dim,):
            raise StateMemoryError("memory.projection_bias must be [key_dim]")
        if logit_bank.dims != (1,):
            raise StateMemoryError("memory.logit_bias must be [1]")

        keys = _rows(keys_bank, count, key_dim)
        for index, key in enumerate(keys):
            norm = math.sqrt(math.fsum(value * value for value in key))
            if abs(norm - 1.0) > _NORM_TOLERANCE:
                raise StateMemoryError(f"memory.keys[{index}] is not unit length")
        values = _token_ids(values_bank.data, n_vocab)
        projection = _rows(projection_bank, key_dim, n_embd)
        projection_bias = tuple(float(value) for value in bias_bank.data)
        logit_bias = float(logit_bank.data[0])
        if not math.isfinite(logit_bias) or logit_bias <= 0.0:
            raise StateMemoryError("memory.logit_bias must be finite and positive")
        if "min_similarity" in present:
            floor_bank = present["min_similarity"]
            if floor_bank.dims != (1,) or len(floor_bank.data) != 1:
                raise StateMemoryError("memory.min_similarity must be [1]")
            min_similarity = float(floor_bank.data[0])
            if (
                not math.isfinite(min_similarity)
                or not -1.0 <= min_similarity <= 1.0
            ):
                raise StateMemoryError("memory.min_similarity must be in [-1, 1]")
        else:
            min_similarity = 0.0

        return cls(
            keys=keys,
            values=values,
            projection=projection,
            projection_bias=projection_bias,
            logit_bias=logit_bias,
            min_similarity=min_similarity,
            input_dim=n_embd,
        )

    def readout(self, hidden: Sequence[float]) -> tuple[int, float] | None:
        """Return ``(token, bias)`` for the nearest key, or ``None``.

        ``None`` means the hidden state is zero or the best cosine is below the
        bank's floor.  A malformed vector is an error, not a silent miss.
        """
        if len(hidden) != self.input_dim:
            raise StateMemoryError(
                f"hidden has {len(hidden)} values, expected {self.input_dim}"
            )
        values = tuple(float(value) for value in hidden)
        if any(not math.isfinite(value) for value in values):
            raise StateMemoryError("hidden contains a non-finite value")
        projected = [
            math.fsum(weight * value for weight, value in zip(row, values)) + bias
            for row, bias in zip(self.projection, self.projection_bias)
        ]
        norm = math.sqrt(math.fsum(value * value for value in projected))
        if norm <= 1e-12:
            return None
        query = tuple(value / norm for value in projected)
        scores = [
            math.fsum(left * right for left, right in zip(query, key))
            for key in self.keys
        ]
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] < self.min_similarity:
            return None
        return self.values[best], self.logit_bias

    def summary(self) -> dict[str, object]:
        """Metadata-only summary; the key and value payloads stay private."""
        return {
            "schema": "pmc-state-memory/1",
            "keys": len(self.keys),
            "key_dim": len(self.keys[0]) if self.keys else 0,
            "input_dim": self.input_dim,
            "logit_bias": self.logit_bias,
            "min_similarity": self.min_similarity,
        }


def _rows(bank: StateBank, rows: int, columns: int) -> tuple[tuple[float, ...], ...]:
    data = list(bank.data)
    return tuple(
        tuple(data[index * columns : (index + 1) * columns])
        for index in range(rows)
    )


def _token_ids(data: Sequence[float], n_vocab: int) -> tuple[int, ...]:
    token_ids: list[int] = []
    for index, value in enumerate(data):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise StateMemoryError(f"memory.values[{index}] is not an integer token id")
        token = int(number)
        if token < 0 or token >= n_vocab:
            raise StateMemoryError(
                f"memory.values[{index}] is outside the vocabulary"
            )
        token_ids.append(token)
    return tuple(token_ids)


__all__ = ["PREFIX", "StateMemory", "StateMemoryError"]
