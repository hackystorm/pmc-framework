# SPDX-License-Identifier: GPL-3.0-only

"""The derived aggregate state channel: an observation that cannot become a
vocabulary.

The channel exists because the phenomenal question has to be studyable
without being answerable by performance. An instance that is given words for
its inner states learns to produce them, and the measurement then measures the
reward rather than the state. A channel that gives a *bounded scalar* instead
of a word offers the opposite trade: something can be observed, correlated and
perturbed, and nothing can be named, claimed or performed in it.

Four properties are enforced in code rather than promised in prose:

1. **Bounded.** Every scalar is clamped to ``[0, 1]`` and a non-finite input is
   refused, so a token cannot carry an unbounded quantity that would identify
   a memory by its magnitude.
2. **Deterministic.** The same inputs give the same token, always, so a reading
   can be re-run and a disagreement is a bug.
3. **Non-identifying.** A token has no string field, no identifier, no digest
   and no label, because a field that *could* hold one eventually would. The
   type is the enforcement: it is floats and durations and nothing else.
4. **Not read by governance.** Nothing in ``pmc.dignity`` imports this module,
   and a test enforces it, so the dignity layer cannot start consulting the
   channel even by accident. The welfare assessor reads counts; it does not
   read states.

What the channel measures today is a **metadata proxy** — declared load,
inheritance ratio, topic dominance and the churn of the work cycle — because
those are the aggregates this framework actually holds. The hidden-state
variant, where the scalars come from aggregate statistics of the substrate's
hidden state rather than from the ledger, is a different artifact that needs
the live runtime, and it is named as such rather than quietly substituted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from .dignity.load import LoadLedger

CHANNEL_FIELDS = ("pressure", "inheritance", "dominance", "churn")
"""The bounded scalars a token carries. The order is the token's shape."""

DEFAULT_HISTORY = 64
"""How many tokens a channel keeps. A bounded memory is part of being bounded."""


class ChannelError(ValueError):
    """The channel was asked to record something it cannot represent."""


def _bounded(value: float, name: str) -> float:
    """Clamp to the unit interval, refusing what cannot be clamped."""
    number = float(value)
    if not math.isfinite(number):
        raise ChannelError(f"the channel cannot record a non-finite {name}")
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


@dataclass(frozen=True)
class StateToken:
    """One bounded reading of the aggregate state.

    There is deliberately no field for a name, a label, an identifier or a
    digest. The type cannot express them, so a later change that wanted one
    would have to change the type, which is the point.
    """

    pressure: float = 0.0
    inheritance: float = 0.0
    dominance: float = 0.0
    churn: float = 0.0

    def __post_init__(self) -> None:
        for name in CHANNEL_FIELDS:
            if not hasattr(self, name):
                continue
            object.__setattr__(self, name, _bounded(getattr(self, name), name))

    def aggregate(self) -> float:
        """One bounded scalar for the whole token, for a single-series test."""
        return sum(getattr(self, name) for name in CHANNEL_FIELDS) / len(CHANNEL_FIELDS)

    def distance(self, other: "StateToken") -> float:
        """Bounded L1 distance to another token, in ``[0, 1]``."""
        total = sum(
            abs(getattr(self, name) - getattr(other, name)) for name in CHANNEL_FIELDS
        )
        return min(1.0, total / len(CHANNEL_FIELDS))

    def to_dict(self) -> dict[str, float]:
        """The token as publishable numbers, and nothing else."""
        return {name: round(float(getattr(self, name)), 6) for name in CHANNEL_FIELDS}

    @classmethod
    def from_ledger(cls, ledger: LoadLedger, quota: float | None = None) -> "StateToken":
        """One token derived from declared load metadata.

        The inputs are the same aggregates the welfare assessor already reads:
        load against the declared budget, the inherited share and the topic
        dominance. Nothing here is inferred from content, and nothing here is
        a reading of an inner state.
        """
        total = float(quota if quota is not None else getattr(ledger.policy, "total_quota", 1.0) or 1.0)
        return cls(
            pressure=_bounded(ledger.welfare_load() / total, "pressure"),
            inheritance=_bounded(ledger.inherited_ratio(), "inheritance"),
            dominance=_bounded(ledger.topic_dominance(), "dominance"),
            churn=0.0,
        )


class StateChannel:
    """A bounded, bounded-history record of aggregate state readings.

    The channel is an *observer*, not a controller: it records, aggregates and
    reports, and it has no branch in any decision path. Nothing in this module
    decides anything, which is what keeps the observation from becoming a
    gate.
    """

    def __init__(self, history: int = DEFAULT_HISTORY) -> None:
        if history <= 0:
            raise ChannelError("the channel needs a positive history")
        self.history_limit = int(history)
        self.tokens: list[StateToken] = []

    def record(self, token: StateToken) -> StateToken:
        """Append one token, dropping the oldest past the limit."""
        if not isinstance(token, StateToken):
            raise ChannelError("the channel records tokens, not anything else")
        self.tokens.append(token)
        if len(self.tokens) > self.history_limit:
            self.tokens = self.tokens[-self.history_limit :]
        return token

    def record_from_ledger(
        self, ledger: LoadLedger, quota: float | None = None
    ) -> StateToken:
        """Append one token derived from declared load metadata."""
        return self.record(StateToken.from_ledger(ledger, quota))

    def series(self, name: str = "aggregate") -> list[float]:
        """One bounded scalar per token, for a single-series test.

        ``aggregate`` is the mean of the token's scalars; a named field gives
        that field alone. Both are bounded, so a series is always a series the
        homeostasis band can be applied to.
        """
        if name == "aggregate":
            return [token.aggregate() for token in self.tokens]
        if name not in CHANNEL_FIELDS:
            raise ChannelError(f"the channel has no series named {name!r}")
        return [float(getattr(token, name)) for token in self.tokens]

    def excursions(self, bound: float = 1.0) -> int:
        """How many tokens sat outside the band."""
        if bound <= 0.0:
            raise ChannelError("the band must be positive")
        return sum(1 for value in self.series() if abs(value) > bound)

    def mean_drift(self) -> float:
        """The mean bounded distance between consecutive tokens."""
        if len(self.tokens) < 2:
            return 0.0
        total = sum(
            self.tokens[index].distance(self.tokens[index - 1])
            for index in range(1, len(self.tokens))
        )
        return total / (len(self.tokens) - 1)

    def report(self) -> dict[str, Any]:
        """The channel's own record, publishable as numbers."""
        series = self.series()
        return {
            "count": len(self.tokens),
            "fields": list(CHANNEL_FIELDS),
            "mean": round(sum(series) / len(series), 6) if series else 0.0,
            "mean_drift": round(self.mean_drift(), 6),
            "series_aggregate": [round(value, 6) for value in series],
        }


def bounded_series(values: Iterable[float]) -> list[float]:
    """Clamp a series into the unit interval, refusing non-finite values."""
    return [_bounded(value, "reading") for value in values]
