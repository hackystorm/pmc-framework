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


# -- the hidden-state source ------------------------------------------------


HIDDEN_FIELDS = ("energy", "drift", "spread", "novelty")
"""The bounded scalars a hidden-state token carries.

Each is an *aggregate* of the hidden vector - its magnitude, the change in
that magnitude since the previous reading, its dispersion, and its distance
from the running mean - and none of them is a dimension, a token, or anything
that could identify what the vector was about.

``drift`` is deliberately the *change in magnitude* rather than the distance
between two readings. A distance would require keeping the previous vector,
and a module that stores hidden vectors is not a channel that observes them:
an earlier version of this source kept one, and the test that says "the source
keeps no vector" caught it. The sacrifice is real and it is documented: the
channel sees how big a state is and how that changed, never the direction it
moved in.
"""

HIDDEN_TO_TOKEN = {
    "energy": "pressure",
    "drift": "inheritance",
    "spread": "dominance",
    "novelty": "churn",
}
"""Where each hidden aggregate lands on the token, and why they may share.

The channel has one token shape, and the hidden source reuses it rather than
adding fields, because a source-specific field would be a source-specific
vocabulary. Which aggregate each slot holds is recorded here, in one place, so
a reader of a token can tell what it was measured from.
"""


def _magnitude(vector: Sequence[float], scale: float) -> float:
    """The vector's L2 norm, scaled by a declared reference magnitude."""
    total = math.sqrt(sum(value * value for value in vector))
    return _bounded(total / scale, "energy")


def _dispersion(vector: Sequence[float], scale: float) -> float:
    """The largest absolute coordinate, scaled, so the token sees spread."""
    peak = max((abs(value) for value in vector), default=0.0)
    return _bounded(peak / scale, "spread")


class HiddenSource:
    """Turns hidden vectors into channel tokens, keeping nothing else.

    This is the source the protocol version of the channel needs, and it is
    the one the metadata proxy cannot substitute for, because the question it
    serves is about the substrate's own state rather than about the ledger.

    The properties that make it safe are the point, and they are enforced
    rather than intended:

    - **No vector is stored.** The token is built and returned; the vector is
      dropped. A token has no field that could hold one, and this class holds
      no reference to one either.
    - **Two vectors with the same aggregates give the same token.** So the
      token cannot distinguish two facts that happen to share a magnitude, a
      drift, a spread and a novelty, which is what stops it being an identity
      function on the hidden state.
    - **Every scalar is bounded and deterministic.** Same vector, same token.
    """

    def __init__(self, scale: float = 64.0, decay: float = 0.9) -> None:
        if not (scale > 0.0):
            raise ChannelError("the reference magnitude must be positive")
        if not (0.0 < decay < 1.0):
            raise ChannelError("the running-mean decay must be inside (0, 1)")
        self.scale = float(scale)
        self.decay = float(decay)
        # Only scalars live past a token: the previous magnitude and the
        # running mean *of the magnitude*. No vector-shaped state is kept, and
        # that is total rather than partial - a vector-shaped running mean with
        # a 0.9 decay is 90% of the previous vector, which is the previous
        # vector. So a restart of the channel is a genuine reset and not a
        # recovery of what it last saw.
        self._previous_magnitude: float | None = None
        self._magnitude_mean: float | None = None
        self._seen = 0

    def reset(self) -> None:
        """Forget the running state, which is all the state there is."""
        self._previous_magnitude = None
        self._magnitude_mean = None
        self._seen = 0

    def token(self, vector: Sequence[float]) -> StateToken:
        """One bounded token from one hidden vector, keeping nothing."""
        if not vector:
            raise ChannelError("the hidden source needs a non-empty vector")
        for value in vector:
            if not math.isfinite(float(value)):
                raise ChannelError("the hidden source refuses a non-finite vector")
        total = math.sqrt(sum(float(value) * float(value) for value in vector))
        energy = _bounded(total / self.scale, "energy")
        spread = _dispersion(vector, self.scale)
        # The drift is the change in magnitude, not the distance between two
        # readings: computing a distance would mean keeping the previous
        # vector, which is the one thing this source must not do.
        if self._previous_magnitude is None:
            drift = 0.0
        else:
            drift = _bounded(
                abs(total - self._previous_magnitude) / self.scale, "drift"
            )
        # Novelty is how far this reading's magnitude sits from the running
        # mean of magnitudes, never a distance from a vector-shaped mean: a
        # mean of recent vectors is most of the last vector.
        if self._magnitude_mean is None:
            novelty = 0.0
        else:
            novelty = _bounded(
                abs(total - self._magnitude_mean) / self.scale, "novelty"
            )
        self._magnitude_mean = (
            total
            if self._magnitude_mean is None
            else self.decay * self._magnitude_mean + (1.0 - self.decay) * total
        )
        self._previous_magnitude = total
        self._seen += 1
        return StateToken(
            pressure=energy,
            inheritance=drift,
            dominance=spread,
            churn=novelty,
        )

    def tokens(self, vectors: Iterable[Sequence[float]]) -> list[StateToken]:
        """The tokens for a run of vectors, in order, keeping nothing else."""
        self.reset()
        return [self.token(vector) for vector in vectors]

    def series(self, vectors: Iterable[Sequence[float]]) -> list[float]:
        """The aggregate series over a run of vectors, as a bounded list."""
        return [token.aggregate() for token in self.tokens(vectors)]
