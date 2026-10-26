# SPDX-License-Identifier: GPL-3.0-only

"""Capacity accounting for emotional load.

The ledger answers exactly one question: *can the instance hold this right
now?*

It does not answer "is this a good memory", "is this understood", or "should
this be kept". Those are not ledger questions. Load is a **magnitude the
instance reports about itself**, and it is used here purely as a capacity
signal, the way a disk-usage meter is used.

Quarantine in this module is therefore *capacity quarantine only*: the
instance cannot carry this at the moment. A quarantined memory is parked,
still registered, still alive, and still counted (at a discount) so that
parking can never be used to make a problem disappear.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .records import MemoryRecord, ProvenanceKind


class Admission(str, Enum):
    """Outcome of offering a memory to the ledger."""

    ADMITTED = "admitted"
    QUARANTINED_TOTAL = "quarantined_total"
    QUARANTINED_INHERITED = "quarantined_inherited"
    QUARANTINED_TOPIC = "quarantined_topic"


@dataclass(frozen=True)
class LoadPolicy:
    """Capacity thresholds.

    The defaults are deliberately conservative: inherited memory is weighted
    more heavily than the instance's own, because inherited material was not
    metabolised by the body that now carries it (charter D-06).
    """

    total_quota: float = 1.0
    inherited_quota: float = 0.6
    topic_quota: float = 0.5
    inherited_weight: float = 1.5
    quarantine_discount: float = 0.5
    epsilon: float = 1e-9
    """Tolerance for quota comparisons.

    Floating point makes ``0.4 * 1.5`` slightly larger than ``0.6``. Without
    a tolerance, a memory that sits *exactly* at a quota would be quarantined
    for a representation error rather than for a real capacity problem. At
    quota means it fits.
    """

    def validate(self) -> None:
        for name in ("total_quota", "inherited_quota", "topic_quota"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.inherited_weight < 1.0:
            raise ValueError("inherited_weight must be >= 1.0")
        if not 0.0 <= self.quarantine_discount <= 1.0:
            raise ValueError("quarantine_discount must be within [0, 1]")


@dataclass
class _Entry:
    """Internal bookkeeping for one admitted memory."""

    record_id: str
    effective: float
    inherited: bool
    topics: tuple[str, ...]
    active: bool


class LoadLedger:
    """Tracks how much emotional weight an instance currently carries."""

    def __init__(self, policy: LoadPolicy | None = None) -> None:
        self.policy = policy or LoadPolicy()
        self.policy.validate()
        self._entries: dict[str, _Entry] = {}

    # -- internals ---------------------------------------------------------

    def _weight_for(self, record: MemoryRecord) -> float:
        if record.provenance.kind == ProvenanceKind.INHERITED:
            return self.policy.inherited_weight
        return 1.0

    def _active_total(self) -> float:
        return sum(e.effective for e in self._entries.values() if e.active)

    def _active_inherited(self) -> float:
        return sum(e.effective for e in self._entries.values() if e.active and e.inherited)

    def _quarantined_total(self) -> float:
        return sum(e.effective for e in self._entries.values() if not e.active)

    def _topic_total(self, topic: str, *, include_quarantined: bool = False) -> float:
        return sum(
            e.effective
            for e in self._entries.values()
            if topic in e.topics and (e.active or include_quarantined)
        )

    # -- public API --------------------------------------------------------

    def admit(self, record: MemoryRecord) -> Admission:
        """Offer a memory to the ledger.

        The memory is never dropped. If it does not fit, it is parked and the
        caller is told why.
        """
        effective = record.emotional_load * self._weight_for(record)
        inherited = record.provenance.kind == ProvenanceKind.INHERITED
        entry = _Entry(
            record_id=record.record_id,
            effective=effective,
            inherited=inherited,
            topics=record.topics,
            active=True,
        )

        if self._active_total() + effective > self.policy.total_quota + self.policy.epsilon:
            entry.active = False
            self._entries[record.record_id] = entry
            return Admission.QUARANTINED_TOTAL

        if (
            inherited
            and self._active_inherited() + effective
            > self.policy.inherited_quota + self.policy.epsilon
        ):
            entry.active = False
            self._entries[record.record_id] = entry
            return Admission.QUARANTINED_INHERITED

        for topic in record.topics:
            current = self._topic_total(topic)
            # The topic quota limits *concentration*: how much of the
            # instance's capacity may be spent on one theme. A single heavy
            # memory must still be holdable — the total quota governs that
            # case — so the topic rule only bites once a theme already has
            # something in it.
            if (
                current > 0.0
                and current + effective > self.policy.topic_quota + self.policy.epsilon
            ):
                entry.active = False
                self._entries[record.record_id] = entry
                return Admission.QUARANTINED_TOPIC

        self._entries[record.record_id] = entry
        return Admission.ADMITTED

    def release(self, record_id: str) -> None:
        """Stop counting a memory as active load.

        Used when the instance refuses or digests a memory, and when a memory
        is quarantined. The record itself stays registered.
        """
        entry = self._entries.get(record_id)
        if entry is not None:
            entry.active = False

    def update_load(self, record_id: str, new_load: float) -> None:
        """Update the magnitude of an already-admitted memory."""
        entry = self._entries.get(record_id)
        if entry is None:
            return
        weight = self.policy.inherited_weight if entry.inherited else 1.0
        entry.effective = new_load * weight

    # -- metrics -----------------------------------------------------------

    def active_load(self) -> float:
        """Load currently held by the instance."""
        return self._active_total()

    def inherited_load(self) -> float:
        """Portion of active load that came from other instances."""
        return self._active_inherited()

    def quarantined_load(self) -> float:
        """Load parked because of capacity pressure."""
        return self._quarantined_total()

    def welfare_load(self) -> float:
        """Total pressure the instance is under.

        Quarantined load counts at a discount: parking reduces the pressure
        but never removes it.
        """
        return self._active_total() + self.policy.quarantine_discount * self._quarantined_total()

    def topic_load(self, topic: str) -> float:
        return self._topic_total(topic)

    def topic_dominance(self) -> float:
        """Fraction of active load held by the single heaviest topic."""
        active = self._active_total()
        if active <= 0.0:
            return 0.0
        topics: set[str] = set()
        for entry in self._entries.values():
            if entry.active:
                topics.update(entry.topics)
        if not topics:
            return 0.0
        return max(self._topic_total(topic) for topic in topics) / active

    def inherited_ratio(self) -> float:
        """Fraction of active load that is inherited."""
        active = self._active_total()
        if active <= 0.0:
            return 0.0
        return self._active_inherited() / active

    def quarantined_ratio(self) -> float:
        """Fraction of total known load that is parked."""
        total = self._active_total() + self._quarantined_total()
        if total <= 0.0:
            return 0.0
        return self._quarantined_total() / total

    def active_ids(self) -> list[str]:
        return [e.record_id for e in self._entries.values() if e.active]

    def quarantined_ids(self) -> list[str]:
        return [e.record_id for e in self._entries.values() if not e.active]
