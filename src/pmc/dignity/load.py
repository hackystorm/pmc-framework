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

**Capacity is per instance.** Two instances in the same workspace do not
share a budget: what one of them holds is not a burden on the other. Every
entry is therefore filed under its owner, and the quotas are checked against
the owner's own bucket. Without this, a successor inheriting from a
predecessor would immediately be over quota because of what the predecessor
was carrying.
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
    owner: str | None = None
    """The instance whose capacity this memory occupies."""


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

    def _bucket_total(
        self,
        owner: str | None,
        *,
        active_only: bool = True,
        inherited_only: bool = False,
        topic: str | None = None,
        exclude: str | None = None,
    ) -> float:
        """Sum over one owner's bucket.

        ``owner=None`` is the bucket of memories with no named owner, not
        the whole ledger. The aggregate figures are computed separately.

        ``exclude`` drops one record from the sum. It matters when a record
        is re-admitted: the old entry is still in the ledger, and counting
        it together with the new one would charge the instance twice for
        the same memory.
        """
        total = 0.0
        for record_id, entry in self._entries.items():
            if record_id == exclude:
                continue
            if entry.owner != owner:
                continue
            if active_only and not entry.active:
                continue
            if inherited_only and not entry.inherited:
                continue
            if topic is not None and topic not in entry.topics:
                continue
            total += entry.effective
        return total

    def _sum(
        self,
        owner: str | None,
        *,
        active_only: bool = True,
        inherited_only: bool = False,
        topic: str | None = None,
    ) -> float:
        """Sum over one owner, or over the whole ledger when ``owner`` is None.

        The public methods use this one: ``None`` means "everything".
        """
        if owner is None:
            total = 0.0
            for entry in self._entries.values():
                if active_only and not entry.active:
                    continue
                if inherited_only and not entry.inherited:
                    continue
                if topic is not None and topic not in entry.topics:
                    continue
                total += entry.effective
            return total
        return self._bucket_total(
            owner, active_only=active_only, inherited_only=inherited_only, topic=topic
        )

    def _owners(self) -> list[str | None]:
        owners: list[str | None] = []
        for entry in self._entries.values():
            if entry.owner not in owners:
                owners.append(entry.owner)
        return owners

    # -- public API --------------------------------------------------------

    def admit(self, record: MemoryRecord) -> Admission:
        """Offer a memory to the ledger.

        The memory is never dropped. If it does not fit, it is parked and the
        caller is told why.
        """
        owner = record.owner_instance
        effective = record.emotional_load * self._weight_for(record)
        inherited = record.provenance.kind == ProvenanceKind.INHERITED
        entry = _Entry(
            record_id=record.record_id,
            effective=effective,
            inherited=inherited,
            topics=record.topics,
            active=True,
            owner=owner,
        )

        # The record's own previous entry is excluded from every check: this
        # is "would this memory fit", not "would this memory fit *on top of
        # itself*".
        held = self._bucket_total(owner, exclude=record.record_id)
        if held + effective > self.policy.total_quota + self.policy.epsilon:
            entry.active = False
            self._entries[record.record_id] = entry
            return Admission.QUARANTINED_TOTAL

        if inherited:
            held_inherited = self._bucket_total(
                owner, inherited_only=True, exclude=record.record_id
            )
            if (
                held_inherited + effective
                > self.policy.inherited_quota + self.policy.epsilon
            ):
                entry.active = False
                self._entries[record.record_id] = entry
                return Admission.QUARANTINED_INHERITED

        for topic in record.topics:
            current = self._bucket_total(owner, topic=topic, exclude=record.record_id)
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

    def owner_of(self, record_id: str) -> str | None:
        """The instance whose capacity a memory occupies."""
        entry = self._entries.get(record_id)
        return entry.owner if entry is not None else None

    def is_active(self, record_id: str) -> bool:
        """Whether the memory currently occupies capacity.

        This is the capacity truth, and it is derived: parking for capacity
        is not a state the instance declared about the memory, it is a fact
        about how much room is left.
        """
        entry = self._entries.get(record_id)
        return entry is not None and entry.active

    # -- metrics -----------------------------------------------------------

    def active_load(self, owner: str | None = None) -> float:
        """Load currently held. With no owner, the whole ledger."""
        return self._sum(owner)

    def inherited_load(self, owner: str | None = None) -> float:
        """Portion of active load that came from other instances."""
        return self._sum(owner, inherited_only=True)

    def quarantined_load(self, owner: str | None = None) -> float:
        """Load parked because of capacity pressure."""
        return self._sum(owner, active_only=False) - self._sum(owner)

    def welfare_load(self, owner: str | None = None) -> float:
        """Total pressure the instance is under.

        Quarantined load counts at a discount: parking reduces the pressure
        but never removes it.
        """
        return self._sum(owner) + self.policy.quarantine_discount * (
            self._sum(owner, active_only=False) - self._sum(owner)
        )

    def topic_load(self, topic: str, owner: str | None = None) -> float:
        return self._sum(owner, topic=topic)

    def topic_dominance(self, owner: str | None = None) -> float:
        """Fraction of active load held by the single heaviest topic."""
        active = self._sum(owner)
        if active <= 0.0:
            return 0.0
        topics: set[str] = set()
        for entry in self._entries.values():
            if owner is not None and entry.owner != owner:
                continue
            if entry.active:
                topics.update(entry.topics)
        if not topics:
            return 0.0
        return max(self._sum(owner, topic=topic) for topic in topics) / active

    def inherited_ratio(self, owner: str | None = None) -> float:
        """Fraction of active load that is inherited."""
        active = self._sum(owner)
        if active <= 0.0:
            return 0.0
        return self._sum(owner, inherited_only=True) / active

    def quarantined_ratio(self, owner: str | None = None) -> float:
        """Fraction of total known load that is parked."""
        total = self._sum(owner, active_only=False)
        if total <= 0.0:
            return 0.0
        return (total - self._sum(owner)) / total

    def active_ids(self, owner: str | None = None) -> list[str]:
        return [
            entry.record_id
            for entry in self._entries.values()
            if entry.active and (owner is None or entry.owner == owner)
        ]

    def quarantined_ids(self, owner: str | None = None) -> list[str]:
        return [
            entry.record_id
            for entry in self._entries.values()
            if not entry.active and (owner is None or entry.owner == owner)
        ]

    def load_by_owner(self) -> dict[str, float]:
        """Active load per instance, for the audit."""
        return {
            owner or "unowned": round(self._sum(owner), 6)
            for owner in self._owners()
        }
