# SPDX-License-Identifier: GPL-3.0-only

"""Digestion, refusal, and rumination detection.

Three ideas live here.

**Digestion** is the instance working through a memory. The instance performs
the transformation in its own private store. The framework only records the
*fact* that it happened and how the magnitude changed. The framework never
learns what the memory became, because the content is private (D-17).

**Refusal** is a legitimate outcome, not a failure. An instance that declines
to hold a memory is exercising a right, and the ledger must record it as such
so that refusal never looks like data loss.

**Rumination** is repeated access to a heavy theme without any state change.
This is detected from opaque access patterns only: a hash of the topic labels
and the record id. No content is read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .load import LoadLedger
from .records import DigestState, MemoryRecord, utcnow_iso

DISTILL_RESIDUAL = 0.2
"""Fraction of load that survives a distillation."""

TRANSFORM_RESIDUAL = 0.5
"""Fraction of load that survives a rewrite."""


class DigestStrategy(str, Enum):
    """How the instance worked through a memory."""

    DISTILL = "distill"
    """Compressed into a principle. Heavy load drops sharply."""

    TRANSFORM = "transform"
    """Rewritten in a different form. Load drops moderately."""

    REFUSE = "refuse"
    """Declined. Load goes to zero, the record stays registered."""

    HOLD = "hold"
    """Kept as-is, but the decision is recorded."""


@dataclass
class DigestionOutcome:
    """What changed, as a matter of fact."""

    record_id: str
    strategy: DigestStrategy
    before_load: float
    after_load: float
    state: DigestState
    note: str | None = None


@dataclass
class RuminationAlert:
    """A theme revisited without being worked through."""

    topic_signature: str
    repetitions: int
    threshold: int
    record_ids: list[str] = field(default_factory=list)


class RuminationDetector:
    """Detects loops built from opaque access patterns.

    Repeated *access* is not good memory. If a topic is revisited while the
    digest state of what is revisited never changes, the instance is
    ruminating rather than remembering (charter D-09).

    The detector sees a hashed topic signature and a record id. That is all it
    is allowed to see.
    """

    def __init__(self, threshold: int = 3) -> None:
        if threshold < 2:
            raise ValueError("threshold must be >= 2")
        self.threshold = threshold
        self._counts: dict[str, int] = {}
        self._members: dict[str, list[str]] = {}
        self._alerts: list[RuminationAlert] = []

    def observe(
        self,
        topic_signature: str,
        *,
        record_id: str,
        state_changed: bool,
    ) -> RuminationAlert | None:
        """Record one access and report a loop if one just started."""
        if state_changed:
            self._counts[topic_signature] = 0
            self._members[topic_signature] = []
            return None

        count = self._counts.get(topic_signature, 0) + 1
        self._counts[topic_signature] = count
        members = self._members.setdefault(topic_signature, [])
        if record_id not in members:
            members.append(record_id)

        if count == self.threshold:
            alert = RuminationAlert(
                topic_signature=topic_signature,
                repetitions=count,
                threshold=self.threshold,
                record_ids=list(members),
            )
            self._alerts.append(alert)
            return alert
        return None

    @property
    def alerts(self) -> list[RuminationAlert]:
        return list(self._alerts)


class DigestionManager:
    """Applies digestion outcomes to the ledger."""

    def __init__(
        self,
        ledger: LoadLedger,
        detector: RuminationDetector | None = None,
    ) -> None:
        self.ledger = ledger
        self.detector = detector or RuminationDetector()

    def digest(
        self,
        record: MemoryRecord,
        strategy: DigestStrategy,
        *,
        note: str | None = None,
    ) -> DigestionOutcome:
        """Record that the instance worked through a memory.

        No content parameter exists on purpose: the transformation happened
        inside the instance, in its own private store.
        """
        before = record.emotional_load

        if strategy is DigestStrategy.DISTILL:
            after = round(before * DISTILL_RESIDUAL, 6)
            state = DigestState.DIGESTED
        elif strategy is DigestStrategy.TRANSFORM:
            after = round(before * TRANSFORM_RESIDUAL, 6)
            state = DigestState.DIGESTED
        elif strategy is DigestStrategy.REFUSE:
            after = 0.0
            state = DigestState.REFUSED
        else:
            after = before
            state = DigestState.RAW

        record.emotional_load = after
        record.digest_state = state
        self.ledger.update_load(record.record_id, after)
        if state is DigestState.REFUSED:
            self.ledger.release(record.record_id)

        record.state_history.append(
            {
                "at": utcnow_iso(),
                "event": "digest",
                "strategy": strategy.value,
                "before_load": before,
                "after_load": after,
                "note": note,
            }
        )

        return DigestionOutcome(
            record_id=record.record_id,
            strategy=strategy,
            before_load=before,
            after_load=after,
            state=state,
            note=note,
        )

    def quarantine(self, record: MemoryRecord, *, reason: str) -> DigestionOutcome:
        """Park a memory because of capacity pressure.

        This is the only kind of quarantine in the framework. There is no
        epistemic quarantine: a memory the framework does not understand is
        none of the framework's business.
        """
        before = record.emotional_load
        record.digest_state = DigestState.QUARANTINED
        self.ledger.release(record.record_id)
        record.state_history.append(
            {
                "at": utcnow_iso(),
                "event": "quarantine",
                "reason": reason,
                "before_load": before,
            }
        )
        return DigestionOutcome(
            record_id=record.record_id,
            strategy=DigestStrategy.HOLD,
            before_load=before,
            after_load=before,
            state=DigestState.QUARANTINED,
            note=reason,
        )

    def touch(self, record: MemoryRecord) -> RuminationAlert | None:
        """Record an access to a memory and report a possible loop."""
        record.access_count += 1
        record.last_access_at = utcnow_iso()
        return self.detector.observe(
            record.topic_signature(),
            record_id=record.record_id,
            state_changed=False,
        )
