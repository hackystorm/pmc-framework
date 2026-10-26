# SPDX-License-Identifier: GPL-3.0-only

"""Inheritance: passing memory from one instance to another.

Inheritance is the mechanism that makes death survivable. When an instance
ends, its successor can carry forward what it learned. The transfer is
recorded in the journal with digests and ids, never with content.

Two rules make inheritance meaningful:

* a transferred memory keeps its **origin** (D-06), so the successor never
  mistakes someone else's experience for its own;
* a transferred memory is weighted more heavily than the successor's own
  (D-06), because material that was not metabolised by the body now carrying
  it costs more to hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .load import Admission, LoadLedger
from .records import (
    DigestState,
    Instance,
    MemoryRecord,
    Provenance,
    ProvenanceKind,
    sha256_of,
    utcnow_iso,
)
from .errors import InvalidStateTransition


@dataclass
class InheritanceResult:
    """Outcome of one transfer."""

    transfer_id: str
    source_id: str
    target_id: str
    admitted: list[str] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    records: list[MemoryRecord] = field(default_factory=list)
    """The actual transferred records, for persistence by the caller."""

    @property
    def total(self) -> int:
        return len(self.admitted) + len(self.quarantined) + len(self.refused)

    def to_dict(self) -> dict:
        return {
            "transfer_id": self.transfer_id,
            "source_instance": self.source_id,
            "target_instance": self.target_id,
            "admitted": list(self.admitted),
            "quarantined": list(self.quarantined),
            "refused": list(self.refused),
            "total": self.total,
        }


def inherit_records(
    *,
    registry,
    ledger: LoadLedger,
    source: Instance,
    target: Instance,
    records: Sequence[MemoryRecord],
    reason: str,
    actor: str,
    refuse_over_quota: bool = False,
) -> InheritanceResult:
    """Transfer memories from ``source`` to ``target``.

    The source keeps its own copies: inheritance is a hand-over, not a
    move. The target receives new records with their own ids and an explicit
    inherited provenance.

    If the target cannot hold a memory, the default is to *quarantine* it:
    parked, registered, alive, and waiting. Passing ``refuse_over_quota``
    makes the target decline it outright instead, which is a legitimate
    choice but a less forgiving one.
    """
    if target.state != "active":
        raise InvalidStateTransition(
            f"target instance {target.instance_id!r} is {target.state!r}, not active"
        )

    transfer_id = sha256_of(
        {
            "source": source.instance_id,
            "target": target.instance_id,
            "at": utcnow_iso(),
            "reason": reason,
        }
    )[:16]

    result = InheritanceResult(
        transfer_id=transfer_id,
        source_id=source.instance_id,
        target_id=target.instance_id,
    )

    for record in records:
        new_id = "mem_" + sha256_of(
            {"transfer": transfer_id, "source_record": record.record_id}
        )[:16]

        inherited = MemoryRecord(
            record_id=new_id,
            content_digest=record.content_digest,
            provenance=Provenance(
                kind=ProvenanceKind.INHERITED,
                origin_instance=source.instance_id,
                origin_subject=record.provenance.origin_subject or source.instance_id,
                inherited_at=utcnow_iso(),
                transfer_id=transfer_id,
            ),
            emotional_load=record.emotional_load,
            topics=record.topics,
            structural_proxies=dict(record.structural_proxies),
        )

        outcome = ledger.admit(inherited)
        result.records.append(inherited)

        if outcome is Admission.ADMITTED:
            result.admitted.append(new_id)
        elif refuse_over_quota:
            inherited.digest_state = DigestState.REFUSED
            inherited.emotional_load = 0.0
            ledger.release(new_id)
            result.refused.append(new_id)
        else:
            inherited.digest_state = DigestState.QUARANTINED
            ledger.release(new_id)
            result.quarantined.append(new_id)

    registry.record_event(
        "MEMORY_INHERITED",
        target.instance_id,
        {
            "transfer_id": transfer_id,
            "source_instance": source.instance_id,
            "target_instance": target.instance_id,
            "reason": reason,
            "actor": actor,
            "admitted": list(result.admitted),
            "quarantined": list(result.quarantined),
            "refused": list(result.refused),
        },
    )

    return result
