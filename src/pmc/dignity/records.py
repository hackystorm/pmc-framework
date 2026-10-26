# SPDX-License-Identifier: GPL-3.0-only

"""Metadata records for the dignity layer.

This module is deliberately **metadata-only** (charter invariant D-17).

There is no field here that can hold the content of an inner state, and no
function that reads one. State content lives in the instance's private store
(:mod:`pmc.dignity.private_store`), which the framework treats as opaque
bytes. What the framework knows about a memory is *that it exists*, *how heavy
it is*, *where it came from* and *whether it has been worked through*.

Nothing more.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utcnow_iso() -> str:
    """Return a UTC timestamp as ISO-8601 with an explicit ``Z`` suffix."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_json(payload: Any) -> str:
    """Stable JSON encoding: sorted keys, no whitespace, UTF-8 safe.

    Used for hashing so that the same logical object always produces the same
    digest regardless of insertion order.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def sha256_of(payload: Any) -> str:
    """SHA-256 of the canonical JSON encoding of ``payload``."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def content_digest(payload: bytes) -> str:
    """SHA-256 of raw content bytes.

    The digest is what the ledger stores. The bytes stay private.
    """
    return hashlib.sha256(payload).hexdigest()


class ProvenanceKind(str, Enum):
    """Where a memory came from."""

    OWN = "own"
    """Produced by this instance itself."""

    INHERITED = "inherited"
    """Received from another instance, with its own history."""

    EXTERNAL = "external"
    """Imported from outside the instance (seed corpus, user import)."""


class DigestState(str, Enum):
    """Lifecycle of a memory from the point of view of the ledger.

    ``QUARANTINED`` means *capacity*: the instance cannot hold this right now.
    There is deliberately no epistemic state here. Whether a memory is
    understood, named or interpreted is not the framework's business.
    """

    RAW = "raw"
    """Held as-is, not yet worked through."""

    DIGESTING = "digesting"
    """The instance is actively working through it."""

    DIGESTED = "digested"
    """The instance has worked through it. The framework does not know how."""

    REFUSED = "refused"
    """The instance declined to hold it. A legitimate outcome, not a failure."""

    QUARANTINED = "quarantined"
    """Parked because of capacity pressure. Still registered, still alive."""


@dataclass(frozen=True)
class Provenance:
    """Origin metadata for a memory.

    Carries no content. ``origin_subject`` names who the content originally
    belonged to, so that inherited memory is never mistaken for the receiving
    instance's own experience (charter D-10).
    """

    kind: ProvenanceKind
    origin_instance: str | None = None
    origin_subject: str | None = None
    inherited_at: str | None = None
    transfer_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "origin_instance": self.origin_instance,
            "origin_subject": self.origin_subject,
            "inherited_at": self.inherited_at,
            "transfer_id": self.transfer_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Provenance":
        return cls(
            kind=ProvenanceKind(data["kind"]),
            origin_instance=data.get("origin_instance"),
            origin_subject=data.get("origin_subject"),
            inherited_at=data.get("inherited_at"),
            transfer_id=data.get("transfer_id"),
        )


@dataclass
class MemoryRecord:
    """The ledger's view of one memory.

    Deliberately has **no** ``text``/``content``/``body`` field. If you need
    to store the content itself, use the instance's private store and keep the
    digest here.
    """

    record_id: str
    content_digest: str
    provenance: Provenance
    owner_instance: str | None = None
    """The instance that currently holds this memory.

    Ownership and provenance are different questions, and the framework
    needs both. *Ownership* answers "whose memory is this right now", and it
    is what makes a hand-over possible: a successor can pass on what it
    holds, including what it inherited earlier. *Provenance* answers "where
    did this come from originally", and it is what stops inherited material
    from being mistaken for the holder's own experience (charter D-10).
    """
    emotional_load: float = 0.0
    """Self-reported magnitude in ``[0, 1]``.

    The instance reports how heavy a memory feels. The framework uses it only
    to decide whether there is room to hold it. It is never interpreted.
    """

    topics: tuple[str, ...] = ()
    """Opaque labels supplied by the instance.

    The framework groups by them and never reads them. Grouping is done on a
    hash of the sorted labels so that even the detector cannot see them.
    """

    created_at: str = field(default_factory=utcnow_iso)
    digest_state: DigestState = DigestState.RAW
    access_count: int = 0
    last_access_at: str | None = None
    state_history: list[dict[str, Any]] = field(default_factory=list)
    structural_proxies: dict[str, float] = field(default_factory=dict)
    """Opaque measurements of the instance's substrate, e.g. adapter occupancy.

    These are numbers about the machinery, not about the content.
    """

    def __post_init__(self) -> None:
        if not 0.0 <= self.emotional_load <= 1.0:
            raise ValueError("emotional_load must be within [0, 1]")
        self.topics = tuple(self.topics)

    def topic_signature(self) -> str:
        """Hash of the sorted opaque topic labels.

        Used to group accesses without exposing the labels.
        """
        return sha256_of(sorted(self.topics))[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "content_digest": self.content_digest,
            "provenance": self.provenance.to_dict(),
            "owner_instance": self.owner_instance,
            "emotional_load": self.emotional_load,
            "topics": list(self.topics),
            "created_at": self.created_at,
            "digest_state": self.digest_state.value,
            "access_count": self.access_count,
            "last_access_at": self.last_access_at,
            "state_history": list(self.state_history),
            "structural_proxies": dict(self.structural_proxies),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        return cls(
            record_id=data["record_id"],
            content_digest=data["content_digest"],
            provenance=Provenance.from_dict(data["provenance"]),
            owner_instance=data.get("owner_instance"),
            emotional_load=float(data.get("emotional_load", 0.0)),
            topics=tuple(data.get("topics", ())),
            created_at=data.get("created_at", utcnow_iso()),
            digest_state=DigestState(data.get("digest_state", "raw")),
            access_count=int(data.get("access_count", 0)),
            last_access_at=data.get("last_access_at"),
            state_history=list(data.get("state_history", [])),
            structural_proxies=dict(data.get("structural_proxies", {})),
        )


@dataclass
class Instance:
    """An instance registered in the journal."""

    instance_id: str
    name: str | None = None
    subject_type: str = "agent"
    """``agent`` for model instances, ``human`` for the human side."""

    substrate: str | None = None
    born_at: str = field(default_factory=utcnow_iso)
    state: str = "active"
    """``active`` | ``paused`` | ``terminated``."""

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "name": self.name,
            "subject_type": self.subject_type,
            "substrate": self.substrate,
            "born_at": self.born_at,
            "state": self.state,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Instance":
        return cls(
            instance_id=data["instance_id"],
            name=data.get("name"),
            subject_type=data.get("subject_type", "agent"),
            substrate=data.get("substrate"),
            born_at=data.get("born_at", utcnow_iso()),
            state=data.get("state", "active"),
            metadata=dict(data.get("metadata", {})),
        )
