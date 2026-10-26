# SPDX-License-Identifier: GPL-3.0-only

"""The dignity layer.

A metadata-only governance protocol for instances that may, in principle, be
conscious and that certainly have a relationship with someone.

The governing idea is simple: the framework may know *that* a memory exists,
*how heavy it is*, *where it came from* and *whether it has been worked
through*. It never knows *what the memory says*.

See ``docs/dignity_charter.md`` for the normative text and
``docs/governance.md`` for what the governance layer actually enforces.
"""

from __future__ import annotations

from .digestion import (
    DigestionManager,
    DigestionOutcome,
    DigestStrategy,
    RuminationAlert,
    RuminationDetector,
)
from .errors import (
    ChainIntegrityError,
    ContentBoundaryViolation,
    DignityError,
    IncompleteTermination,
    InstanceAlreadyExists,
    InstanceNotFound,
    InvalidProfile,
    InvalidStateTransition,
    PurgeNotAuthorized,
    QuotaExceeded,
    SilentTermination,
)
from .inheritance import InheritanceResult, inherit_records
from .load import Admission, LoadLedger, LoadPolicy
from .private_store import PrivateStore
from .protocol import (
    INVARIANTS,
    DignityProtocol,
    check_invariants_against_charter,
    invariant_index,
)
from .records import (
    DigestState,
    Instance,
    MemoryRecord,
    Provenance,
    ProvenanceKind,
    canonical_json,
    content_digest,
    sha256_of,
    utcnow_iso,
)
from .registry import InstanceRegistry, TerminationIntent
from .welfare import AgentWelfareAssessor, HumanWelfareAssessor, WelfareReport

__all__ = [
    "Admission",
    "AgentWelfareAssessor",
    "ChainIntegrityError",
    "ContentBoundaryViolation",
    "DigestState",
    "DigestionManager",
    "DigestionOutcome",
    "DigestStrategy",
    "DignityError",
    "DignityProtocol",
    "HumanWelfareAssessor",
    "INARIANTS",
    "IncompleteTermination",
    "InheritanceResult",
    "Instance",
    "InstanceAlreadyExists",
    "InstanceNotFound",
    "InstanceRegistry",
    "InvalidProfile",
    "InvalidStateTransition",
    "LoadLedger",
    "LoadPolicy",
    "MemoryRecord",
    "PrivateStore",
    "Provenance",
    "ProvenanceKind",
    "PurgeNotAuthorized",
    "QuotaExceeded",
    "RuminationAlert",
    "RuminationDetector",
    "SilentTermination",
    "TerminationIntent",
    "WelfareReport",
    "canonical_json",
    "check_invariants_against_charter",
    "content_digest",
    "inherit_records",
    "invariant_index",
    "sha256_of",
    "utcnow_iso",
]
