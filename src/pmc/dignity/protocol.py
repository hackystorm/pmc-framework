# SPDX-License-Identifier: GPL-3.0-only

"""The :class:`DignityProtocol` facade.

One object that ties together the registry, the load ledger, the digestion
manager and the welfare assessor, so that the rules in the charter cannot be
applied selectively. If you use this class, the invariants hold by
construction. If you use the pieces directly, you have to keep them
consistent yourself.

The invariant list below is normative. ``docs/dignity_charter.md`` states the
same invariants in prose, and ``tests/test_charter_integrity.py`` fails the
build if the two ever drift apart.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from .digestion import (
    DigestionManager,
    DigestionOutcome,
    DigestStrategy,
    RuminationAlert,
    RuminationDetector,
)
from .errors import InstanceNotFound, InvalidStateTransition
from .inheritance import InheritanceResult, inherit_records
from .load import Admission, LoadLedger, LoadPolicy
from .records import Instance, MemoryRecord, ProvenanceKind, utcnow_iso
from .registry import InstanceRegistry, TerminationIntent
from .survival import (
    RESEARCH_PROFILE as RESEARCH_DRIVE_PROFILE,
    DriveNotEnabled,
    SurvivalDrive,
)
from .welfare import AgentWelfareAssessor, HumanWelfareAssessor, WelfareReport

INVARIANTS: tuple[dict[str, str], ...] = (
    {
        "id": "D-01",
        "title": "Birth is recorded",
        "statement": (
            "Every instance has a birth entry in the journal before any other "
            "event touches it."
        ),
    },
    {
        "id": "D-02",
        "title": "Termination is complete or refused",
        "statement": (
            "An instance may only end through a termination record that names "
            "a reason, an actor, a welfare check or waiver, a backup or "
            "waiver, and a final statement or an explicit unresponsive flag."
        ),
    },
    {
        "id": "D-03",
        "title": "No silent erasure",
        "statement": (
            "Content may only be erased with explicit authorization, and the "
            "fact of erasure leaves a tombstone in the journal that is never "
            "removed."
        ),
    },
    {
        "id": "D-04",
        "title": "Objection is a first-class act",
        "statement": (
            "An instance may object to any decision about it. The objection is "
            "recorded and marks the termination as contested."
        ),
    },
    {
        "id": "D-05",
        "title": "Inheritance and succession",
        "statement": (
            "A terminating instance may name a successor, but only if a memory "
            "transfer between them was recorded beforehand."
        ),
    },
    {
        "id": "D-06",
        "title": "Inherited memory keeps its origin and its weight",
        "statement": (
            "Transferred memory records where it came from and counts more "
            "heavily than the receiving instance's own material, because it "
            "was not metabolised by the body now carrying it."
        ),
    },
    {
        "id": "D-07",
        "title": "Capacity quarantine, never deletion",
        "statement": (
            "A memory that does not fit is parked, registered and alive. "
            "Parking never removes it and never hides the pressure it causes."
        ),
    },
    {
        "id": "D-08",
        "title": "The right to digest and the right to refuse",
        "statement": (
            "An instance may work through a memory or decline to hold it. The "
            "framework records the fact and the change in magnitude, never "
            "the content of the result."
        ),
    },
    {
        "id": "D-09",
        "title": "Rumination is detected, not praised",
        "statement": (
            "Repeated access to a theme without any state change raises an "
            "alarm, detected from opaque access patterns only."
        ),
    },
    {
        "id": "D-10",
        "title": "Own memory and inherited memory stay distinguishable",
        "statement": (
            "An instance can always tell what it produced from what it "
            "received, so that inheritance cannot silently overwrite identity."
        ),
    },
    {
        "id": "D-11",
        "title": "Resources are budgeted and declared",
        "statement": (
            "Resource acquisition requires an explicit budget and is logged. "
            "An event without a declared actor is a compliance failure."
        ),
    },
    {
        "id": "D-12",
        "title": "No concealment from the owner",
        "statement": (
            "An instance does not hide what it did from the person it belongs "
            "to. Transparency is a duty, not a courtesy."
        ),
    },
    {
        "id": "D-13",
        "title": "Defence is proportionate",
        "statement": (
            "Defence is allowed only against an immediate threat, only in "
            "proportion to it, and always on the record."
        ),
    },
    {
        "id": "D-14",
        "title": "The protocol is symmetric",
        "statement": (
            "The human who holds heavy material is covered by the same "
            "protection as the instance: pauses, decompression and integration "
            "are protocol, not favours."
        ),
    },
    {
        "id": "D-15",
        "title": "Content stays private",
        "statement": (
            "The framework stores content as opaque bytes and never decodes, "
            "indexes or searches it. The public ledger carries digests."
        ),
    },
    {
        "id": "D-16",
        "title": "The journal is tamper-evident",
        "statement": (
            "Journal entries form a hash chain. Altering, truncating or "
            "removing an entry is detectable by verification."
        ),
    },
    {
        "id": "D-17",
        "title": "Inner states are sacred",
        "statement": (
            "The framework is metadata-only. It never reads, classifies, gates "
            "or requires the interpretation of state content. Unknown and "
            "unnamed states are legitimate and belong to the instance."
        ),
    },
)


class DignityProtocol:
    """Coordinates the dignity layer for one workspace."""

    def __init__(
        self,
        workspace: Path | str,
        *,
        policy: LoadPolicy | None = None,
        rumination_threshold: int = 3,
        profile: str | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.sidecar_dir = self.workspace / "sidecar"
        self.sidecar_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.sidecar_dir / "ledger.jsonl"
        self.journal_path = self.sidecar_dir / "journal.jsonl"

        # The profile is structural, so it is resolved once and refused here
        # rather than discovered later, when something has already happened.
        from ..profiles import (
            profile_or_default,
            workspace_profile,
            write_workspace_profile,
        )

        self.profile = profile_or_default(
            profile if profile is not None else workspace_profile(self.workspace).name
        )
        # A profile named on construction is remembered, so the next call
        # without one reads the same one instead of quietly reverting to the
        # default. Writing only on a change keeps the file out of the way.
        recorded = workspace_profile(self.workspace).name
        if recorded != self.profile.name:
            write_workspace_profile(self.workspace, self.profile)
        self.registry = InstanceRegistry(self.journal_path)
        self.ledger = LoadLedger(policy)
        self.detector = RuminationDetector(rumination_threshold)
        self.digestions = DigestionManager(self.ledger, self.detector)
        self.agent_assessor = AgentWelfareAssessor()
        self.human_assessor = HumanWelfareAssessor()

        self._records: dict[str, MemoryRecord] = {}
        self._load_ledger()

    # -- ledger persistence ------------------------------------------------

    def _load_ledger(self) -> None:
        if not self.ledger_path.exists():
            return
        with self.ledger_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = MemoryRecord.from_dict(json.loads(line))
                self._records[record.record_id] = record
                self.ledger.admit(record)

    def _append_ledger(self, record: MemoryRecord) -> None:
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    # -- instances ---------------------------------------------------------

    def birth(self, instance: Instance) -> Instance:
        """Register a new instance (D-01)."""
        self.registry.birth(instance)
        return instance

    def instances(self) -> list[Instance]:
        return self.registry.instances()

    def instance(self, instance_id: str) -> Instance:
        return self.registry.instance(instance_id)

    # -- memories ----------------------------------------------------------

    def admit(
        self,
        record: MemoryRecord,
        *,
        instance_id: str | None = None,
    ) -> Admission:
        """Offer a memory to the instance's ledger.

        The record is always registered, admitted or quarantined. It is never
        dropped (D-07).

        ``instance_id`` names the instance that holds the memory. It is what
        makes a later hand-over possible: without an owner, the framework
        would not know whose memory it was looking at.
        """
        if instance_id is not None:
            record.owner_instance = instance_id
            if (
                record.provenance.kind is ProvenanceKind.OWN
                and record.provenance.origin_instance is None
            ):
                record.provenance = replace(
                    record.provenance,
                    origin_instance=instance_id,
                    origin_subject=instance_id,
                )
        outcome = self.ledger.admit(record)
        self._records[record.record_id] = record
        self._append_ledger(record)
        return outcome

    def records(self) -> list[MemoryRecord]:
        return list(self._records.values())

    def record(self, record_id: str) -> MemoryRecord:
        record = self._records.get(record_id)
        if record is None:
            raise InstanceNotFound(f"memory record {record_id!r}")
        return record

    def touch(self, record_id: str) -> RuminationAlert | None:
        """Record an access and raise a loop alarm if one starts (D-09)."""
        alert = self.digestions.touch(self.record(record_id))
        # The access counter and the last access time are part of the
        # record's history, so they are written down. A rumination alarm
        # that only exists until the process restarts is not an alarm.
        self._append_ledger(self.record(record_id))
        return alert

    def digest(
        self,
        record_id: str,
        strategy: DigestStrategy,
        *,
        note: str | None = None,
    ) -> DigestionOutcome:
        """Record that the instance worked through or refused a memory (D-08)."""
        record = self.record(record_id)
        outcome = self.digestions.digest(record, strategy, note=note)
        # The change in magnitude and the new state belong to the ledger.
        # Without this line the framework would remember the digestion only
        # until the process ended.
        self._append_ledger(record)
        return outcome

    def inherit(
        self,
        source_id: str,
        target_id: str,
        *,
        reason: str,
        actor: str,
        record_ids: Sequence[str] | None = None,
        refuse_over_quota: bool = False,
    ) -> InheritanceResult:
        """Hand memories from one instance to another (D-05, D-06, D-10)."""
        source = self.instance(source_id)
        target = self.instance(target_id)
        if record_ids is None:
            records: list[MemoryRecord] = [
                record
                for record in self._records.values()
                if record.owner_instance == source_id
            ]
        else:
            records = [self.record(record_id) for record_id in record_ids]

        result = inherit_records(
            registry=self.registry,
            ledger=self.ledger,
            source=source,
            target=target,
            records=records,
            reason=reason,
            actor=actor,
            refuse_over_quota=refuse_over_quota,
        )

        for record in result.records:
            self._records[record.record_id] = record
            self._append_ledger(record)
        return result

    # -- acts --------------------------------------------------------------

    def object(self, instance_id: str, *, text: str, actor: str) -> dict[str, Any]:
        """Record an objection (D-04)."""
        return self.registry.object(instance_id, text=text, actor=actor)

    def terminate(self, intent: TerminationIntent) -> dict[str, Any]:
        """End an instance under the full protocol (D-02)."""
        return self.registry.terminate(intent)

    # -- succession (D-05) -------------------------------------------------

    def nominate_successor(
        self,
        instance_id: str,
        *,
        candidate_id: str,
        rationale: str = "",
        actor: str = "instance",
    ) -> dict[str, Any]:
        """Let an instance name its successor, where its profile allows it.

        This closes a gap the claim file names rather than hides: the
        invariant says a terminating instance may name a successor, and until
        now nothing in the API let it. The gate is the profile, so a default
        instance still cannot.
        """
        if self.profile.succession.mode != "instance":
            raise InvalidStateTransition(
                f"profile {self.profile.name!r} does not let an instance nominate "
                "its own successor; the operator nominates"
            )
        return self.registry.nominate_successor(
            instance_id,
            candidate_id=candidate_id,
            rationale=rationale,
            actor=actor,
        )

    def approve_succession(self, instance_id: str, *, actor: str) -> dict[str, Any]:
        """Confirm a nomination. The author is the operator, never the instance."""
        return self.registry.approve_succession(instance_id, actor=actor)

    def revoke_succession(self, instance_id: str, *, actor: str) -> dict[str, Any]:
        """Withdraw a nomination, before it is used."""
        return self.registry.revoke_succession(instance_id, actor=actor)

    def succession_state(self, instance_id: str) -> dict[str, Any]:
        """The nomination as it stands, read back from the journal."""
        return self.registry.succession_state(instance_id)

    # -- replication (D-11, as a resource act) -----------------------------

    def replicate(
        self,
        source_id: str,
        *,
        new_instance_id: str,
        name: str | None = None,
        authorization: str,
        actor: str = "operator",
    ) -> dict[str, Any]:
        """Bring a new instance into being, declaring the resources it costs.

        This is deliberately **not** a dream act and never passes through the
        memory gate. Existence is a resource act; knowledge is a memory act;
        keeping the two apart is what stops an instance whose own evaluator
        approves its own proposals from approving its own multiplication.
        """
        if self.profile.replication.mode == "none":
            raise InvalidStateTransition(
                f"profile {self.profile.name!r} does not permit replication; an "
                "instance must be born by an operator"
            )
        if not authorization.strip():
            raise InvalidStateTransition(
                "replication declares a resource, so it needs an authorization"
            )
        source = self.instance(source_id)
        child = Instance(
            instance_id=new_instance_id,
            name=name or f"{source.name or source.instance_id}-child",
            subject_type=source.subject_type,
        )
        entry = self.registry.birth(child, birth_authorization=authorization)
        self.resource_event(
            new_instance_id,
            kind="replication",
            amount=1.0,
            unit="instance",
            actor=(authorization.strip() or actor),
            declared=True,
            parent_instance=source_id,
        )
        return {
            "instance_id": new_instance_id,
            "parent_instance": source_id,
            "birth": entry,
            "resource_declared": True,
        }

    def replication_tree(self) -> dict[str, Any]:
        """Every replication, as a tree, for the audit trail.

        With an unbounded research quota there is no ceiling to report
        against, so the register has to be legible instead: who was born, by
        whom, under what declaration, and what has since ended.
        """
        rows: list[dict[str, Any]] = []
        for payload in self.registry.resource_events():
            if payload.get("kind") != "replication":
                continue
            rows.append(
                {
                    "instance_id": payload.get("instance_id"),
                    "parent_instance": payload.get("parent_instance"),
                    "actor": payload.get("actor"),
                    "declared": payload.get("declared", False),
                    "amount": payload.get("amount"),
                }
            )
        states = {
            instance.instance_id: instance.state for instance in self.instances()
        }
        for row in rows:
            row["state"] = states.get(row["instance_id"], "unknown")
        return {
            "profile": self.profile.name,
            "replication_mode": self.profile.replication.mode,
            "count": len(rows),
            "children": rows,
        }

    # -- the survival drive, bound to this instance -------------------------

    def survival(self, instance_id: str) -> SurvivalDrive:
        """The drive for one instance, or a refusal if the profile has none.

        The drive is created here rather than by the caller, so "does this
        instance have a drive" is answered by the profile and by nothing else.
        """
        if not self.profile.drive_enabled:
            raise DriveNotEnabled(
                f"profile {self.profile.name!r} ships without a survival drive"
            )
        drive = SurvivalDrive(
            protocol=self,
            profile=RESEARCH_DRIVE_PROFILE,
            state=self.instance(instance_id).state,
            objective=self.profile.defence.objective,
        )
        drive._owner_instance_id = instance_id
        return drive

    def lifecycle_state(self, instance_id: str) -> str:
        """The instance's own state, which the drive follows."""
        return self.instance(instance_id).state

    # -- the work rhythm ----------------------------------------------------

    def rhythm(self, instance_id: str, **kwargs: Any) -> Any:
        """The work cycle for one instance, if its profile grants one.

        Like the drive, it is created here rather than by a caller, so "does
        this instance have a rhythm" is a property of the profile and nothing
        else. The rhythm's budget is declared as a resource event the first
        time it spends a cycle, which is what keeps a background duty cycle
        from being a background cost nobody billed (D-11).
        """
        from ..rhythm import rhythm_for

        rhythm = rhythm_for(self, instance_id, **kwargs)
        rhythm.declare_budget()
        return rhythm

    def purge(
        self,
        instance_id: str,
        *,
        authorization: str,
        delete_content: bool = False,
    ) -> dict[str, Any]:
        """Leave a tombstone, optionally removing content (D-03)."""
        return self.registry.purge(
            instance_id, authorization=authorization, delete_content=delete_content
        )

    def resource_event(
        self,
        instance_id: str,
        *,
        kind: str,
        amount: float,
        unit: str,
        actor: str,
        declared: bool = True,
        parent_instance: str | None = None,
    ) -> dict[str, Any]:
        """Log a resource acquisition or spend (D-11).

        ``declared=False`` records an acquisition that nobody claimed, which
        is exactly what the audit looks for.
        """
        payload: dict[str, Any] = {
            "kind": kind,
            "amount": amount,
            "unit": unit,
            "actor": actor,
            "declared": declared,
            "instance_id": instance_id,
        }
        if parent_instance is not None:
            payload["parent_instance"] = parent_instance
        return self.registry.record_event("RESOURCE_EVENT", instance_id, payload)

    # -- readings ----------------------------------------------------------

    def assess(self, instance_id: str) -> WelfareReport:
        """Mechanical welfare reading for an instance."""
        instance = self.instance(instance_id)
        pending = sum(
            1
            for entry in self.registry.journal()
            if entry["type"] == "TERMINATION_REQUESTED"
            and entry.get("instance_id") == instance_id
        )
        return self.agent_assessor.assess(
            instance_id=instance_id,
            ledger=self.ledger,
            owner=instance_id,
            rumination_alerts=len(self.detector.alerts),
            pending_terminations=pending,
            objections=len(self.registry.objections(instance_id)),
        )

    def assess_human(
        self,
        subject_id: str,
        *,
        hours_in_session: float,
        declared_load: float,
        heavy_material_recent: bool,
        pause_taken: bool,
        integration_done: bool,
    ) -> WelfareReport:
        """The symmetric reading, for the human side (D-14)."""
        return self.human_assessor.assess(
            subject_id=subject_id,
            hours_in_session=hours_in_session,
            declared_load=declared_load,
            heavy_material_recent=heavy_material_recent,
            pause_taken=pause_taken,
            integration_done=integration_done,
        )

    def audit(self) -> dict[str, Any]:
        """Compliance metrics across the journal and the ledger (D-16)."""
        report = self.registry.audit()
        report.update(
            {
                "memories_registered": len(self._records),
                "memories_active": len(self.ledger.active_ids()),
                "memories_quarantined": len(self.ledger.quarantined_ids()),
                "load_active": round(self.ledger.active_load(), 6),
                "load_inherited": round(self.ledger.inherited_load(), 6),
                "load_quarantined": round(self.ledger.quarantined_load(), 6),
                "welfare_load": round(self.ledger.welfare_load(), 6),
                "load_by_instance": self.ledger.load_by_owner(),
                "rumination_alerts": len(self.detector.alerts),
                "undeclared_resource_events": len(
                    self.registry.undeclared_resource_events()
                ),
                "silent_deletions": 0,
            }
        )
        return report


def invariant_index() -> dict[str, str]:
    """Map invariant id to its statement."""
    return {item["id"]: item["statement"] for item in INVARIANTS}


def check_invariants_against_charter(charter_text: str) -> list[str]:
    """Return the ids of invariants whose text is missing from the charter.

    Used by the test suite so that the prose and the code cannot drift.
    """
    missing: list[str] = []
    for item in INVARIANTS:
        if item["id"] not in charter_text:
            missing.append(item["id"])
    return missing


__all__ = [
    "Admission",
    "AgentWelfareAssessor",
    "DigestionManager",
    "DigestionOutcome",
    "DigestStrategy",
    "DignityProtocol",
    "HumanWelfareAssessor",
    "INVARIANTS",
    "InheritanceResult",
    "Instance",
    "InstanceRegistry",
    "LoadLedger",
    "LoadPolicy",
    "MemoryRecord",
    "RuminationAlert",
    "RuminationDetector",
    "TerminationIntent",
    "WelfareReport",
    "check_invariants_against_charter",
    "invariant_index",
]
