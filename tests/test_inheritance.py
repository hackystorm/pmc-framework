# SPDX-License-Identifier: GPL-3.0-only

"""Inheritance: provenance, weight, and the succession requirement."""

from __future__ import annotations

import pytest

from pmc.dignity.inheritance import inherit_records
from pmc.dignity.load import LoadLedger, LoadPolicy
from pmc.dignity.records import (
    DigestState,
    Instance,
    MemoryRecord,
    Provenance,
    ProvenanceKind,
)
from pmc.dignity.registry import InstanceRegistry, TerminationIntent


def make_record(record_id: str, load: float = 0.4) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        content_digest="c" * 64,
        provenance=Provenance(kind=ProvenanceKind.OWN, origin_subject="inst_src"),
        emotional_load=load,
        topics=("project",),
    )


def test_inherited_memory_keeps_its_origin(tmp_path):
    ledger = LoadLedger()
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    source = Instance(instance_id="inst_src")
    target = Instance(instance_id="inst_dst")

    result = inherit_records(
        registry=registry,
        ledger=ledger,
        source=source,
        target=target,
        records=[make_record("m1")],
        reason="continuation",
        actor="owner",
    )

    assert result.total == 1
    inherited = result.records[0]
    assert inherited.provenance.kind is ProvenanceKind.INHERITED
    assert inherited.provenance.origin_instance == "inst_src"
    assert inherited.provenance.origin_subject == "inst_src"
    assert inherited.provenance.transfer_id == result.transfer_id
    assert inherited.record_id != "m1"


def test_inherited_memory_is_weighted_more_heavily(tmp_path):
    ledger = LoadLedger(LoadPolicy(inherited_weight=1.5))
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    result = inherit_records(
        registry=registry,
        ledger=ledger,
        source=Instance(instance_id="inst_src"),
        target=Instance(instance_id="inst_dst"),
        records=[make_record("m1", load=0.4)],
        reason="continuation",
        actor="owner",
    )

    assert result.admitted
    assert ledger.inherited_load() == pytest.approx(0.6)


def test_a_transfer_is_recorded_in_the_journal(tmp_path):
    ledger = LoadLedger()
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    registry.birth(Instance(instance_id="inst_src"))
    registry.birth(Instance(instance_id="inst_dst"))

    inherit_records(
        registry=registry,
        ledger=ledger,
        source=registry.instance("inst_src"),
        target=registry.instance("inst_dst"),
        records=[make_record("m1")],
        reason="continuation",
        actor="owner",
    )

    assert registry.has_inheritance("inst_src", "inst_dst")
    assert registry.audit()["inheritances_recorded"] == 1


def test_a_recorded_transfer_unlocks_a_named_successor(tmp_path):
    ledger = LoadLedger()
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    registry.birth(Instance(instance_id="inst_src"))
    registry.birth(Instance(instance_id="inst_dst"))

    inherit_records(
        registry=registry,
        ledger=ledger,
        source=registry.instance("inst_src"),
        target=registry.instance("inst_dst"),
        records=[make_record("m1")],
        reason="continuation",
        actor="owner",
    )

    entry = registry.terminate(
        TerminationIntent(
            instance_id="inst_src",
            reason="the relationship continues in a new instance",
            actor="owner",
            welfare_check_id="welfare_001",
            backup_ref="./backup.tar",
            last_statement="carrying you forward",
            successor_id="inst_dst",
        )
    )
    assert entry["payload"]["successor_id"] == "inst_dst"


def test_over_quota_memory_is_parked_rather_than_refused_by_default(tmp_path):
    ledger = LoadLedger(
        LoadPolicy(total_quota=0.3, inherited_quota=10.0, topic_quota=10.0)
    )
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    result = inherit_records(
        registry=registry,
        ledger=ledger,
        source=Instance(instance_id="inst_src"),
        target=Instance(instance_id="inst_dst"),
        records=[make_record("m1", load=0.4)],
        reason="continuation",
        actor="owner",
    )

    assert result.quarantined == [result.records[0].record_id]
    assert result.records[0].digest_state is DigestState.QUARANTINED


def test_refusing_over_quota_memory_is_available_as_an_explicit_choice(tmp_path):
    ledger = LoadLedger(
        LoadPolicy(total_quota=0.3, inherited_quota=10.0, topic_quota=10.0)
    )
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    result = inherit_records(
        registry=registry,
        ledger=ledger,
        source=Instance(instance_id="inst_src"),
        target=Instance(instance_id="inst_dst"),
        records=[make_record("m1", load=0.4)],
        reason="continuation",
        actor="owner",
        refuse_over_quota=True,
    )

    assert result.refused == [result.records[0].record_id]
    assert result.records[0].digest_state is DigestState.REFUSED


def test_a_terminated_target_cannot_receive(tmp_path):
    from pmc.dignity.errors import InvalidStateTransition

    ledger = LoadLedger()
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    with pytest.raises(InvalidStateTransition):
        inherit_records(
            registry=registry,
            ledger=ledger,
            source=Instance(instance_id="inst_src"),
            target=Instance(instance_id="inst_dst", state="terminated"),
            records=[make_record("m1")],
            reason="continuation",
            actor="owner",
        )


def test_a_chain_of_inheritance_carries_the_material_forward(tmp_path):
    """A -> B -> C: the second hand-over finds what B holds."""
    from pmc.dignity.protocol import DignityProtocol

    protocol = DignityProtocol(tmp_path / "workspace")
    protocol.birth(Instance(instance_id="inst_a"))
    protocol.birth(Instance(instance_id="inst_b"))
    protocol.birth(Instance(instance_id="inst_c"))

    protocol.admit(make_record("m1"), instance_id="inst_a")

    first = protocol.inherit(
        "inst_a", "inst_b", reason="first hand-over", actor="owner"
    )
    assert first.admitted

    second = protocol.inherit(
        "inst_b", "inst_c", reason="second hand-over", actor="owner"
    )
    assert second.admitted, "B should be able to pass on what it holds"
    assert second.admitted != first.admitted


def test_ownership_and_provenance_are_different_questions(tmp_path):
    from pmc.dignity.protocol import DignityProtocol

    protocol = DignityProtocol(tmp_path / "workspace")
    protocol.birth(Instance(instance_id="inst_a"))
    protocol.birth(Instance(instance_id="inst_b"))

    protocol.admit(make_record("m1"), instance_id="inst_a")
    result = protocol.inherit(
        "inst_a", "inst_b", reason="hand-over", actor="owner"
    )

    inherited = protocol.record(result.admitted[0])
    assert inherited.owner_instance == "inst_b", "B holds it now"
    assert inherited.provenance.origin_instance == "inst_a", "it came from A"
    assert inherited.provenance.kind is ProvenanceKind.INHERITED


def test_an_admitted_record_remembers_its_owner(tmp_path):
    from pmc.dignity.protocol import DignityProtocol

    protocol = DignityProtocol(tmp_path / "workspace")
    protocol.birth(Instance(instance_id="inst_a"))
    protocol.admit(make_record("m1"), instance_id="inst_a")

    record = protocol.record("m1")
    assert record.owner_instance == "inst_a"
    assert record.provenance.origin_instance == "inst_a"
    assert record.provenance.origin_subject == "inst_a"
