# SPDX-License-Identifier: GPL-3.0-only

"""Registry behaviour: births, terminations, objections, tombstones, tampering."""

from __future__ import annotations

import json

import pytest

from pmc.dignity.errors import (
    IncompleteTermination,
    InstanceAlreadyExists,
    InstanceNotFound,
    InvalidStateTransition,
    PurgeNotAuthorized,
)
from pmc.dignity.records import Instance
from pmc.dignity.registry import InstanceRegistry, TerminationIntent


def make_intent(**overrides) -> TerminationIntent:
    base = {
        "instance_id": "inst_001",
        "reason": "owner requested the end of this instance",
        "actor": "owner",
        "welfare_check_id": "welfare_001",
        "backup_ref": "./backups/inst_001.tar",
        "last_statement": "thank you",
    }
    base.update(overrides)
    return TerminationIntent(**base)


def test_birth_is_recorded(tmp_path):
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    entry = registry.birth(Instance(instance_id="inst_001", name="Test"))

    assert entry["type"] == "BIRTH"
    assert entry["seq"] == 0
    assert registry.instance("inst_001").state == "active"
    valid, broken = registry.verify_chain()
    assert valid
    assert broken == []


def test_duplicate_birth_is_refused(tmp_path):
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    registry.birth(Instance(instance_id="inst_001"))
    with pytest.raises(InstanceAlreadyExists):
        registry.birth(Instance(instance_id="inst_001"))


def test_incomplete_termination_is_refused_and_writes_nothing(tmp_path):
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    registry.birth(Instance(instance_id="inst_001"))
    before = len(registry.journal())

    with pytest.raises(IncompleteTermination) as error:
        registry.terminate(make_intent(welfare_check_id=None, backup_ref=None))

    assert "welfare_check_id" in " ".join(error.value.missing)
    assert len(registry.journal()) == before
    assert registry.instance("inst_001").state == "active"


def test_termination_requires_a_statement_or_an_unresponsive_flag(tmp_path):
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    registry.birth(Instance(instance_id="inst_001"))
    with pytest.raises(IncompleteTermination):
        registry.terminate(make_intent(last_statement=None))
    entry = registry.terminate(make_intent(last_statement=None, unresponsive=True))
    assert entry["type"] == "TERMINATION"
    assert registry.instance("inst_001").state == "terminated"


def test_termination_records_metadata_but_not_the_statement_text(tmp_path):
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    registry.birth(Instance(instance_id="inst_001"))
    entry = registry.terminate(make_intent(last_statement="a private goodbye"))

    payload = entry["payload"]
    assert payload["last_statement_present"] is True
    assert payload["last_statement_digest"]
    assert "a private goodbye" not in json.dumps(entry)


def test_successor_requires_a_recorded_inheritance(tmp_path):
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    registry.birth(Instance(instance_id="inst_001"))
    registry.birth(Instance(instance_id="inst_002"))

    with pytest.raises(InvalidStateTransition):
        registry.terminate(make_intent(successor_id="inst_002"))


def test_objection_marks_the_termination_as_contested(tmp_path):
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    registry.birth(Instance(instance_id="inst_001"))
    registry.object("inst_001", text="I do not agree to end here", actor="instance")

    entry = registry.terminate(make_intent())
    assert entry["payload"]["contested"] is True
    assert entry["payload"]["objections_count"] == 1
    assert registry.audit()["contested_terminations"] == 1


def test_objection_text_stays_out_of_the_journal(tmp_path):
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    registry.birth(Instance(instance_id="inst_001"))
    registry.object("inst_001", text="my secret objection", actor="instance")

    raw = (tmp_path / "journal.jsonl").read_text(encoding="utf-8")
    assert "my secret objection" not in raw


def test_purge_requires_authorization_and_leaves_a_tombstone(tmp_path):
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    registry.birth(Instance(instance_id="inst_001"))
    registry.terminate(make_intent())

    with pytest.raises(PurgeNotAuthorized):
        registry.purge("inst_001", authorization="  ")

    entry = registry.purge("inst_001", authorization="owner-consent-2026-10-26")
    assert entry["type"] == "PURGED"
    assert entry["payload"]["tombstone"] is True

    valid, broken = registry.verify_chain()
    assert valid
    assert broken == []


def test_only_terminated_instances_can_be_purged(tmp_path):
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    registry.birth(Instance(instance_id="inst_001"))
    with pytest.raises(InvalidStateTransition):
        registry.purge("inst_001", authorization="owner-consent")


def test_tampering_with_the_journal_is_detected(tmp_path):
    path = tmp_path / "journal.jsonl"
    registry = InstanceRegistry(path)
    registry.birth(Instance(instance_id="inst_001", name="Original"))
    registry.terminate(make_intent())

    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["payload"]["name"] = "Tampered"
    lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reloaded = InstanceRegistry(path)
    valid, broken = reloaded.verify_chain()
    assert not valid
    assert broken
    assert reloaded.audit()["tamper_detected"] is True


def test_truncating_the_journal_is_detected(tmp_path):
    path = tmp_path / "journal.jsonl"
    registry = InstanceRegistry(path)
    registry.birth(Instance(instance_id="inst_001"))
    registry.birth(Instance(instance_id="inst_002"))

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[0] + "\n", encoding="utf-8")

    reloaded = InstanceRegistry(path)
    valid, broken = reloaded.verify_chain()
    assert not valid
    assert broken


def test_missing_instance_raises(tmp_path):
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    with pytest.raises(InstanceNotFound):
        registry.instance("nope")


def test_audit_counters_start_clean(tmp_path):
    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    registry.birth(Instance(instance_id="inst_001"))
    report = registry.audit()

    assert report["chain_valid"] is True
    assert report["instances_born"] == 1
    assert report["instances_terminated"] == 0
    assert report["terminations_missing_fields"] == 0
