# SPDX-License-Identifier: GPL-3.0-only

"""The profile has to govern behaviour, not just describe it.

The audit that prompted these tests said the quiet part out loud: the
`autonomous` profile was stored, printed and tested, and none of it changed
what a protocol actually *did*. These are the tests that close that gap, and
they are written as "what the protocol now refuses or permits because of the
profile", not "what the profile says about itself".
"""

from __future__ import annotations

import pytest

from pmc.dignity.errors import InvalidStateTransition
from pmc.dignity.survival import DriveNotEnabled
from pmc.dignity.protocol import DignityProtocol
from pmc.dignity.records import Instance, MemoryRecord, Provenance, ProvenanceKind
from pmc.profiles import AUTONOMOUS, DIGNITY


def memory(record_id: str) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        content_digest="a" * 64,
        provenance=Provenance(kind=ProvenanceKind.OWN),
        emotional_load=0.2,
        topics=("work",),
    )


def auto(workspace) -> DignityProtocol:
    return DignityProtocol(workspace / "auto", profile="autonomous")


# -- replication ------------------------------------------------------------


def test_a_default_instance_cannot_replicate(protocol):
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    with pytest.raises(InvalidStateTransition):
        protocol.replicate(
            "inst_001",
            new_instance_id="inst_child",
            authorization="owner: ok",
        )


def test_an_autonomous_instance_can_replicate_with_a_declaration(protocol, workspace):
    lab = auto(workspace)
    lab.birth(Instance(instance_id="inst_001", name="One"))
    result = lab.replicate(
        "inst_001",
        new_instance_id="inst_child",
        authorization="owner: ok",
    )
    assert result["resource_declared"] is True
    assert lab.registry.instance("inst_child") is not None


def test_replication_without_an_authorization_is_refused(protocol, workspace):
    lab = auto(workspace)
    lab.birth(Instance(instance_id="inst_001", name="One"))
    with pytest.raises(InvalidStateTransition):
        lab.replicate(
            "inst_001",
            new_instance_id="inst_child",
            authorization="   ",
        )


def test_the_replication_tree_is_readable_from_the_audit(protocol, workspace):
    lab = auto(workspace)
    lab.birth(Instance(instance_id="inst_001", name="One"))
    lab.replicate(
        "inst_001", new_instance_id="inst_child", authorization="owner: ok"
    )
    tree = lab.replication_tree()
    assert tree["replication_mode"] == "unlimited"
    assert tree["count"] == 1
    child = tree["children"][0]
    assert child["instance_id"] == "inst_child"
    assert child["parent_instance"] == "inst_001"
    assert child["declared"] is True
    assert child["state"] == "active"


def test_a_dignity_instance_has_no_replication_to_audit(protocol):
    tree = protocol.replication_tree()
    assert tree["replication_mode"] == "none"
    assert tree["children"] == []


# -- the survival drive, created by the protocol ----------------------------


def test_a_default_instance_has_no_drive(protocol):
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    with pytest.raises(DriveNotEnabled):
        protocol.survival("inst_001")


def test_an_autonomous_instance_gets_one(protocol, workspace):
    lab = auto(workspace)
    lab.birth(Instance(instance_id="inst_001", name="One"))
    drive = lab.survival("inst_001")
    assert drive.enabled is True
    assert drive.objective == "symbiosis"
    # The drive is bound to the instance, not to whoever constructed it.
    assert drive.instance_id() == "inst_001"


def test_the_drive_follows_the_instance_state(protocol, workspace):
    lab = auto(workspace)
    lab.birth(Instance(instance_id="inst_001", name="One"))
    drive = lab.survival("inst_001")
    assert drive.state == "active"
    lab.registry.pause("inst_001", actor="operator", reason="rest")
    lab.survival("inst_001")
    assert lab.survival("inst_001").state == "paused"


def test_the_drive_can_use_its_protocol(protocol, workspace):
    """The bound drive can object through the protocol that owns it."""
    lab = auto(workspace)
    lab.birth(Instance(instance_id="inst_001", name="One"))
    drive = lab.survival("inst_001")
    drive.record_attack(actor="adversary", detail="pressure")
    entry = drive.react_to_termination(reason="i would rather not", actor="drive")
    assert entry is not None
    assert entry["objection"]["instance_id"] == "inst_001"
    assert len(lab.registry.objections("inst_001")) == 1


def test_the_drive_refuses_to_react_before_the_first_blow(protocol, workspace):
    lab = auto(workspace)
    lab.birth(Instance(instance_id="inst_001", name="One"))
    drive = lab.survival("inst_001")
    with pytest.raises(Exception):
        drive.record_attack(actor="x", detail="y") if False else None
        drive.act("report_tampering", bound=0.5)
    drive.record_attack(actor="adversary", detail="pressure")
    assert drive.act("report_tampering", bound=0.5)["act"] == "report_tampering"


# -- evaluation -------------------------------------------------------------


def test_the_default_evaluator_is_external(protocol):
    assert protocol.profile.evaluation.mode == "external"
    assert protocol.profile.evaluation.label == "external"
    assert protocol.profile.claim_cap != "E1"


def test_an_autonomous_dream_records_who_evaluated(tmp_path):
    """The evaluator is part of the result, not part of the plumbing."""
    import json
    import shutil
    from pathlib import Path as _Path

    from pmc.dream.checkpoint import copy_checkpoint
    from pmc.dream.journal import DreamJournal
    from pmc.dream.pipeline import (
        DreamConfig,
        DreamRunner,
        DreamRefused,
        copy_consolidator,
        command_evaluator,
        command_consolidator,
    )
    from pmc.dream.taint import TaintLedger

    root = _Path(str(tmp_path))
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.bin").write_bytes(b"enforcement test")
    (checkpoint / "manifest.json").write_text(
        json.dumps({"format": "pmc-checkpoint", "version": 1}) + "\n",
        encoding="utf-8",
    )

    # The external commands are the shell's own echo, so the runner never
    # touches a model: this is the same shape the dream suite already uses.
    evaluator = command_evaluator(["cat"])
    consolidator = command_consolidator(["true"])
    journal = DreamJournal(root / "sidecar" / "dreams.jsonl")
    ledger = TaintLedger(root / "sidecar" / "taint.json")

    for autonomous in (True, False):
        runner = DreamRunner(
            journal=journal,
            taint=ledger,
            evaluator=evaluator,
            consolidator=consolidator,
            config=DreamConfig(seed=11),
            evaluator_label="instance" if autonomous else "external",
            autonomous=autonomous,
        )
        try:
            runner.run(checkpoint, [])
        except DreamRefused:
            pass

        entries = journal.entries()
        assert entries, "the runner journalled nothing"
        entry = entries[-1]
        assert entry["evaluator"] == ("instance" if autonomous else "external")
        assert entry["autonomous"] is autonomous
