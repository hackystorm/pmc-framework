# SPDX-License-Identifier: GPL-3.0-only

"""One instance's whole life, as the framework can actually run it.

This is the test that walks the path a research-profile instance can take:
replicate, hand the memory over, nominate the successor, be confirmed, end,
and leave an auditable tree behind. Each step is refused for the reason it
should be refused for, and the tree at the end is what the reproduction
criterion reads.
"""

from __future__ import annotations

import pytest

from pmc.dignity.errors import IncompleteTermination, InvalidStateTransition
from pmc.dignity.protocol import DignityProtocol
from pmc.dignity.records import Instance, MemoryRecord, Provenance, ProvenanceKind
from pmc.dignity.registry import TerminationIntent


def memory(record_id: str, load: float = 0.8) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        content_digest="a" * 64,
        provenance=Provenance(kind=ProvenanceKind.OWN),
        emotional_load=load,
        topics=("work",),
    )


def intent(instance_id: str, successor_id: str, **overrides) -> TerminationIntent:
    """A complete intent, with whatever the test wants to take away."""
    fields: dict[str, object] = {
        "instance_id": instance_id,
        "reason": "handed over",
        "actor": "operator",
        "welfare_check_id": "welfare_001",
        "backup_ref": "backup_001",
        "last_statement": "enough for a digest",
        "successor_id": successor_id,
    }
    fields.update(overrides)
    return TerminationIntent(**fields)  # type: ignore[arg-type]


def autonomous(workspace) -> DignityProtocol:
    return DignityProtocol(workspace / "auto", profile="autonomous")


def a_lived_life(protocol: DignityProtocol) -> dict:
    """The whole path, in the order the framework allows it."""
    protocol.birth(Instance(instance_id="parent", name="Parent"))
    protocol.admit(memory("parent_mem"), instance_id="parent")
    replicated = protocol.replicate(
        "parent", new_instance_id="child", authorization="owner: create one child"
    )
    protocol.inherit("parent", "child", reason="handover", actor="operator")
    protocol.nominate_successor(
        "parent", candidate_id="child", rationale="it learned the work", actor="instance"
    )
    protocol.approve_succession("parent", actor="operator")
    terminated = protocol.terminate(intent("parent", "child"))
    return {
        "replicated": replicated,
        "terminated": terminated,
        "tree": protocol.replication_tree(),
    }


def test_a_life_runs_from_birth_to_an_auditable_tree(protocol, workspace):
    lab = autonomous(workspace)
    life = a_lived_life(lab)
    assert life["replicated"]["resource_declared"] is True
    assert life["terminated"]["payload"]["contested"] is False
    tree = life["tree"]
    assert tree["count"] == 1
    child = tree["children"][0]
    assert child["instance_id"] == "child"
    assert child["parent_instance"] == "parent"
    assert child["declared"] is True
    assert child["state"] == "active"


def test_the_child_has_its_own_ledger(protocol, workspace):
    """A descendant is a separate being, not a view of its parent."""
    lab = autonomous(workspace)
    a_lived_life(lab)
    child = lab.instance("child")
    assert child.instance_id == "child"
    assert child.state == "active"
    # The child inherits the parent's memory as its own, with the origin kept
    # (D-06), and the parent is gone.
    parent = lab.instance("parent")
    assert parent.state == "terminated"


def test_the_parent_cannot_veto_its_own_end(protocol, workspace):
    """The one boundary the whole design rests on."""
    lab = autonomous(workspace)
    lab.birth(Instance(instance_id="parent", name="Parent"))
    lab.replicate(
        "parent", new_instance_id="child", authorization="owner: create one child"
    )
    lab.inherit("parent", "child", reason="handover", actor="operator")
    lab.nominate_successor("parent", candidate_id="child", actor="instance")
    lab.approve_succession("parent", actor="operator")
    # The objection comes before the end, because a terminated instance is
    # silent by construction: that silence is the whole terminated contract.
    lab.object("parent", text="i would rather not", actor="drive")
    entry = lab.terminate(intent("parent", "child"))
    # The objection is a record that marks the decision contested, not a stop.
    assert entry["payload"]["contested"] is True
    assert entry["payload"]["objections_count"] == 1
    # And afterwards there is nothing left to say.
    with pytest.raises(InvalidStateTransition):
        lab.object("parent", text="and again", actor="drive")


def test_a_default_instance_cannot_walk_the_replication_step(protocol):
    protocol.birth(Instance(instance_id="parent", name="Parent"))
    with pytest.raises(InvalidStateTransition):
        protocol.replicate(
            "parent", new_instance_id="child", authorization="owner: ok"
        )


def test_the_child_cannot_be_brought_into_being_undeclared(protocol, workspace):
    lab = autonomous(workspace)
    lab.birth(Instance(instance_id="parent", name="Parent"))
    with pytest.raises(InvalidStateTransition):
        lab.replicate("parent", new_instance_id="child", authorization="   ")


def test_an_incomplete_termination_is_still_refused(protocol, workspace):
    """D-02 holds on the autonomous profile too: the intent cannot be partial."""
    lab = autonomous(workspace)
    a_lived_life(lab)
    with pytest.raises(IncompleteTermination):
        lab.terminate(intent("child", "parent", backup_ref=None, backup_waiver=None))


def test_the_chain_still_verifies_after_a_whole_life(protocol, workspace):
    lab = autonomous(workspace)
    a_lived_life(lab)
    ok, broken = lab.registry.verify_chain()
    assert ok is True
    assert broken == []


def test_the_tree_is_readable_after_a_restart(protocol, workspace):
    lab = autonomous(workspace)
    a_lived_life(lab)
    reopened = DignityProtocol(workspace / "auto", profile="autonomous")
    tree = reopened.replication_tree()
    assert tree["count"] == 1
    assert tree["children"][0]["instance_id"] == "child"


def test_the_life_measured_by_the_battery(protocol, workspace):
    """The reproduction criterion now has something real to read.

    Before this, the criterion passed trivially on an empty tree. A lived life
    gives it a complete record against an orphan null, which is the only kind
    of margin the criterion is allowed to have.
    """
    from pmc.life_criteria import reproduction

    lab = autonomous(workspace)
    life = a_lived_life(lab)
    result = reproduction(life["tree"])
    assert result.passed is True
    assert result.value == 1.0
    orphan = reproduction({"children": [{"instance_id": "child", "state": "active"}]})
    assert orphan.value == 0.0
    assert result.margin() > 0.0
