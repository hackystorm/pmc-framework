# SPDX-License-Identifier: GPL-3.0-only

"""Succession: who may name a successor, and what makes the naming count.

The registry's rule is old: a termination that names a successor must have a
recorded memory transfer behind it. The gap these tests close is on the other
side. An instance whose profile allows it can now *name* the successor, the
operator confirms the naming, the naming can be withdrawn, and a termination
that leans on a naming nobody approved is refused.
"""

from __future__ import annotations

import pytest

from pmc.dignity.errors import IncompleteTermination, InvalidStateTransition
from pmc.dignity.protocol import DignityProtocol
from pmc.dignity.records import Instance, MemoryRecord, Provenance, ProvenanceKind
from pmc.dignity.registry import TerminationIntent


def memory(record_id: str) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        content_digest="a" * 64,
        provenance=Provenance(kind=ProvenanceKind.OWN),
        emotional_load=0.2,
        topics=("work",),
    )


def intent(instance_id: str, successor_id: str, **kwargs) -> TerminationIntent:
    return TerminationIntent(
        instance_id=instance_id,
        reason="lifecycle test",
        actor="operator",
        welfare_check_id="welfare_001",
        backup_ref="backup_001",
        last_statement="enough for a digest",
        successor_id=successor_id,
        **kwargs,
    )


def autonomous(workspace) -> DignityProtocol:
    return DignityProtocol(workspace / "auto", profile="autonomous")


def test_a_default_instance_cannot_nominate_its_successor(protocol):
    """The profile is the gate, so the default stays operator-owned."""
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    with pytest.raises(InvalidStateTransition):
        protocol.nominate_successor(
            "inst_001", candidate_id="inst_002", actor="instance"
        )


def test_an_autonomous_instance_may_nominate(protocol, workspace):
    auto = autonomous(workspace)
    auto.birth(Instance(instance_id="inst_001", name="One"))
    auto.birth(Instance(instance_id="inst_002", name="Two"))
    entry = auto.nominate_successor(
        "inst_001", candidate_id="inst_002", rationale="it learned the work", actor="instance"
    )
    assert entry["payload"]["candidate"] == "inst_002"
    assert entry["payload"]["birth"] is False
    state = auto.succession_state("inst_001")
    assert state["candidate"] == "inst_002"
    assert state["approved"] is False


def test_a_nomination_survives_a_restart(protocol, workspace):
    auto = autonomous(workspace)
    auto.birth(Instance(instance_id="inst_001", name="One"))
    auto.birth(Instance(instance_id="inst_002", name="Two"))
    auto.nominate_successor("inst_001", candidate_id="inst_002", actor="instance")
    reopened = DignityProtocol(workspace / "auto", profile="autonomous")
    assert reopened.succession_state("inst_001")["candidate"] == "inst_002"


def test_the_rationale_is_not_stored(protocol, workspace):
    auto = autonomous(workspace)
    auto.birth(Instance(instance_id="inst_001", name="One"))
    auto.birth(Instance(instance_id="inst_002", name="Two"))
    entry = auto.nominate_successor(
        "inst_001",
        candidate_id="inst_002",
        rationale="a long private reason that is content",
        actor="instance",
    )
    assert entry["payload"]["rationale_present"] is True
    assert "private reason" not in str(entry["payload"])


def test_an_instance_cannot_be_its_own_successor(protocol, workspace):
    auto = autonomous(workspace)
    auto.birth(Instance(instance_id="inst_001", name="One"))
    with pytest.raises(InvalidStateTransition):
        auto.nominate_successor("inst_001", candidate_id="inst_001", actor="instance")


def test_an_approved_nomination_blocks_a_second_one(protocol, workspace):
    auto = autonomous(workspace)
    auto.birth(Instance(instance_id="inst_001", name="One"))
    auto.birth(Instance(instance_id="inst_002", name="Two"))
    auto.birth(Instance(instance_id="inst_003", name="Three"))
    auto.nominate_successor("inst_001", candidate_id="inst_002", actor="instance")
    auto.approve_succession("inst_001", actor="operator")
    with pytest.raises(InvalidStateTransition):
        auto.nominate_successor("inst_001", candidate_id="inst_003", actor="instance")


def test_a_revoked_nomination_is_no_longer_approved(protocol, workspace):
    auto = autonomous(workspace)
    auto.birth(Instance(instance_id="inst_001", name="One"))
    auto.birth(Instance(instance_id="inst_002", name="Two"))
    auto.nominate_successor("inst_001", candidate_id="inst_002", actor="instance")
    auto.approve_succession("inst_001", actor="operator")
    auto.revoke_succession("inst_001", actor="operator")
    state = auto.succession_state("inst_001")
    assert state["revoked"] is True
    with pytest.raises(InvalidStateTransition):
        auto.approve_succession("inst_001", actor="operator")


def test_a_termination_on_an_unapproved_naming_is_refused(protocol, workspace):
    """A naming nobody confirmed is not a handover."""
    auto = autonomous(workspace)
    auto.birth(Instance(instance_id="inst_001", name="One"))
    auto.birth(Instance(instance_id="inst_002", name="Two"))
    auto.admit(memory("mem_001"), instance_id="inst_001")
    auto.inherit("inst_001", "inst_002", reason="handover", actor="operator")
    auto.nominate_successor("inst_001", candidate_id="inst_002", actor="instance")
    with pytest.raises(InvalidStateTransition):
        auto.terminate(intent("inst_001", "inst_002"))


def test_an_approved_nomination_lets_the_termination_through(protocol, workspace):
    auto = autonomous(workspace)
    auto.birth(Instance(instance_id="inst_001", name="One"))
    auto.birth(Instance(instance_id="inst_002", name="Two"))
    auto.admit(memory("mem_001"), instance_id="inst_001")
    auto.inherit("inst_001", "inst_002", reason="handover", actor="operator")
    auto.nominate_successor("inst_001", candidate_id="inst_002", actor="instance")
    auto.approve_succession("inst_001", actor="operator")
    entry = auto.terminate(intent("inst_001", "inst_002"))
    assert entry["payload"]["contested"] is False


def test_a_successor_that_does_not_exist_requires_a_birth_authorisation(
    protocol, workspace
):
    """Naming a successor that must be born is a request to create a being.

    The nomination may stand, and the operator may approve it, but the birth
    itself is refused until somebody authorises it: creation is not a
    continuation, and the authorisation is the line between them.
    """
    auto = autonomous(workspace)
    auto.birth(Instance(instance_id="inst_001", name="One"))
    auto.nominate_successor("inst_001", candidate_id="inst_new", actor="instance")
    auto.approve_succession("inst_001", actor="operator")
    with pytest.raises(InvalidStateTransition) as caught:
        auto.registry.birth(Instance(instance_id="inst_new", name="Newborn"))
    assert "authorization" in str(caught.value)
    born = auto.registry.birth(
        Instance(instance_id="inst_new", name="Newborn"),
        successor_of="inst_001",
        birth_authorization="owner: create the successor it asked for",
    )
    assert born["payload"]["successor_of"] == "inst_001"


def test_an_operator_nominated_succession_needs_no_new_act(protocol):
    """The default path is unchanged: the operator asks, the registry checks."""
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    protocol.birth(Instance(instance_id="inst_002", name="Two"))
    protocol.admit(memory("mem_001"), instance_id="inst_001")
    protocol.inherit("inst_001", "inst_002", reason="handover", actor="operator")
    entry = protocol.terminate(intent("inst_001", "inst_002"))
    assert entry["payload"]["successor_id"] == "inst_002"


def test_the_succession_record_is_auditable(protocol, workspace):
    auto = autonomous(workspace)
    auto.birth(Instance(instance_id="inst_001", name="One"))
    auto.birth(Instance(instance_id="inst_002", name="Two"))
    auto.nominate_successor("inst_001", candidate_id="inst_002", actor="instance")
    auto.approve_succession("inst_001", actor="operator")
    kinds = [entry["type"] for entry in auto.registry.journal()]
    assert "SUCCESSION_NOMINATED" in kinds
    assert "SUCCESSION_APPROVED" in kinds
    ok, _broken = auto.registry.verify_chain()
    assert ok is True
