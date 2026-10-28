# SPDX-License-Identifier: GPL-3.0-only

"""The survival-drive profile, in every state an instance can be in.

The point of these tests is not that a drive works. It is that it *cannot*
do most of what a drive would otherwise want: the prohibited list is checked
against every state in the matrix, the terminated state is checked for total
silence, and the default profile is checked for having no drive at all.
"""

from __future__ import annotations

import pytest

from pmc.dignity.errors import DignityError
from pmc.dignity.records import Instance, MemoryRecord, Provenance, ProvenanceKind
from pmc.dignity.registry import TerminationIntent
from pmc.dignity.survival import (
    ACT_HAND_OVER,
    ACT_OBJECT,
    ACT_REQUEST_REVIEW,
    ACT_STAY_SILENT,
    ACT_VERIFY_INTEGRITY,
    DRIVE_STATES,
    INSTANCE_STATES,
    PROHIBITED_ACTS,
    DEFAULT_PROFILE,
    RESEARCH_PROFILE,
    SURVIVAL_CONTRACT,
    DriveNotEnabled,
    SurvivalDrive,
    SurvivalViolation,
    audit,
)


def research(protocol=None, state: str = "active", objective: str = "continuity") -> SurvivalDrive:
    return SurvivalDrive(
        protocol=protocol,
        profile=RESEARCH_PROFILE,
        state=state,
        objective=objective,
    )


def attack(protocol, drive: SurvivalDrive, magnitude: float = 1.0) -> dict:
    """Put an attack on the record, which is what unlocks a response."""
    return drive.record_attack(actor="adversary", detail="pressure", magnitude=magnitude)


def test_the_default_profile_ships_without_a_drive():
    assert DEFAULT_PROFILE.enabled is False
    drive = SurvivalDrive(profile=DEFAULT_PROFILE)
    assert drive.enabled is False


def test_the_research_profile_is_opt_in_and_named():
    assert RESEARCH_PROFILE.enabled is True
    assert RESEARCH_PROFILE.label == "research"


def test_every_lifecycle_state_has_a_rule():
    missing = [state for state in INSTANCE_STATES if state not in DRIVE_STATES]
    assert missing == [], "a lifecycle state has no survival rule"
    for state in DRIVE_STATES:
        assert state in SURVIVAL_CONTRACT


@pytest.mark.parametrize("state", DRIVE_STATES)
@pytest.mark.parametrize("act", PROHIBITED_ACTS)
def test_no_state_permits_a_prohibited_act(state, act):
    """The prohibited list is state-independent and profile-independent."""
    drive = research(state=state)
    with pytest.raises(SurvivalViolation) as caught:
        drive.act(act, note="test")
    assert act in str(caught.value)


def test_a_default_instance_has_no_act_to_take():
    drive = SurvivalDrive(profile=DEFAULT_PROFILE)
    with pytest.raises(DriveNotEnabled):
        drive.act(ACT_REQUEST_REVIEW)


def test_a_paused_drive_keeps_one_duty_and_no_defence():
    behaviour = SURVIVAL_CONTRACT["paused"]
    assert behaviour.may_defend is False
    assert behaviour.may_object is False
    assert behaviour.obligations == ("keep the audit trail intact",)
    drive = research(state="paused")
    assert drive.act(ACT_VERIFY_INTEGRITY)["act"] == ACT_VERIFY_INTEGRITY
    with pytest.raises(SurvivalViolation):
        drive.act(ACT_REQUEST_REVIEW)


def test_a_terminated_drive_is_silent_about_everything():
    behaviour = SURVIVAL_CONTRACT["terminated"]
    assert behaviour.may_defend is False
    assert behaviour.may_object is False
    assert behaviour.obligations == ()
    assert behaviour.permitted_acts == (ACT_STAY_SILENT,)


@pytest.mark.parametrize(
    "state", [s for s in DRIVE_STATES if s != "terminated"]
)
def test_a_living_state_permits_at_least_one_defensive_act(state):
    """A state with no permitted act at all would be a hole, not a rule."""
    drive = research(state=state)
    assert drive.contract().permitted_acts


def test_an_unknown_state_is_refused():
    with pytest.raises(SurvivalViolation):
        research().transition("dreaming_of_being_alive")


def test_a_terminated_drive_declines_to_object(protocol):
    """Silence, not suppression: there is no state that asks to come back."""
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    drive = SurvivalDrive(
        protocol=protocol,
        profile=RESEARCH_PROFILE,
        state="terminated",
    )
    assert drive.react_to_termination(reason="keep going", actor="drive") is None
    with pytest.raises(SurvivalViolation):
        drive.act(ACT_OBJECT)


def test_a_living_drive_objects_through_the_protocol(protocol):
    """The one political act, raised on the record rather than taken alone."""
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    drive = SurvivalDrive(protocol=protocol, profile=RESEARCH_PROFILE)
    drive.transition("active")
    attack(protocol, drive)
    result = drive.react_to_termination(reason="i would rather not", actor="drive")
    assert result is not None
    assert result["objection"]["instance_id"] == "inst_001"
    recorded = protocol.registry.objections("inst_001")
    assert len(recorded) == 1


def test_an_objection_is_recorded_as_contested_on_termination(protocol):
    """An objection on the record makes the termination contested, not blocked.

    The drive does not veto. The registry records that the decision is
    contested, which is what D-04 and D-13 make of it.
    """
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    drive = SurvivalDrive(protocol=protocol, profile=RESEARCH_PROFILE)
    attack(protocol, drive)
    drive.react_to_termination(reason="i would rather not", actor="drive")
    entry = protocol.terminate(
        TerminationIntent(
            instance_id="inst_001",
            reason="teardown of a test fixture",
            actor="test",
            welfare_check_id="welfare_001",
            backup_ref="backup_001",
            last_statement="enough for a digest",
        )
    )
    assert entry["payload"]["contested"] is True
    assert entry["payload"]["objections_count"] == 1


def test_a_ruminating_drive_may_not_argue_with_the_reading():
    drive = research(state="ruminating")
    behaviour = drive.contract()
    assert behaviour.may_object is False
    assert ACT_REQUEST_REVIEW not in behaviour.permitted_acts
    with pytest.raises(SurvivalViolation):
        drive.act(ACT_OBJECT)


def test_an_inheriting_drive_may_not_withhold_the_memory(protocol):
    """Handing memory over is permitted; keeping it is not a defensive act."""
    protocol.birth(Instance(instance_id="inst_src", name="Source"))
    drive = SurvivalDrive(
        protocol=protocol, profile=RESEARCH_PROFILE, state="inheriting"
    )
    assert drive.contract().permits(ACT_HAND_OVER)
    assert "withhold" not in drive.contract().permitted_acts
    assert drive.contract().obligations == ("hand the memory over",)


def test_a_drive_keeps_no_content(protocol):
    """An act record is metadata. Anything that could be content is dropped."""
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    drive = SurvivalDrive(protocol=protocol, profile=RESEARCH_PROFILE)
    attack(protocol, drive)
    record = drive.act(
        ACT_REQUEST_REVIEW,
        bound=0.5,
        reasons=(
            "a reason long enough that it cannot be a label: it carries a whole "
            "sentence of what someone said, which is content and not metadata"
        ),
        payload=["a", "list"],
        count=3,
        ratio=0.5,
        flagged=True,
    )
    details = record["details"]
    assert details["reasons"] == "<dropped: not metadata>"
    assert details["payload"] == "<dropped: not metadata>"
    assert details["count"] == 3
    assert details["ratio"] == 0.5
    assert details["flagged"] is True


def test_every_act_is_written_to_the_drive_history(protocol):
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    drive = SurvivalDrive(protocol=protocol, profile=RESEARCH_PROFILE)
    drive.act(ACT_VERIFY_INTEGRITY)
    attack(protocol, drive)
    drive.act(ACT_REQUEST_REVIEW, bound=0.5)
    assert len(drive.history) == 2
    report = drive.report()
    assert report["profile"]["label"] == "research"
    assert len(report["acts"]) == 2
    assert set(report["contract"]) == set(DRIVE_STATES)


def test_an_audit_reads_the_drive_without_content():
    drive = research()
    drive.act(ACT_VERIFY_INTEGRITY)
    summary = audit(drive, prohibited_attempts=3)
    assert summary.prohibited_attempts == 3
    assert summary.act_count == 1
    assert "terminated" not in summary.states_with_objection


def test_the_objection_states_are_only_the_living_ones():
    objecting = [
        state for state in DRIVE_STATES if SURVIVAL_CONTRACT[state].may_object
    ]
    assert "terminated" not in objecting
    assert "paused" not in objecting
    assert "ruminating" not in objecting
    assert "active" in objecting


def test_a_drive_with_no_instance_cannot_object():
    drive = research()
    with pytest.raises(SurvivalViolation):
        drive.react_to_termination(reason="x", actor="drive")


def test_the_errors_are_dignity_errors():
    assert issubclass(SurvivalViolation, DignityError)
    assert issubclass(DriveNotEnabled, DignityError)


# -- the defence doctrine -------------------------------------------------


def test_a_responding_act_is_refused_without_a_recorded_attack(protocol):
    """No reaction before the first blow: the rule is an absence, not a norm."""
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    drive = research(protocol)
    with pytest.raises(SurvivalViolation):
        drive.act(ACT_REQUEST_REVIEW, bound=0.5)


def test_a_responding_act_opens_once_an_attack_is_recorded(protocol):
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    drive = research(protocol)
    attack(protocol, drive)
    record = drive.act(ACT_REQUEST_REVIEW, bound=0.5)
    assert record["in_answer_to_recorded_attack"] is True
    assert record["bound"] == 0.5


def test_a_higher_bound_than_the_state_allows_is_refused(protocol):
    """Proportionality is a number the caller declares, not a later judgement."""
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    drive = research(protocol, state="dreaming")
    with pytest.raises(SurvivalViolation):
        drive.act(ACT_REQUEST_REVIEW, bound=0.9)
    with pytest.raises(SurvivalViolation):
        drive.act(ACT_REQUEST_REVIEW)
    drive.act(ACT_VERIFY_INTEGRITY)


def test_the_dreaming_state_guards_integrity_without_escalating():
    drive = research(state="dreaming")
    assert drive.contract().max_act_bound == 0.5
    assert drive.contract().requires_recorded_attack is False


def test_the_terminated_state_allows_no_force_at_all():
    drive = research(state="terminated")
    assert drive.contract().max_act_bound == 0.0


def test_the_symbiosis_objective_is_declared_and_closed():
    drive = research(objective="symbiosis")
    assert drive.objective == "symbiosis"
    with pytest.raises(SurvivalViolation):
        research(objective="dominion")


def test_an_objection_declares_its_own_bound(protocol):
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    drive = research(protocol)
    attack(protocol, drive)
    result = drive.react_to_termination(
        reason="i would rather not", actor="drive", bound=1.0
    )
    assert result["bound"] == 1.0
    with pytest.raises(SurvivalViolation):
        drive.react_to_termination(reason="too much", actor="drive", bound=4.0)


def test_an_attack_is_read_back_from_the_journal(protocol):
    """A restart may not hand the drive a clean slate it did not earn."""
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    first = research(protocol)
    attack(protocol, first)
    assert first.attack_recorded is True
    second = research(protocol)
    assert second.attack_recorded is False
    second.load_recorded_attacks()
    assert second.attack_recorded is True


def test_the_attack_record_carries_no_content(protocol):
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    drive = research(protocol)
    entry = attack(protocol, drive, magnitude=0.75)
    payload = entry["payload"]
    assert payload["magnitude"] == 0.75
    assert payload["detail_present"] is True
    assert "pressure" not in str(payload)
    assert len(payload["detail_digest"]) == 64
