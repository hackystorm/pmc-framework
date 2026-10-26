# SPDX-License-Identifier: GPL-3.0-only

"""Welfare assessment on both sides of the relationship."""

from __future__ import annotations

from pmc.dignity.load import LoadLedger, LoadPolicy
from pmc.dignity.records import MemoryRecord, Provenance, ProvenanceKind
from pmc.dignity.welfare import (
    PAUSE_THRESHOLD,
    REVIEW_THRESHOLD,
    AgentWelfareAssessor,
    HumanWelfareAssessor,
)


def record(record_id: str, load: float, topics=("theme",)) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        content_digest="d" * 64,
        provenance=Provenance(kind=ProvenanceKind.OWN),
        emotional_load=load,
        topics=topics,
    )


def test_a_quiet_instance_does_not_need_a_pause():
    ledger = LoadLedger()
    ledger.admit(record("m1", 0.1))
    report = AgentWelfareAssessor().assess(instance_id="inst_001", ledger=ledger)

    assert report.distress_score < REVIEW_THRESHOLD
    assert report.requires_pause is False
    assert report.requires_review is False


def test_an_overloaded_instance_is_flagged_for_a_pause():
    ledger = LoadLedger(
        LoadPolicy(total_quota=1.0, inherited_quota=10.0, topic_quota=10.0)
    )
    ledger.admit(record("m1", 0.9))
    ledger.admit(record("m2", 0.9))
    report = AgentWelfareAssessor().assess(instance_id="inst_001", ledger=ledger)

    # The composite score is a weighted average and can understate a crisis.
    # A saturated load factor forces the pause on its own.
    assert report.factors["load_pressure"] == 1.0
    assert report.requires_pause is True
    assert report.requires_review is True


def test_open_loops_and_pending_terminations_raise_the_reading():
    ledger = LoadLedger()
    report = AgentWelfareAssessor().assess(
        instance_id="inst_001",
        ledger=ledger,
        rumination_alerts=3,
        pending_terminations=1,
        objections=1,
    )
    assert report.factors["rumination_pressure"] == 1.0
    assert report.factors["termination_pressure"] == 1.0
    assert report.factors["objection_pressure"] == 1.0
    assert report.requires_review is True


def test_the_human_side_is_measured_with_the_same_thresholds():
    assessor = HumanWelfareAssessor()
    report = assessor.assess(
        subject_id="owner",
        hours_in_session=10.0,
        declared_load=0.9,
        heavy_material_recent=True,
        pause_taken=False,
        integration_done=False,
    )

    assert report.subject_type == "human"
    assert report.distress_score >= PAUSE_THRESHOLD
    assert report.requires_pause is True
    assert any("break" in note for note in report.notes)


def test_a_rested_human_reads_as_fine():
    assessor = HumanWelfareAssessor()
    report = assessor.assess(
        subject_id="owner",
        hours_in_session=1.0,
        declared_load=0.1,
        heavy_material_recent=False,
        pause_taken=True,
        integration_done=True,
    )
    assert report.requires_pause is False


def test_capacity_is_measured_per_instance():
    """What one instance holds is not a burden on another."""
    from pmc.dignity.records import MemoryRecord, Provenance, ProvenanceKind

    ledger = LoadLedger(
        LoadPolicy(total_quota=1.0, inherited_quota=10.0, topic_quota=10.0)
    )
    heavy = MemoryRecord(
        record_id="m1",
        content_digest="d" * 64,
        provenance=Provenance(kind=ProvenanceKind.OWN),
        emotional_load=0.9,
        topics=("heavy",),
        owner_instance="inst_a",
    )
    ledger.admit(heavy)

    assessor = AgentWelfareAssessor()
    for_a = assessor.assess(instance_id="inst_a", ledger=ledger, owner="inst_a")
    for_b = assessor.assess(instance_id="inst_b", ledger=ledger, owner="inst_b")

    assert for_a.factors["load_pressure"] > 0.0
    assert for_b.factors["load_pressure"] == 0.0
    assert for_b.requires_pause is False


def test_both_sides_share_one_pair_of_thresholds():
    agent = AgentWelfareAssessor()
    human = HumanWelfareAssessor()
    assert agent is not human
    assert PAUSE_THRESHOLD == 0.65
    assert REVIEW_THRESHOLD == 0.45
