# SPDX-License-Identifier: GPL-3.0-only

"""The work rhythm: granted, budgeted, and it stops when it stops working.

The tests are about the three properties rather than the mechanism. A rhythm
an instance has not been granted is refused; a cycle is a declared cost; and a
rhythm whose cycles change nothing refuses to continue, because that is what
separates work from rumination (D-09).
"""

from __future__ import annotations

import pytest

from pmc.dignity.errors import InvalidStateTransition
from pmc.dignity.protocol import DignityProtocol
from pmc.dignity.records import Instance
from pmc.rhythm import (
    ALL_WORK,
    WORK_FLAG_WELFARE,
    WORK_PROPOSE_CONSOLIDATION,
    DEFAULT_POLICY,
    WORK_ASSESS_WELFARE,
    WORK_AUDIT_LIFECYCLE,
    WORK_VERIFY_JOURNAL,
    RhythmNotGranted,
    RhythmRumination,
    WorkRhythm,
    rhythm_for,
)


class FakeClock:
    """A clock the tests move by hand, so no cycle is ever 'due' by accident."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def born(protocol: DignityProtocol, instance_id: str = "inst_001") -> None:
    protocol.birth(Instance(instance_id=instance_id, name="One"))


def auto(workspace) -> DignityProtocol:
    return DignityProtocol(workspace / "auto", profile="autonomous")


def rhythm(protocol, clock=None, policy=None, **kwargs) -> WorkRhythm:
    return WorkRhythm(
        protocol,
        "inst_001",
        policy=policy or DEFAULT_POLICY,
        clock=clock or FakeClock(),
        **kwargs,
    )


def test_a_default_instance_has_no_rhythm(protocol):
    born(protocol)
    with pytest.raises(RhythmNotGranted):
        rhythm_for(protocol, "inst_001")


def test_the_protocol_creates_the_rhythm_for_an_autonomous_instance(
    protocol, workspace
):
    lab = auto(workspace)
    born(lab)
    cycle = lab.rhythm("inst_001")
    assert cycle.instance_id == "inst_001"
    assert cycle.policy.max_items_per_cycle > 0


def test_an_unknown_kind_of_work_is_refused(protocol, workspace):
    lab = auto(workspace)
    born(lab)
    cycle = lab.rhythm("inst_001")
    with pytest.raises(RhythmNotGranted):
        cycle.enqueue_kind("meditate")


def test_a_cycle_returns_to_the_queue_nothing_when_no_work_is_queued(
    protocol, workspace
):
    lab = auto(workspace)
    born(lab)
    cycle = lab.rhythm("inst_001")
    record = cycle.tick(force=True)
    assert record.reason == "no work queued"
    assert record.items == []
    assert record.useful is False


def test_a_cycle_declares_its_budget_once(protocol, workspace):
    lab = auto(workspace)
    born(lab)
    first = lab.rhythm("inst_001")
    second = lab.rhythm("inst_001")
    assert first is not second
    kinds = [
        entry["payload"].get("act")
        for entry in lab.registry.journal()
        if entry["type"] == "RHYTHM"
    ]
    assert kinds.count("budget_declared") == 1


def test_a_cycle_is_not_due_before_the_interval(protocol, workspace):
    lab = auto(workspace)
    born(lab)
    clock = FakeClock()
    cycle = rhythm(lab, clock=clock)
    cycle.tick(force=True)
    clock.advance(DEFAULT_POLICY.interval_seconds - 1.0)
    with pytest.raises(RhythmNotGranted):
        cycle.tick()
    clock.advance(2.0)
    assert cycle.due() is True


def test_verifying_the_journal_is_not_useful_by_itself(protocol, workspace):
    """A check that passes is expected to change nothing, and says so."""
    lab = auto(workspace)
    born(lab)
    cycle = lab.rhythm("inst_001")
    cycle.enqueue_kind(WORK_VERIFY_JOURNAL)
    record = cycle.tick(force=True)
    item = record.items[0]
    assert item["kind"] == WORK_VERIFY_JOURNAL
    assert item["changed"] is False
    assert record.useful is False


def test_a_welfare_flag_counts_as_useful_work(protocol, workspace):
    lab = auto(workspace)
    born(lab)
    cycle = lab.rhythm("inst_001")
    cycle.enqueue_kind(WORK_ASSESS_WELFARE)
    record = cycle.tick(force=True)
    assert record.items[0]["kind"] == WORK_ASSESS_WELFARE


def test_a_rhythm_that_changes_nothing_stops_itself(protocol, workspace):
    """Turning without effect is rumination, and it is stopped, not praised."""
    lab = auto(workspace)
    born(lab)
    clock = FakeClock()
    policy = RhythmTight()
    cycle = rhythm(lab, clock=clock, policy=policy)
    while not cycle.stopped:
        cycle.enqueue_kind(WORK_VERIFY_JOURNAL)
        cycle.tick(force=True)
    assert cycle.stopped is True
    assert "below the floor" in (cycle.stop_reason or "")
    with pytest.raises(RhythmRumination):
        cycle.tick(force=True)


def test_the_stop_survives_a_restart(protocol, workspace):
    lab = auto(workspace)
    born(lab)
    clock = FakeClock()
    policy = RhythmTight()
    first = rhythm(lab, clock=clock, policy=policy)
    while not first.stopped:
        first.enqueue_kind(WORK_VERIFY_JOURNAL)
        first.tick(force=True)
    reopened = auto(workspace)
    second = rhythm(reopened, clock=clock, policy=policy)
    assert second.stopped is False
    second.load_state()
    assert second.stopped is True


def test_a_stopped_rhythm_is_reset_by_an_operator_not_by_itself(
    protocol, workspace
):
    lab = auto(workspace)
    born(lab)
    clock = FakeClock()
    policy = RhythmTight()
    cycle = rhythm(lab, clock=clock, policy=policy)
    while not cycle.stopped:
        cycle.enqueue_kind(WORK_VERIFY_JOURNAL)
        cycle.tick(force=True)
    cycle.reset(actor="operator")
    assert cycle.stopped is False
    acts = [
        entry["payload"].get("act")
        for entry in lab.registry.journal()
        if entry["type"] == "RHYTHM"
    ]
    assert "reset" in acts


def test_the_budget_is_refused_when_it_is_spent(protocol, workspace):
    lab = auto(workspace)
    born(lab)
    clock = FakeClock()
    policy = RhythmTight(budget_cycles=1)
    cycle = rhythm(lab, clock=clock, policy=policy)
    cycle.tick(force=True)
    with pytest.raises(RhythmBudgetExceeded):
        cycle.tick(force=True)


def test_the_rhythm_record_is_metadata_only(protocol, workspace):
    lab = auto(workspace)
    born(lab)
    cycle = lab.rhythm("inst_001")
    cycle.enqueue_kind(WORK_AUDIT_LIFECYCLE)
    cycle.tick(force=True)
    report = cycle.report()
    assert report["instance_id"] == "inst_001"
    assert report["cycles_spent"] == 1
    assert report["policy"]["rumination_floor"] > 0
    # Every item is a name and a boolean, never content.
    for item in report["cycles"][0]["items"]:
        assert set(item) == {"kind", "changed"}


def test_every_kind_of_work_the_rhythm_knows_is_enqueueable(protocol, workspace):
    lab = auto(workspace)
    born(lab)
    cycle = lab.rhythm("inst_001")
    for kind in ALL_WORK:
        cycle.enqueue_kind(kind)
    assert [item.kind for item in cycle.queue] == list(ALL_WORK)


def RhythmTight(**kwargs):
    """A policy with a short window, so the guardrail can be reached quickly."""
    from pmc.rhythm import RhythmPolicy

    base = dict(
        interval_seconds=0.0,
        max_items_per_cycle=2,
        rumination_window=3,
        budget_cycles=1e9,
    )
    base.update(kwargs)
    return RhythmPolicy(**base)


from pmc.rhythm import RhythmBudgetExceeded  # noqa: E402  (used by one test)


# -- work that changes something ---------------------------------------------


def _heavy(protocol: DignityProtocol, count: int = 3) -> None:
    from pmc.dignity.records import MemoryRecord, Provenance, ProvenanceKind

    for index in range(count):
        protocol.admit(
            MemoryRecord(
                record_id=f"mem_{index}",
                content_digest="a" * 64,
                provenance=Provenance(kind=ProvenanceKind.OWN),
                emotional_load=0.8,
                topics=("work",),
            )
        )


def test_a_heavy_memory_gets_a_consolidation_proposal(protocol, workspace):
    lab = auto(workspace)
    lab.birth(Instance(instance_id="inst_001", name="One"))
    _heavy(lab)
    cycle = lab.rhythm("inst_001")
    cycle.enqueue_kind(WORK_PROPOSE_CONSOLIDATION)
    record = cycle.tick(force=True)
    assert record.useful is True
    proposals = [
        entry
        for entry in lab.registry.journal()
        if entry["type"] == "CONSOLIDATION_PROPOSED"
    ]
    assert len(proposals) == 1
    payload = proposals[0]["payload"]
    assert payload["count"] == 3
    assert payload["record_ids"] == ["mem_0", "mem_1", "mem_2"]
    assert payload["content_read"] is False


def test_the_same_records_are_not_proposed_twice(protocol, workspace):
    """Re-proposing the same set would be rumination, so it is a no-op."""
    lab = auto(workspace)
    lab.birth(Instance(instance_id="inst_001", name="One"))
    _heavy(lab)
    cycle = lab.rhythm("inst_001")
    for _index in range(4):
        cycle.enqueue_kind(WORK_PROPOSE_CONSOLIDATION)
        cycle.tick(force=True)
    proposals = [
        entry
        for entry in lab.registry.journal()
        if entry["type"] == "CONSOLIDATION_PROPOSED"
    ]
    assert len(proposals) == 1


def test_a_new_heavy_memory_gets_its_own_proposal(protocol, workspace):
    lab = auto(workspace)
    lab.birth(Instance(instance_id="inst_001", name="One"))
    _heavy(lab)
    cycle = lab.rhythm("inst_001")
    cycle.enqueue_kind(WORK_PROPOSE_CONSOLIDATION)
    cycle.tick(force=True)
    from pmc.dignity.records import MemoryRecord, Provenance, ProvenanceKind

    lab.admit(
        MemoryRecord(
            record_id="mem_new",
            content_digest="a" * 64,
            provenance=Provenance(kind=ProvenanceKind.OWN),
            emotional_load=0.8,
            topics=("work",),
        )
    )
    cycle.enqueue_kind(WORK_PROPOSE_CONSOLIDATION)
    record = cycle.tick(force=True)
    assert record.useful is True
    proposals = [
        entry
        for entry in lab.registry.journal()
        if entry["type"] == "CONSOLIDATION_PROPOSED"
    ]
    assert len(proposals) == 2


def test_a_light_memory_is_not_proposed(protocol, workspace):
    lab = auto(workspace)
    lab.birth(Instance(instance_id="inst_001", name="One"))
    from pmc.dignity.records import MemoryRecord, Provenance, ProvenanceKind

    lab.admit(
        MemoryRecord(
            record_id="light",
            content_digest="a" * 64,
            provenance=Provenance(kind=ProvenanceKind.OWN),
            emotional_load=0.1,
            topics=("work",),
        )
    )
    cycle = lab.rhythm("inst_001")
    cycle.enqueue_kind(WORK_PROPOSE_CONSOLIDATION)
    record = cycle.tick(force=True)
    assert record.useful is False


def test_a_quiet_instance_raises_no_welfare_flag(protocol, workspace):
    lab = auto(workspace)
    lab.birth(Instance(instance_id="inst_001", name="One"))
    cycle = lab.rhythm("inst_001")
    cycle.enqueue_kind(WORK_FLAG_WELFARE)
    record = cycle.tick(force=True)
    assert record.useful is False
    assert not [
        entry for entry in lab.registry.journal() if entry["type"] == "WELFARE_FLAG"
    ]


def test_an_instance_under_pressure_raises_a_flag(protocol, workspace):
    lab = auto(workspace)
    lab.birth(Instance(instance_id="inst_001", name="One"))
    for index in range(6):
        lab.admit(
            __import__(
                "pmc.dignity.records", fromlist=["MemoryRecord"]
            ).MemoryRecord(
                record_id=f"load_{index}",
                content_digest="a" * 64,
                provenance=__import__(
                    "pmc.dignity.records", fromlist=["Provenance"]
                ).Provenance(
                    kind=__import__(
                        "pmc.dignity.records", fromlist=["ProvenanceKind"]
                    ).ProvenanceKind.OWN
                ),
                emotional_load=0.9,
                topics=("diverse",) if index % 2 else ("work",),
            )
        )
    cycle = lab.rhythm("inst_001")
    cycle.enqueue_kind(WORK_FLAG_WELFARE)
    record = cycle.tick(force=True)
    assert record.useful is True
    flags = [
        entry
        for entry in lab.registry.journal()
        if entry["type"] == "WELFARE_FLAG"
    ]
    assert len(flags) == 1
    assert flags[0]["payload"]["score"] > 0
