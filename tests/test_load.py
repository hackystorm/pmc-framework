# SPDX-License-Identifier: GPL-3.0-only

"""Capacity accounting: quotas, inherited weight, and quarantine."""

from __future__ import annotations

import pytest

from pmc.dignity.load import Admission, LoadLedger, LoadPolicy
from pmc.dignity.records import MemoryRecord, Provenance, ProvenanceKind


def record(
    record_id: str,
    load: float,
    *,
    kind: ProvenanceKind = ProvenanceKind.OWN,
    topics: tuple[str, ...] = (),
) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        content_digest="d" * 64,
        provenance=Provenance(kind=kind),
        emotional_load=load,
        topics=topics,
    )


def test_a_memory_within_quota_is_admitted():
    ledger = LoadLedger()
    assert ledger.admit(record("m1", 0.4)) is Admission.ADMITTED
    assert ledger.active_load() == pytest.approx(0.4)


def test_inherited_memory_is_weighted_more_heavily():
    ledger = LoadLedger(
        LoadPolicy(total_quota=2.0, inherited_quota=2.0, topic_quota=1.0)
    )
    ledger.admit(record("m1", 0.5))
    ledger.admit(record("m2", 0.5, kind=ProvenanceKind.INHERITED))

    assert ledger.active_load() == pytest.approx(1.25)
    assert ledger.inherited_load() == pytest.approx(0.75)
    assert ledger.inherited_ratio() == pytest.approx(0.6)


def test_exceeding_the_total_quota_parks_instead_of_dropping():
    ledger = LoadLedger(LoadPolicy(total_quota=1.0, topic_quota=10.0, inherited_quota=10.0))
    ledger.admit(record("m1", 0.5))
    ledger.admit(record("m2", 0.5))
    outcome = ledger.admit(record("m3", 0.5))

    assert outcome is Admission.QUARANTINED_TOTAL
    assert ledger.quarantined_ids() == ["m3"]
    assert "m3" in ledger.active_ids() or ledger.quarantined_ids() == ["m3"]


def test_exceeding_the_inherited_quota_parks_the_inherited_memory():
    ledger = LoadLedger(
        LoadPolicy(total_quota=10.0, inherited_quota=0.5, topic_quota=10.0)
    )
    outcome = ledger.admit(record("m1", 0.5, kind=ProvenanceKind.INHERITED))
    assert outcome is Admission.QUARANTINED_INHERITED


def test_exceeding_the_topic_quota_parks_the_second_memory():
    ledger = LoadLedger(
        LoadPolicy(total_quota=10.0, inherited_quota=10.0, topic_quota=0.35)
    )
    assert ledger.admit(record("m1", 0.3, topics=("heavy",))) is Admission.ADMITTED
    outcome = ledger.admit(record("m2", 0.3, topics=("heavy",)))
    assert outcome is Admission.QUARANTINED_TOPIC


def test_quarantined_load_still_counts_toward_welfare_pressure():
    ledger = LoadLedger(
        LoadPolicy(total_quota=0.5, inherited_quota=10.0, topic_quota=10.0)
    )
    ledger.admit(record("m1", 0.5))
    ledger.admit(record("m2", 0.5))

    assert ledger.active_load() == pytest.approx(0.5)
    assert ledger.quarantined_load() == pytest.approx(0.5)
    assert ledger.welfare_load() == pytest.approx(0.75)
    assert ledger.quarantined_ratio() == pytest.approx(0.5)


def test_topic_dominance_reports_concentration():
    ledger = LoadLedger(
        LoadPolicy(total_quota=10.0, inherited_quota=10.0, topic_quota=10.0)
    )
    ledger.admit(record("m1", 0.6, topics=("heavy",)))
    ledger.admit(record("m2", 0.2, topics=("light",)))
    assert ledger.topic_dominance() == pytest.approx(0.75)


def test_release_and_update_change_the_accounting():
    ledger = LoadLedger()
    ledger.admit(record("m1", 0.5))
    ledger.update_load("m1", 0.1)
    assert ledger.active_load() == pytest.approx(0.1)

    ledger.release("m1")
    assert ledger.active_load() == pytest.approx(0.0)
    assert ledger.quarantined_ids() == ["m1"]


def test_invalid_policies_are_rejected():
    with pytest.raises(ValueError):
        LoadPolicy(total_quota=0.0).validate()
    with pytest.raises(ValueError):
        LoadPolicy(inherited_weight=0.5).validate()
    with pytest.raises(ValueError):
        LoadPolicy(quarantine_discount=2.0).validate()
