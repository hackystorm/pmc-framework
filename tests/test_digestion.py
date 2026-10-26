# SPDX-License-Identifier: GPL-3.0-only

"""Digestion, refusal, and rumination detection."""

from __future__ import annotations

import pytest

from pmc.dignity.digestion import (
    DigestionManager,
    DigestStrategy,
    RuminationDetector,
)
from pmc.dignity.load import LoadLedger
from pmc.dignity.records import DigestState, MemoryRecord, Provenance, ProvenanceKind


def make_record(load: float = 1.0, topics: tuple[str, ...] = ("theme",)) -> MemoryRecord:
    return MemoryRecord(
        record_id="m1",
        content_digest="d" * 64,
        provenance=Provenance(kind=ProvenanceKind.OWN),
        emotional_load=load,
        topics=topics,
    )


def test_distillation_drops_the_load_sharply():
    ledger = LoadLedger()
    manager = DigestionManager(ledger)
    record = make_record(1.0)
    ledger.admit(record)

    outcome = manager.digest(record, DigestStrategy.DISTILL)

    assert outcome.after_load == pytest.approx(0.2)
    assert outcome.state is DigestState.DIGESTED
    assert ledger.active_load() == pytest.approx(0.2)


def test_transformation_drops_the_load_moderately():
    ledger = LoadLedger()
    manager = DigestionManager(ledger)
    record = make_record(1.0)
    ledger.admit(record)

    outcome = manager.digest(record, DigestStrategy.TRANSFORM)
    assert outcome.after_load == pytest.approx(0.5)


def test_refusal_zeroes_the_load_and_releases_the_memory():
    ledger = LoadLedger()
    manager = DigestionManager(ledger)
    record = make_record(0.8)
    ledger.admit(record)

    outcome = manager.digest(record, DigestStrategy.REFUSE)

    assert outcome.after_load == 0.0
    assert outcome.state is DigestState.REFUSED
    assert ledger.active_load() == pytest.approx(0.0)
    assert ledger.quarantined_ids() == ["m1"]


def test_hold_keeps_the_memory_as_it_was():
    ledger = LoadLedger()
    manager = DigestionManager(ledger)
    record = make_record(0.8)
    ledger.admit(record)

    outcome = manager.digest(record, DigestStrategy.HOLD)
    assert outcome.after_load == pytest.approx(0.8)
    assert outcome.state is DigestState.RAW


def test_the_digestion_interface_has_no_content_parameter():
    import inspect

    signature = inspect.signature(DigestionManager.digest)
    assert "text" not in signature.parameters
    assert "content" not in signature.parameters


def test_rumination_alert_fires_at_the_threshold():
    detector = RuminationDetector(threshold=3)
    assert detector.observe("sig", record_id="m1", state_changed=False) is None
    assert detector.observe("sig", record_id="m1", state_changed=False) is None
    alert = detector.observe("sig", record_id="m1", state_changed=False)

    assert alert is not None
    assert alert.repetitions == 3
    assert len(detector.alerts) == 1


def test_a_state_change_resets_the_loop_counter():
    detector = RuminationDetector(threshold=3)
    detector.observe("sig", record_id="m1", state_changed=False)
    detector.observe("sig", record_id="m1", state_changed=False)
    assert detector.observe("sig", record_id="m1", state_changed=True) is None
    assert detector.observe("sig", record_id="m1", state_changed=False) is None


def test_touching_a_record_increments_its_access_count():
    ledger = LoadLedger()
    manager = DigestionManager(ledger)
    record = make_record(0.5)
    ledger.admit(record)

    manager.touch(record)
    manager.touch(record)

    assert record.access_count == 2
    assert record.last_access_at is not None
