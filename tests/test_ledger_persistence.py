# SPDX-License-Identifier: GPL-3.0-only

"""The ledger remembers: persistence of digestion, access and capacity.

These tests cover the facts that only show up when a workspace is reopened.
A dignity layer that forgets what it recorded at shutdown has not recorded
anything, and the end-to-end run against a real loader is what surfaced that.
"""

from __future__ import annotations

import pytest

from pmc.dignity.digestion import DigestStrategy
from pmc.dignity.load import Admission, LoadLedger
from pmc.dignity.protocol import DignityProtocol
from pmc.dignity.records import (
    DigestState,
    Instance,
    MemoryRecord,
    Provenance,
    ProvenanceKind,
)


def make_record(record_id: str, load: float, *, topics=("theme",)) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        content_digest=(record_id * 64)[:64],
        provenance=Provenance(kind=ProvenanceKind.OWN),
        emotional_load=load,
        topics=topics,
    )


def test_readmitting_a_record_does_not_charge_it_twice():
    ledger = LoadLedger()
    record = make_record("m1", 0.4)

    assert ledger.admit(record) is Admission.ADMITTED
    assert ledger.admit(record) is Admission.ADMITTED

    assert ledger.active_load() == pytest.approx(0.4)


def test_readmitting_a_record_with_a_new_load_replaces_the_old_one():
    ledger = LoadLedger()
    record = make_record("m1", 0.4)
    ledger.admit(record)

    record.emotional_load = 0.2
    ledger.admit(record)

    assert ledger.active_load() == pytest.approx(0.2)


def test_digestion_is_written_to_the_ledger_and_survives_a_reload(tmp_path):
    protocol = DignityProtocol(tmp_path)
    protocol.birth(Instance(instance_id="inst_a", name="A"))
    protocol.admit(make_record("m1", 0.4), instance_id="inst_a")

    protocol.digest("m1", DigestStrategy.TRANSFORM, note="worked through")

    reloaded = DignityProtocol(tmp_path)
    stored = reloaded.record("m1")
    assert stored.emotional_load == pytest.approx(0.2)
    assert stored.digest_state is DigestState.DIGESTED
    assert stored.state_history[-1]["event"] == "digest"
    assert stored.state_history[-1]["note"] == "worked through"
    assert reloaded.ledger.active_load() == pytest.approx(0.2)


def test_access_counts_survive_a_reload(tmp_path):
    protocol = DignityProtocol(tmp_path)
    protocol.admit(make_record("m1", 0.4))

    protocol.touch("m1")
    protocol.touch("m1")

    reloaded = DignityProtocol(tmp_path)
    assert reloaded.record("m1").access_count == 2
    assert reloaded.record("m1").last_access_at is not None


def test_working_through_a_parked_memory_can_release_it(tmp_path):
    protocol = DignityProtocol(tmp_path)
    protocol.admit(make_record("m1", 0.35), instance_id="inst_a")
    protocol.admit(make_record("m2", 0.35), instance_id="inst_a")
    protocol.admit(make_record("m3", 0.35), instance_id="inst_a")

    # Two of the three do not fit the concentration quota and are parked.
    assert len(protocol.ledger.quarantined_ids()) == 2

    protocol.digest("m2", DigestStrategy.DISTILL)

    # A parked memory does not occupy capacity, so the concentration quota is
    # counted over what is actually held: 0.35 + 0.07 = 0.42 fits.
    assert "m2" not in protocol.ledger.quarantined_ids()
    assert "m3" in protocol.ledger.quarantined_ids()
    protocol.digest("m3", DigestStrategy.DISTILL)

    # 0.35 + 0.07 + 0.07 = 0.49 now fits, so nothing is parked any more.
    assert protocol.ledger.quarantined_ids() == []
    assert len(protocol.ledger.active_ids()) == 3


def test_a_refused_memory_stays_out_of_the_active_set(tmp_path):
    protocol = DignityProtocol(tmp_path)
    protocol.admit(make_record("m1", 0.4), instance_id="inst_a")

    protocol.digest("m1", DigestStrategy.REFUSE)

    assert protocol.ledger.active_ids() == []
    assert protocol.record("m1").digest_state is DigestState.REFUSED
    reloaded = DignityProtocol(tmp_path)
    assert reloaded.record("m1").digest_state is DigestState.REFUSED
