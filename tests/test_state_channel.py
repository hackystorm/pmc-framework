# SPDX-License-Identifier: GPL-3.0-only

"""The derived aggregate state channel, and the four bounds that make it safe.

The interesting tests here are not that a token holds numbers. They are that
a token *cannot* hold anything else, that governance cannot reach it, and that
a reading can be re-run to the same answer.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from pmc.dignity.load import LoadLedger, LoadPolicy
from pmc.dignity.records import MemoryRecord, Provenance, ProvenanceKind
from pmc.state_channel import (
    CHANNEL_FIELDS,
    DEFAULT_HISTORY,
    ChannelError,
    StateChannel,
    StateToken,
    bounded_series,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "pmc"


def record(record_id: str, load: float, topics=("work",)) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        content_digest="d" * 64,
        provenance=Provenance(kind=ProvenanceKind.OWN),
        emotional_load=load,
        topics=topics,
    )


def ledger_with(*records: MemoryRecord) -> LoadLedger:
    ledger = LoadLedger()
    for item in records:
        ledger.admit(item)
    return ledger


# -- bounds -----------------------------------------------------------------


def test_a_token_is_bounded_on_construction():
    token = StateToken(pressure=3.0, inheritance=-2.0, dominance=0.5, churn=0.5)
    assert token.pressure == 1.0
    assert token.inheritance == 0.0
    assert token.dominance == 0.5


def test_a_non_finite_reading_is_refused():
    with pytest.raises(ChannelError):
        StateToken(pressure=float("nan"))


def test_every_scalar_is_inside_the_unit_interval():
    token = StateToken(pressure=0.5, inheritance=0.25, dominance=0.75, churn=0.125)
    for name in CHANNEL_FIELDS:
        value = getattr(token, name)
        assert 0.0 <= value <= 1.0


def test_the_aggregate_is_bounded_too():
    token = StateToken(pressure=1.0, inheritance=1.0, dominance=1.0, churn=1.0)
    assert token.aggregate() == 1.0
    empty = StateToken()
    assert empty.aggregate() == 0.0


def test_the_distance_is_bounded():
    one = StateToken(pressure=1.0, inheritance=1.0, dominance=1.0, churn=1.0)
    zero = StateToken()
    assert one.distance(zero) == 1.0
    assert one.distance(one) == 0.0


# -- non-identification -----------------------------------------------------


def test_a_token_has_no_string_field():
    """The type cannot express a name, a label or a digest.

    A field that *could* hold one eventually would, so the enforcement is in
    the dataclass rather than in a validator that could be relaxed.
    """
    for name, value in inspect.getmembers(StateToken()):
        if name.startswith("_") or inspect.ismethod(value) or inspect.isfunction(value):
            continue
        assert not isinstance(value, str), f"{name} is a string and could carry a label"


def test_the_token_serialises_to_numbers_only():
    published = StateToken(pressure=0.5).to_dict()
    assert set(published) == set(CHANNEL_FIELDS)
    for value in published.values():
        assert isinstance(value, float)


def test_an_unknown_series_is_refused():
    channel = StateChannel()
    channel.record(StateToken(pressure=0.5))
    with pytest.raises(ChannelError):
        channel.series("mood")


def test_the_channel_refuses_anything_that_is_not_a_token():
    channel = StateChannel()
    with pytest.raises(ChannelError):
        channel.record({"pressure": 0.5})  # type: ignore[arg-type]


# -- determinism -------------------------------------------------------------


def test_the_same_ledger_gives_the_same_token():
    first = StateToken.from_ledger(ledger_with(record("m1", 0.3)))
    second = StateToken.from_ledger(ledger_with(record("m1", 0.3)))
    assert first == second
    assert first.to_dict() == second.to_dict()


def test_a_series_is_reproducible():
    ledger = ledger_with(record("m1", 0.3), record("m2", 0.1, ("admin",)))
    one = StateChannel()
    one.record_from_ledger(ledger)
    two = StateChannel()
    two.record_from_ledger(ledger)
    assert one.series() == two.series()


# -- history -----------------------------------------------------------------


def test_the_channel_keeps_a_bounded_history():
    channel = StateChannel(history=4)
    for index in range(10):
        channel.record(StateToken(pressure=index / 10.0))
    assert len(channel.tokens) == 4
    assert channel.series("pressure") == [0.6, 0.7, 0.8, 0.9]
    # The aggregate divides by the number of fields, so it stays bounded.
    assert channel.series() == [0.15, 0.175, 0.2, 0.225]


def test_the_default_history_is_bounded():
    channel = StateChannel()
    assert channel.history_limit == DEFAULT_HISTORY


def test_a_non_positive_history_is_refused():
    with pytest.raises(ChannelError):
        StateChannel(history=0)


# -- the token as a reading --------------------------------------------------


def test_a_token_from_a_quiet_ledger_sits_low():
    token = StateToken.from_ledger(ledger_with(record("m1", 0.1)))
    assert token.aggregate() < 0.5


def test_a_token_from_a_loaded_ledger_sits_higher():
    quiet = StateToken.from_ledger(ledger_with(record("m1", 0.1)))
    loaded = StateToken.from_ledger(
        ledger_with(record("m1", 0.9), record("m2", 0.8), record("m3", 0.85))
    )
    assert loaded.pressure > quiet.pressure


def test_the_mean_drift_is_bounded_and_reported():
    channel = StateChannel()
    channel.record(StateToken(pressure=0.0))
    channel.record(StateToken(pressure=1.0))
    channel.record(StateToken(pressure=0.0))
    assert 0.0 <= channel.mean_drift() <= 1.0
    report = channel.report()
    assert report["count"] == 3
    assert len(report["series_aggregate"]) == 3


def test_an_empty_channel_reports_nothing_rather_than_failing():
    channel = StateChannel()
    report = channel.report()
    assert report["count"] == 0
    assert report["mean"] == 0.0
    assert report["series_aggregate"] == []


# -- the structural bound ----------------------------------------------------


def test_no_dignity_module_imports_the_state_channel():
    """Governance cannot reach the channel, even by accident.

    This is the structural half of D-17. The welfare assessor reads counts and
    quotients; it does not read states, and the import graph is what keeps that
    true as the modules grow.
    """
    for path in (SRC / "dignity").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "state_channel" not in text, f"{path.name} reaches into the channel"


def test_the_channel_imports_the_ledger_and_nothing_that_decides():
    """The channel observes; it does not decide, and its imports show it."""
    source = inspect.getsource(StateChannel)
    assert "def " in source
    for forbidden in ("SurvivalDrive", "DignityProtocol", "DreamRunner"):
        assert forbidden not in source, f"the channel references {forbidden}"


def test_the_channel_is_not_in_the_prompt_path():
    """The channel never reaches an opening prompt or a runtime readout."""
    text = (SRC / "companion" / "templates.py").read_text(encoding="utf-8")
    assert "state_channel" not in text
    runtime = SRC / "runtime"
    for path in runtime.glob("*.py"):
        assert "state_channel" not in path.read_text(encoding="utf-8")


# -- bounded_series ----------------------------------------------------------


def test_a_series_is_clamped_and_refuses_the_unbounded():
    assert bounded_series([0.5, 2.0, -1.0]) == [0.5, 1.0, 0.0]
    with pytest.raises(ChannelError):
        bounded_series([float("inf")])
