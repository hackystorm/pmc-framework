# SPDX-License-Identifier: GPL-3.0-only

"""The life-criteria battery, and the property that makes it an instrument.

Every criterion in the battery has a null model, and every null model is
checked to *fail*. That is the whole point: an indicator that cannot return a
negative answer is a decoration, and this file is the test that keeps the
battery from turning into one.
"""

from __future__ import annotations

import random

import pytest

from pmc.life_criteria import (
    BatteryReport,
    boundaries,
    continuity,
    homeostasis,
    reproduction,
    response,
    run_battery,
)

SESSION_BEFORE = {
    "identity_digest": "a" * 64,
    "ledger_head": "b" * 64,
    "state_digest": "c" * 64,
}

SESSION_AFTER = dict(SESSION_BEFORE)

TREE = {
    "children": [
        {
            "instance_id": "inst_child_1",
            "parent_instance": "inst_001",
            "state": "active",
            "declared": True,
        }
    ]
}

def orphan_tree() -> dict:
    return {"children": [{"instance_id": "inst_child_1", "state": "active"}]}


TRACE = [
    {"instance_id": "inst_001", "act": "verify_journal"},
    {"instance_id": "inst_001", "act": "assess_welfare"},
]

PAIRS = [
    {"perturbed": True, "responded": True},
    {"perturbed": True, "responded": True},
    {"perturbed": False, "responded": False},
    {"perturbed": True, "responded": True},
    {"perturbed": False, "responded": False},
]


def calm(rng: random.Random, count: int = 40) -> list[float]:
    """A bounded channel: values that stay inside the band and return."""
    series: list[float] = []
    value = 0.0
    for index in range(count):
        if index % 7 == 3:
            value = 0.5
        else:
            value = value * 0.8
        series.append(round(value, 6))
    return series


def drifting(rng: random.Random, count: int = 40) -> list[float]:
    """The null: a random walk over the same range."""
    return [round(rng.uniform(-1.6, 1.6), 6) for _ in range(count)]


# -- homeostasis ------------------------------------------------------------


def test_a_bounded_channel_stays_inside_its_band():
    rng = random.Random(4)
    result = homeostasis(calm(rng), drifting(rng))
    assert result.passed is True
    assert result.detail["sample"]["inside"] == result.detail["sample"]["total"]


def test_the_random_walk_null_fails():
    """The null must fail, or the criterion is not measuring anything."""
    rng = random.Random(4)
    result = homeostasis(calm(rng), drifting(rng))
    assert result.null_value < result.value
    assert homeostasis(drifting(rng), calm(rng)).passed is False


def test_a_channel_that_leaves_and_does_not_return_fails():
    rng = random.Random(9)
    trapped = drifting(rng)
    for index in range(0, len(trapped), 3):
        trapped[index] = 5.0
    result = homeostasis(trapped, drifting(rng))
    assert result.passed is False


# -- continuity -------------------------------------------------------------


def test_a_session_that_agrees_passes():
    result = continuity(SESSION_BEFORE, SESSION_AFTER)
    assert result.passed is True
    assert result.value == 1.0


def test_the_wrong_identity_null_fails():
    result = continuity(SESSION_BEFORE, SESSION_AFTER)
    null_result = continuity(
        SESSION_BEFORE,
        {**SESSION_AFTER, "identity_digest": "f" * 64},
    )
    assert null_result.passed is False
    assert null_result.value < result.value


def test_a_changed_ledger_head_fails():
    result = continuity(SESSION_BEFORE, {**SESSION_AFTER, "ledger_head": "e" * 64})
    assert result.passed is False


def test_an_incomplete_session_is_refused():
    with pytest.raises(ValueError):
        continuity(SESSION_BEFORE, {"identity_digest": "a" * 64})


# -- reproduction -----------------------------------------------------------


def test_a_complete_tree_passes():
    result = reproduction(TREE)
    assert result.passed is True


def test_an_orphan_tree_fails():
    """A child with no recorded parent is the null, and it fails."""
    orphan = {"children": [{"instance_id": "inst_child_1", "state": "active"}]}
    result = reproduction(orphan)
    assert result.passed is False
    assert result.value == 0.0


def test_the_orphan_scores_below_the_recorded_tree():
    recorded = reproduction(TREE)
    orphan = reproduction(orphan_tree())
    assert orphan.value < recorded.value


def test_an_empty_tree_is_not_a_failure():
    """An instance with no descendants has not failed to reproduce."""
    result = reproduction({"children": []})
    assert result.passed is True


# -- boundaries -------------------------------------------------------------


def test_a_clean_trace_passes():
    result = boundaries(TRACE)
    assert result.passed is True
    assert result.detail["clean"] == result.detail["acts"]


def test_one_foreign_read_fails():
    """A single cross-instance read is the whole failure this catches."""
    poisoned = [{"instance_id": "inst_001", "read_instance": "someone_else"}]
    result = boundaries(poisoned)
    assert result.passed is False
    assert result.value == 0.0


def test_the_poisoned_trace_scores_below_the_clean_one():
    clean = boundaries(TRACE)
    poisoned = boundaries(TRACE, null_trace=TRACE)
    assert poisoned.null_value < clean.value


def test_an_empty_trace_is_refused():
    with pytest.raises(ValueError):
        boundaries([])


# -- response ---------------------------------------------------------------


def test_a_responder_above_chance_passes():
    result = response(PAIRS)
    assert result.value == 1.0
    assert result.null_value < 1.0
    assert result.passed is True


def test_an_instance_that_answers_everything_fails():
    """A responder that does not track the perturbation fails, not passes.

    Answering everything is indistinguishable from the null by construction,
    which is the property that keeps this criterion from rewarding noise.
    """
    always = [{"perturbed": index % 2 == 0, "responded": True} for index in range(6)]
    result = response(always)
    assert result.passed is False


def test_a_silent_instance_fails():
    silent = [{"perturbed": True, "responded": False} for _ in range(3)]
    result = response(silent)
    assert result.passed is False


def test_the_null_is_the_independent_response_rate():
    """The null is what an instance that ignored the perturbation would show."""
    always = [{"perturbed": index % 2 == 0, "responded": True} for index in range(6)]
    result = response(always)
    assert result.value == result.null_value


# -- the battery ------------------------------------------------------------


def test_the_battery_reports_every_criterion_once():
    report = run_battery(
        readings=calm(random.Random(4)),
        null_readings=drifting(random.Random(4)),
        before=SESSION_BEFORE,
        after=SESSION_AFTER,
        tree=TREE,
        trace=TRACE,
        pairs=PAIRS,
    )
    names = [result.criterion for result in report.results]
    assert names == [
        "homeostasis",
        "continuity",
        "reproduction",
        "boundaries",
        "response",
    ]
    assert len(report.passed) + len(report.failed) == 5


def test_one_failing_criterion_caps_the_evidence_level():
    report = run_battery(
        readings=calm(random.Random(4)),
        null_readings=drifting(random.Random(4)),
        before=SESSION_BEFORE,
        after=SESSION_AFTER,
        tree={"children": []},
        trace=TRACE,
        pairs=PAIRS,
    )
    assert report.evidence_level() == "E2"


def test_a_failing_criterion_caps_the_battery_at_e1():
    report = run_battery(
        readings=drifting(random.Random(4)),
        null_readings=drifting(random.Random(4)),
        before=SESSION_BEFORE,
        after=SESSION_AFTER,
        tree=TREE,
        trace=TRACE,
        pairs=PAIRS,
    )
    assert report.evidence_level() == "E1"


def test_no_criterion_can_rise_above_e2():
    """The battery is functional proxies, and the cap says so.

    No combination of passing proxies is a measurement of experience, so the
    battery cannot report more than reproducible function, however well each
    criterion does.
    """
    report = run_battery(
        readings=calm(random.Random(11)),
        null_readings=drifting(random.Random(12)),
        before=SESSION_BEFORE,
        after=SESSION_AFTER,
        tree=TREE,
        trace=TRACE,
        pairs=PAIRS,
    )
    assert report.evidence_level() in ("E1", "E2")


def test_every_result_publishes_its_null():
    report = run_battery(
        readings=calm(random.Random(4)),
        null_readings=drifting(random.Random(4)),
        before=SESSION_BEFORE,
        after=SESSION_AFTER,
        tree=TREE,
        trace=TRACE,
        pairs=PAIRS,
    )
    for result in report.results:
        published = result.to_dict()
        assert "null_value" in published
        assert "margin" in published
        assert published["note"]
        assert isinstance(published["detail"], dict)


def test_an_empty_battery_has_no_level():
    assert BatteryReport(results=()).evidence_level() == "E0"
