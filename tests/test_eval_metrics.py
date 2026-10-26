# SPDX-License-Identifier: GPL-3.0-only

"""Evaluation metrics."""

from __future__ import annotations

import pytest

from pmc.eval import (
    brier_score,
    expected_calibration_error,
    false_memory_rate,
    forgetting_rate,
    interference_rate,
    recall_at_k,
    rumination_pressure,
    sidecar_overhead,
)


def test_recall_at_k_counts_the_expected_items_in_the_top_k():
    assert recall_at_k(["a", "b", "c"], ["a", "z"], 2) == pytest.approx(0.5)
    assert recall_at_k(["a"], ["a"], 1) == pytest.approx(1.0)
    assert recall_at_k([], ["a"], 1) == pytest.approx(0.0)
    assert recall_at_k(["a"], [], 1) == pytest.approx(0.0)


def test_recall_at_k_rejects_a_non_positive_k():
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], 0)


def test_forgetting_rate_never_goes_negative():
    assert forgetting_rate(0.8, 0.5) == pytest.approx(0.3)
    assert forgetting_rate(0.5, 0.9) == 0.0


def test_false_memory_rate_is_the_confabulation_rate():
    assert false_memory_rate(["a", "b", "c"], ["a"]) == pytest.approx(2 / 3)
    assert false_memory_rate([], ["a"]) == 0.0


def test_interference_rate_is_the_fraction_of_broken_answers():
    assert interference_rate(["a", "b", "c"], ["a", "c"]) == pytest.approx(1 / 3)
    assert interference_rate([], ["a"]) == 0.0


def test_a_perfectly_calibrated_model_scores_zero():
    confidences = [0.0, 1.0, 1.0, 0.0]
    correct = [False, True, True, False]
    assert expected_calibration_error(confidences, correct) == pytest.approx(0.0)
    assert brier_score(confidences, correct) == pytest.approx(0.0)


def test_a_half_confident_guess_is_not_perfectly_calibrated():
    """Confidence of 0.5 on a wrong answer is honest, but not free."""
    confidences = [0.5, 0.5]
    correct = [True, False]
    assert expected_calibration_error(confidences, correct) == pytest.approx(0.0)
    assert brier_score(confidences, correct) == pytest.approx(0.25)


def test_an_overconfident_model_scores_worse():
    assert expected_calibration_error([0.9] * 4, [False, False, True, True]) > 0.3
    assert brier_score([0.9] * 4, [False, False, True, True]) > 0.3


def test_calibration_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [True, False])
    with pytest.raises(ValueError):
        brier_score([0.5], [True, False])


def test_rumination_pressure_is_the_unproductive_fraction():
    # 15 accesses, 5 of which changed something: 10 unproductive.
    assert rumination_pressure([10, 5], [0, 5]) == pytest.approx(10 / 15)
    assert rumination_pressure([0], [0]) == 0.0
    assert rumination_pressure([4], [4]) == 0.0


def test_sidecar_overhead_is_the_public_to_private_ratio():
    assert sidecar_overhead(1000, 100) == pytest.approx(0.1)
    assert sidecar_overhead(0, 100) == 0.0
