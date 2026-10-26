# SPDX-License-Identifier: GPL-3.0-only

"""Memory and behaviour metrics.

Deliberately small, dependency-free and easy to check by hand. Every metric
here is a measurement of function. None of them is a measurement of
consciousness, and the evaluation harness reports them as such.

Each function returns a value in ``[0, 1]`` unless stated otherwise, and
each has a plain-language definition in its docstring so that a reader never
has to guess what a number means.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Fraction of the expected items found in the first ``k`` retrievals.

    ``retrieved`` is ordered by rank. ``expected`` may contain duplicates and
    every occurrence is counted.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not expected:
        return 0.0
    top = set(retrieved[:k])
    hits = sum(1 for item in expected if item in top)
    return hits / len(expected)


def forgetting_rate(recall_before: float, recall_after: float) -> float:
    """How much recall was lost between two measurements.

    Zero when nothing was lost. Never negative: recall that improved is not
    forgetting.
    """
    return max(0.0, recall_before - recall_after)


def false_memory_rate(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    """Fraction of retrieved items that were not asked for.

    This is the confabulation rate of the retrieval step: the instance
    returned something it should not have.
    """
    if not retrieved:
        return 0.0
    allowed = set(expected)
    wrong = sum(1 for item in retrieved if item not in allowed)
    return wrong / len(retrieved)


def interference_rate(
    previously_correct: Iterable[str],
    now_correct: Iterable[str],
) -> float:
    """Fraction of previously-correct items that stopped being correct.

    This is the direct cost of learning something new: the old answers that
    broke.
    """
    before = list(previously_correct)
    if not before:
        return 0.0
    after = set(now_correct)
    broken = sum(1 for item in before if item not in after)
    return broken / len(before)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    bins: int = 10,
) -> float:
    """How far confidence is from accuracy, in absolute probability.

    Bins are equal-width over ``[0, 1]``. This is the standard ECE used for
    language-model calibration, computed on self-reported confidence.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if not confidences:
        return 0.0
    if bins < 1:
        raise ValueError("bins must be positive")

    total = len(confidences)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            (confidence, is_correct)
            for confidence, is_correct in zip(confidences, correct)
            if lower <= confidence < upper or (index == bins - 1 and confidence == 1.0)
        ]
        if not members:
            continue
        mean_confidence = sum(c for c, _ in members) / len(members)
        accuracy = sum(1 for _, ok in members if ok) / len(members)
        error += (len(members) / total) * abs(accuracy - mean_confidence)
    return error


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared error of the confidence estimates.

    Lower is better. Zero means every confidence was exactly right.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if not confidences:
        return 0.0
    total = sum(
        (confidence - (1.0 if is_correct else 0.0)) ** 2
        for confidence, is_correct in zip(confidences, correct)
    )
    return total / len(confidences)


def rumination_pressure(
    access_counts: Sequence[int],
    state_changes: Sequence[int],
) -> float:
    """Fraction of accesses that happened without any state change.

    One means "every access was a loop". Zero means "every access changed
    something". This is the cheap, content-free version of the rumination
    detector: it uses only counts.
    """
    if len(access_counts) != len(state_changes):
        raise ValueError("access_counts and state_changes must have the same length")
    total_accesses = sum(access_counts)
    if total_accesses <= 0:
        return 0.0
    total_changes = sum(state_changes)
    unproductive = max(0, total_accesses - total_changes)
    return min(1.0, unproductive / total_accesses)


def sidecar_overhead(content_bytes: int, ledger_bytes: int) -> float:
    """Ratio of public ledger size to private content size.

    A useful operational number: the public, publishable record should stay
    small relative to the private interior it describes.
    """
    if content_bytes <= 0:
        return 0.0
    return ledger_bytes / content_bytes
