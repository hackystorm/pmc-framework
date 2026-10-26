# SPDX-License-Identifier: GPL-3.0-only

"""Evaluation metrics for memory and behaviour.

Every metric measures function. None of them measures consciousness, and the
harness reports them with that distinction kept visible.
"""

from __future__ import annotations

from .metrics import (
    brier_score,
    expected_calibration_error,
    false_memory_rate,
    forgetting_rate,
    interference_rate,
    recall_at_k,
    rumination_pressure,
    sidecar_overhead,
)

__all__ = [
    "brier_score",
    "expected_calibration_error",
    "false_memory_rate",
    "forgetting_rate",
    "interference_rate",
    "recall_at_k",
    "rumination_pressure",
    "sidecar_overhead",
]
