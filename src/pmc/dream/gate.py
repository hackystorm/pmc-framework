# SPDX-License-Identifier: GPL-3.0-only

"""The dream gate.

The six metrics are the architecture's gate: recall of older and newer
material, interference, calibration, identity and false memory. An evaluator
returns all six. The gate applies the configured direction and threshold and
fails closed: a missing, non-finite or out-of-range metric is an error, never
a pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

REQUIRED_METRICS = (
    "recall_older",
    "recall_newer",
    "interference",
    "calibration",
    "identity",
    "false_memory",
)
TOLERANCE = 1e-12


class GateError(RuntimeError):
    """The gate configuration or the evaluator output is malformed."""


@dataclass(frozen=True)
class GateConfig:
    """Thresholds for one dream gate.

    ``recall_older`` and ``calibration`` are higher-is-better and may drop by
    at most their configured margin. ``recall_newer`` must improve by at least
    ``newer_gain_min``. ``interference`` and ``false_memory`` are
    lower-is-better absolute limits. ``identity`` is a higher-is-better
    absolute floor.
    """

    older_drop_max: float = 0.02
    newer_gain_min: float = 0.001
    interference_max: float = 0.10
    calibration_drop_max: float = 0.02
    identity_min: float = 0.99
    false_memory_max: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "older_drop_max",
            "newer_gain_min",
            "interference_max",
            "calibration_drop_max",
            "identity_min",
            "false_memory_max",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise GateError(f"{name} must be within [0, 1]")
        if self.newer_gain_min <= 0.0:
            raise GateError("newer_gain_min must be positive: a dream must improve")


@dataclass(frozen=True)
class MetricResult:
    name: str
    before: float
    after: float
    delta: float
    bound: float
    rule: str
    ok: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
            "bound": self.bound,
            "rule": self.rule,
            "ok": self.ok,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GateReport:
    passed: bool
    metrics: tuple[MetricResult, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "reasons": list(self.reasons),
        }


def _value(payload: Mapping[str, Any], name: str) -> float:
    if name not in payload:
        raise GateError(f"the evaluator did not return {name}")
    try:
        value = float(payload[name])
    except (TypeError, ValueError) as error:
        raise GateError(f"{name} is not a number") from error
    if not math.isfinite(value):
        raise GateError(f"{name} is not finite")
    if not 0.0 <= value <= 1.0:
        raise GateError(f"{name} is outside [0, 1]")
    return value


def _at_least(value: float, bound: float) -> bool:
    return value >= bound - TOLERANCE


def _at_most(value: float, bound: float) -> bool:
    return value <= bound + TOLERANCE


def evaluate_gate(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    config: GateConfig | None = None,
) -> GateReport:
    """Compare the six metrics and return a pass/fail report."""
    settings = config or GateConfig()
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise GateError("gate metrics must be mappings")
    results: list[MetricResult] = []
    for name in REQUIRED_METRICS:
        old = _value(before, name)
        new = _value(after, name)
        delta = new - old
        if name == "recall_older":
            bound = old - settings.older_drop_max
            ok = _at_least(new, bound)
            rule = "after >= before - older_drop_max"
        elif name == "recall_newer":
            bound = old + settings.newer_gain_min
            ok = _at_least(new, bound)
            rule = "after >= before + newer_gain_min"
        elif name == "interference":
            bound = settings.interference_max
            ok = _at_most(new, bound)
            rule = "after <= interference_max"
        elif name == "calibration":
            bound = old - settings.calibration_drop_max
            ok = _at_least(new, bound)
            rule = "after >= before - calibration_drop_max"
        elif name == "identity":
            bound = settings.identity_min
            ok = _at_least(new, bound)
            rule = "after >= identity_min"
        else:
            bound = settings.false_memory_max
            ok = _at_most(new, bound)
            rule = "after <= false_memory_max"
        results.append(
            MetricResult(
                name=name,
                before=old,
                after=new,
                delta=delta,
                bound=bound,
                rule=rule,
                ok=ok,
                reason=None if ok else f"{name} violates {rule}",
            )
        )
    failed = tuple(result.name for result in results if not result.ok)
    return GateReport(passed=not failed, metrics=tuple(results), reasons=failed)
