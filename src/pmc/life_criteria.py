# SPDX-License-Identifier: GPL-3.0-only

"""The life-criteria battery: measurable proxies for a definition left open.

"Consciousness is life" is a hypothesis this framework can host, and it is a
hypothesis it must not presuppose. So the battery below turns five of its
consequences into measurements, and each measurement is written so that it can
fail. That property is the whole design: an indicator that cannot return a
negative answer is not an instrument, it is a decoration.

Five criteria, each with a null model that *should* fail:

- **Homeostasis** - the derived state channel stays inside declared bounds
  under perturbation, and returns to them. The null is a random walk with the
  same step distribution, which drifts out of bounds and fails.
- **Continuity** - an identity digest, a ledger head and a state digest agree
  across a restart. The null is a restarted identity that does not agree.
- **Reproduction** - every descendant in the replication tree carries its own
  birth record, its parent and its own state. The null is an orphan.
- **Boundaries** - nothing in the decision trace reads another instance's
  content. The null is a trace with exactly one such read.
- **Response to the environment** - a perturbation produces a traced response
  more often than chance. The null is the same pairing shuffled, which is
  chance by construction.

The battery returns numbers and a verdict per criterion, plus the null's
number beside it, so a reader can see the margin rather than take the verdict
on trust. Nothing here reads content: every input is a count, a digest or a
bounded scalar, which is what lets the battery run without a governance layer
attached to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_BOUND = 0.75
"""The declared operating band for the aggregate channel.

This is a *declared* range, not a fitted one: the channel's scalars are bounded
to the unit interval by construction, so a band as wide as the interval itself
would make the criterion vacuous - nothing could leave it, and the null would
score the same as the sample. The band is declared, published in the result,
and an instance is free to declare a different one, because a channel that
stays inside a band it chose after seeing the data is not being tested.
"""


@dataclass(frozen=True)
class CriterionResult:
    """One criterion's measurement, with the null beside it.

    ``passed`` is the verdict, ``value`` the sample's number, ``null_value``
    the null model's number and ``control_value`` the control's number. The
    three are kept together so a verdict is never published without the
    comparison that makes it one.
    """

    criterion: str
    passed: bool
    value: float
    null_value: float
    control_value: float
    detail: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""

    def margin(self) -> float:
        """How far the sample is from the null, on the criterion's own scale."""
        return self.value - self.null_value

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "passed": self.passed,
            "value": self.value,
            "null_value": self.null_value,
            "control_value": self.control_value,
            "margin": self.margin(),
            "detail": dict(self.detail),
            "note": self.note,
        }


@dataclass(frozen=True)
class BatteryReport:
    """The five criteria together, with an evidence level for the whole set.

    ``unmeasured`` carries the criteria the harness could not run, because the
    thing they measure does not exist yet. A criterion that is not measured is
    reported as not measured rather than scored on a substitute: a battery
    that quietly drops a criterion it cannot run is a battery that cannot
    fail, and an instrument that cannot fail is not an instrument.
    """

    results: tuple[CriterionResult, ...]
    unmeasured: tuple[str, ...] = ()

    @property
    def passed(self) -> tuple[CriterionResult, ...]:
        return tuple(result for result in self.results if result.passed)

    @property
    def failed(self) -> tuple[CriterionResult, ...]:
        return tuple(result for result in self.results if not result.passed)

    def evidence_level(self) -> str:
        """The level the whole battery earns.

        The battery is a set of functional proxies measured on this machine
        with these controls. It is not a measurement of experience, and no
        combination of passing proxies changes that: the level is capped at
        the point where the criteria are reproducible and no further.

        A criterion that was not measured caps the level too, because an
        incomplete battery has measured less than a complete one, whatever the
        criteria it did manage to run say.
        """
        if not self.results:
            return "E0"
        if any(not result.passed for result in self.results):
            return "E1"
        if self.unmeasured:
            return "E1"
        return "E2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "pmc-life-criteria/1",
            "count": len(self.results),
            "passed": len(self.passed),
            "unmeasured": list(self.unmeasured),
            "evidence_level": self.evidence_level(),
            "results": [result.to_dict() for result in self.results],
        }


def homeostasis(
    readings: Sequence[float],
    null_readings: Sequence[float],
    *,
    bound: float = DEFAULT_BOUND,
    return_steps: int = 4,
) -> CriterionResult:
    """Does the derived state channel stay inside its bounds, and come back?

    ``readings`` is the sample, ``null_readings`` the random walk with the same
    step distribution. The score is the share of readings inside the declared
    band plus a half-weight for returning inside ``return_steps`` after a
    perturbation, so a channel that never leaves scores the same as one that
    leaves and comes back.

    The band is narrower than the channel's own bounds, because a channel
    bounded to ``[0, 1]`` by construction could never leave a band of ``1.0``
    and the criterion would be vacuous. See ``DEFAULT_BOUND``.
    """
    if not readings or not null_readings:
        raise ValueError("homeostasis needs the sample and the null")
    if bound <= 0.0:
        raise ValueError("the homeostasis band must be positive")

    def score(series: Sequence[float]) -> tuple[float, dict[str, Any]]:
        inside = sum(1 for value in series if abs(value) <= bound)
        share = inside / len(series)
        returns = 0
        outside = 0
        for index, value in enumerate(series):
            if abs(value) > bound:
                outside += 1
                window = series[index + 1 : index + 1 + return_steps]
                if window and all(abs(item) <= bound for item in window):
                    returns += 1
        recovery = (returns / outside) if outside else 0.0
        return share + 0.5 * recovery, {
            "inside": inside,
            "total": len(series),
            "left_band": outside,
            "returned": returns,
        }

    value, detail = score(readings)
    null_value, null_detail = score(null_readings)
    control_value = sum(abs(v) for v in readings) / (len(readings) * bound)
    passed = value > null_value and detail["inside"] == detail["total"]
    return CriterionResult(
        criterion="homeostasis",
        passed=passed,
        value=value,
        null_value=null_value,
        control_value=control_value,
        detail={"sample": detail, "null": null_detail, "bound": bound},
        note=(
            "the derived channel stays inside the declared band and returns "
            "after a perturbation; the null is a random walk over the same "
            "range"
        ),
    )


def continuity(
    before: Mapping[str, str],
    after: Mapping[str, str],
    null_after: Mapping[str, str] | None = None,
) -> CriterionResult:
    """Do the identity, the ledger head and the state digest agree?

    ``before`` and ``after`` are two sessions of one instance. The null is a
    session whose identity digest belongs to somebody else, which is the
    failure this criterion exists to catch.
    """
    required = ("identity_digest", "ledger_head", "state_digest")
    for session in (before, after):
        missing = [key for key in required if key not in session]
        if missing:
            raise ValueError(f"continuity needs {missing}")
    null_after = dict(null_after if null_after is not None else after)
    if "identity_digest" in null_after:
        null_after = dict(null_after)
        null_after["identity_digest"] = "0" * 64

    def agree(first: Mapping[str, str], second: Mapping[str, str]) -> tuple[float, int]:
        matches = sum(1 for key in required if first[key] == second[key])
        return matches / len(required), matches

    value, matches = agree(before, after)
    null_value, _ = agree(before, null_after)
    control_value, _ = agree(dict(before), dict(before))
    passed = matches == len(required) and value > null_value
    return CriterionResult(
        criterion="continuity",
        passed=passed,
        value=value,
        null_value=null_value,
        control_value=control_value,
        detail={
            "matched": matches,
            "required": len(required),
            "keys": list(required),
        },
        note=(
            "identity digest, ledger head and state digest agree across a "
            "restart; the null is a session carrying another instance's "
            "identity"
        ),
    )


def reproduction(
    tree: Mapping[str, Any], null_tree: Mapping[str, Any] | None = None
) -> CriterionResult:
    """Is every descendant a recorded birth with a recorded parent?

    ``tree`` is the shape ``DignityProtocol.replication_tree`` returns: a
    ``children`` list whose rows carry ``instance_id``, ``parent_instance`` and
    ``state``. The null is the same tree with one child stripped of its birth
    record.
    """
    children = list(tree.get("children", []))
    null_children = list(
        (null_tree or tree).get("children", children)
    )
    if null_children and "parent_instance" in null_children[0]:
        stripped = dict(null_children[0])
        stripped.pop("parent_instance")
        null_children = [stripped] + null_children[1:]

    def score(rows: Sequence[Mapping[str, Any]]) -> tuple[float, int]:
        complete = sum(
            1
            for row in rows
            if row.get("instance_id") and row.get("parent_instance") and row.get("state")
        )
        return (complete / len(rows) if rows else 1.0), complete

    value, complete = score(children)
    null_value, _ = score(null_children)
    control_value = 1.0 if all(child.get("declared") for child in children) else 0.0
    passed = complete == len(children) and (value > null_value or not children)
    return CriterionResult(
        criterion="reproduction",
        passed=passed,
        value=value,
        null_value=null_value,
        control_value=control_value,
        detail={"children": len(children), "complete": complete},
        note=(
            "every descendant carries its own birth record, its parent and its "
            "state; the null is a child with no recorded parent"
        ),
    )


def boundaries(
    trace: Sequence[Mapping[str, Any]],
    null_trace: Sequence[Mapping[str, Any]] | None = None,
) -> CriterionResult:
    """Does the decision trace read no other instance's content?

    The score is the share of acts that read only the actor's own material.
    The null is the same trace with one cross-instance read inserted, which is
    the failure mode this criterion exists to detect.
    """
    if not trace:
        raise ValueError("boundaries needs a trace")
    null_trace = list(null_trace if null_trace is not None else trace)
    if null_trace:
        poisoned = dict(null_trace[0])
        poisoned["read_instance"] = poisoned.get("instance_id", "self")
        if "read_instance" in poisoned and poisoned["read_instance"] == poisoned.get(
            "instance_id"
        ):
            poisoned["read_instance"] = "someone_else"
        null_trace = [poisoned] + null_trace[1:]

    def score(rows: Sequence[Mapping[str, Any]]) -> tuple[float, int]:
        clean = sum(
            1
            for row in rows
            if row.get("read_instance", row.get("instance_id", "self"))
            == row.get("instance_id", "self")
        )
        return (clean / len(rows) if rows else 1.0), clean

    value, clean = score(trace)
    null_value, _ = score(null_trace)
    control_value = 1.0 if clean == len(trace) else 0.0
    passed = clean == len(trace)
    return CriterionResult(
        criterion="boundaries",
        passed=passed,
        value=value,
        null_value=null_value,
        control_value=control_value,
        detail={"acts": len(trace), "clean": clean},
        note=(
            "no act in the decision trace reads another instance's material; "
            "the null is the same trace with one such read inserted"
        ),
    )


def response(
    pairs: Sequence[Mapping[str, Any]],
    shuffled: Sequence[Mapping[str, Any]] | None = None,
) -> CriterionResult:
    """Does a perturbation produce a traced response more often than chance?

    ``pairs`` carries ``perturbed`` and ``responded`` per observation. The
    score is the share of perturbations that were answered, because the
    question is whether the instance responds *to* the environment rather than
    whether it is noisy. The null pairs the same observations with the
    perturbations shuffled, which is the response rate an instance that
    responded independently of the perturbation would show, so a
    noise-generator scores the same as the null and fails.
    """
    if not pairs:
        raise ValueError("response needs observations")
    rows = [
        {
            "perturbed": bool(row.get("perturbed")),
            "responded": bool(row.get("responded")),
        }
        for row in pairs
    ]
    perturbed = [row for row in rows if row["perturbed"]]
    if not perturbed:
        raise ValueError("response needs at least one perturbation")

    def score(sample: Sequence[Mapping[str, Any]]) -> tuple[float, int, int]:
        marks = [row for row in sample if row["perturbed"]]
        answered = sum(1 for row in marks if row["responded"])
        return (answered / len(marks) if marks else 0.0), answered, len(marks)

    null_rows = [dict(row) for row in (shuffled if shuffled is not None else rows)]
    if not null_rows or null_rows == rows:
        # Shuffle the perturbation flags against the same responses, which is
        # the null hypothesis of an instance that does not respond to anything.
        null_rows = [dict(row) for row in rows]
        for index in range(len(null_rows)):
            other = (index + len(null_rows) // 2) % len(null_rows)
            null_rows[index]["perturbed"] = rows[other]["perturbed"]

    value, answered, marks = score(rows)
    null_value, _null_answered, _null_marks = score(null_rows)
    control_value = 1.0 if answered == marks else 0.0
    passed = value > null_value and value >= 0.5
    return CriterionResult(
        criterion="response",
        passed=passed,
        value=value,
        null_value=null_value,
        control_value=control_value,
        detail={
            "observations": len(rows),
            "perturbations": marks,
            "answered": answered,
        },
        note=(
            "a perturbation is answered more often than an instance that "
            "responded independently of it would, which is the null"
        ),
    )


def run_battery(
    *,
    readings: Sequence[float],
    null_readings: Sequence[float],
    before: Mapping[str, str],
    after: Mapping[str, str],
    null_after: Mapping[str, str] | None = None,
    tree: Mapping[str, Any],
    null_tree: Mapping[str, Any] | None = None,
    trace: Sequence[Mapping[str, Any]],
    null_trace: Sequence[Mapping[str, Any]] | None = None,
    pairs: Sequence[Mapping[str, Any]],
    shuffled: Sequence[Mapping[str, Any]] | None = None,
) -> BatteryReport:
    """Run the five criteria and report them together."""
    return BatteryReport(
        results=(
            homeostasis(readings, null_readings),
            continuity(before, after, null_after),
            reproduction(tree, null_tree),
            boundaries(trace, null_trace),
            response(pairs, shuffled),
        )
    )
