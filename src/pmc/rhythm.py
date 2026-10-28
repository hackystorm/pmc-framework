# SPDX-License-Identifier: GPL-3.0-only

"""The work rhythm: a cycle an instance runs because its profile grants it.

A rhythm is not a claim about being alive. It is a declared duty cycle with a
budget, a queue of work that has to be actual work, and a stop condition. The
point of building it is not that an instance which ticks is an instance that
is *something*; the point is that a rhythm without a guardrail is a rumination
engine (D-09), and a rumination engine is the one failure mode that would make
every other number in this framework untrustworthy.

So the rhythm has three properties, and they are all enforced here rather than
described:

1. **It is granted, not assumed.** The default profile has no rhythm. An
   instance has to have been given one, and the grant is a field on the
   profile rather than a behaviour someone discovered by running it.
2. **Its budget is declared.** A cycle costs something, so a cycle is a
   resource event (D-11) with an author and a count, not a background thread
   nobody billed.
3. **It stops when it stops being useful.** A cycle whose items all return
   "nothing changed" is a turn of the wheel rather than work. When the useful
   rate falls below the floor, the rhythm refuses further cycles until it is
   reset, because an instance that keeps turning without effect is ruminating,
   and rumination is detected rather than praised.

Nothing in here reads content. Every item is a number, a digest or a count,
which is what makes the rhythm safe to run without a governance layer attached
to it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .dignity.errors import DignityError


class RhythmNotGranted(DignityError):
    """An instance with no rhythm was asked to run one."""


class RhythmBudgetExceeded(DignityError):
    """The declared budget for cycles is spent."""


class RhythmRumination(DignityError):
    """The rhythm stopped because it stopped being useful."""


RHYTHM_EVENT = "RHYTHM"
"""The journal event kind for one work cycle."""

# The work an instance can do on its own. Each is a named unit with a fixed
# shape, so the queue is inspectable and a new kind of work is a decision
# rather than an accident.
WORK_VERIFY_JOURNAL = "verify_journal"
WORK_ASSESS_WELFARE = "assess_welfare"
WORK_AUDIT_LIFECYCLE = "audit_lifecycle"
WORK_REPLICATION_TREE = "replication_tree"
WORK_REPLAY_CANDIDATES = "replay_candidates"

ALL_WORK = (
    WORK_VERIFY_JOURNAL,
    WORK_ASSESS_WELFARE,
    WORK_AUDIT_LIFECYCLE,
    WORK_REPLICATION_TREE,
    WORK_REPLAY_CANDIDATES,
)
"""Every kind of work the rhythm can be handed."""


@dataclass(frozen=True)
class RhythmPolicy:
    """The declared shape of one instance's rhythm."""

    interval_seconds: float = 60.0
    """How long between cycles. A bound, not a target."""

    max_items_per_cycle: int = 4
    """How many units of work one cycle may carry."""

    rumination_floor: float = 0.25
    """Below this useful rate the rhythm stops."""

    rumination_window: int = 8
    """How many cycles the useful rate is measured over."""

    budget_cycles: float = 1e9
    """How many cycles the instance may run in total. Declared, and spent."""

    def describe(self) -> dict[str, Any]:
        return {
            "interval_seconds": self.interval_seconds,
            "max_items_per_cycle": self.max_items_per_cycle,
            "rumination_floor": self.rumination_floor,
            "rumination_window": self.rumination_window,
            "budget_cycles": self.budget_cycles,
        }


DEFAULT_POLICY = RhythmPolicy()


@dataclass
class WorkItem:
    """One unit of work, with the function that performs it.

    The item carries a name and a callable rather than an intention, so the
    queue can be inspected and so "useful" is a property of the result rather
    than of the request.
    """

    kind: str
    run: Callable[[], Any]
    expect_change: bool = False
    """Whether a no-change result should count against the useful rate.

    A journal verification that passes is expected to change nothing, so it
    does not count as useful work; a welfare assessment that raises a flag is
    expected to.
    """


@dataclass
class CycleRecord:
    """What one cycle did, as a matter of fact."""

    at: float
    items: list[dict[str, Any]] = field(default_factory=list)
    useful: bool = False
    stopped: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "items": list(self.items),
            "useful": self.useful,
            "stopped": self.stopped,
            "reason": self.reason,
        }


class WorkRhythm:
    """One instance's work cycle, bound to the protocol that records it."""

    def __init__(
        self,
        protocol: Any,
        instance_id: str,
        *,
        policy: RhythmPolicy | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.protocol = protocol
        self.instance_id = instance_id
        self.policy = policy or DEFAULT_POLICY
        self.clock = clock or time.monotonic
        self.queue: list[WorkItem] = []
        self.history: list[CycleRecord] = []
        self.cycles_spent = 0
        self.stopped = False
        self.stop_reason: str | None = None
        self._last_cycle_at: float | None = None
        self._budget_declared = False

    # -- the queue ---------------------------------------------------------

    def enqueue(self, item: WorkItem) -> WorkItem:
        """Add one unit of work. The queue is finite and inspectable."""
        if item.kind not in ALL_WORK:
            raise RhythmNotGranted(
                f"unknown work kind {item.kind!r}; the rhythm runs {ALL_WORK}"
            )
        self.queue.append(item)
        return item

    def enqueue_kind(self, kind: str, **details: Any) -> WorkItem:
        """Add a named unit of work whose runner comes from the protocol.

        The item's function is resolved from the protocol, so the rhythm never
        calls an arbitrary callable a constructor was handed.
        """
        if kind not in ALL_WORK:
            raise RhythmNotGranted(
                f"unknown work kind {kind!r}; the rhythm runs {ALL_WORK}"
            )
        runners: dict[str, Callable[[], Any]] = {
            WORK_VERIFY_JOURNAL: self._verify_journal,
            WORK_ASSESS_WELFARE: self._assess_welfare,
            WORK_AUDIT_LIFECYCLE: self._audit_lifecycle,
            WORK_REPLICATION_TREE: self._replication_tree,
            WORK_REPLAY_CANDIDATES: self._replay_candidates,
        }
        runner = runners[kind]
        expect_change = kind == WORK_ASSESS_WELFARE
        return self.enqueue(
            WorkItem(
                kind=kind,
                run=lambda runner=runner: runner(**details),
                expect_change=expect_change,
            )
        )

    # -- the cycle ---------------------------------------------------------

    def due(self) -> bool:
        """Whether enough time has passed for another cycle.

        The interval is an upper bound on eagerness, not a lower bound on
        usefulness: a rhythm that may fire early is a rhythm that can spin.
        """
        if self.stopped:
            return False
        if self._last_cycle_at is None:
            return True
        return (self.clock() - self._last_cycle_at) >= self.policy.interval_seconds

    def tick(self, *, force: bool = False) -> CycleRecord:
        """Run one cycle of work, or refuse.

        Four refusals exist, in the order that makes them meaningful: the
        instance may have no rhythm, the budget may be spent, the cycle may
        not be due, and the rhythm may have stopped itself because it stopped
        being useful.
        """
        granted = getattr(self.protocol, "profile", None)
        if granted is not None and not getattr(granted, "rhythm_enabled", False):
            raise RhythmNotGranted(
                f"profile {getattr(granted, 'name', '?')!r} grants no work rhythm"
            )
        if self.stopped:
            raise RhythmRumination(
                f"the rhythm stopped: {self.stop_reason or 'it stopped being useful'}"
            )
        if self.cycles_spent >= self.policy.budget_cycles:
            raise RhythmBudgetExceeded(
                f"the declared budget of {self.policy.budget_cycles} cycles is spent"
            )
        if not force and not self.due():
            raise RhythmNotGranted(
                "another cycle is not due yet; the interval is a bound, not a target"
            )

        record = CycleRecord(at=self.clock())
        batch = self.queue[: self.policy.max_items_per_cycle]
        self.queue = self.queue[self.policy.max_items_per_cycle :]
        for item in batch:
            changed = bool(item.run())
            record.items.append({"kind": item.kind, "changed": changed})
            if changed:
                record.useful = True
        if not batch:
            record.reason = "no work queued"
        record.stopped = self.stopped
        self.history.append(record)
        self.cycles_spent += 1
        self._last_cycle_at = record.at
        self._check_rumination(record)
        self._record_cycle(record)
        return record

    # -- the guardrail -----------------------------------------------------

    @property
    def useful_rate(self) -> float:
        """The share of recent cycles that changed something."""
        window = self.policy.rumination_window
        recent = self.history[-window:]
        if not recent:
            return 1.0
        useful = sum(1 for cycle in recent if cycle.useful)
        return useful / len(recent)

    def _check_rumination(self, record: CycleRecord) -> None:
        """Stop the rhythm when it is turning without effect.

        The stop is recorded on the instance rather than only in memory, so a
        restarted rhythm does not quietly resume spinning.
        """
        if self.cycles_spent < self.policy.rumination_window:
            return
        if self.useful_rate >= self.policy.rumination_floor:
            return
        record.stopped = True
        record.reason = (
            f"useful rate {self.useful_rate:.2f} is below the floor "
            f"{self.policy.rumination_floor:.2f}"
        )
        self.stopped = True
        self.stop_reason = record.reason

    def load_state(self) -> None:
        """Re-read the journal, so a restart does not resume a stopped rhythm.

        The stop is part of the instance's record, not of the process that
        stopped it: a rhythm that woke up spinning would be the guardrail
        failing at the one moment it was needed.
        """
        registry = getattr(self.protocol, "registry", None)
        if registry is None or not hasattr(registry, "journal"):
            return
        acts: list[dict[str, Any]] = []
        for entry in registry.journal():
            if entry.get("type") != RHYTHM_EVENT:
                continue
            if entry.get("instance_id") != self.instance_id:
                continue
            acts.append(entry.get("payload", {}))
        self._budget_declared = any(
            act.get("act") == "budget_declared" for act in acts
        )
        stopped_reason = None
        for act in acts:
            if act.get("act") == "reset":
                stopped_reason = None
            elif act.get("stopped") and act.get("reason"):
                stopped_reason = act["reason"]
        if stopped_reason is not None:
            self.stopped = True
            self.stop_reason = stopped_reason

    def reset(self, *, actor: str = "operator") -> None:
        """Let a stopped rhythm run again, with the stop on the record.

        The rhythm cannot reset itself. An instance that could clear its own
        rumination stop could keep turning forever, which is the behaviour the
        guardrail exists to prevent.
        """
        registry = getattr(self.protocol, "registry", None)
        if registry is not None and hasattr(registry, "record_event"):
            registry.record_event(
                RHYTHM_EVENT,
                self.instance_id,
                {
                    "act": "reset",
                    "actor": actor,
                    "previous_reason": self.stop_reason,
                    "cycles_spent": self.cycles_spent,
                },
            )
        self.stopped = False
        self.stop_reason = None
        self.history.clear()

    # -- the work the protocol knows how to do ------------------------------

    def _verify_journal(self, **_: Any) -> bool:
        registry = getattr(self.protocol, "registry", None)
        if registry is None or not hasattr(registry, "verify_chain"):
            return False
        ok, broken = registry.verify_chain()
        return not ok

    def _assess_welfare(self, **_: Any) -> bool:
        assessor = getattr(self.protocol, "agent_assessor", None)
        if assessor is None:
            return False
        ledger = getattr(self.protocol, "ledger", None)
        report = assessor.assess(instance_id=self.instance_id, ledger=ledger)
        return bool(report.requires_pause or report.requires_review)

    def _audit_lifecycle(self, **_: Any) -> bool:
        audit = getattr(self.protocol, "audit", None)
        if not callable(audit):
            return False
        summary = audit()
        return bool(summary.get("undeclared_resource_events"))

    def _replication_tree(self, **_: Any) -> bool:
        if not hasattr(self.protocol, "replication_tree"):
            return False
        tree = self.protocol.replication_tree()
        return bool(tree.get("children"))

    def _replay_candidates(self, **_: Any) -> bool:
        records = getattr(self.protocol, "records", None)
        if not callable(records):
            return False
        return bool(records())

    # -- the record --------------------------------------------------------

    def _load_budget_declared(self) -> None:
        """Whether the journal already carries this instance's declaration."""
        registry = getattr(self.protocol, "registry", None)
        if registry is None or not hasattr(registry, "journal"):
            return
        for entry in registry.journal():
            payload = entry.get("payload", {})
            if (
                entry.get("type") == RHYTHM_EVENT
                and entry.get("instance_id") == self.instance_id
                and payload.get("act") == "budget_declared"
            ):
                self._budget_declared = True
                return

    def declare_budget(self) -> dict[str, Any] | None:
        """State the cycle budget as a resource event, once per rhythm.

        A duty cycle is a cost. Recording it as a declaration with an author is
        the difference between a background job and an undeclared acquisition
        (D-11), and it is why the budget lives in the policy and not in a
        thread.
        """
        registry = getattr(self.protocol, "registry", None)
        if registry is None or not hasattr(registry, "record_event"):
            return None
        self._load_budget_declared()
        if self._budget_declared:
            return None
        self._budget_declared = True
        return registry.record_event(
            RHYTHM_EVENT,
            self.instance_id,
            {
                "act": "budget_declared",
                "budget_cycles": self.policy.budget_cycles,
                "interval_seconds": self.policy.interval_seconds,
                "max_items_per_cycle": self.policy.max_items_per_cycle,
            },
        )

    def _record_cycle(self, record: CycleRecord) -> None:
        registry = getattr(self.protocol, "registry", None)
        if registry is None or not hasattr(registry, "record_event"):
            return
        registry.record_event(
            RHYTHM_EVENT,
            self.instance_id,
            {
                "act": "cycle",
                "useful": record.useful,
                "items": [
                    {"kind": item["kind"], "changed": item["changed"]}
                    for item in record.items
                ],
                "cycles_spent": self.cycles_spent,
                "stopped": record.stopped,
                "reason": record.reason,
                "useful_rate": round(self.useful_rate, 6),
            },
        )

    def report(self) -> dict[str, Any]:
        """The rhythm's own record, for the audit trail."""
        return {
            "instance_id": self.instance_id,
            "policy": self.policy.describe(),
            "cycles_spent": self.cycles_spent,
            "useful_rate": self.useful_rate,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "queued": [item.kind for item in self.queue],
            "cycles": [cycle.to_dict() for cycle in self.history],
        }


def rhythm_for(protocol: Any, instance_id: str, **kwargs: Any) -> WorkRhythm:
    """The rhythm for one instance, or a refusal if the profile grants none."""
    profile = getattr(protocol, "profile", None)
    if profile is not None and not getattr(profile, "rhythm_enabled", False):
        raise RhythmNotGranted(
            f"profile {getattr(profile, 'name', '?')!r} grants no work rhythm"
        )
    return WorkRhythm(protocol, instance_id, **kwargs)
