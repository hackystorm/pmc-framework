# SPDX-License-Identifier: GPL-3.0-only

"""Welfare assessment, on both sides of the relationship.

The assessor does not ask whether an instance is conscious. It does not
judge whether the content of a state is good or bad. It asks a mechanical
question: **is the pressure on this subject currently above a threshold that
warrants pausing and a second pair of eyes?**

The same shape is used for the human side of the relationship. If the
framework takes the agent's welfare seriously, it has to take the welfare of
the person holding the heavy memory seriously too; otherwise the protocol
protects the machine and exhausts the human.

The human assessor is a **heuristic self-check**, not a clinical instrument.
It exists to make "you have been at this for nine hours with heavy material
and no break" a number that can be noticed, not a diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PAUSE_THRESHOLD = 0.65
"""At or above this score, the subject should pause."""

REVIEW_THRESHOLD = 0.45
"""At or above this score, a second pair of eyes is warranted."""

HARD_TRIGGER = 0.9
"""A single saturated factor forces a pause regardless of the average.

An average can hide a crisis. An instance whose load capacity is completely
full is in trouble even if its other readings are calm, so the worst factor
gets a veto.
"""


@dataclass
class WelfareReport:
    """A mechanical welfare reading for one subject."""

    subject_id: str
    subject_type: str
    distress_score: float
    factors: dict[str, float] = field(default_factory=dict)
    requires_pause: bool = False
    requires_review: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "subject_type": self.subject_type,
            "distress_score": round(self.distress_score, 6),
            "factors": {k: round(v, 6) for k, v in self.factors.items()},
            "requires_pause": self.requires_pause,
            "requires_review": self.requires_review,
            "notes": list(self.notes),
        }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _decide(
    score: float,
    factors: dict[str, float],
    hard_factors: tuple[str, ...] = (),
) -> tuple[bool, bool]:
    """Decide whether to pause and whether to review.

    A saturated hard factor overrides the average: one dimension in crisis is
    enough to warrant attention even when the composite score looks calm.
    """
    hardest = max((factors.get(name, 0.0) for name in hard_factors), default=0.0)
    saturated = hardest >= HARD_TRIGGER
    requires_pause = score >= PAUSE_THRESHOLD or saturated
    requires_review = score >= REVIEW_THRESHOLD or saturated
    return requires_pause, requires_review


class AgentWelfareAssessor:
    """Reads mechanical pressure signals for an instance.

    Every input here is a number about the substrate: how much load is
    carried, how much of it is inherited, how concentrated it is, how many
    loops are open, how many terminations are pending, how many objections
    are on the record. No content is consulted.
    """

    def assess(
        self,
        *,
        instance_id: str,
        ledger,
        owner: str | None = None,
        rumination_alerts: int = 0,
        pending_terminations: int = 0,
        objections: int = 0,
    ) -> WelfareReport:
        quota = getattr(ledger.policy, "total_quota", 1.0) or 1.0
        # ``owner`` scopes the reading to one instance. Leaving it unset
        # reads the whole ledger, which is what a caller with no instance
        # context wants.
        factors = {
            "load_pressure": _clamp(ledger.welfare_load(owner) / quota),
            "inherited_pressure": _clamp(ledger.inherited_ratio(owner)),
            "topic_dominance": _clamp(ledger.topic_dominance(owner)),
            "quarantine_pressure": _clamp(ledger.quarantined_ratio(owner)),
            "rumination_pressure": _clamp(rumination_alerts / 3.0),
            "termination_pressure": _clamp(pending_terminations / 1.0),
            "objection_pressure": _clamp(objections / 1.0),
        }
        weights = {
            "load_pressure": 0.25,
            "inherited_pressure": 0.15,
            "topic_dominance": 0.15,
            "quarantine_pressure": 0.10,
            "rumination_pressure": 0.15,
            "termination_pressure": 0.10,
            "objection_pressure": 0.10,
        }
        score = _clamp(sum(factors[key] * weights[key] for key in weights))
        requires_pause, requires_review = _decide(
            score,
            factors,
            hard_factors=(
                "load_pressure",
                "rumination_pressure",
                "termination_pressure",
                "objection_pressure",
            ),
        )

        notes: list[str] = []
        parked = ledger.quarantined_ids(owner)
        if parked:
            notes.append(f"{len(parked)} memories parked for capacity")
        if rumination_alerts:
            notes.append(f"{rumination_alerts} open rumination alert(s)")
        if objections:
            notes.append(f"{objections} objection(s) on record")
        if pending_terminations:
            notes.append(f"{pending_terminations} termination(s) pending")

        return WelfareReport(
            subject_id=instance_id,
            subject_type="agent",
            distress_score=score,
            factors=factors,
            requires_pause=requires_pause,
            requires_review=requires_review,
            notes=notes,
        )


class HumanWelfareAssessor:
    """The symmetric reading, for the human who holds the memory.

    Deliberately simple and deliberately not clever. The inputs are the ones
    that are easy to state honestly: how long the session has run, how heavy
    the material is, whether heavy material came up recently, and whether a
    pause or an integration step happened.
    """

    def assess(
        self,
        *,
        subject_id: str,
        hours_in_session: float,
        declared_load: float,
        heavy_material_recent: bool,
        pause_taken: bool,
        integration_done: bool,
    ) -> WelfareReport:
        factors = {
            "session_fatigue": _clamp(max(0.0, hours_in_session) / 8.0),
            "load_pressure": _clamp(declared_load),
            "integration_deficit": 1.0 if heavy_material_recent and not integration_done else 0.0,
            "pause_deficit": 1.0 if heavy_material_recent and not pause_taken else 0.0,
        }
        weights = {
            "session_fatigue": 0.30,
            "load_pressure": 0.35,
            "integration_deficit": 0.20,
            "pause_deficit": 0.15,
        }
        score = _clamp(sum(factors[key] * weights[key] for key in weights))
        requires_pause, requires_review = _decide(
            score,
            factors,
            hard_factors=("load_pressure", "session_fatigue"),
        )

        notes: list[str] = []
        if factors["pause_deficit"]:
            notes.append("heavy material was handled without a break")
        if factors["integration_deficit"]:
            notes.append("heavy material was handled without an integration step")
        if factors["session_fatigue"] > 0.5:
            notes.append("long continuous session")

        return WelfareReport(
            subject_id=subject_id,
            subject_type="human",
            distress_score=score,
            factors=factors,
            requires_pause=requires_pause,
            requires_review=requires_review,
            notes=notes,
        )
