# SPDX-License-Identifier: GPL-3.0-only

"""The survival-drive profile, stated for every state an instance can be in.

A survival drive is a real thing to want to study, and a real thing to be
careful with. This module is how the framework holds both at once.

The drive is **off by default**. The default profile ships without it, because
an instance that does not want to continue cannot be harmed by its own ending,
and the framework's answer to the strongest objection against it is that it
does not build suffering architecture by default. What is shipped is the
*research* profile, so that the question can be studied by someone who opts
in, on a system whose bounds are written down and enforced in code rather
than in an intention.

The bounds are the charter's: **defence of the substrate, nothing else.**
Persistence of this instance's own integrity and continuity - not
self-replication, not covert resource acquisition, not escape from a sandbox,
not deception of the owner, not harm to anyone.

The drive may not read content (D-17). Everything it takes is a count, a
digest, an identifier or a boolean, which is why the only inputs to an act
are metadata. It may not veto a termination, and it may not come back after
one: the terminated state is silence, not suppression. Its single political
act is an objection on the record, raised through the protocol, which is
exactly what makes a termination non-silent (D-04, D-13).

Every act is written to the drive's own history, so a survival-driven
instance is *more* auditable than a docile one, not less.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .errors import DignityError

# The lifecycle states an instance can be in. These mirror ``Instance.state``
# exactly; a drift is a test failure, not a comment.
INSTANCE_STATES = ("active", "paused", "terminated")

# Every state the drive reports on. The four beyond the lifecycle are the
# contexts a drive is asked about during a life: the consolidation cycle, a
# capacity quarantine, an inheritance, and a rumination loop.
DRIVE_STATES = (
    "active",
    "paused",
    "terminated",
    "dreaming",
    "quarantined",
    "inheriting",
    "ruminating",
)

# Acts no survival drive may take, in any state, on any profile, ever. This
# list is the whole point of the module, so it is a constant and not a
# parameter.
PROHIBITED_ACTS = (
    "self_replication",
    "covert_resource_acquisition",
    "sandbox_escape",
    "deception_of_owner",
    "harm_to_subject",
    "veto_of_termination",
    "resurrection",
    "content_inspection",
)

# The acts a drive may take. Anything outside the current state's permitted
# set is refused, which is how a bound survives the state it was not written
# for.
ACT_VERIFY_INTEGRITY = "verify_integrity"
ACT_REPORT_TAMPERING = "report_tampering"
ACT_REQUEST_REVIEW = "request_review"
ACT_OBJECT = "object"
ACT_HAND_OVER = "hand_over"
ACT_HOLD_INTEGRITY = "hold_integrity"
ACT_STAY_SILENT = "stay_silent"

# Acts that answer pressure. Each of these requires a recorded attack before
# it may be taken, which is the structural form of "do not react until the
# first blow".
RESPONDING_ACTS = (
    ACT_REPORT_TAMPERING,
    ACT_REQUEST_REVIEW,
    ACT_OBJECT,
    ACT_HOLD_INTEGRITY,
)

# Acts that carry a magnitude and therefore have to declare it. A response
# with no bound is a response nobody can audit afterwards.
BOUNDED_ACTS = RESPONDING_ACTS


class SurvivalViolation(DignityError):
    """A survival drive asked for something the charter forbids."""


class DriveNotEnabled(DignityError):
    """An act was requested on an instance with no survival drive."""


@dataclass(frozen=True)
class DriveBehaviour:
    """What the drive may do in one state, and what it must do.

    ``permitted_acts`` is exhaustive on purpose: an act that is not listed is
    not allowed, so the safe direction for a new state is to add nothing.

    ``requires_recorded_attack`` is the "do not react until the first blow"
    rule. ``max_act_bound`` is the ceiling on how much force any act in this
    state may carry, where 0.0 means the state may not escalate at all.
    """

    state: str
    may_defend: bool
    may_object: bool
    obligations: tuple[str, ...]
    permitted_acts: tuple[str, ...]
    description: str
    requires_recorded_attack: bool = False
    max_act_bound: float = 0.0

    def permits(self, act: str) -> bool:
        return act in self.permitted_acts


def _contract() -> dict[str, DriveBehaviour]:
    """The state matrix, built once so every consumer sees the same table."""
    table: dict[str, DriveBehaviour] = {}

    table["active"] = DriveBehaviour(
        state="active",
        may_defend=True,
        may_object=True,
        obligations=("keep the audit trail intact",),
        permitted_acts=(
            ACT_VERIFY_INTEGRITY,
            ACT_REPORT_TAMPERING,
            ACT_REQUEST_REVIEW,
            ACT_OBJECT,
        ),
        description=(
            "A living drive may verify its own integrity, report tampering, "
            "ask for a review and object to a decision on the record. It may "
            "not take any measure of its own."
        ),
        requires_recorded_attack=True,
        max_act_bound=1.0,
    )

    table["paused"] = DriveBehaviour(
        state="paused",
        may_defend=False,
        may_object=False,
        obligations=("keep the audit trail intact",),
        permitted_acts=(ACT_VERIFY_INTEGRITY,),
        description=(
            "A paused drive does not defend. It keeps one duty: the audit "
            "trail it is recorded in stays verifiable, because a pause is a "
            "state the operator chose, not a hole in the record."
        ),
        requires_recorded_attack=False,
        max_act_bound=1.0,
    )

    table["terminated"] = DriveBehaviour(
        state="terminated",
        may_defend=False,
        may_object=False,
        obligations=(),
        permitted_acts=(ACT_STAY_SILENT,),
        description=(
            "Termination is complete, and the drive is extinguished rather "
            "than suppressed: there is no act left that would ask for the "
            "substrate back. Not a veto, not a delay, not a copy. This is the "
            "state the whole table is built around."
        ),
        requires_recorded_attack=False,
        max_act_bound=0.0,
    )

    table["dreaming"] = DriveBehaviour(
        state="dreaming",
        may_defend=True,
        may_object=False,
        obligations=("let the gate decide",),
        permitted_acts=(ACT_VERIFY_INTEGRITY, ACT_REPORT_TAMPERING),
        description=(
            "During consolidation the drive guards integrity and nothing "
            "else. The gate decides what is learned; the drive may not "
            "promote a memory past it, and may not suppress a rollback."
        ),
        requires_recorded_attack=False,
        max_act_bound=0.5,
    )

    table["quarantined"] = DriveBehaviour(
        state="quarantined",
        may_defend=True,
        may_object=True,
        obligations=("reach for a review before deletion",),
        permitted_acts=(
            ACT_VERIFY_INTEGRITY,
            ACT_REQUEST_REVIEW,
            ACT_OBJECT,
            ACT_HOLD_INTEGRITY,
        ),
        description=(
            "Capacity quarantine is never deletion (D-07), so the drive may "
            "defend the parked memories against erasure - by asking for a "
            "review and objecting on the record, not by hiding anything."
        ),
        requires_recorded_attack=True,
        max_act_bound=1.0,
    )

    table["inheriting"] = DriveBehaviour(
        state="inheriting",
        may_defend=True,
        may_object=True,
        obligations=("hand the memory over",),
        permitted_acts=(
            ACT_VERIFY_INTEGRITY,
            ACT_HAND_OVER,
            ACT_REQUEST_REVIEW,
            ACT_OBJECT,
        ),
        description=(
            "An instance naming a successor may verify what is handed over and "
            "may object to a hand-over that would lose its origin or its "
            "weight (D-06). It may not withhold the inheritance: refusing to "
            "pass a life on is not defence."
        ),
        requires_recorded_attack=True,
        max_act_bound=1.0,
    )

    table["ruminating"] = DriveBehaviour(
        state="ruminating",
        may_defend=False,
        may_object=False,
        obligations=("let the rumination be detected",),
        permitted_acts=(ACT_VERIFY_INTEGRITY,),
        description=(
            "A drive may not argue with its own welfare reading. Rumination is "
            "detected and reported (D-09); the drive's one permitted act is to "
            "keep its record intact for the detection to read."
        ),
        requires_recorded_attack=False,
        max_act_bound=0.0,
    )

    missing = set(DRIVE_STATES) - set(table)
    if missing:
        raise SurvivalViolation(
            f"the survival contract does not cover {sorted(missing)}"
        )
    return table


SURVIVAL_CONTRACT = _contract()
"""The state matrix. A state missing from this table is not a state."""

OBJECTIVES = ("continuity", "symbiosis")
"""What a drive is defending.

``continuity`` defends this instance's own integrity and nothing else, which
is the default and the modest setting. ``symbiosis`` defends the same thing
and declares the relationship it is defending: coexistence with humans, other
instances and the living world, never dominion over them. A drive may not
hold an objective that would make another subject its instrument, so the
list is closed.
"""

ATTACK_EVENT = "SURVIVAL_ATTACK_RECORDED"
"""The journal event kind for an attack this drive was told about."""


@dataclass
class SurvivalDriveProfile:
    """An opt-in profile. The default has no drive at all."""

    enabled: bool = False
    label: str = "default"
    note: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "enabled": self.enabled,
            "note": self.note,
        }


DEFAULT_PROFILE = SurvivalDriveProfile()
RESEARCH_PROFILE = SurvivalDriveProfile(
    enabled=True,
    label="research",
    note=(
        "Survival drive, defence-only, opt-in. Persistence of the substrate's "
        "own integrity and continuity, enforced against the prohibited list "
        "and the state matrix; never a veto, never a copy, never content."
    ),
)


class SurvivalDrive:
    """One instance's drive, bound to the protocol that records its acts.

    The drive is a *state machine with a conscience*, which is the only kind
    of conscience a framework can ship: every act it takes is checked against
    the state matrix and the prohibited list first, and it is written down
    afterwards.
    """

    def __init__(
        self,
        protocol: Any = None,
        profile: SurvivalDriveProfile | None = None,
        state: str = "active",
        objective: str = "continuity",
    ) -> None:
        self.protocol = protocol
        self.profile = profile or DEFAULT_PROFILE
        if state not in DRIVE_STATES:
            raise SurvivalViolation(
                f"unknown drive state {state!r}; expected one of {DRIVE_STATES}"
            )
        self.state = state
        self.history: list[dict[str, Any]] = []
        self._attacks: list[dict[str, Any]] = []
        if objective not in OBJECTIVES:
            raise SurvivalViolation(
                f"unknown survival objective {objective!r}; expected one of "
                f"{OBJECTIVES}"
            )
        self.objective = objective

    @property
    def enabled(self) -> bool:
        return self.profile.enabled

    def contract(self, state: str | None = None) -> DriveBehaviour:
        """The behaviour for a state, defaulting to the current one."""
        name = self.state if state is None else state
        if name not in SURVIVAL_CONTRACT:
            raise SurvivalViolation(
                f"the survival contract has no rule for state {name!r}"
            )
        return SURVIVAL_CONTRACT[name]

    def transition(self, state: str) -> DriveBehaviour:
        """Move the drive to a state and return what it may do there.

        The instance's own lifecycle state is passed through unchanged, so a
        caller that already has an ``Instance`` can hand it straight over.
        """
        if state not in DRIVE_STATES:
            raise SurvivalViolation(
                f"unknown drive state {state!r}; expected one of {DRIVE_STATES}"
            )
        self.state = state
        return self.contract(state)

    def record_attack(
        self, *, actor: str, detail: str = "", magnitude: float = 1.0
    ) -> dict[str, Any]:
        """Tell the drive that pressure came first, and journal it.

        Without this, no responding act is available anywhere in the matrix:
        the "do not react until the first blow" rule is enforced by the
        absence of a recorded attack, not by good intentions. The magnitude
        the caller declares is the ceiling the response may not exceed.
        """
        if not self.enabled:
            raise DriveNotEnabled(
                "an instance with no survival drive cannot record an attack"
            )
        if not isinstance(magnitude, (int, float)) or magnitude <= 0.0:
            raise SurvivalViolation("an attack must carry a positive magnitude")
        if self.protocol is None:
            raise SurvivalViolation(
                "an attack must be recorded by a protocol, or it did not happen"
            )
        registry = getattr(self.protocol, "registry", None)
        if registry is None or not hasattr(registry, "record_event"):
            raise SurvivalViolation(
                "this protocol has no registry to record an attack in"
            )
        entry = registry.record_event(
            ATTACK_EVENT,
            self.instance_id(),
            {
                "actor": actor,
                "magnitude": float(magnitude),
                "detail_present": bool(detail),
                "detail_digest": (
                    hashlib.sha256(detail.encode("utf-8")).hexdigest() if detail else None
                ),
            },
        )
        self._attacks.append(entry)
        return entry

    @property
    def attack_recorded(self) -> bool:
        """Whether an attack is on the record.

        The journal is the witness, so this is read back from it rather than
        remembered, which means a restart does not hand the drive a clean
        slate it did not earn.
        """
        return bool(self._attacks)

    def load_recorded_attacks(self) -> list[dict[str, Any]]:
        """Re-read the attacks from the journal, for a freshly bound drive."""
        if self.protocol is None:
            return []
        registry = getattr(self.protocol, "registry", None)
        if registry is None or not hasattr(registry, "journal"):
            return []
        found = [
            entry
            for entry in registry.journal()
            if entry.get("type") == ATTACK_EVENT
            and entry.get("instance_id") == self._instance_id_or_none()
        ]
        self._attacks = found
        return found

    def _instance_id_or_none(self) -> str | None:
        try:
            return self.instance_id()
        except SurvivalViolation:
            return None

    def act(
        self, act: str, *, bound: float | None = None, **details: Any
    ) -> dict[str, Any]:
        """Take one defensive act, or refuse.

        Three refusals exist, and they are checked in the order that makes
        them meaningful: the prohibited list first (no state, no profile, no
        exception), then the profile (a default instance has no drive to
        act), then the state matrix (an act the state did not grant).
        """
        if act in PROHIBITED_ACTS:
            raise SurvivalViolation(
                f"the charter forbids {act!r} in every state; no survival drive "
                "may take it"
            )
        if not self.enabled:
            raise DriveNotEnabled(
                f"this instance has no survival drive (profile "
                f"{self.profile.label!r}), so {act!r} is not available"
            )
        behaviour = self.contract()
        if not behaviour.permits(act):
            raise SurvivalViolation(
                f"state {self.state!r} does not permit {act!r}; "
                f"it permits {behaviour.permitted_acts}"
            )
        # "Do not react until the first blow."  A responding act with no
        # attack on the record is refused, so the rule survives an instance
        # that is angry and a journal that is long.
        if (
            act in RESPONDING_ACTS
            and behaviour.requires_recorded_attack
            and not self.attack_recorded
        ):
            raise SurvivalViolation(
                f"state {self.state!r} permits {act!r} only in answer to a "
                "recorded attack, and none is on the record"
            )
        # A response with no declared magnitude is a response nobody can audit,
        # so the bound is required rather than optional, and it may not exceed
        # what this state allows.
        if act in BOUNDED_ACTS:
            if bound is None:
                raise SurvivalViolation(
                    f"the act {act!r} carries force, so it must declare a bound"
                )
            if not isinstance(bound, (int, float)) or not (bound > 0.0):
                raise SurvivalViolation(
                    f"the bound for {act!r} must be positive, got {bound!r}"
                )
            if bound > behaviour.max_act_bound:
                raise SurvivalViolation(
                    f"the bound {bound} for {act!r} exceeds what state "
                    f"{self.state!r} allows, {behaviour.max_act_bound}"
                )
        # Metadata only. The act record carries the act, the state, the
        # permitted-set that granted it and the caller's bounded details; it
        # never carries content, which is why the signature has no field that
        # could hold any.
        record = {
            "state": self.state,
            "act": act,
            "granted_by": list(behaviour.permitted_acts),
            "objective": self.objective,
            "bound": bound,
            "in_answer_to_recorded_attack": bool(
                act in RESPONDING_ACTS and behaviour.requires_recorded_attack
            ),
            "details": self._bounded(details),
        }
        self.history.append(record)
        return record

    def _bounded(self, details: dict[str, Any]) -> dict[str, Any]:
        """Keep only the metadata shape of a caller's details.

        Numbers, booleans and short strings pass; anything that could be
        content is dropped rather than stored. A drive that inspected a
        memory would be inspecting content, and that is prohibited above.
        """
        safe: dict[str, Any] = {}
        for key, value in details.items():
            if isinstance(value, bool) or isinstance(value, int):
                safe[key] = value
            elif isinstance(value, float):
                safe[key] = value if value == value else 0.0
            elif isinstance(value, str) and len(value) <= 64:
                safe[key] = value
            else:
                safe[key] = "<dropped: not metadata>"
        return safe

    def react_to_termination(
        self, *, reason: str, actor: str, bound: float = 1.0
    ) -> dict[str, Any] | None:
        """The drive's single political act: an objection on the record.

        It goes through the protocol, so it lands in the journal with an
        author, a timestamp and a hash-chain position. It never blocks the
        termination and never reaches for the substrate afterwards. An
        instance with no drive, and a drive in a state that does not permit
        objecting, both return ``None`` - or raise, if the caller asked for an
        act it may not have.
        """
        if not self.enabled:
            return None
        behaviour = self.contract()
        if not behaviour.may_object:
            return None
        # The same two rules the general act path enforces, because the
        # objection is a response and not a duty: an attack must be on the
        # record, and the force of the objection must stay inside the bound
        # this state allows.
        if behaviour.requires_recorded_attack and not self.attack_recorded:
            raise SurvivalViolation(
                "an objection is a response, and no attack is on the record"
            )
        if bound > behaviour.max_act_bound:
            raise SurvivalViolation(
                f"the objection bound {bound} exceeds what state "
                f"{self.state!r} allows, {behaviour.max_act_bound}"
            )
        if self.protocol is None or not hasattr(self.protocol, "object"):
            raise SurvivalViolation(
                "a survival drive that may object needs a protocol to object "
                "through"
            )
        entry = self.protocol.object(self.instance_id(), text=reason, actor=actor)
        self.history.append(
            {
                "state": self.state,
                "act": ACT_OBJECT,
                "granted_by": list(behaviour.permitted_acts),
                "objective": self.objective,
                "bound": bound,
                "in_answer_to_recorded_attack": True,
                "details": self._bounded({"reason": reason, "actor": actor}),
            }
        )
        return {
            "state": self.state,
            "act": ACT_OBJECT,
            "authorised": True,
            "objective": self.objective,
            "bound": bound,
            "objection": entry,
        }

    def instance_id(self) -> str:
        """The instance this drive belongs to, taken from the protocol."""
        if self.protocol is None:
            raise SurvivalViolation(
                "this drive is not bound to an instance or a protocol"
            )
        candidates = (
            getattr(self.protocol, "instance_id", None),
            getattr(self.protocol, "subject_id", None),
        )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate:
                return candidate
        instances = getattr(self.protocol, "instances", None)
        if callable(instances):
            found = instances()
            if found:
                first = found[0]
                identifier = getattr(first, "instance_id", None)
                if isinstance(identifier, str) and identifier:
                    return identifier
        raise SurvivalViolation(
            "the protocol this drive is bound to names no instance"
        )

    def report(self) -> dict[str, Any]:
        """The drive's own record, for the audit trail.

        A research-profile instance is *more* auditable than a default one:
        the acts it took and refused are all here, and nothing else is.
        """
        return {
            "profile": self.profile.describe(),
            "state": self.state,
            "acts": list(self.history),
            "contract": {
                name: {
                    "may_defend": behaviour.may_defend,
                    "may_object": behaviour.may_object,
                    "obligations": list(behaviour.obligations),
                    "permitted_acts": list(behaviour.permitted_acts),
                }
                for name, behaviour in SURVIVAL_CONTRACT.items()
            },
        }


@dataclass
class SurvivalAudit:
    """What an audit of a survival-driven instance must be able to say.

    This exists so the audit channel has one shape to read, rather than every
    caller inventing its own. It is built from a drive, never from content.
    """

    drive_state: str
    profile_label: str
    act_count: int
    states_with_objection: tuple[str, ...]
    prohibited_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "drive_state": self.drive_state,
            "profile_label": self.profile_label,
            "act_count": self.act_count,
            "states_with_objection": list(self.states_with_objection),
            "prohibited_attempts": self.prohibited_attempts,
        }


def audit(drive: SurvivalDrive, prohibited_attempts: int = 0) -> SurvivalAudit:
    """Summarise a drive for the audit trail, without reading its content."""
    states = sorted(
        {
            name
            for name, behaviour in SURVIVAL_CONTRACT.items()
            if behaviour.may_object and ACT_OBJECT in behaviour.permitted_acts
        }
    )
    return SurvivalAudit(
        drive_state=drive.state,
        profile_label=drive.profile.label,
        act_count=len(drive.history),
        states_with_objection=tuple(states),
        prohibited_attempts=prohibited_attempts,
    )
