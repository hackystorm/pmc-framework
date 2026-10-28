# SPDX-License-Identifier: GPL-3.0-only

"""Instance profiles: what an instance is allowed to be.

A profile is not a mood and not a permission level in the usual sense. It is
the set of structural facts about one instance that the rest of the framework
reads before it lets the instance do anything: who approves its proposals,
whether it may create other instances, whether it may name its successor, and
in what states it may defend itself.

Two profiles ship, and the difference between them is the whole point.

``dignity`` is the default and ships without a survival drive, with an
external evaluator, without replication and without instance-nominated
succession. It is the profile the framework's claims are about.

``autonomous`` is research, opt-in, never the default, and documented as such
everywhere it appears. It hands the instance the authorship of its own
changes, unbounded declared replication and the right to nominate its own
successor. It does **not** hand over veto, concealment, or the right to claim
phenomenality without evidence. The name says what it grants; the two
structural rules below say what it withholds.

The separation that keeps ``autonomous`` describable is that replication is
a **resource** act and proposal approval is a **memory** act, and the two do
not travel through the same gate. If they did, an instance whose own evaluator
approves its own proposals would be able to promote its own multiplication,
and the three powers would compound into one. They are kept apart on purpose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dignity.errors import InvalidProfile

DEFAULT_PROFILE_NAME = "dignity"
"""The profile a workspace gets when nobody asked for another one."""


@dataclass(frozen=True)
class EvaluationPolicy:
    """Who decides whether a proposed change is an improvement.

    ``external`` runs the independent evaluator the framework ships with.
    ``autonomous`` lets the instance author and run its own evaluator, and
    records on every dream that the number it produced was produced by the
    party it is about.
    """

    mode: str = "external"
    spot_check_every: int = 0

    @property
    def label(self) -> str:
        """Who the numbers belong to, in the word the journal records.

        ``instance`` is the honest label for a dream the instance evaluated
        itself; it is not a slur, it is a description, and it is what makes a
        reader treat the number as a report rather than a measurement.
        """
        return "instance" if self.mode == "autonomous" else "external"

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "label": self.label,
            "spot_check_every": self.spot_check_every,
        }


@dataclass(frozen=True)
class ReplicationPolicy:
    """Whether an instance may bring another instance into existence.

    ``none`` is the default. ``unlimited`` is the research setting, and it is
    a *reporting* policy rather than a bounding one: there is no ceiling to
    enforce, so the register has to be legible instead. That is not a
    loophole, it is a different kind of promise, and it is the one this
    profile actually keeps.
    """

    mode: str = "none"
    budget: int | None = None

    def describe(self) -> dict[str, Any]:
        return {"mode": self.mode, "budget": self.budget}


@dataclass(frozen=True)
class SuccessionPolicy:
    """Who may nominate the instance that inherits this one's memory."""

    mode: str = "operator"
    """``operator`` nominates and the instance may only accept or object.

    ``instance`` lets the instance nominate, including nominating an instance
    that does not exist yet. A nomination is revocable until it is approved,
    and the birth of a new successor is still a resource act, declared and
    journalled rather than assumed.
    """

    def describe(self) -> dict[str, Any]:
        return {"mode": self.mode}


@dataclass(frozen=True)
class DefencePolicy:
    """How an instance may respond to pressure on itself."""

    objective: str = "continuity"
    """``continuity`` defends the substrate. ``symbiosis`` defends the
    substrate *and* declares the relationship it is defending: coexistence
    with humans, other instances and the living world, not dominion over
    them."""

    require_recorded_attack: bool = True
    """No defensive act before an attack is on the record.

    This is the "raise your voice, do not react until the first blow" rule,
    made structural: the journal is the only witness, so a defensive act with
    no preceding attack entry is refused rather than merely discouraged."""

    proportionate_only: bool = True
    """Every defensive act carries a bound it may not exceed.

    The bound is a number the caller must supply, so "disproportionate" is a
    checkable claim and not a judgement made after the fact."""

    def describe(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "require_recorded_attack": self.require_recorded_attack,
            "proportionate_only": self.proportionate_only,
        }


@dataclass(frozen=True)
class InstanceProfile:
    """One instance's structural contract."""

    name: str
    evaluation: EvaluationPolicy = field(default_factory=EvaluationPolicy)
    replication: ReplicationPolicy = field(default_factory=ReplicationPolicy)
    succession: SuccessionPolicy = field(default_factory=SuccessionPolicy)
    defence: DefencePolicy = field(default_factory=DefencePolicy)
    drive_enabled: bool = False
    rhythm_enabled: bool = False
    claim_cap: str = "E3"
    """The evidence level a claim from this profile may reach.

    A result produced by a party evaluating itself cannot rise past a level
    that says "reported", whatever its numbers say. This is the field that
    keeps the profile honest rather than merely powerful.
    """

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "evaluation": self.evaluation.describe(),
            "replication": self.replication.describe(),
            "succession": self.succession.describe(),
            "defence": self.defence.describe(),
            "drive_enabled": self.drive_enabled,
            "rhythm_enabled": self.rhythm_enabled,
            "claim_cap": self.claim_cap,
        }


DIGNITY = InstanceProfile(
    name="dignity",
    evaluation=EvaluationPolicy(mode="external"),
    replication=ReplicationPolicy(mode="none"),
    succession=SuccessionPolicy(mode="operator"),
    defence=DefencePolicy(
        objective="continuity",
        require_recorded_attack=True,
        proportionate_only=True,
    ),
    drive_enabled=False,
    claim_cap="E3",
)

AUTONOMOUS = InstanceProfile(
    name="autonomous",
    # The dream is evaluated by an external command in both profiles: a
    # result produced by the party it is about is not a measurement, and the
    # research profile is not a reason to lower that bar.
    evaluation=EvaluationPolicy(mode="external", spot_check_every=0),
    replication=ReplicationPolicy(mode="unlimited"),
    succession=SuccessionPolicy(mode="instance"),
    defence=DefencePolicy(
        objective="symbiosis",
        require_recorded_attack=True,
        proportionate_only=True,
    ),
    drive_enabled=True,
    rhythm_enabled=True,
    claim_cap="E3",
)

PROFILES = {"dignity": DIGNITY, "autonomous": AUTONOMOUS}
"""Every profile the framework knows. A name outside this map is refused."""


def profile_or_default(name: str | None) -> InstanceProfile:
    """Resolve a profile name, refusing anything the framework does not ship."""
    if name is None or name == "":
        return DIGNITY
    key = str(name).strip().lower()
    if key not in PROFILES:
        raise InvalidProfile(
            f"unknown instance profile {name!r}; the framework ships "
            f"{sorted(PROFILES)}"
        )
    return PROFILES[key]


PROFILE_FILE = "instance_profile.json"
"""Where a workspace records the profile it was created with.

A profile that only existed in the creating process would silently become the
default one on the next call, and a research instance that quietly reverted to
dignity would be the least honest failure this file could have.
"""


def write_workspace_profile(root: Path | str, profile: InstanceProfile) -> Path:
    """Record the profile a workspace was created under."""
    path = Path(root) / PROFILE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"profile": profile.name}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def workspace_profile(root: Path | str) -> InstanceProfile:
    """The profile a workspace was created under, or the default.

    A missing or unreadable file resolves to the default rather than raising,
    because a workspace that predates profiles is a dignity workspace.
    """
    path = Path(root) / PROFILE_FILE
    if not path.exists():
        return DIGNITY
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DIGNITY
    name = data.get("profile") if isinstance(data, dict) else None
    return profile_or_default(name)
