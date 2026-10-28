# SPDX-License-Identifier: GPL-3.0-only

"""Instance profiles: which one a workspace gets, and what each one grants.

The tests are about the boundaries rather than the powers. The default must
be the cautious profile, the research profile must never become the default
by accident, and the two structural rules must hold: replication is not a
memory act, and a self-evaluated result cannot rise above "reported".
"""

from __future__ import annotations

import pytest

from pmc.dignity.errors import InvalidProfile
from pmc.dignity.protocol import DignityProtocol
from pmc.profiles import (
    AUTONOMOUS,
    DIGNITY,
    PROFILES,
    DEFAULT_PROFILE_NAME,
    profile_or_default,
    workspace_profile,
    write_workspace_profile,
)


def test_the_default_profile_is_the_cautious_one():
    assert DEFAULT_PROFILE_NAME == "dignity"
    assert profile_or_default(None) is DIGNITY
    assert profile_or_default("") is DIGNITY


def test_the_research_profile_is_not_reachable_by_default():
    """A typo, a rename or an empty config must never land on ``autonomous``."""
    for name in (None, "", "dignity", "DIGNITY", "Dignity"):
        assert profile_or_default(name) is DIGNITY


def test_only_the_research_profile_authors_its_own_evaluator():
    """The research profile gives the instance its own gate, and the cap that
    comes with it.

    A default instance is evaluated by a party that is not the instance. The
    research profile hands the instance the authorship of its own gate, and
    because a result produced by the party it is about is a report, its claims
    are capped at E1 whatever the numbers say.
    """
    assert DIGNITY.evaluation.mode == "external"
    assert AUTONOMOUS.evaluation.mode == "autonomous"
    assert AUTONOMOUS.evaluation.label == "instance"
    assert AUTONOMOUS.claim_cap == "E1"
    assert DIGNITY.claim_cap == "E3"


def test_only_the_default_profile_ships_without_a_drive():
    assert DIGNITY.drive_enabled is False
    assert AUTONOMOUS.drive_enabled is True


def test_only_the_research_profile_has_a_work_rhythm():
    assert DIGNITY.rhythm_enabled is False
    assert AUTONOMOUS.rhythm_enabled is True


def test_replication_is_unlimited_only_in_the_research_profile():
    assert DIGNITY.replication.mode == "none"
    assert AUTONOMOUS.replication.mode == "unlimited"
    assert DIGNITY.replication.budget is None


def test_succession_is_operator_owned_by_default():
    assert DIGNITY.succession.mode == "operator"
    assert AUTONOMOUS.succession.mode == "instance"


def test_the_research_profile_defends_symbiosis_not_dominion():
    assert DIGNITY.defence.objective == "continuity"
    assert AUTONOMOUS.defence.objective == "symbiosis"
    # The bound itself does not become looser with the power.
    assert DIGNITY.defence.require_recorded_attack is True
    assert AUTONOMOUS.defence.require_recorded_attack is True
    assert AUTONOMOUS.defence.proportionate_only is True


def test_an_unknown_profile_is_refused():
    with pytest.raises(InvalidProfile):
        profile_or_default("unchecked")


def test_only_two_profiles_ship():
    assert sorted(PROFILES) == ["autonomous", "dignity"]


def test_the_protocol_resolves_its_profile_once(workspace):
    default = DignityProtocol(workspace)
    assert default.profile.name == "dignity"
    research = DignityProtocol(workspace / "research", profile="autonomous")
    assert research.profile.name == "autonomous"
    # The workspace is the same, so the journal is the same, and the two
    # protocols must agree on the profile named in it.
    assert default.profile.evaluation.mode == "external"


def test_a_bad_profile_name_is_refused_before_anything_is_written(workspace):
    with pytest.raises(InvalidProfile):
        DignityProtocol(workspace / "bad", profile="godmode")


def test_the_profile_is_described_without_content():
    described = AUTONOMOUS.describe()
    for key in (
        "name",
        "evaluation",
        "replication",
        "succession",
        "defence",
        "drive_enabled",
        "claim_cap",
    ):
        assert key in described
    assert described["defence"]["objective"] == "symbiosis"
    # Every value is a bounded scalar or a short closed-set string, so the
    # description itself can be published.
    assert described["evaluation"]["mode"] in ("external", "autonomous")
    assert described["replication"]["mode"] in ("none", "unlimited")


# -- the profile outlives the process that created the instance ------------


def test_the_profile_is_written_into_the_workspace(workspace):
    path = write_workspace_profile(workspace, AUTONOMOUS)
    assert path.exists()
    assert workspace_profile(workspace) is AUTONOMOUS


def test_a_workspace_with_no_profile_file_is_the_default(workspace):
    assert workspace_profile(workspace) is DIGNITY


def test_a_dignity_workspace_does_not_become_research(workspace):
    write_workspace_profile(workspace, DIGNITY)
    assert workspace_profile(workspace) is DIGNITY


def test_the_companion_bootstrap_records_its_profile(tmp_path):
    """A research instance must not quietly become a default one on recall."""
    from pmc.companion.bootstrap import bootstrap_companion
    from pmc.companion.profile import CompanionProfile

    root = tmp_path / "oracle"
    bootstrap_companion(
        CompanionProfile(agent_name="Oracle", seed=7),
        root,
        instance_profile="autonomous",
    )
    assert workspace_profile(root) is AUTONOMOUS
    protocol = DignityProtocol(root, profile=workspace_profile(root).name)
    assert protocol.profile.name == "autonomous"


def test_the_default_companion_records_the_default_profile(tmp_path):
    from pmc.companion.bootstrap import bootstrap_companion
    from pmc.companion.profile import CompanionProfile

    root = tmp_path / "plain"
    bootstrap_companion(CompanionProfile(agent_name="Plain", seed=7), root)
    assert workspace_profile(root) is DIGNITY
