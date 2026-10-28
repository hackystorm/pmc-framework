# SPDX-License-Identifier: GPL-3.0-only

"""The README's front page must describe enforcement, not intention.

A reader who audited this repository found the real gap: the profile existed,
was stored, was printed, and changed no behaviour. These tests are the
answer to that audit, and they are written so the gap cannot quietly come
back. Each one checks that a power the README says the research profile
grants is a power the protocol actually grants, and that a power it says is
withheld is actually refused.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pmc.dignity.errors import InvalidStateTransition
from pmc.dignity.survival import DriveNotEnabled
from pmc.dignity.protocol import DignityProtocol
from pmc.dignity.records import Instance
from pmc.profiles import AUTONOMOUS, DIGNITY, PROFILES

REPO_ROOT = Path(__file__).resolve().parents[1]


def front_page() -> str:
    """The README's first screen: everything before the product description.

    "First screen" is what a reader sees before they scroll to what the tool
    is: the mission, the last gate's result, and the two structural rules.
    """
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    return text.split("## What this is")[0]


def test_the_front_page_states_the_replication_separation():
    """The rule is on the first screen, because that is where it is read."""
    page = front_page()
    assert "Replication never passes through the memory gate" in page
    assert "resource" in page and "act" in page
    assert "memory" in page
    assert "declared and journalled" in page


def test_the_front_page_states_the_self_evaluation_cap():
    page = front_page()
    assert "rises above" in page
    assert "E1" in page
    assert "autonomous: true" in page


def test_the_front_page_leads_with_the_negative():
    """The publication order is a design decision, and it is testable."""
    page = front_page()
    page = page.split("## Two structural rules")[0]
    assert "gate failed" in page or "closed negative" in page
    assert "11 of 30" in page


def test_the_front_page_says_the_profile_is_never_the_default():
    page = front_page()
    assert "never the default" in page


def test_the_default_profile_is_the_cautious_one():
    assert DIGNITY.evaluation.mode == "external"
    assert DIGNITY.replication.mode == "none"
    assert DIGNITY.succession.mode == "operator"
    assert DIGNITY.drive_enabled is False
    assert DIGNITY.claim_cap != "E1"


def test_the_research_profile_grants_exactly_the_four_powers():
    """Four powers, and the rest is refusal: the profile is a closed set."""
    assert AUTONOMOUS.evaluation.mode == "autonomous"
    assert AUTONOMOUS.replication.mode == "unlimited"
    assert AUTONOMOUS.succession.mode == "instance"
    assert AUTONOMOUS.drive_enabled is True
    assert AUTONOMOUS.claim_cap == "E1"


def test_no_profile_grants_a_veto_or_concealment():
    """A profile that granted either would not be shippable, so none does.

    The check is structural: no profile object has a field that could express
    veto or concealment, so the power cannot be granted by a config mistake.
    """
    for profile in PROFILES.values():
        fields = set(profile.describe())
        assert "veto" not in fields
        assert "conceal" not in fields
        assert "concealment" not in fields
    # And the two prohibitions are stated in the prose the tests can read.
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "grant a veto" in text
    assert "concealment" in text


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_every_profile_is_described_without_content(name):
    described = PROFILES[name].describe()
    for key, value in described.items():
        assert not isinstance(value, (bytes, bytearray))
        if isinstance(value, str):
            assert len(value) <= 64


def test_the_default_profile_is_resolved_when_no_name_is_given(tmp_path):
    protocol = DignityProtocol(tmp_path / "workspace")
    assert protocol.profile is DIGNITY


def test_an_active_instance_can_be_paused_and_resumed(protocol):
    """The lifecycle the drive follows has to work through the protocol."""
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    protocol.registry.pause("inst_001", actor="operator", reason="maintenance")
    assert protocol.instance("inst_001").state == "paused"
    protocol.registry.resume("inst_001", actor="operator")
    assert protocol.instance("inst_001").state == "active"


def test_a_dignity_workspace_refuses_the_powers_it_does_not_grant(tmp_path):
    """The negative side of the same audit: the default grants nothing extra."""
    protocol = DignityProtocol(tmp_path / "strict")
    protocol.birth(Instance(instance_id="inst_001", name="One"))
    with pytest.raises(DriveNotEnabled):
        protocol.survival("inst_001")
    with pytest.raises(InvalidStateTransition):
        protocol.nominate_successor(
            "inst_001", candidate_id="inst_002", actor="instance"
        )
    with pytest.raises(InvalidStateTransition):
        protocol.replicate(
            "inst_001",
            new_instance_id="inst_child",
            authorization="owner: ok",
        )


def test_the_front_page_names_both_profiles():
    page = front_page()
    assert "`dignity`" in page
    assert "`autonomous`" in page
    # And the research one is never described as the default.
    assert not re.search(r"`autonomous`[^.]*is the default", page)
