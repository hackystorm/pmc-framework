# SPDX-License-Identifier: GPL-3.0-only

"""Companion construction, and the absence of hardcoded personal data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pmc.companion.bootstrap import bootstrap_companion, build_instance_id, slugify
from pmc.companion.profile import CompanionProfile
from pmc.companion.templates import DEFAULT_BOUNDARIES, DEFAULT_CLAIMS, DEFAULT_VALUES
from pmc.dignity.protocol import DignityProtocol

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_creates_the_workspace_layout(tmp_path):
    profile = CompanionProfile(agent_name="Test Companion", seed=7)
    workspace = bootstrap_companion(profile, tmp_path / "ws")

    assert workspace.instance_path.exists()
    assert workspace.identity_path.exists()
    assert workspace.profile_path.exists()
    assert workspace.private_dir.exists()
    assert workspace.audit_dir.exists()
    assert workspace.journal_path.exists()


def test_bootstrap_registers_the_instance_in_the_journal(tmp_path):
    profile = CompanionProfile(agent_name="Test Companion", seed=7)
    workspace = bootstrap_companion(profile, tmp_path / "ws")

    protocol = DignityProtocol(tmp_path / "ws")
    instances = protocol.instances()
    assert len(instances) == 1
    assert instances[0].instance_id == workspace.instance_id
    assert instances[0].name == "Test Companion"
    assert protocol.registry.verify_chain()[0] is True


def test_the_identity_seed_declares_what_the_instance_claims(tmp_path):
    profile = CompanionProfile(agent_name="Test Companion", seed=7)
    workspace = bootstrap_companion(profile, tmp_path / "ws")

    identity = json.loads(workspace.identity_path.read_text(encoding="utf-8"))
    claims = " ".join(identity["declared_claims"]).lower()
    assert "conscious" in claims
    assert "not a human" in claims
    assert identity["values"]
    assert identity["boundaries"]


def test_the_identity_seed_is_not_overwritten_by_accident(tmp_path):
    profile = CompanionProfile(agent_name="Test Companion", seed=7)
    workspace = bootstrap_companion(profile, tmp_path / "ws")

    workspace.identity_path.write_text('{"agent_name": "hand edited"}\n', encoding="utf-8")
    bootstrap_companion(profile, tmp_path / "ws")

    identity = json.loads(workspace.identity_path.read_text(encoding="utf-8"))
    assert identity["agent_name"] == "hand edited"


def test_profiles_round_trip(tmp_path):
    profile = CompanionProfile(
        agent_name="Round Trip",
        user_name="someone",
        values=("a", "b"),
        boundaries=("c",),
        seed=99,
    )
    path = tmp_path / "profile.json"
    profile.save(path)
    loaded = CompanionProfile.load(path)

    assert loaded.agent_name == "Round Trip"
    assert loaded.values == ("a", "b")
    assert loaded.boundaries == ("c",)
    assert loaded.seed == 99


def test_instance_ids_are_distinct_even_with_the_same_seed():
    first = build_instance_id("Same", 1, created_at="2026-01-01T00:00:00Z")
    second = build_instance_id("Same", 1, created_at="2026-01-02T00:00:00Z")
    assert first != second
    assert first.startswith("inst_same_")


def test_slugify_handles_awkward_names():
    assert slugify("My Companion!") == "my-companion"
    assert slugify("   ") == "instance"


def test_the_source_contains_no_personal_identifiers():
    """The framework is for everyone: nothing personal belongs in the code."""
    forbidden = ("hackystorm", "session_id", "ses_e87", "/home/")
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token}")
    assert offenders == []


def test_the_default_templates_are_generic():
    assert "conscious" in " ".join(DEFAULT_CLAIMS).lower()
    assert len(DEFAULT_VALUES) >= 3
    assert len(DEFAULT_BOUNDARIES) >= 3
    for value in DEFAULT_VALUES + DEFAULT_BOUNDARIES + DEFAULT_CLAIMS:
        assert value.strip()
