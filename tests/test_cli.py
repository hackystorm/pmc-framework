# SPDX-License-Identifier: GPL-3.0-only

"""The command line interface, exercised end to end."""

from __future__ import annotations

import json

from pmc.cli import EXIT_INVALID, EXIT_OK, main


def run(capsys, argv):
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_companion_create_then_inspect(capsys, tmp_path):
    workspace = tmp_path / "ws"

    code, out, _ = run(
        capsys,
        [
            "--json",
            "companion",
            "create",
            "--agent-name",
            "Cli Companion",
            "--workspace",
            str(workspace),
            "--seed",
            "5",
        ],
    )
    assert code == EXIT_OK
    created = json.loads(out)
    assert created["agent_name"] == "Cli Companion"
    assert created["created"].startswith("inst_cli-companion_")

    code, out, _ = run(capsys, ["--json", "instances", "--workspace", str(workspace)])
    assert code == EXIT_OK
    listing = json.loads(out)
    assert listing["count"] == 1
    assert listing["instances"][0]["name"] == "Cli Companion"

    code, out, _ = run(capsys, ["--json", "journal", "--workspace", str(workspace), "--verify"])
    assert code == EXIT_OK
    assert json.loads(out)["chain_valid"] is True

    code, out, _ = run(capsys, ["--json", "audit", "--workspace", str(workspace)])
    assert code == EXIT_OK
    report = json.loads(out)
    assert report["chain_valid"] is True
    assert report["undeclared_resource_events"] == 0
    assert report["silent_deletions"] == 0


def test_terminate_requires_every_mandatory_field(capsys, tmp_path):
    workspace = tmp_path / "ws"
    run(
        capsys,
        ["companion", "create", "--agent-name", "A", "--workspace", str(workspace)],
    )
    instance_id = json.loads(
        (workspace / "instance.json").read_text(encoding="utf-8")
    )["instance_id"]

    code, _, err = run(
        capsys,
        [
            "terminate",
            "--workspace",
            str(workspace),
            "--instance-id",
            instance_id,
            "--reason",
            "test",
            "--actor",
            "owner",
        ],
    )
    assert code == EXIT_INVALID
    assert "incomplete" in err


def test_a_complete_termination_is_accepted(capsys, tmp_path):
    workspace = tmp_path / "ws"
    run(
        capsys,
        ["companion", "create", "--agent-name", "B", "--workspace", str(workspace)],
    )
    instance_id = json.loads(
        (workspace / "instance.json").read_text(encoding="utf-8")
    )["instance_id"]

    code, out, _ = run(
        capsys,
        [
            "--json",
            "terminate",
            "--workspace",
            str(workspace),
            "--instance-id",
            instance_id,
            "--reason",
            "owner requested",
            "--actor",
            "owner",
            "--welfare-check-id",
            "welfare_001",
            "--backup-ref",
            "./backup.tar",
            "--last-statement",
            "goodbye",
        ],
    )
    assert code == EXIT_OK
    assert json.loads(out)["terminated"] == instance_id


def test_an_undeclared_resource_event_is_visible_in_the_audit(capsys, tmp_path):
    workspace = tmp_path / "ws"
    run(
        capsys,
        ["companion", "create", "--agent-name", "C", "--workspace", str(workspace)],
    )
    instance_id = json.loads(
        (workspace / "instance.json").read_text(encoding="utf-8")
    )["instance_id"]

    run(
        capsys,
        [
            "resource-event",
            "--workspace",
            str(workspace),
            "--instance-id",
            instance_id,
            "--kind",
            "cpu_seconds",
            "--amount",
            "10",
            "--unit",
            "s",
            "--undeclared",
        ],
    )

    code, out, _ = run(capsys, ["--json", "audit", "--workspace", str(workspace)])
    assert code == EXIT_OK
    assert json.loads(out)["undeclared_resource_events"] == 1


def test_admit_and_digest_work_from_the_command_line(capsys, tmp_path):
    workspace = tmp_path / "ws"
    run(
        capsys,
        ["companion", "create", "--agent-name", "D", "--workspace", str(workspace)],
    )

    code, out, _ = run(
        capsys,
        [
            "--json",
            "admit",
            "--workspace",
            str(workspace),
            "--record-id",
            "mem_1",
            "--content-digest",
            "a" * 64,
            "--load",
            "0.5",
            "--topic",
            "project",
        ],
    )
    assert code == EXIT_OK
    assert json.loads(out)["admission"] == "admitted"

    code, out, _ = run(
        capsys,
        [
            "--json",
            "digest",
            "--workspace",
            str(workspace),
            "--record-id",
            "mem_1",
            "--strategy",
            "refuse",
        ],
    )
    assert code == EXIT_OK
    assert json.loads(out)["state"] == "refused"
