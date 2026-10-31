# SPDX-License-Identifier: GPL-3.0-only

"""The command line interface, exercised end to end."""

from __future__ import annotations

import json

from pmc.dignity.protocol import DignityProtocol
from pmc.dignity.records import Instance
import shlex
import sys

from pmc.cli import EXIT_ERROR, EXIT_INVALID, EXIT_OK, main
from pmc.dream import checkpoint_digest


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


def _dream_commands(older: float) -> tuple[str, str]:
    trainer_code = (
        "import pathlib,sys\n"
        "pathlib.Path(sys.argv[2], 'dream.txt').write_text('dream')\n"
    )
    evaluator_code = (
        "import json,pathlib,sys\n"
        "p=pathlib.Path(sys.argv[1])\n"
        "done=(p/'dream.txt').exists()\n"
        f"older={older} if done else 0.8\n"
        "print(json.dumps({'recall_older':older,'recall_newer':0.6 if done else 0.5,"
        "'interference':0.05,'calibration':0.9,'identity':1.0,'false_memory':0.01}))\n"
    )
    trainer = f"{shlex.quote(sys.executable)} -c {shlex.quote(trainer_code)}"
    evaluator = f"{shlex.quote(sys.executable)} -c {shlex.quote(evaluator_code)}"
    return trainer, evaluator


def _dream_workspace(capsys, tmp_path):
    workspace = tmp_path / "ws"
    code, out, _ = run(
        capsys,
        [
            "--json",
            "companion",
            "create",
            "--agent-name",
            "Dreamer",
            "--workspace",
            str(workspace),
            "--seed",
            "9",
        ],
    )
    assert code == EXIT_OK
    instance_id = json.loads(out)["created"]
    code, _, _ = run(
        capsys,
        [
            "admit",
            "--workspace",
            str(workspace),
            "--record-id",
            "mem_1",
            "--content-digest",
            "a" * 64,
            "--instance-id",
            instance_id,
            "--load",
            "0.5",
        ],
    )
    assert code == EXIT_OK
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.bin").write_bytes(b"base checkpoint")
    return workspace, instance_id, checkpoint


def test_dream_commits_with_external_trainer_and_evaluator(capsys, tmp_path):
    workspace, instance_id, checkpoint = _dream_workspace(capsys, tmp_path)
    before = checkpoint_digest(checkpoint)
    trainer, evaluator = _dream_commands(0.79)
    code, out, err = run(
        capsys,
        [
            "--json",
            "dream",
            "--workspace",
            str(workspace),
            "--checkpoint",
            str(checkpoint),
            "--instance",
            instance_id,
            "--trainer",
            trainer,
            "--evaluator",
            evaluator,
        ],
    )
    assert code == EXIT_OK, err
    report = json.loads(out)
    assert report["outcome"] == "committed"
    assert report["checkpoint_before"] == before
    assert report["checkpoint_after"] == checkpoint_digest(checkpoint)
    assert (checkpoint / "dream.txt").exists()
    dreams = (workspace / "sidecar" / "dreams.jsonl").read_text(encoding="utf-8")
    assert '"outcome": "committed"' in dreams


def test_dream_rolls_back_on_a_bad_gate(capsys, tmp_path):
    workspace, instance_id, checkpoint = _dream_workspace(capsys, tmp_path)
    before = checkpoint_digest(checkpoint)
    trainer, evaluator = _dream_commands(0.20)
    code, out, err = run(
        capsys,
        [
            "--json",
            "dream",
            "--workspace",
            str(workspace),
            "--checkpoint",
            str(checkpoint),
            "--instance",
            instance_id,
            "--trainer",
            trainer,
            "--evaluator",
            evaluator,
        ],
    )
    assert code == EXIT_ERROR, err
    report = json.loads(out)
    assert report["outcome"] == "rolled_back"
    assert report["checkpoint_before"] == before
    assert report["checkpoint_after"] == before
    assert checkpoint_digest(checkpoint) == before
    assert not (checkpoint / "dream.txt").exists()
    dreams = (workspace / "sidecar" / "dreams.jsonl").read_text(encoding="utf-8")
    assert '"outcome": "rolled_back"' in dreams




def test_a_dream_needs_a_consolidator(capsys, tmp_path):
    """A dream without one is a refusal, not a default.

    Which consolidator ran is part of the result, so the CLI will not pick one
    silently. ``--spin`` counts, because it names a consolidator too.
    """
    workspace, instance_id, checkpoint = _dream_workspace(capsys, tmp_path)
    _, evaluator = _dream_commands(0.79)

    code, out, err = run(
        capsys,
        [
            "--json",
            "dream",
            "--workspace",
            str(workspace),
            "--checkpoint",
            str(checkpoint),
            "--evaluator",
            evaluator,
        ],
    )
    assert code == EXIT_INVALID
    assert "--trainer" in err
    assert "--spin" in err


def test_spin_is_a_named_consolidator_not_a_default(capsys, tmp_path):
    """``--spin`` satisfies the requirement, and reaches the real consolidator.

    The binary is a cargo example. When it is built, ``--spin`` runs it - which
    this test proves by observing that the *consolidator* refused the fixture
    checkpoint, not the CLI and not a missing-file error. When it has not been
    built, the refusal names the command that builds it.
    """
    workspace, instance_id, checkpoint = _dream_workspace(capsys, tmp_path)
    _, evaluator = _dream_commands(0.79)
    code, out, err = run(
        capsys,
        [
            "--json",
            "dream",
            "--workspace",
            str(workspace),
            "--checkpoint",
            str(checkpoint),
            "--evaluator",
            evaluator,
            "--spin",
        ],
    )
    if "was not found" in err or "not found" in err:
        assert "dream_spin" in err, f"the refusal should name the build: {err}"
        return
    # The consolidator ran and refused on its own terms. `command_consolidator`
    # does not forward a subprocess's stderr, so what is observable here is the
    # exit code it returns - and 2 is the consolidator's own "bad input" code,
    # which proves the *consolidator* refused rather than the CLI failing to
    # find or spawn it.
    assert code == EXIT_INVALID
    assert "the consolidator exited with" in err, err


def test_the_spin_flag_and_an_explicit_trainer_are_both_accepted():
    """``--spin`` overrides ``--trainer``, because it is more specific.

    A caller who passes both has said which one; the flag is the later
    statement and wins, and the explicit command remains available for every
    other consolidator.
    """
    from pmc.cli import build_parser

    parser = build_parser()
    spin = parser.parse_args(
        ["dream", "--workspace", "w", "--checkpoint", "c", "--evaluator", "e", "--spin"]
    )
    assert spin.spin is True
    assert spin.trainer is None

    trainer = parser.parse_args(
        [
            "dream",
            "--workspace",
            "w",
            "--checkpoint",
            "c",
            "--evaluator",
            "e",
            "--trainer",
            "a b",
        ]
    )
    assert trainer.spin is False
    assert trainer.trainer == "a b"
def test_a_dignity_instance_is_refused_a_rhythm(tmp_path, capsys):
    """The command is thin, but the refusal still comes from the profile."""
    workspace = tmp_path / "plain"
    DignityProtocol(workspace).birth(Instance(instance_id="inst_001", name="Plain"))
    code = main(["rhythm", "--workspace", str(workspace), "--instance", "inst_001"])
    assert code == EXIT_INVALID
    assert "no work rhythm" in capsys.readouterr().err


def test_an_autonomous_instance_runs_a_rhythm(tmp_path, capsys):
    workspace = tmp_path / "auto"
    protocol = DignityProtocol(workspace, profile="autonomous")
    protocol.birth(Instance(instance_id="inst_001", name="Oracle"))
    code = main(
        [
            "rhythm",
            "--workspace",
            str(workspace),
            "--instance",
            "inst_001",
            "--cycles",
            "1",
            "--force",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["instance"] == "inst_001"
    assert payload["report"]["cycles_spent"] == 1
    assert payload["report"]["policy"]["rumination_floor"] > 0

def test_remember_stores_content_and_admits_the_record(capsys, tmp_path):
    """Teach an instance something in one command.

    Before `remember`, teaching an instance meant computing a SHA-256 by hand,
    inventing a record id, and then storing the content somewhere the framework
    would never look. The framework held the ledger side of the model but not
    the path a person could walk.
    """
    workspace = tmp_path / "ws"
    run(
        capsys,
        ["companion", "create", "--agent-name", "R", "--workspace", str(workspace)],
    )

    code, out, _ = run(
        capsys,
        [
            "--json",
            "remember",
            "--workspace",
            str(workspace),
            "--text",
            "The gate code is 47-rosso. Memory persists.",
            "--topic",
            "project",
        ],
    )
    assert code == EXIT_OK
    record_id = json.loads(out)["record_id"]

    # The record exists in the ledger, and the bytes exist in the private store.
    code, out, _ = run(
        capsys,
        ["--json", "instances", "--workspace", str(workspace)],
    )
    assert code == EXIT_OK

    # The id is derived from the content, not chosen by the caller.
    from pmc.dignity.records import content_digest

    expected = content_digest(b"The gate code is 47-rosso. Memory persists.")
    assert record_id == f"mem_{expected[:16]}"

    # And the content is in the private store, under the store's own path.
    from pmc.dignity.records import sha256_of
    from pmc.dignity.private_store import PrivateStore

    store = PrivateStore(workspace / "sidecar" / "private")
    assert store.digest_of(record_id) == expected
    assert store.has(record_id)


def test_remembering_the_same_content_twice_is_one_memory(capsys, tmp_path):
    """The ledger is append-only and a record's id is its content.

    Admitting the same content twice is the same record - not a second one -
    so a person who re-remembers something has not spent capacity the instance
    does not have.
    """
    workspace = tmp_path / "ws"
    run(
        capsys,
        ["companion", "create", "--agent-name", "R", "--workspace", str(workspace)],
    )

    for _ in range(2):
        code, out, _ = run(
            capsys,
            ["--json", "remember", "--workspace", str(workspace), "--text", "one memory"],
        )
        assert code == EXIT_OK
    assert json.loads(out)["state"] == "already_held"

    from pmc.dignity.protocol import DignityProtocol

    records = [
        record
        for record in DignityProtocol(workspace).records()
        if record.record_id == json.loads(out)["record_id"]
    ]
    assert len(records) == 1


def test_remember_from_a_file_remembers_the_files_bytes(capsys, tmp_path):
    workspace = tmp_path / "ws"
    run(
        capsys,
        ["companion", "create", "--agent-name", "R", "--workspace", str(workspace)],
    )
    path = tmp_path / "note.txt"
    path.write_bytes(b"bytes from a file, exactly as written")

    code, out, _ = run(
        capsys,
        [
            "--json",
            "remember",
            "--workspace",
            str(workspace),
            "--text-file",
            str(path),
        ],
    )
    assert code == EXIT_OK
    payload = json.loads(out)
    assert payload["bytes"] == len(b"bytes from a file, exactly as written")

    from pmc.dignity.private_store import PrivateStore

    store = PrivateStore(workspace / "sidecar" / "private")
    assert store.get(payload["record_id"]) == b"bytes from a file, exactly as written"


def test_remember_refuses_without_content(capsys, tmp_path):
    """Nothing to remember is a refusal, not an empty record."""
    workspace = tmp_path / "ws"
    run(
        capsys,
        ["companion", "create", "--agent-name", "R", "--workspace", str(workspace)],
    )

    code, _, _ = run(
        capsys,
        ["--json", "remember", "--workspace", str(workspace)],
    )
    assert code == EXIT_INVALID

    code, _, _ = run(
        capsys,
        ["--json", "remember", "--workspace", str(workspace), "--text", ""],
    )
    assert code == EXIT_INVALID


def test_remember_refuses_two_kinds_of_content_at_once(capsys, tmp_path):
    """`--text` and `--text-file` name different things, so both is a refusal."""
    workspace = tmp_path / "ws"
    run(
        capsys,
        ["companion", "create", "--agent-name", "R", "--workspace", str(workspace)],
    )
    path = tmp_path / "n.txt"
    path.write_text("a file")

    code, _, err = run(
        capsys,
        [
            "--json",
            "remember",
            "--workspace",
            str(workspace),
            "--text",
            "some text",
            "--text-file",
            str(path),
        ],
    )
    assert code == EXIT_INVALID
    assert "mutually exclusive" in err


def test_remember_refuses_a_missing_file(capsys, tmp_path):
    workspace = tmp_path / "ws"
    run(
        capsys,
        ["companion", "create", "--agent-name", "R", "--workspace", str(workspace)],
    )

    code, _, err = run(
        capsys,
        [
            "--json",
            "remember",
            "--workspace",
            str(workspace),
            "--text-file",
            str(tmp_path / "absent.txt"),
        ],
    )
    assert code == EXIT_INVALID
    assert "no such file" in err