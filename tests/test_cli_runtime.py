# SPDX-License-Identifier: GPL-3.0-only

"""The runtime commands on the command line.

The runtime talks to a loader, so these tests point it at a stub endpoint
rather than a network.
"""

from __future__ import annotations

import json

from pmc.cli import EXIT_INVALID, EXIT_OK, main


def run(capsys, argv):
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def make_instance(capsys, tmp_path):
    workspace = tmp_path / "ws"
    code, out, _ = run(
        capsys,
        [
            "--json",
            "companion",
            "create",
            "--agent-name",
            "Runtime Companion",
            "--workspace",
            str(workspace),
            "--seed",
            "13",
        ],
    )
    assert code == EXIT_OK
    return workspace, json.loads(out)["created"]


def test_runtime_state_on_an_empty_instance(capsys, tmp_path):
    workspace, instance_id = make_instance(capsys, tmp_path)

    code, out, _ = run(
        capsys,
        [
            "--json",
            "runtime",
            "state",
            "--workspace",
            str(workspace),
            "--model",
            "stub",
            "--endpoint",
            "http://127.0.0.1:1/v1",
        ],
    )

    assert code == EXIT_OK
    state = json.loads(out)
    assert state["instance_id"] == instance_id
    assert state["memories_held"] == 0
    assert state["state"] == ""


def test_runtime_state_on_a_workspace_without_an_instance(capsys, tmp_path):
    code, out, _ = run(
        capsys,
        [
            "--json",
            "runtime",
            "state",
            "--workspace",
            str(tmp_path / "empty"),
            "--model",
            "stub",
        ],
    )

    assert code == EXIT_INVALID
    assert "error" in json.loads(out)


def test_runtime_ask_reaches_the_loader(capsys, tmp_path, monkeypatch):
    workspace, instance_id = make_instance(capsys, tmp_path)
    seen: dict = {}

    class FakeReply:
        text = "the answer"
        prompt_tokens = 11
        completion_tokens = 2

    def fake_complete(self, messages, *, temperature=None, max_tokens=None):
        seen["messages"] = messages
        return FakeReply()

    monkeypatch.setattr(
        "pmc.runtime.ChatClient.complete", fake_complete, raising=True
    )

    code, out, _ = run(
        capsys,
        [
            "--json",
            "runtime",
            "ask",
            "what do you know?",
            "--workspace",
            str(workspace),
            "--model",
            "stub",
            "--endpoint",
            "http://127.0.0.1:1/v1",
        ],
    )

    assert code == EXIT_OK
    payload = json.loads(out)
    assert payload["answer"] == "the answer"
    assert payload["instance_id"] == instance_id
    assert payload["state_injected"] is True
    assert seen["messages"][0]["role"] == "system"
    assert seen["messages"][-1]["content"] == "what do you know?"


def test_runtime_ask_can_skip_the_state(capsys, tmp_path, monkeypatch):
    workspace, _ = make_instance(capsys, tmp_path)
    seen: dict = {}

    class FakeReply:
        text = "no state"
        prompt_tokens = 5
        completion_tokens = 1

    def fake_complete(self, messages, *, temperature=None, max_tokens=None):
        seen["messages"] = messages
        return FakeReply()

    monkeypatch.setattr(
        "pmc.runtime.ChatClient.complete", fake_complete, raising=True
    )

    code, out, _ = run(
        capsys,
        [
            "--json",
            "runtime",
            "ask",
            "hello",
            "--workspace",
            str(workspace),
            "--model",
            "stub",
            "--endpoint",
            "http://127.0.0.1:1/v1",
            "--no-state",
        ],
    )

    assert code == EXIT_OK
    assert json.loads(out)["state_injected"] is False
    assert len(seen["messages"]) == 2
