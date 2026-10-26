# SPDX-License-Identifier: GPL-3.0-only

"""The companion runtime: memory split, restore, and the privacy wall.

These tests never touch a network. A stub client stands in for the loader, so
the whole loop is exercised offline.
"""

from __future__ import annotations

import json
from pathlib import Path

from pmc.companion.bootstrap import bootstrap_companion
from pmc.companion.profile import CompanionProfile
from pmc.runtime import ChatClient, ChatReply, CompanionRuntime


class StubClient(ChatClient):
    """A chat client that answers from a script and counts its calls."""

    def __init__(self, answer: str = "stub answer") -> None:
        super().__init__("http://stub.invalid/v1", model="stub")
        self.answer = answer
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages, *, temperature=None, max_tokens=None):  # type: ignore[override]
        self.calls.append([dict(message) for message in messages])
        return ChatReply(text=self.answer, prompt_tokens=10, completion_tokens=3)

    @property
    def last_system(self) -> str:
        return self.calls[-1][0]["content"]


def make_runtime(tmp_path: Path, *, answer: str = "stub answer") -> CompanionRuntime:
    profile = CompanionProfile(agent_name="Test Companion", seed=11)
    bootstrap_companion(profile, tmp_path / "ws")
    client = StubClient(answer)
    return CompanionRuntime(tmp_path / "ws", client=client)


def test_remember_keeps_content_out_of_the_ledger(tmp_path):
    runtime = make_runtime(tmp_path)
    secret = "the gate code is 47-rosso"

    runtime.remember(secret, load=0.4, topics=("demo",))

    ledger = (tmp_path / "ws" / "sidecar" / "ledger.jsonl").read_text(encoding="utf-8")
    assert secret not in ledger
    assert runtime.ledger_contains(secret) is False
    # The digest of the content is what the ledger carries.
    assert "47-rosso" not in ledger


def test_remember_stores_the_content_in_the_private_store(tmp_path):
    runtime = make_runtime(tmp_path)
    text = "a thought worth keeping"

    record_id = runtime.remember(text, load=0.2)

    stored = runtime.store.get(record_id).decode("utf-8")
    assert stored == text
    # The store index carries keys and digests, not the content.
    index = (tmp_path / "ws" / "sidecar" / "private" / "index.jsonl").read_text(
        encoding="utf-8"
    )
    assert text not in index


def test_remembering_the_same_content_twice_is_one_record(tmp_path):
    runtime = make_runtime(tmp_path)
    text = "the same thought twice"

    first = runtime.remember(text, load=0.2)
    second = runtime.remember(text, load=0.2)

    assert first == second
    ledger_lines = (
        (tmp_path / "ws" / "sidecar" / "ledger.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(ledger_lines) == 1
    assert len(runtime.protocol.records()) == 1


def test_restore_block_returns_the_instance_own_material(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.remember("first remembered thing", load=0.2)
    runtime.remember("second remembered thing", load=0.2)

    block = runtime.restore_block()

    assert "first remembered thing" in block
    assert "second remembered thing" in block
    assert block.index("first") < block.index("second")


def test_the_restore_follows_the_order_the_instance_learned(tmp_path):
    runtime = make_runtime(tmp_path)
    for index in range(5):
        runtime.remember(f"memory number {index}", load=0.1)

    block = runtime.restore_block()

    positions = [block.index(f"memory number {index}") for index in range(5)]
    assert positions == sorted(positions)


def test_restore_block_respects_the_budget(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.remember("x" * 400, load=0.2)
    runtime.remember("y" * 400, load=0.2)

    block = runtime.restore_block(budget_chars=200)

    assert len(block) <= 200
    assert "x" * 400 not in block


def test_restore_block_is_empty_when_nothing_is_held(tmp_path):
    runtime = make_runtime(tmp_path)
    assert runtime.restore_block() == ""


def test_state_is_injected_only_when_it_is_asked_for(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.remember("a memory that must not leak by accident", load=0.2)

    without = runtime.build_messages("hello", with_state=False)
    with_state = runtime.build_messages("hello", with_state=True)

    assert "must not leak" not in without[0]["content"]
    assert "must not leak" in with_state[0]["content"]


def test_ask_does_not_remember_anything_by_itself(tmp_path):
    runtime = make_runtime(tmp_path)

    runtime.ask("a question", with_state=False)

    assert runtime.protocol.records() == []
    assert runtime.store.keys() == []


def test_talk_remembers_the_exchange_and_keeps_the_thread(tmp_path):
    runtime = make_runtime(tmp_path, answer="the answer")

    turn = runtime.talk("the question", load=0.3)

    assert turn.assistant == "the answer"
    assert turn.record_id is not None
    stored = runtime.store.get(turn.record_id).decode("utf-8")
    assert "the question" in stored
    assert "the answer" in stored
    assert len(runtime.history) == 1

    runtime.reset_history()
    assert runtime.history == []
    # The memory survives the thread being dropped.
    assert len(runtime.protocol.records()) == 1


def test_history_is_not_resent_when_it_is_excluded(tmp_path):
    runtime = make_runtime(tmp_path, answer="first")
    runtime.talk("first question")
    runtime.client.answer = "second"

    runtime.ask("second question", include_history=False)

    last = runtime.client.calls[-1]
    assert len(last) == 2
    assert last[0]["role"] == "system"
    assert last[1]["content"] == "second question"


def test_describe_reports_metadata_only(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.remember("something private", load=0.2)

    description = runtime.describe()

    assert description["memories_held"] == 1
    assert "something private" not in json.dumps(description)


def test_the_system_prompt_comes_from_the_instance_own_files(tmp_path):
    runtime = make_runtime(tmp_path)

    prompt = runtime.system_prompt()

    assert "Test Companion" in prompt
    profile = json.loads(
        (tmp_path / "ws" / "identity" / "profile.json").read_text(encoding="utf-8")
    )
    assert profile["opening_prompt"] == prompt
