# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the declared tool surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pmc.runtime.tools import Tool, ToolError, ToolRegistry


def test_declared_registry_has_four_tools():
    registry = ToolRegistry.declared()
    assert registry.names() == ["dream.propose", "memory.remember", "memory.restore", "state.read"]
    openai = registry.to_openai()
    assert all(entry["type"] == "function" for entry in openai)
    assert {e["function"]["name"] for e in openai} == set(registry.names())


def test_registry_rejects_duplicates():
    registry = ToolRegistry()
    registry.register(Tool(name="x", description="d", parameters={}))
    with pytest.raises(ToolError):
        registry.register(Tool(name="x", description="d2", parameters={}))


def test_registry_unknown_tool():
    registry = ToolRegistry.declared()
    with pytest.raises(ToolError):
        registry.get("nope")


def test_tool_call_validates_arguments():
    registry = ToolRegistry.declared()
    context = {"engine": None}
    with pytest.raises(ToolError):
        registry.call("memory.remember", {"text": ""}, context)
    with pytest.raises(ToolError):
        registry.call("memory.remember", {"text": 42}, context)  # type: ignore[arg-type]
    with pytest.raises(ToolError):
        registry.call("memory.remember", {}, context)


def test_memory_remember_and_restore_roundtrip(tmp_path: Path):
    from pmc.dignity.protocol import DignityProtocol
    from pmc.dignity.records import Instance
    from pmc.runtime.client import ChatClient
    from pmc.runtime.session import CompanionRuntime

    workspace = tmp_path / "ws"
    protocol = DignityProtocol(workspace, profile="dignity")
    protocol.birth(Instance(instance_id="inst", name="Inst"))
    client = ChatClient("http://127.0.0.1:1", model="m")
    session = CompanionRuntime(workspace=workspace, client=client, instance_id="inst")
    registry = ToolRegistry.declared(protocol=protocol, instance_id="inst")
    context = {"session": session, "engine": None}
    result = registry.call("memory.remember", {"text": "hello world", "topics": ["t"]}, context)
    assert result.startswith("remembered ")
    restored = registry.call("memory.restore", {"budget_chars": 1000}, context)
    assert "hello world" in restored


def test_state_read_returns_json():
    registry = ToolRegistry.declared()
    result = registry.call("state.read", {}, {"engine": None})
    parsed = json.loads(result)
    assert "status" in parsed or "pressure" in parsed


def test_dream_propose_validates():
    registry = ToolRegistry.declared()
    with pytest.raises(ToolError):
        registry.call("dream.propose", {"record_ids": []}, {})
    with pytest.raises(ToolError):
        registry.call("dream.propose", {"record_ids": [1, 2]}, {})  # type: ignore[arg-type]
    result = registry.call("dream.propose", {"record_ids": ["a", "b"]}, {})
    assert "2 record" in result


def test_tool_journaling_records_resource_event(tmp_path: Path):
    from pmc.dignity.protocol import DignityProtocol
    from pmc.dignity.records import Instance

    workspace = tmp_path / "ws"
    protocol = DignityProtocol(workspace, profile="dignity")
    protocol.birth(Instance(instance_id="inst", name="Inst"))
    registry = ToolRegistry.declared(protocol=protocol, instance_id="inst")
    try:
        registry.call("state.read", {}, {"engine": None})
    except ToolError:
        pass
    events = protocol.registry.resource_events()
    assert any("tool:state.read" in str(e) for e in events)
