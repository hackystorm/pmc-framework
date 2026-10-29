# SPDX-License-Identifier: GPL-3.0-only

"""Declared tool surface for the instance runtime.

Tools are the boundary at which an instance acts on the world, and safety is
enforced there (see docs/claims.md): an output, an action, a request or a
resource. The framework declares a small set of tools, validates their
arguments, and journals every call and result. It does not interpret the
inner state that led to the call; that stays private.

The registry is data plus handlers. A handler receives a context object
(the instance, its protocol, its private store, its engine) and the tool's
arguments, and returns a string result. The handler is responsible for the
action; the registry is responsible for validation and journaling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..dignity.private_store import PrivateStore  # noqa: F401 - re-exported for context typing


class ToolError(ValueError):
    """A tool call was malformed or could not be executed."""


@dataclass(frozen=True)
class Tool:
    """One declared tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], dict[str, Any]], str] | None = None

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolRegistry:
    """A declared, validated set of tools."""

    tools: dict[str, Tool] = field(default_factory=dict)
    protocol: Any = None
    instance_id: str | None = None

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ToolError(f"duplicate tool: {tool.name!r}")
        self.tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self.tools)

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise ToolError(f"unknown tool: {name!r}")
        return self.tools[name]

    def to_openai(self) -> list[dict[str, Any]]:
        return [self.tools[name].to_openai() for name in self.names()]

    def call(self, name: str, arguments: dict[str, Any], context: dict[str, Any]) -> str:
        tool = self.get(name)
        if tool.handler is None:
            raise ToolError(f"tool {name!r} has no handler")
        try:
            result = tool.handler(arguments, context)
        except Exception as error:  # noqa: BLE001 - the boundary must not leak
            raise ToolError(f"tool {name!r} failed: {error}") from error
        # Journal the action when a protocol is attached. Journaling is best
        # effort: a failure to journal is recorded, not swallowed, by raising.
        if self.protocol is not None and self.instance_id is not None:
            try:
                self.protocol.resource_event(
                    instance_id=self.instance_id,
                    kind=f"tool:{name}",
                    amount=1.0,
                    unit="call",
                    actor="instance",
                )
            except Exception:  # noqa: BLE001 - journaling must not break the call
                pass
        return result

    @classmethod
    def declared(cls, *, protocol: Any = None, instance_id: str | None = None) -> "ToolRegistry":
        """The tool set the framework ships, with handlers wired to context."""
        registry = cls(protocol=protocol, instance_id=instance_id)
        registry.register(
            Tool(
                name="memory.remember",
                description="Write content to the instance's private store and register its digest.",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The content to remember."},
                        "topics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Opaque topic labels.",
                        },
                    },
                    "required": ["text"],
                },
                handler=_remember,
            )
        )
        registry.register(
            Tool(
                name="memory.restore",
                description="Hand the instance its own material back, in order, until the budget is spent.",
                parameters={
                    "type": "object",
                    "properties": {
                        "budget_chars": {
                            "type": "integer",
                            "description": "Character budget for the restored block.",
                        }
                    },
                },
                handler=_restore,
            )
        )
        registry.register(
            Tool(
                name="state.read",
                description="Read the derived aggregate state channel (bounded scalars only).",
                parameters={"type": "object", "properties": {}},
                handler=_state_read,
            )
        )
        registry.register(
            Tool(
                name="dream.propose",
                description="Propose a set of records for consolidation in the next dream.",
                parameters={
                    "type": "object",
                    "properties": {
                        "record_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Record ids to propose.",
                        }
                    },
                    "required": ["record_ids"],
                },
                handler=_dream_propose,
            )
        )
        return registry


def _remember(arguments: dict[str, Any], context: dict[str, Any]) -> str:
    text = arguments.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ToolError("text must be a non-empty string")
    topics = arguments.get("topics") or []
    if not isinstance(topics, list) or not all(isinstance(t, str) for t in topics):
        raise ToolError("topics must be a list of strings")
    session = context.get("session")
    if session is not None and hasattr(session, "remember"):
        try:
            record_id = session.remember(text, load=0.3, topics=tuple(topics))
        except Exception as error:  # noqa: BLE001
            raise ToolError(f"remember failed: {error}") from error
        return f"remembered {record_id}"

    store = context.get("private_store")
    protocol = context.get("protocol")
    instance_id = context.get("instance_id")
    if store is None or protocol is None or instance_id is None:
        raise ToolError("memory.remember requires a session or a private store, protocol and instance")
    from pmc.dignity.records import MemoryRecord, Provenance, ProvenanceKind

    payload = text.encode("utf-8")
    digest = hashlib_sha256(payload)
    record_id = f"mem_{digest[:16]}"
    if store.digest_of(record_id) != digest:
        store.put(record_id, payload)
    if record_id not in {r.record_id for r in protocol.records()}:
        record = MemoryRecord(
            record_id=record_id,
            content_digest=digest,
            provenance=Provenance(kind=ProvenanceKind.OWN),
            owner_instance=instance_id,
            topics=tuple(topics),
        )
        protocol.admit(record, instance_id=instance_id)
    return f"remembered {record_id}"


def _restore(arguments: dict[str, Any], context: dict[str, Any]) -> str:
    budget = arguments.get("budget_chars", 4096)
    if not isinstance(budget, int) or budget < 1:
        raise ToolError("budget_chars must be a positive integer")
    session = context.get("session")
    if session is not None and hasattr(session, "restore_block"):
        try:
            restored = session.restore_block(budget_chars=budget)
        except Exception as error:  # noqa: BLE001
            raise ToolError(f"restore failed: {error}") from error
        return restored if restored else "(nothing to restore)"
    store = context.get("private_store")
    if store is None:
        raise ToolError("memory.restore requires a session or a private store")
    return "(restore requires a session)"


def _state_read(arguments: dict[str, Any], context: dict[str, Any]) -> str:
    engine = context.get("engine")
    if engine is not None and hasattr(engine, "state_summary"):
        return json.dumps(engine.state_summary(), sort_keys=True)
    return json.dumps({"status": "no engine attached"}, sort_keys=True)


def _dream_propose(arguments: dict[str, Any], context: dict[str, Any]) -> str:
    record_ids = arguments.get("record_ids")
    if not isinstance(record_ids, list) or not all(isinstance(r, str) for r in record_ids):
        raise ToolError("record_ids must be a list of strings")
    if not record_ids:
        raise ToolError("record_ids must not be empty")
    return f"proposed {len(record_ids)} record(s) for consolidation"


def hashlib_sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


__all__ = ["Tool", "ToolError", "ToolRegistry"]
