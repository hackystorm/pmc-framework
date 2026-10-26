# SPDX-License-Identifier: GPL-3.0-only

"""The companion runtime: the part of an instance that actually talks.

The framework proper is metadata-only. This module is the *instance side* of
the wall, and it is where content is allowed to exist:

* ``remember`` writes the content into the instance's private store as opaque
  bytes, and puts only a digest, a magnitude and opaque labels into the
  ledger. The governance layer never sees the content.
* ``restore_block`` reads the instance's own store back and rebuilds the part
  of the context that the model cannot carry across a restart. This is a
  state restore, not a retrieval system: there is no similarity search and no
  index, the instance simply gets its own material handed back to it.
* ``ask`` sends a conversation to whatever loader the user runs.

The governance layer (:mod:`pmc.dignity`) holds no reference to a store and no
content ever reaches the ledger. That separation is the point: the instance
can be inspected without being read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..companion.profile import CompanionProfile
from ..dignity.records import (
    MemoryRecord,
    Provenance,
    ProvenanceKind,
    content_digest,
)
from ..dignity.private_store import PrivateStore
from ..dignity.protocol import DignityProtocol
from .client import ChatClient, ChatReply

DEFAULT_STATE_HEADER = "REMEMBERED STATE"
"""Label used when the instance rebuilds its own context.

The header is cosmetic and belongs to the instance, not to the framework: a
user running an Italian companion simply passes a different label.
"""


@dataclass
class Turn:
    """One exchange, kept by the instance so it can continue the thread."""

    user: str
    assistant: str
    record_id: str | None = None

    def as_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": self.user},
            {"role": "assistant", "content": self.assistant},
        ]


@dataclass
class CompanionRuntime:
    """Drives one instance against a chat endpoint."""

    workspace: Path | str
    client: ChatClient
    instance_id: str | None = None
    state_header: str = DEFAULT_STATE_HEADER
    state_budget_chars: int = 1200
    history: list[Turn] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace)
        self.protocol = DignityProtocol(self.workspace)
        self.store = PrivateStore(self.workspace / "sidecar" / "private")
        if self.instance_id is None:
            instances = self.protocol.instances()
            if not instances:
                raise ValueError("workspace holds no registered instance")
            self.instance_id = instances[0].instance_id

    # -- identity ----------------------------------------------------------

    def profile(self) -> CompanionProfile:
        """The declared constitution of this instance."""
        path = self.workspace / "identity" / "profile.json"
        return CompanionProfile.load(path)

    def system_prompt(self) -> str:
        """The prompt the instance starts from, rendered from its own files."""
        return self.profile().resolved_opening_prompt()

    # -- memory ------------------------------------------------------------

    def remember(
        self,
        text: str,
        *,
        load: float = 0.3,
        topics: Sequence[str] = (),
    ) -> str:
        """Hold a memory: content to the private store, metadata to the ledger.

        Returns the record id. The content is never handed to the ledger, and
        the framework never reads it back.
        """
        payload = text.encode("utf-8")
        digest = content_digest(payload)
        record_id = f"mem_{digest[:16]}"
        if self.store.digest_of(record_id) != digest:
            self.store.put(record_id, payload)
        if record_id in {record.record_id for record in self.protocol.records()}:
            # The ledger is append-only: admitting the same content twice is
            # the same record, not two records. Re-admitting it would make the
            # ledger claim a capacity that was never spent twice.
            return record_id
        record = MemoryRecord(
            record_id=record_id,
            content_digest=digest,
            provenance=Provenance(kind=ProvenanceKind.OWN),
            emotional_load=float(load),
            topics=tuple(topics),
        )
        self.protocol.admit(record, instance_id=self.instance_id)
        return record_id

    def own_records(self) -> list[MemoryRecord]:
        """The records this instance holds, in the order it learned them.

        The order is the ledger's order, which is the order of learning. The
        timestamps are not used for it: they have one-second resolution, so
        several memories learned in the same second would come back in hash
        order instead of the order the instance actually met them.
        """
        return [
            record
            for record in self.protocol.records()
            if record.owner_instance == self.instance_id
        ]

    def active_records(self) -> list[MemoryRecord]:
        """The records this instance currently carries."""
        active = set(self.protocol.ledger.active_ids(self.instance_id))
        return [record for record in self.own_records() if record.record_id in active]

    def restore_block(self, *, budget_chars: int | None = None) -> str:
        """Rebuild the instance's own memory as a context block.

        This is a restore, not a lookup: every memory the instance holds is
        offered back to it, in order, until the budget is spent. The ledger
        supplies the metadata (which records are held) and the private store
        supplies the bytes (what they say).
        """
        budget = self.state_budget_chars if budget_chars is None else budget_chars
        if budget <= 0:
            return ""
        header = f"{self.state_header}:\n"
        # The header is part of the budget: a caller that asks for 200
        # characters must not receive 220.
        room = budget - len(header)
        if room <= 0:
            return ""
        lines: list[str] = []
        used = 0
        for record in self.active_records():
            try:
                raw = self.store.get(record.record_id)
            except KeyError:
                continue
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            entry = "- " + text.replace("\n", "\n  ")
            cost = len(entry) + 1
            if used + cost > room:
                remaining = room - used - 1
                if remaining > 40:
                    lines.append(entry[:remaining])
                break
            lines.append(entry)
            used += cost
        if not lines:
            return ""
        return header + "\n".join(lines)

    # -- conversation ------------------------------------------------------

    def build_messages(
        self,
        question: str,
        *,
        with_state: bool = False,
        include_history: bool = True,
    ) -> list[dict[str, str]]:
        """Assemble the message list for one turn."""
        system = self.system_prompt()
        if with_state:
            block = self.restore_block()
            if block:
                system = f"{system}\n\n{block}"
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        if include_history:
            for turn in self.history:
                messages.extend(turn.as_messages())
        messages.append({"role": "user", "content": question})
        return messages

    def ask(
        self,
        question: str,
        *,
        with_state: bool = False,
        include_history: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatReply:
        """Send one question and get the answer. Nothing is remembered."""
        messages = self.build_messages(
            question, with_state=with_state, include_history=include_history
        )
        return self.client.complete(
            messages, temperature=temperature, max_tokens=max_tokens
        )

    def talk(
        self,
        text: str,
        *,
        with_state: bool = False,
        load: float = 0.3,
        topics: Sequence[str] = (),
    ) -> Turn:
        """Exchange one turn and remember it."""
        reply = self.ask(text, with_state=with_state)
        record_id = self.remember(
            f"USER: {text}\nASSISTANT: {reply.text}", load=load, topics=topics
        )
        turn = Turn(user=text, assistant=reply.text, record_id=record_id)
        self.history.append(turn)
        return turn

    def reset_history(self) -> None:
        """Drop the conversation thread, keeping the memories."""
        self.history.clear()

    # -- introspection -----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Metadata about this instance. Never content."""
        return {
            "instance_id": self.instance_id,
            "workspace": str(self.workspace),
            "memories_held": len(self.active_records()),
            "memories_registered": len(self.own_records()),
            "state_chars": len(self.restore_block()),
        }

    def ledger_contains(self, needle: str) -> bool:
        """Whether a literal string shows up in the public ledger.

        This is a self-check for the privacy split: the ledger must never
        carry the content it points at.
        """
        path = self.workspace / "sidecar" / "ledger.jsonl"
        if not path.exists():
            return False
        return needle in path.read_text(encoding="utf-8")


def load_runtime_config(path: Path | str) -> dict[str, Any]:
    """Read a small JSON runtime configuration file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "CompanionRuntime",
    "DEFAULT_STATE_HEADER",
    "Turn",
    "load_runtime_config",
]
