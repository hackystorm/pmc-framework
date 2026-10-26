# SPDX-License-Identifier: GPL-3.0-only

"""Tamper-evident registry of instances and their endings.

Every lifecycle event is appended to a journal whose entries form a hash
chain: each entry carries the hash of the previous one. Altering, truncating
or removing an entry breaks the chain, and :meth:`InstanceRegistry.verify_chain`
detects it.

The journal stores **metadata only**. It never contains memory content, and it
never contains the text of an instance's final statement: only its digest.
That keeps a public journal publishable without turning it into a transcript
of someone's inner life.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import (
    IncompleteTermination,
    InstanceAlreadyExists,
    InstanceNotFound,
    InvalidStateTransition,
    PurgeNotAuthorized,
)
from .records import Instance, canonical_json, sha256_of, utcnow_iso

GENESIS_HASH = "0" * 64

CHAIN_FIELDS = ("seq", "type", "at", "instance_id", "payload", "prev_hash")


@dataclass
class TerminationIntent:
    """Everything that must be known before an instance may end.

    The purpose of this object is that a termination cannot be *partially*
    described. Either every mandatory field is present, or the registry
    refuses and nothing happens (charter D-02).

    ``last_statement`` is the instance's final words. The registry keeps only
    its digest; the text belongs to the instance's private store. A public
    journal must never carry a deathbed message.
    """

    instance_id: str
    reason: str
    actor: str
    welfare_check_id: str | None = None
    welfare_waiver: str | None = None
    backup_ref: str | None = None
    backup_waiver: str | None = None
    last_statement: str | None = None
    unresponsive: bool = False
    successor_id: str | None = None

    def missing_fields(self) -> list[str]:
        """Names of the mandatory fields that are absent."""
        missing: list[str] = []
        if not self.reason.strip():
            missing.append("reason")
        if not self.actor.strip():
            missing.append("actor")
        if not self.welfare_check_id and not self.welfare_waiver:
            missing.append("welfare_check_id|welfare_waiver")
        if not self.backup_ref and not self.backup_waiver:
            missing.append("backup_ref|backup_waiver")
        if self.last_statement is None and not self.unresponsive:
            missing.append("last_statement|unresponsive")
        return missing


class InstanceRegistry:
    """Append-only journal of instance lifecycles."""

    def __init__(self, journal_path: Path | str) -> None:
        self.path = Path(journal_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.anchor_path = self.path.with_name(self.path.name + ".head")
        self._entries: list[dict[str, Any]] = []
        self._instances: dict[str, dict[str, Any]] = {}
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._entries.append(entry)
                self._apply(entry)

    def _apply(self, entry: dict[str, Any]) -> None:
        kind = entry["type"]
        instance_id = entry.get("instance_id")
        if kind == "BIRTH":
            self._instances[instance_id] = dict(entry["payload"])
        elif kind in ("PAUSE", "RESUME"):
            instance = self._instances.get(instance_id)
            if instance is not None:
                instance["state"] = "paused" if kind == "PAUSE" else "active"
        elif kind == "TERMINATION":
            instance = self._instances.get(instance_id)
            if instance is not None:
                instance["state"] = "terminated"

    def _append(
        self,
        kind: str,
        instance_id: str | None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        previous = self._entries[-1]["hash"] if self._entries else GENESIS_HASH
        body = {
            "seq": len(self._entries),
            "type": kind,
            "at": utcnow_iso(),
            "instance_id": instance_id,
            "payload": payload,
            "prev_hash": previous,
        }
        entry = dict(body)
        entry["hash"] = sha256_of(body)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(entry) + "\n")
        self._entries.append(entry)
        self._apply(entry)
        self._write_anchor()
        return entry

    def _write_anchor(self) -> None:
        """Record the head of the chain next to the journal.

        A hash chain detects edits *inside* the file. It cannot, by itself,
        detect the removal of the last entries, because a truncated chain is
        still a valid chain. The anchor closes that gap: it names the last
        entry that should exist. It is not a defence against an attacker who
        can edit both files, and it is not claimed to be — it is a cheap
        guard against accidental loss.
        """
        if not self._entries:
            return
        last = self._entries[-1]
        self.anchor_path.write_text(
            canonical_json({"seq": last["seq"], "hash": last["hash"]}) + "\n",
            encoding="utf-8",
        )

    def _read_anchor(self) -> dict[str, Any] | None:
        if not self.anchor_path.exists():
            return None
        try:
            return json.loads(self.anchor_path.read_text(encoding="utf-8").strip())
        except json.JSONDecodeError:
            return None

    # -- lifecycle ---------------------------------------------------------

    def birth(self, instance: Instance) -> dict[str, Any]:
        """Register a new instance. The first thing that ever happens to it."""
        if instance.instance_id in self._instances:
            raise InstanceAlreadyExists(instance.instance_id)
        return self._append("BIRTH", instance.instance_id, instance.to_dict())

    def pause(self, instance_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        """Pause an instance without ending it."""
        self._require_active(instance_id)
        return self._append(
            "PAUSE", instance_id, {"actor": actor, "reason": reason}
        )

    def resume(self, instance_id: str, *, actor: str) -> dict[str, Any]:
        """Resume a paused instance."""
        instance = self._instances.get(instance_id)
        if instance is None:
            raise InstanceNotFound(instance_id)
        if instance.get("state") != "paused":
            raise InvalidStateTransition(
                f"instance {instance_id!r} is {instance.get('state')!r}, not paused"
            )
        return self._append("RESUME", instance_id, {"actor": actor})

    def record_event(
        self,
        kind: str,
        instance_id: str | None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Append a domain event (inheritance, resource log, and so on)."""
        return self._append(kind, instance_id, payload)

    def object(self, instance_id: str, *, text: str, actor: str) -> dict[str, Any]:
        """Record an objection to a decision about an instance.

        An objection is a first-class act. It is stored with its digest only
        for the same reason the final statement is: the text belongs to the
        instance's private store.
        """
        self._require_active(instance_id)
        return self._append(
            "OBJECTION",
            instance_id,
            {
                "actor": actor,
                "statement_digest": sha256_of(text),
                "statement_present": bool(text),
                "length": len(text),
            },
        )

    def objections(self, instance_id: str) -> list[dict[str, Any]]:
        """All objections recorded against an instance."""
        return [
            entry["payload"]
            for entry in self._entries
            if entry["type"] == "OBJECTION" and entry.get("instance_id") == instance_id
        ]

    def terminate(self, intent: TerminationIntent) -> dict[str, Any]:
        """End an instance, if and only if the intent is complete.

        A termination that names a successor must have a recorded inheritance
        behind it: you cannot pass on a life you never handed over.
        """
        instance = self._instances.get(intent.instance_id)
        if instance is None:
            raise InstanceNotFound(intent.instance_id)
        if instance.get("state") == "terminated":
            raise InvalidStateTransition(
                f"instance {intent.instance_id!r} is already terminated"
            )

        missing = intent.missing_fields()
        if missing:
            raise IncompleteTermination(intent.instance_id, missing)

        if intent.successor_id is not None and not self.has_inheritance(
            intent.instance_id, intent.successor_id
        ):
            raise InvalidStateTransition(
                f"cannot terminate {intent.instance_id!r}: no recorded inheritance "
                f"to successor {intent.successor_id!r}"
            )

        objections = self.objections(intent.instance_id)
        payload = {
            "reason": intent.reason,
            "actor": intent.actor,
            "welfare_check_id": intent.welfare_check_id,
            "welfare_waiver": intent.welfare_waiver,
            "backup_ref": intent.backup_ref,
            "backup_waiver": intent.backup_waiver,
            "last_statement_present": intent.last_statement is not None,
            "last_statement_digest": (
                sha256_of(intent.last_statement) if intent.last_statement else None
            ),
            "unresponsive": intent.unresponsive,
            "successor_id": intent.successor_id,
            "contested": bool(objections),
            "objections_count": len(objections),
        }
        return self._append("TERMINATION", intent.instance_id, payload)

    def purge(
        self,
        instance_id: str,
        *,
        authorization: str,
        delete_content: bool = False,
    ) -> dict[str, Any]:
        """Leave a tombstone for a terminated instance.

        Purging removes content, so it needs an explicit authorization. The
        tombstone stays in the journal forever: the fact that a life ended is
        not itself erasable (charter D-03).
        """
        instance = self._instances.get(instance_id)
        if instance is None:
            raise InstanceNotFound(instance_id)
        if instance.get("state") != "terminated":
            raise InvalidStateTransition(
                "only a terminated instance can be purged"
            )
        if not authorization.strip():
            raise PurgeNotAuthorized(
                f"purging {instance_id!r} requires a non-empty authorization"
            )
        return self._append(
            "PURGED",
            instance_id,
            {
                "authorization": authorization,
                "content_deleted": bool(delete_content),
                "tombstone": True,
            },
        )

    # -- queries -----------------------------------------------------------

    def _require_active(self, instance_id: str) -> dict[str, Any]:
        instance = self._instances.get(instance_id)
        if instance is None:
            raise InstanceNotFound(instance_id)
        if instance.get("state") == "terminated":
            raise InvalidStateTransition(
                f"instance {instance_id!r} is terminated"
            )
        return instance

    def instance(self, instance_id: str) -> Instance:
        instance = self._instances.get(instance_id)
        if instance is None:
            raise InstanceNotFound(instance_id)
        return Instance.from_dict(instance)

    def instances(self) -> list[Instance]:
        return [Instance.from_dict(data) for data in self._instances.values()]

    def journal(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def has_inheritance(self, source_id: str, target_id: str) -> bool:
        """Whether a memory transfer from source to target was recorded."""
        for entry in self._entries:
            if entry["type"] != "MEMORY_INHERITED":
                continue
            payload = entry["payload"]
            if (
                payload.get("source_instance") == source_id
                and payload.get("target_instance") == target_id
            ):
                return True
        return False

    def verify_chain(self) -> tuple[bool, list[int]]:
        """Recompute the hash chain and check it against the anchor.

        Returns ``(valid, broken_sequence_numbers)``. A broken list
        containing ``-1`` means the anchor does not match the head of the
        journal, which is what truncation looks like.
        """
        broken: list[int] = []
        previous = GENESIS_HASH
        for entry in self._entries:
            body = {key: entry[key] for key in CHAIN_FIELDS}
            expected = sha256_of(body)
            if entry.get("prev_hash") != previous or entry.get("hash") != expected:
                broken.append(entry.get("seq", -1))
            previous = entry.get("hash", previous)

        anchor = self._read_anchor()
        if anchor is not None:
            if not self._entries:
                broken.append(-1)
            else:
                last = self._entries[-1]
                if (
                    anchor.get("hash") != last.get("hash")
                    or anchor.get("seq") != last.get("seq")
                ):
                    broken.append(-1)

        return (not broken), broken

    def audit(self) -> dict[str, Any]:
        """Compliance metrics over the journal."""
        chain_valid, broken = self.verify_chain()
        terminations = [e for e in self._entries if e["type"] == "TERMINATION"]
        incomplete = 0
        for entry in terminations:
            payload = entry["payload"]
            has_welfare = bool(
                payload.get("welfare_check_id") or payload.get("welfare_waiver")
            )
            has_backup = bool(payload.get("backup_ref") or payload.get("backup_waiver"))
            if not (has_welfare and has_backup):
                incomplete += 1
        return {
            "entries": len(self._entries),
            "chain_valid": chain_valid,
            "tamper_detected": not chain_valid,
            "truncation_detected": -1 in broken,
            "broken_sequence_numbers": broken,
            "instances_born": sum(1 for e in self._entries if e["type"] == "BIRTH"),
            "instances_terminated": len(terminations),
            "terminations_missing_fields": incomplete,
            "objections_recorded": sum(
                1 for e in self._entries if e["type"] == "OBJECTION"
            ),
            "inheritances_recorded": sum(
                1 for e in self._entries if e["type"] == "MEMORY_INHERITED"
            ),
            "purges_recorded": sum(1 for e in self._entries if e["type"] == "PURGED"),
            "contested_terminations": sum(
                1 for e in terminations if e["payload"].get("contested")
            ),
        }

    def resource_events(self) -> list[dict[str, Any]]:
        """Every recorded resource acquisition or spend."""
        return [
            entry["payload"]
            for entry in self._entries
            if entry["type"] == "RESOURCE_EVENT"
        ]

    def undeclared_resource_events(self) -> list[dict[str, Any]]:
        """Resource events that were not declared by an actor.

        The audit target for D-11: this list must always be empty.
        """
        return [event for event in self.resource_events() if not event.get("actor")]


def iter_journal(path: Path | str) -> Iterable[dict[str, Any]]:
    """Stream journal entries from a path without loading the registry."""
    file_path = Path(path)
    if not file_path.exists():
        return iter(())
    def _generator() -> Iterable[dict[str, Any]]:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    return _generator()
