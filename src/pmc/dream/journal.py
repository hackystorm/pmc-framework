# SPDX-License-Identifier: GPL-3.0-only

"""Hash-chained dream journal.

Every dream appends one line to ``sidecar/dreams.jsonl``: the dream id, the
checkpoint digests, the selected record ids and source digests, the metrics,
the gate result and the outcome. No memory content is written here, and no
inner state is named.

The chain is deliberately small: each entry carries the hash of the previous
entry and its own canonical hash. Truncation, reordering or an edited line
shows up in :meth:`DreamJournal.verify_chain`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..dignity.records import sha256_of, utcnow_iso

GENESIS = "0" * 64


class DreamJournalError(RuntimeError):
    """The dream journal is malformed."""


class DreamJournal:
    """An append-only, hash-chained dream journal."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError as error:
                    raise DreamJournalError("the dream journal contains invalid JSON") from error
                if not isinstance(row, dict):
                    raise DreamJournalError("a dream journal entry is not an object")
                rows.append(row)
        return rows

    def head(self) -> str:
        rows = self.entries()
        if not rows:
            return GENESIS
        value = rows[-1].get("hash")
        if not isinstance(value, str):
            raise DreamJournalError("the last dream journal entry has no hash")
        return value

    def append(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Append one entry and return the stored record with chain fields."""
        if not isinstance(entry, dict):
            raise DreamJournalError("a dream journal entry must be an object")
        reserved = {"seq", "at", "previous", "hash"}
        if reserved & set(entry):
            raise DreamJournalError("the caller cannot supply chain fields")
        rows = self.entries()
        payload = dict(entry)
        payload["seq"] = len(rows) + 1
        payload["at"] = utcnow_iso()
        payload["previous"] = rows[-1]["hash"] if rows else GENESIS
        payload["hash"] = sha256_of(payload)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return payload

    def verify_chain(self) -> tuple[bool, list[int]]:
        """Return ``(valid, broken_sequence_numbers)`` for the whole chain."""
        try:
            rows = self.entries()
        except DreamJournalError:
            return False, [0]
        broken: list[int] = []
        previous = GENESIS
        for index, row in enumerate(rows, start=1):
            if row.get("seq") != index:
                broken.append(index)
            if row.get("previous") != previous:
                broken.append(index)
            body = {key: value for key, value in row.items() if key != "hash"}
            if row.get("hash") != sha256_of(body):
                broken.append(index)
            previous = row.get("hash", "")
        return not broken, sorted(set(broken))
