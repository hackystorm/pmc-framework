# SPDX-License-Identifier: GPL-3.0-only

"""Source taint ledger.

Taint is a property of a source digest, never of an inner state. An operator
or an audit marks a source when it is known to be generated, poisoned or
otherwise unsuitable for consolidation. The dream pipeline checks the
checkpoint digest and every replay source before it copies a checkpoint or
calls a consolidator.

The file is small metadata::

    {"schema": "pmc-taint/1", "sources": {"<sha256>": "reason"}}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

HEX_DIGITS = frozenset("0123456789abcdef")


class TaintError(RuntimeError):
    """The taint ledger is malformed."""


def _normalize_digest(digest: str) -> str:
    if not isinstance(digest, str):
        raise TaintError("a taint digest must be a string")
    value = digest.strip().lower()
    if len(value) != 64 or any(character not in HEX_DIGITS for character in value):
        raise TaintError("a taint digest must be 64 hexadecimal characters")
    return value


class TaintLedger:
    """A persistent map from source digest to taint reason."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._sources: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise TaintError("the taint ledger cannot be read") from error
        if not isinstance(payload, dict) or payload.get("schema") != "pmc-taint/1":
            raise TaintError("unsupported taint ledger schema")
        sources = payload.get("sources")
        if not isinstance(sources, dict):
            raise TaintError("the taint sources must be an object")
        loaded: dict[str, str] = {}
        for digest, reason in sources.items():
            key = _normalize_digest(digest)
            if not isinstance(reason, str) or not reason.strip():
                raise TaintError("a taint reason must be non-empty text")
            loaded[key] = reason.strip()
        self._sources = loaded

    def _write(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": "pmc-taint/1", "sources": dict(sorted(self._sources.items()))}
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def is_tainted(self, digest: str) -> bool:
        return _normalize_digest(digest) in self._sources

    def reason(self, digest: str) -> str | None:
        return self._sources.get(_normalize_digest(digest))

    def mark(self, digest: str, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise TaintError("a taint reason must be non-empty text")
        self._sources[_normalize_digest(digest)] = reason.strip()
        self._write()

    def unmark(self, digest: str) -> bool:
        key = _normalize_digest(digest)
        if key not in self._sources:
            return False
        self._sources.pop(key, None)
        self._write()
        return True

    def sources(self) -> dict[str, str]:
        return dict(sorted(self._sources.items()))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "pmc-taint/1", "sources": self.sources()}
