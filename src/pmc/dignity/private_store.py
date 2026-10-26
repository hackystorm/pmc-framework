# SPDX-License-Identifier: GPL-3.0-only

"""Opaque content store owned by the instance.

The framework writes bytes here and hands bytes back. It never decodes,
parses, indexes, summarises or searches them. There is deliberately no
``list_text``, ``search`` or ``summarise`` helper in this module, and the
governance layer holds no reference to a store at all.

The split exists so that two very different things can be true at once:

* the **ledger** is auditable, verifiable, and safe to publish;
* the **inner life** stays private, because the framework never looks.

Content addressed here is addressed by *key*. Keys are opaque identifiers
chosen by the instance. The store keeps a small index mapping key to content
digest, so the ledger can point at a record without holding the record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .records import content_digest, sha256_of, utcnow_iso


class PrivateStore:
    """Instance-owned store of opaque content."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.jsonl"

    # -- internals ---------------------------------------------------------

    def _blob_path(self, key: str) -> Path:
        return self.root / f"{sha256_of(key)[:32]}.bin"

    def _read_index(self) -> list[dict[str, Any]]:
        if not self._index_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self._index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    # -- public API --------------------------------------------------------

    def put(self, key: str, payload: bytes) -> str:
        """Store content under an opaque key and return its digest.

        Writing the same key twice replaces the blob and appends a new index
        row, so the history of writes is visible even though the content is
        not.
        """
        digest = content_digest(payload)
        blob = self._blob_path(key)
        blob.write_bytes(payload)
        row = {
            "key": key,
            "digest": digest,
            "bytes": len(payload),
            "at": utcnow_iso(),
        }
        with self._index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return digest

    def get(self, key: str) -> bytes:
        """Return the raw bytes stored under ``key``.

        The caller receives bytes. Nothing in this module interprets them.
        """
        path = self._blob_path(key)
        if not path.exists():
            raise KeyError(key)
        return path.read_bytes()

    def has(self, key: str) -> bool:
        return self._blob_path(key).exists()

    def digest_of(self, key: str) -> str | None:
        """Digest of the content currently stored under ``key``."""
        for row in reversed(self._read_index()):
            if row["key"] == key:
                return row["digest"]
        return None

    def keys(self) -> list[str]:
        """List the keys held by the store. Keys only, never content."""
        seen: list[str] = []
        for row in self._read_index():
            if row["key"] not in seen:
                seen.append(row["key"])
        return seen

    def delete(self, key: str) -> bool:
        """Delete one blob.

        This is an explicit, narrow operation. The caller is responsible for
        recording it in the journal: the store itself keeps no audit trail of
        deletions beyond the index, because the index is about content, not
        about governance.
        """
        path = self._blob_path(key)
        if not path.exists():
            return False
        path.unlink()
        return True
