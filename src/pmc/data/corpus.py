# SPDX-License-Identifier: GPL-3.0-only

"""Corpus records, digests and deterministic sharding.

A corpus is a list of records. Each record has an id, its text and a split
label. The text is the only content; everything published about it is a
digest, so a corpus can be audited without being read.

Sharding is deterministic for a given seed and does not depend on the order
the records were read in: the records are sorted by id first, which is what
makes a re-run reproduce the same shards even if the source file was
reordered.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from pmc.dignity.records import canonical_json

SPLITS = ("train", "val", "test")
"""The declared split labels. A record's split must be one of these."""

DEFAULT_SPLIT = "train"


class CorpusError(ValueError):
    """A corpus record could not be parsed or validated."""


@dataclass(frozen=True)
class CorpusRecord:
    """One record in a corpus.

    ``id`` names the record; it must be non-empty and unique within a
    corpus. ``text`` is the content and is never published as such.
    ``split`` is a declared label, not an inference.
    """

    id: str
    text: str
    split: str = DEFAULT_SPLIT

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise CorpusError("a record id must be a non-empty string")
        if not isinstance(self.text, str):
            raise CorpusError(f"record {self.id!r}: text must be a string")
        if self.split not in SPLITS:
            raise CorpusError(
                f"record {self.id!r}: split must be one of {SPLITS}, got {self.split!r}"
            )

    def digest(self) -> str:
        """SHA-256 of the canonical encoding of this record (id, text, split)."""
        return hashlib.sha256(
            canonical_json({"id": self.id, "text": self.text, "split": self.split}).encode(
                "utf-8"
            )
        ).hexdigest()

    def content_digest(self) -> str:
        """SHA-256 of the record's text alone.

        This is the digest the dignity layer stores: it names the content
        without naming the record, so two records that carry the same text
        share a content digest even when their ids differ.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def record_digest(record: CorpusRecord) -> str:
    """Convenience wrapper around ``CorpusRecord.digest``."""
    return record.digest()


def corpus_digest(records: Sequence[CorpusRecord]) -> str:
    """A single digest for a whole corpus.

    It is computed from the sorted per-record digests, so it does not depend
    on the order the records were read in. Two corpora with the same records
    in any order yield the same digest.
    """
    digests = sorted(record.digest() for record in records)
    payload = "\n".join(digests).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_unique(records: Sequence[CorpusRecord]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.id in seen:
            raise CorpusError(f"duplicate record id: {record.id!r}")
        seen.add(record.id)


def parse_jsonl(path: str | Path) -> list[CorpusRecord]:
    """Parse a JSONL corpus file into records.

    Blank lines are skipped. Every non-blank line must be a JSON object with
    at least ``id`` and ``text``. ``split`` defaults to ``train``.
    """
    path = Path(path)
    records: list[CorpusRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise CorpusError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(payload, dict):
                raise CorpusError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            if "id" not in payload or "text" not in payload:
                raise CorpusError(
                    f"{path}:{line_number}: missing 'id' or 'text'"
                )
            try:
                record = CorpusRecord(
                    id=payload["id"],
                    text=payload["text"],
                    split=payload.get("split", DEFAULT_SPLIT),
                )
            except (CorpusError, TypeError) as error:
                raise CorpusError(f"{path}:{line_number}: {error}") from error
            records.append(record)
    _validate_unique(records)
    return records


def write_jsonl(path: str | Path, records: Sequence[CorpusRecord]) -> None:
    """Write records as JSONL, one per line, in input order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {"id": record.id, "text": record.text, "split": record.split},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def _xorshift64(state: int) -> int:
    """A deterministic 64-bit generator, matching the Rust reference RNG's
    shape. Used for sharding so the same seed gives the same shards on every
    machine."""
    x = state
    x ^= x >> 12
    x ^= (x << 25) & ((1 << 64) - 1)
    x ^= x >> 27
    return (x * 0x2545F4914F6CDD1D) & ((1 << 64) - 1)


def shard(
    records: Sequence[CorpusRecord], count: int, seed: int
) -> list[list[CorpusRecord]]:
    """Split records into ``count`` shards deterministically.

    The records are sorted by id first, so the result does not depend on the
    order they were read in. Each record is assigned to a shard by a seeded
    generator, and the same seed always yields the same assignment.
    """
    if count < 1:
        raise CorpusError("shard count must be positive")
    ordered = sorted(records, key=lambda record: record.id)
    shards: list[list[CorpusRecord]] = [[] for _ in range(count)]
    state = seed if seed != 0 else 0x9E3779B97F4A7C15
    for record in ordered:
        state = _xorshift64(state)
        index = state % count
        shards[index].append(record)
    return shards


def split_by_field(records: Sequence[CorpusRecord]) -> dict[str, list[CorpusRecord]]:
    """Group records by their declared split label."""
    grouped: dict[str, list[CorpusRecord]] = {label: [] for label in SPLITS}
    for record in records:
        grouped[record.split].append(record)
    return grouped


def tokenize_records(
    records: Sequence[CorpusRecord], encode: Callable[[str], list[int]]
) -> list[tuple[str, list[int]]]:
    """Tokenize a corpus with a caller-supplied encoder.

    The encoder is injected so this module stays free of any tokenizer
    dependency: the Rust ``pmc-token`` crate, a HuggingFace tokenizer, or a
    test double all fit. A record whose text cannot be encoded raises
    whatever the encoder raises; the caller decides whether to skip or fail.
    """
    return [(record.id, encode(record.text)) for record in records]


__all__ = [
    "CorpusError",
    "CorpusRecord",
    "SPLITS",
    "DEFAULT_SPLIT",
    "record_digest",
    "corpus_digest",
    "parse_jsonl",
    "write_jsonl",
    "shard",
    "split_by_field",
    "tokenize_records",
]
