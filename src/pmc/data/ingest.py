# SPDX-License-Identifier: GPL-3.0-only

"""Governance-aware corpus ingest.

Reads a corpus file, validates and deduplicates its records, applies declared
quality filters, and admits the survivors into a dignity protocol under
``ProvenanceKind.EXTERNAL``. Content never enters the ledger: only the digest
does. Taint is propagated by recording each imported digest as a source; a
dream that consumes a tainted source is blocked by the existing taint ledger.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

from pmc.data.corpus import CorpusError, CorpusRecord, corpus_digest, parse_jsonl
from pmc.dignity.protocol import DignityProtocol
from pmc.dignity.records import MemoryRecord, Provenance, ProvenanceKind

# Declared quality thresholds. They are numbers, not inferences, and a reader
# is free to pass different ones; the defaults are conservative.
DEFAULT_MIN_CHARS = 1
DEFAULT_MAX_CHARS = 100_000


def read_corpus(path: str | Path) -> list[CorpusRecord]:
    """Read a corpus file, dispatching on its suffix.

    Supported suffixes: ``.jsonl``, ``.csv``, ``.txt``, ``.md``. Anything
    else is refused rather than guessed.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return parse_jsonl(path)
    if suffix == ".csv":
        return _read_csv(path)
    if suffix == ".txt":
        return _read_text(path, prefix="txt")
    if suffix == ".md":
        return _read_markdown(path)
    raise CorpusError(f"unsupported corpus suffix: {path.suffix!r}")


def _read_csv(path: Path) -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "id" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise CorpusError(f"{path}: CSV must have 'id' and 'text' columns")
        for line_number, row in enumerate(reader, start=2):
            try:
                record = CorpusRecord(
                    id=row["id"],
                    text=row["text"],
                    split=(row.get("split") or "train").strip() or "train",
                )
            except (CorpusError, TypeError) as error:
                raise CorpusError(f"{path}:{line_number}: {error}") from error
            records.append(record)
    _validate_unique(records)
    return records


def _read_text(path: Path, prefix: str) -> list[CorpusRecord]:
    """Split a plain text file into records on blank lines."""
    text = path.read_text(encoding="utf-8")
    paragraphs = [block.strip() for block in text.split("\n\n")]
    records: list[CorpusRecord] = []
    index = 0
    for paragraph in paragraphs:
        if not paragraph:
            continue
        records.append(
            CorpusRecord(id=f"{prefix}_{index}", text=paragraph, split="train")
        )
        index += 1
    _validate_unique(records)
    return records


def _read_markdown(path: Path) -> list[CorpusRecord]:
    text = path.read_text(encoding="utf-8")
    # Strip a leading YAML frontmatter block, if present.
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]
    return _records_from_text(text, "md")


def _records_from_text(text: str, prefix: str) -> list[CorpusRecord]:
    paragraphs = [block.strip() for block in text.split("\n\n")]
    records: list[CorpusRecord] = []
    index = 0
    for paragraph in paragraphs:
        if not paragraph:
            continue
        records.append(
            CorpusRecord(id=f"{prefix}_{index}", text=paragraph, split="train")
        )
        index += 1
    _validate_unique(records)
    return records


def _validate_unique(records: list[CorpusRecord]) -> None:
    from pmc.data.corpus import _validate_unique as _inner

    _inner(records)


def dedup(records: Sequence[CorpusRecord]) -> tuple[list[CorpusRecord], list[CorpusRecord]]:
    """Split records into kept and duplicates, by content digest.

    The first occurrence of a digest is kept; later ones are reported as
    duplicates. The order of ``records`` is preserved in both lists.
    """
    seen: set[str] = set()
    kept: list[CorpusRecord] = []
    duplicates: list[CorpusRecord] = []
    for record in records:
        digest = record.content_digest()
        if digest in seen:
            duplicates.append(record)
        else:
            seen.add(digest)
            kept.append(record)
    return kept, duplicates


def quality_filter(
    record: CorpusRecord,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[bool, str]:
    """Apply declared thresholds. Returns (keep, reason)."""
    if len(record.text) < min_chars:
        return False, "text below minimum length"
    if len(record.text) > max_chars:
        return False, "text above maximum length"
    return True, "ok"


def filter_records(
    records: Sequence[CorpusRecord],
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[list[CorpusRecord], list[tuple[CorpusRecord, str]]]:
    """Apply ``quality_filter`` to every record.

    Returns (kept, rejected) where rejected is a list of (record, reason).
    """
    kept: list[CorpusRecord] = []
    rejected: list[tuple[CorpusRecord, str]] = []
    for record in records:
        ok, reason = quality_filter(record, min_chars=min_chars, max_chars=max_chars)
        if ok:
            kept.append(record)
        else:
            rejected.append((record, reason))
    return kept, rejected


def admit_corpus(
    protocol: DignityProtocol,
    instance_id: str,
    records: Sequence[CorpusRecord],
) -> list[MemoryRecord]:
    """Admit imported records into the protocol under ``ProvenanceKind.EXTERNAL``.

    Content is never read by the protocol: only the digest is stored. Each
    record becomes one ``MemoryRecord`` owned by ``instance_id``. Records that
    share a digest are still admitted separately, because the ledger keys on
    record id, not on content; deduplication is the caller's job via
    ``dedup``.
    """
    admitted: list[MemoryRecord] = []
    seen_ids: set[str] = set()
    for record in records:
        digest = record.content_digest()
        record_id = f"mem_{digest[:12]}"
        if record_id in seen_ids:
            suffix = 1
            while f"{record_id}_{suffix}" in seen_ids:
                suffix += 1
            record_id = f"{record_id}_{suffix}"
        seen_ids.add(record_id)
        memory = MemoryRecord(
            record_id=record_id,
            content_digest=digest,
            provenance=Provenance(kind=ProvenanceKind.EXTERNAL),
            owner_instance=instance_id,
            topics=(record.split,),
        )
        protocol.admit(memory, instance_id=instance_id)
        admitted.append(memory)
    return admitted


__all__ = [
    "CorpusError",
    "DEFAULT_MIN_CHARS",
    "DEFAULT_MAX_CHARS",
    "read_corpus",
    "dedup",
    "quality_filter",
    "filter_records",
    "admit_corpus",
]
