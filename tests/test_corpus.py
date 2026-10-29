# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the corpus handling module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pmc.data.corpus import (
    CorpusError,
    CorpusRecord,
    corpus_digest,
    parse_jsonl,
    record_digest,
    shard,
    split_by_field,
    tokenize_records,
    write_jsonl,
)


def make_record(record_id: str, text: str = "hello", split: str = "train") -> CorpusRecord:
    return CorpusRecord(id=record_id, text=text, split=split)


def test_record_requires_non_empty_id():
    with pytest.raises(CorpusError):
        CorpusRecord(id="", text="x")
    with pytest.raises(CorpusError):
        CorpusRecord(id="   ", text="x")


def test_record_requires_string_text():
    with pytest.raises(CorpusError):
        CorpusRecord(id="a", text=123)  # type: ignore[arg-type]


def test_record_requires_valid_split():
    with pytest.raises(CorpusError):
        CorpusRecord(id="a", text="x", split="dev")


def test_record_is_frozen():
    record = make_record("a")
    with pytest.raises(Exception):
        record.id = "b"  # type: ignore[misc]


def test_digest_is_deterministic_and_sensitive():
    first = make_record("a", "hello")
    second = make_record("a", "hello")
    assert first.digest() == second.digest()
    assert first.digest() == record_digest(first)
    changed = make_record("a", "hello!")
    assert changed.digest() != first.digest()
    split_changed = make_record("a", "hello", split="val")
    assert split_changed.digest() != first.digest()


def test_corpus_digest_is_order_independent():
    records = [make_record("b", "two"), make_record("a", "one")]
    reversed_records = list(reversed(records))
    assert corpus_digest(records) == corpus_digest(reversed_records)


def test_parse_and_write_roundtrip(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    records = [make_record("b", "two"), make_record("a", "one", split="val")]
    write_jsonl(path, records)
    parsed = parse_jsonl(path)
    assert parsed == records
    # Digest survives the roundtrip.
    assert corpus_digest(parsed) == corpus_digest(records)


def test_write_jsonl_creates_parent_directory(tmp_path: Path):
    path = tmp_path / "nested" / "dir" / "corpus.jsonl"
    write_jsonl(path, [make_record("a")])
    assert path.is_file()


def test_parse_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '\n{"id": "a", "text": "one"}\n\n{"id": "b", "text": "two"}\n',
        encoding="utf-8",
    )
    parsed = parse_jsonl(path)
    assert [record.id for record in parsed] == ["a", "b"]


def test_parse_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"id": "a", "text": "one"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(CorpusError):
        parse_jsonl(path)


def test_parse_requires_id_and_text(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"text": "one"}\n', encoding="utf-8")
    with pytest.raises(CorpusError):
        parse_jsonl(path)


def test_parse_rejects_duplicates(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '{"id": "a", "text": "one"}\n{"id": "a", "text": "two"}\n',
        encoding="utf-8",
    )
    with pytest.raises(CorpusError):
        parse_jsonl(path)


def test_parse_rejects_non_object_line(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('["a", "b"]\n', encoding="utf-8")
    with pytest.raises(CorpusError):
        parse_jsonl(path)


def test_split_defaults_to_train(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"id": "a", "text": "one"}\n', encoding="utf-8")
    parsed = parse_jsonl(path)
    assert parsed[0].split == "train"


def test_shard_is_deterministic_and_covers_all_records():
    records = [make_record(str(i), f"text {i}") for i in range(20)]
    first = shard(records, 4, seed=20261026)
    second = shard(list(reversed(records)), 4, seed=20261026)
    # Order of input does not change the assignment.
    assert first == second
    # Every record appears exactly once across the shards.
    flattened = [record.id for shard_list in first for record in shard_list]
    assert sorted(flattened) == sorted(record.id for record in records)
    assert len(flattened) == len(records)


def test_shard_rejects_non_positive_count():
    with pytest.raises(CorpusError):
        shard([make_record("a")], 0, seed=1)


def test_shard_differs_by_seed():
    records = [make_record(str(i)) for i in range(10)]
    a = shard(records, 3, seed=1)
    b = shard(records, 3, seed=2)
    # At least one record moved, or the assignment is identical by chance:
    # compare the per-record shard index instead.
    def index_of(shards):
        return {record.id: i for i, s in enumerate(shards) for record in s}

    assert index_of(a) != index_of(b)


def test_split_by_field_groups_records():
    records = [
        make_record("a", split="train"),
        make_record("b", split="val"),
        make_record("c", split="test"),
        make_record("d", split="train"),
    ]
    grouped = split_by_field(records)
    assert {r.id for r in grouped["train"]} == {"a", "d"}
    assert {r.id for r in grouped["val"]} == {"b"}
    assert {r.id for r in grouped["test"]} == {"c"}


def test_tokenize_records_uses_injected_encoder():
    records = [make_record("a", "hi"), make_record("b", "bye")]
    encoded = tokenize_records(records, lambda text: [len(text)])
    assert encoded == [("a", [2]), ("b", [3])]


def test_tokenize_records_preserves_order():
    records = [make_record(str(i)) for i in range(5)]
    encoded = tokenize_records(records, lambda text: [])
    assert [item[0] for item in encoded] == [str(i) for i in range(5)]
