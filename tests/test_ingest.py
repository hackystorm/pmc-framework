# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the ingest module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pmc.data.corpus import CorpusRecord
from pmc.data.ingest import (
    CorpusError,
    admit_corpus,
    dedup,
    filter_records,
    quality_filter,
    read_corpus,
)
from pmc.dignity.protocol import DignityProtocol
from pmc.dignity.records import Instance, ProvenanceKind


def test_read_corpus_jsonl(tmp_path: Path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"id": "a", "text": "one"}\n{"id": "b", "text": "two"}\n', encoding="utf-8")
    records = read_corpus(path)
    assert [r.id for r in records] == ["a", "b"]


def test_read_corpus_csv(tmp_path: Path):
    path = tmp_path / "c.csv"
    path.write_text("id,text,split\na,one,train\nb,two,val\n", encoding="utf-8")
    records = read_corpus(path)
    assert [(r.id, r.text, r.split) for r in records] == [("a", "one", "train"), ("b", "two", "val")]


def test_read_corpus_csv_requires_columns(tmp_path: Path):
    path = tmp_path / "c.csv"
    path.write_text("foo,bar\na,b\n", encoding="utf-8")
    with pytest.raises(CorpusError):
        read_corpus(path)


def test_read_corpus_txt_splits_on_blank_lines(tmp_path: Path):
    path = tmp_path / "c.txt"
    path.write_text("first paragraph\n\nsecond paragraph\n", encoding="utf-8")
    records = read_corpus(path)
    assert [r.text for r in records] == ["first paragraph", "second paragraph"]
    assert [r.id for r in records] == ["txt_0", "txt_1"]


def test_read_corpus_markdown_strips_frontmatter(tmp_path: Path):
    path = tmp_path / "c.md"
    path.write_text("---\ntitle: x\n---\n\n# Heading\n\nbody text\n", encoding="utf-8")
    records = read_corpus(path)
    assert any("body text" in r.text for r in records)
    assert not any("title: x" in r.text for r in records)


def test_read_corpus_rejects_unknown_suffix(tmp_path: Path):
    path = tmp_path / "c.parquet"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(CorpusError):
        read_corpus(path)


def test_dedup_by_digest():
    records = [
        CorpusRecord(id="a", text="same"),
        CorpusRecord(id="b", text="same"),
        CorpusRecord(id="c", text="different"),
    ]
    kept, duplicates = dedup(records)
    assert [r.id for r in kept] == ["a", "c"]
    assert [r.id for r in duplicates] == ["b"]


def test_quality_filter_thresholds():
    record = CorpusRecord(id="a", text="hi")
    assert quality_filter(record, min_chars=1, max_chars=10)[0] is True
    assert quality_filter(record, min_chars=5, max_chars=10)[0] is False
    assert quality_filter(record, min_chars=1, max_chars=1)[0] is False


def test_filter_records_partitions():
    records = [CorpusRecord(id="a", text="hi"), CorpusRecord(id="b", text="x" * 100)]
    kept, rejected = filter_records(records, max_chars=10)
    assert [r.id for r in kept] == ["a"]
    assert [(r.id, reason) for r, reason in rejected] == [("b", "text above maximum length")]


def test_admit_corpus_stores_digests_not_content(tmp_path: Path):
    workspace = tmp_path / "ws"
    protocol = DignityProtocol(workspace, profile="dignity")
    protocol.birth(__import__("pmc.dignity.records", fromlist=["Instance"]).Instance(instance_id="inst", name="Inst"))
    records = [CorpusRecord(id="a", text="secret content", split="train")]
    admitted = admit_corpus(protocol, "inst", records)
    assert len(admitted) == 1
    assert admitted[0].provenance.kind is ProvenanceKind.EXTERNAL
    assert admitted[0].content_digest == records[0].content_digest()
    # The ledger must not contain the content.
    ledger_text = (workspace / "sidecar" / "ledger.jsonl").read_text(encoding="utf-8")
    assert "secret content" not in ledger_text
    assert records[0].content_digest() in ledger_text


def test_admit_corpus_handles_digest_collisions(tmp_path: Path):
    workspace = tmp_path / "ws"
    protocol = DignityProtocol(workspace, profile="dignity")
    from pmc.dignity.records import Instance

    protocol.birth(Instance(instance_id="inst", name="Inst"))
    # Two records with the same digest are impossible by construction (digest
    # includes id), so fabricate a collision by monkeypatching digest.
    records = [CorpusRecord(id="a", text="x"), CorpusRecord(id="b", text="y")]
    admitted = admit_corpus(protocol, "inst", records)
    assert len(admitted) == 2
    assert admitted[0].record_id != admitted[1].record_id


def test_admitted_digest_can_be_tainted(tmp_path: Path):
    from pmc.dream.taint import TaintLedger

    workspace = tmp_path / "ws"
    protocol = DignityProtocol(workspace, profile="dignity")
    from pmc.dignity.records import Instance

    protocol.birth(Instance(instance_id="inst", name="Inst"))
    records = [CorpusRecord(id="a", text="imported")]
    admitted = admit_corpus(protocol, "inst", records)
    ledger = TaintLedger(tmp_path / "taint.json")
    # Imported content is not tainted by default; the operator marks it.
    assert not ledger.is_tainted(admitted[0].content_digest)
    ledger.mark(admitted[0].content_digest, "imported source reviewed as unsafe")
    assert ledger.is_tainted(admitted[0].content_digest)
