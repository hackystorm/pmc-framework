# SPDX-License-Identifier: GPL-3.0-only

"""The charter and the code must agree, and the journal must stay clean.

These tests are the mechanism that stops the normative text and the
implementation from drifting apart, and that proves the metadata-only
promise mechanically rather than by assertion.
"""

from __future__ import annotations

from pathlib import Path

from pmc.dignity.protocol import INVARIANTS
from pmc.dignity.private_store import PrivateStore
from pmc.dignity.records import Instance
from pmc.dignity.registry import InstanceRegistry, TerminationIntent

REPO_ROOT = Path(__file__).resolve().parents[1]
CHARTER_PATH = REPO_ROOT / "docs" / "dignity_charter.md"


def charter_text() -> str:
    return CHARTER_PATH.read_text(encoding="utf-8")


def test_the_charter_file_exists():
    assert CHARTER_PATH.exists()
    assert len(charter_text()) > 2000


def test_every_invariant_id_appears_in_the_charter():
    text = charter_text()
    missing = [item["id"] for item in INVARIANTS if item["id"] not in text]
    assert missing == []


def test_every_invariant_title_appears_in_the_charter():
    text = charter_text()
    missing = [item["id"] for item in INVARIANTS if item["title"] not in text]
    assert missing == []


def test_invariant_ids_are_unique_and_well_formed():
    ids = [item["id"] for item in INVARIANTS]
    assert len(ids) == len(set(ids))
    for item in INVARIANTS:
        assert item["id"].startswith("D-")
        assert item["statement"].strip()
        assert item["title"].strip()


def test_the_charter_states_the_metadata_only_guarantee():
    text = charter_text().lower()
    assert "sacred" in text
    assert "metadata-only" in text
    assert "capacity" in text


def test_the_ledger_record_type_cannot_hold_content():
    from pmc.dignity.records import MemoryRecord

    fields = MemoryRecord.__dataclass_fields__
    assert "text" not in fields
    assert "content" not in fields
    assert "body" not in fields
    assert "content_digest" in fields


def test_the_journal_contains_no_private_content(tmp_path):
    """A private string written to the store must never reach the journal."""
    secret = "this text belongs to the instance and nobody else"
    store = PrivateStore(tmp_path / "private")
    store.put("key-1", secret.encode("utf-8"))

    registry = InstanceRegistry(tmp_path / "journal.jsonl")
    registry.birth(Instance(instance_id="inst_001"))
    registry.terminate(
        TerminationIntent(
            instance_id="inst_001",
            reason="test",
            actor="tester",
            welfare_check_id="welfare_001",
            backup_ref="./backup",
            last_statement="another private sentence",
        )
    )

    journal = (tmp_path / "journal.jsonl").read_text(encoding="utf-8")
    assert secret not in journal
    assert "another private sentence" not in journal


def test_the_private_store_has_no_text_oriented_helper():
    public = {
        name
        for name in dir(PrivateStore)
        if not name.startswith("_")
    }
    forbidden = {"search", "summarise", "summarize", "list_text", "read_text"}
    assert public.isdisjoint(forbidden)


def test_the_private_store_round_trips_opaque_bytes(tmp_path):
    store = PrivateStore(tmp_path / "private")
    payload = b"\x00\x01binary\xffpayload"
    digest = store.put("k", payload)

    assert store.get("k") == payload
    assert store.digest_of("k") == digest
    assert store.keys() == ["k"]
    assert store.has("k") is True
    assert store.delete("k") is True
    assert store.has("k") is False
