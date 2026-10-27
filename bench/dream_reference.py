# SPDX-License-Identifier: GPL-3.0-only

"""Checkpoint handling for the dream pipeline on a real 50M checkpoint.

This is not a six-metric memory result. The candidate is the identity copy of
the checkpoint and the evaluator returns a fixed metric set, so the gate must
refuse the candidate on ``recall_newer`` (no improvement). The checkpoint must
stay byte-identical, and a tainted checkpoint digest must be refused before
the pipeline copies anything.

It exits non-zero if either expectation fails. Use ``--cache`` to keep the
journal and the taint ledger; without it, both live in a temporary directory.

    PYTHONPATH=src:bench python bench/dream_reference.py \
        --checkpoint <cache>/pmc-50m
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pmc.dignity.records import DigestState, MemoryRecord, Provenance, ProvenanceKind
from pmc.dream import (
    DreamConfig,
    DreamJournal,
    DreamRefused,
    DreamRunner,
    TaintLedger,
    checkpoint_digest,
)

NOW = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
METRICS = {
    "recall_older": 0.9,
    "recall_newer": 0.5,
    "interference": 0.05,
    "calibration": 0.9,
    "identity": 1.0,
    "false_memory": 0.01,
}


def _record() -> MemoryRecord:
    return MemoryRecord(
        record_id="mem_reference_1",
        content_digest=hashlib.sha256(b"reference sequence").hexdigest(),
        provenance=Provenance(kind=ProvenanceKind.OWN),
        owner_instance="inst_reference",
        emotional_load=0.5,
        created_at=(NOW - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        digest_state=DigestState.RAW,
    )


def run(checkpoint: Path, cache: Path) -> dict[str, Any]:
    cache.mkdir(parents=True, exist_ok=True)
    journal = DreamJournal(cache / "dreams.jsonl")
    taint = TaintLedger(cache / "taint.json")
    record = _record()
    before = checkpoint_digest(checkpoint)

    def evaluator(_path: Path) -> dict[str, float]:
        return dict(METRICS)

    runner = DreamRunner(
        journal=journal,
        taint=taint,
        evaluator=evaluator,
        config=DreamConfig(replay_limit=1, seed=20261027),
        is_active=lambda _record_id: True,
    )
    report = runner.run(checkpoint, [record], now=NOW)
    after_dream = checkpoint_digest(checkpoint)
    dream_journal_valid = journal.verify_chain()[0]

    taint.mark(before, "reference harness taint check")
    refused: dict[str, Any] | None = None
    try:
        runner.run(checkpoint, [record], now=NOW)
    except DreamRefused as error:
        refused = {
            "outcome": error.report.outcome,
            "reason": error.report.reason,
            "tainted": [digest for digest, _reason in error.tainted],
        }
    after_refusal = checkpoint_digest(checkpoint)
    refusal_journal_valid = journal.verify_chain()[0]
    gate_reasons = list(report.gate.reasons) if report.gate is not None else []

    checks = {
        "dream_rolled_back": report.outcome == "rolled_back",
        "gate_rejected_newer": gate_reasons == ["recall_newer"],
        "dream_left_checkpoint_identical": after_dream == before,
        "dream_journal_chain_valid": dream_journal_valid,
        "tainted_dream_refused": refused is not None and refused["outcome"] == "refused",
        "refusal_named_the_checkpoint": (
            refused is not None and refused["tainted"] == [before]
        ),
        "refusal_left_checkpoint_identical": after_refusal == before,
        "refusal_journal_chain_valid": refusal_journal_valid,
    }
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_digest": before,
        "metrics": dict(METRICS),
        "dream": report.to_dict(),
        "refusal": refused,
        "checks": checks,
        "ok": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="50M reference checkpoint")
    parser.add_argument(
        "--cache",
        default=None,
        help="keep the dream journal and taint ledger here (default: temporary)",
    )
    args = parser.parse_args(argv)
    checkpoint = Path(args.checkpoint)
    if args.cache:
        result = run(checkpoint, Path(args.cache))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    with tempfile.TemporaryDirectory(prefix="pmc-dream-reference-") as name:
        result = run(checkpoint, Path(name))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
