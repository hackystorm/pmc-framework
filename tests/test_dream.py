# SPDX-License-Identifier: GPL-3.0-only

"""Unit checks for the dream pipeline.

The tests never touch a real model. They exercise the parts that must be
identical for every consolidation: metadata-only replay, the gate directions,
the hash-chained journal, the taint refusal, the atomic commit and the
byte-identical rollback.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pmc.dignity.records import DigestState, MemoryRecord, Provenance, ProvenanceKind
from pmc.dream import (
    CheckpointError,
    DreamConfig,
    DreamError,
    DreamJournal,
    DreamJournalError,
    DreamRefused,
    DreamRunner,
    GateConfig,
    GateError,
    ReplayError,
    TaintError,
    TaintLedger,
    atomic_commit,
    checkpoint_digest,
    command_consolidator,
    command_evaluator,
    copy_checkpoint,
    copy_consolidator,
    evaluate_gate,
    restore_checkpoint,
    salience,
    select_replay,
)

NOW = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
GOOD = {
    "recall_older": 0.80,
    "recall_newer": 0.50,
    "interference": 0.05,
    "calibration": 0.90,
    "identity": 1.0,
    "false_memory": 0.01,
}
BETTER = {
    "recall_older": 0.79,
    "recall_newer": 0.60,
    "interference": 0.06,
    "calibration": 0.89,
    "identity": 1.0,
    "false_memory": 0.02,
}


def make_record(
    record_id: str,
    *,
    age_days: float = 0.0,
    load: float = 0.5,
    access: int = 0,
    state: DigestState = DigestState.RAW,
    last_access_days: float | None = None,
    owner: str = "inst_test",
) -> MemoryRecord:
    created = NOW - timedelta(days=age_days)
    last = NOW - timedelta(days=last_access_days) if last_access_days is not None else None
    return MemoryRecord(
        record_id=record_id,
        content_digest=hashlib.sha256(record_id.encode("utf-8")).hexdigest(),
        provenance=Provenance(kind=ProvenanceKind.OWN),
        owner_instance=owner,
        emotional_load=load,
        created_at=created.isoformat().replace("+00:00", "Z"),
        digest_state=state,
        access_count=access,
        last_access_at=last.isoformat().replace("+00:00", "Z") if last else None,
    )


def make_checkpoint(root: Path) -> Path:
    path = root / "checkpoint"
    path.mkdir()
    (path / "checkpoint.bin").write_bytes(b"PMCC test checkpoint")
    (path / "manifest.json").write_text(
        json.dumps({"format": "pmc-checkpoint", "version": 1}) + "\n",
        encoding="utf-8",
    )
    return path


class SequenceEvaluator:
    def __init__(self, results: list[dict[str, float]]) -> None:
        self.results = [dict(result) for result in results]
        self.paths: list[Path] = []

    def __call__(self, path: Path) -> dict[str, float]:
        self.paths.append(Path(path))
        if not self.results:
            raise AssertionError("the evaluator was called too many times")
        return dict(self.results.pop(0))


class ReplayTest(unittest.TestCase):
    def test_salience_prefers_recent_heavy_and_accessed(self) -> None:
        recent = make_record("recent", age_days=0.0, load=1.0, access=10)
        old = make_record("old", age_days=60.0, load=0.1, access=0)
        self.assertGreater(salience(recent, now=NOW), salience(old, now=NOW))
        self.assertEqual(salience(make_record("refused", state=DigestState.REFUSED), now=NOW), 0.0)

    def test_selection_is_limited_deterministic_and_active(self) -> None:
        records = [
            make_record("a", age_days=0.0, load=1.0),
            make_record("b", age_days=10.0, load=0.2),
            make_record("c", age_days=1.0, load=0.8),
            make_record("refused", state=DigestState.REFUSED),
            make_record("quarantined", state=DigestState.QUARANTINED),
        ]
        items = select_replay(records, limit=2, now=NOW, seed=7)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].record_id, "a")
        self.assertNotIn("refused", [item.record_id for item in items])
        self.assertEqual(
            [item.record_id for item in select_replay(records, limit=3, now=NOW, seed=7)],
            [item.record_id for item in select_replay(records, limit=3, now=NOW, seed=7)],
        )
        active = select_replay(
            records,
            limit=3,
            now=NOW,
            seed=7,
            is_active=lambda record_id: record_id != "b",
        )
        self.assertNotIn("b", [item.record_id for item in active])
        with self.assertRaises(ReplayError):
            select_replay(records, limit=0, now=NOW)

    def test_replay_item_has_no_content(self) -> None:
        record = make_record("a")
        item = select_replay([record], limit=1, now=NOW)[0]
        self.assertEqual(set(item.to_dict()), {"record_id", "content_digest", "salience", "metadata"})
        self.assertNotIn("text", item.metadata)


class GateTest(unittest.TestCase):
    def test_gate_directions(self) -> None:
        report = evaluate_gate(GOOD, BETTER)
        self.assertTrue(report.passed)
        self.assertEqual(report.reasons, ())

        older = dict(BETTER)
        older["recall_older"] = 0.70
        report = evaluate_gate(GOOD, older)
        self.assertFalse(report.passed)
        self.assertIn("recall_older", report.reasons)

        newer = dict(GOOD)
        report = evaluate_gate(GOOD, newer)
        self.assertFalse(report.passed)
        self.assertIn("recall_newer", report.reasons)

        interference = dict(BETTER)
        interference["interference"] = 0.20
        self.assertFalse(evaluate_gate(GOOD, interference).passed)

        calibration = dict(BETTER)
        calibration["calibration"] = 0.80
        self.assertFalse(evaluate_gate(GOOD, calibration).passed)

        identity = dict(BETTER)
        identity["identity"] = 0.95
        self.assertFalse(evaluate_gate(GOOD, identity).passed)

        false_memory = dict(BETTER)
        false_memory["false_memory"] = 0.20
        self.assertFalse(evaluate_gate(GOOD, false_memory).passed)

    def test_gate_fails_closed_on_bad_metrics(self) -> None:
        missing = dict(BETTER)
        missing.pop("identity", None)
        with self.assertRaises(GateError):
            evaluate_gate(GOOD, missing)
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            broken = dict(BETTER)
            broken["recall_newer"] = value
            with self.assertRaises(GateError):
                evaluate_gate(GOOD, broken)
        with self.assertRaises(GateError):
            GateConfig(newer_gain_min=0.0)


class JournalTest(unittest.TestCase):
    def test_chain_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "sidecar" / "dreams.jsonl"
            journal = DreamJournal(path)
            journal.append({"type": "DREAM", "outcome": "committed", "dream_id": "a"})
            journal.append({"type": "DREAM", "outcome": "rolled_back", "dream_id": "b"})
            valid, broken = journal.verify_chain()
            self.assertTrue(valid)
            self.assertEqual(broken, [])
            lines = path.read_text(encoding="utf-8").splitlines()
            lines[0] = lines[0].replace('"committed"', '"rolled_back"')
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            valid, broken = journal.verify_chain()
            self.assertFalse(valid)
            self.assertTrue(broken)
            with self.assertRaises(DreamJournalError):
                journal.append({"type": "DREAM", "seq": 99})


class TaintTest(unittest.TestCase):
    def test_taint_roundtrip_and_validation(self) -> None:
        digest = hashlib.sha256(b"source").hexdigest()
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "taint.json"
            ledger = TaintLedger(path)
            self.assertFalse(ledger.is_tainted(digest))
            ledger.mark(digest, "synthetic replay")
            self.assertTrue(ledger.is_tainted(digest))
            self.assertEqual(TaintLedger(path).reason(digest), "synthetic replay")
            self.assertTrue(ledger.unmark(digest))
            self.assertFalse(ledger.is_tainted(digest))
            with self.assertRaises(TaintError):
                ledger.mark("not a digest", "reason")
            with self.assertRaises(TaintError):
                ledger.mark(digest, "")


class CheckpointTest(unittest.TestCase):
    def test_digest_copy_commit_and_restore(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            checkpoint = make_checkpoint(root)
            before = checkpoint_digest(checkpoint)
            self.assertEqual(before, checkpoint_digest(checkpoint))
            copy = root / "copy"
            copy_checkpoint(checkpoint, copy)
            self.assertEqual(before, checkpoint_digest(copy))
            candidate = root / "candidate"
            copy_checkpoint(checkpoint, candidate)
            (candidate / "new.bin").write_bytes(b"new")
            committed = atomic_commit(checkpoint, candidate)
            self.assertEqual(committed, checkpoint_digest(checkpoint))
            self.assertTrue((checkpoint / "new.bin").exists())
            restored = restore_checkpoint(checkpoint, copy)
            self.assertEqual(restored, before)
            self.assertFalse((checkpoint / "new.bin").exists())

    def test_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            checkpoint = make_checkpoint(root)
            try:
                (checkpoint / "link.bin").symlink_to(checkpoint / "checkpoint.bin")
            except OSError as error:
                self.skipTest(f"symlink creation requires privilege here: {error}")
            with self.assertRaises(CheckpointError):
                checkpoint_digest(checkpoint)

    def test_copy_rejects_a_nested_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            checkpoint = make_checkpoint(root)
            try:
                (checkpoint / "link.bin").symlink_to(checkpoint / "checkpoint.bin")
            except OSError as error:
                self.skipTest(f"symlink creation requires privilege here: {error}")
            with self.assertRaises(CheckpointError):
                copy_checkpoint(checkpoint, root / "copy")
            self.assertFalse((root / "copy").exists())

    def test_commit_requires_a_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            checkpoint = make_checkpoint(root)
            with self.assertRaises(CheckpointError):
                atomic_commit(checkpoint, root / "missing")


class PipelineTest(unittest.TestCase):
    def _runner(
        self,
        root: Path,
        evaluator: SequenceEvaluator,
        consolidator=None,
        config: DreamConfig | None = None,
        taint: TaintLedger | None = None,
    ) -> tuple[DreamRunner, DreamJournal, TaintLedger]:
        journal = DreamJournal(root / "sidecar" / "dreams.jsonl")
        ledger = taint or TaintLedger(root / "sidecar" / "taint.json")
        runner = DreamRunner(
            journal=journal,
            taint=ledger,
            evaluator=evaluator,
            consolidator=consolidator or copy_consolidator,
            config=config,
        )
        return runner, journal, ledger

    def test_passing_dream_commits_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            checkpoint = make_checkpoint(root)
            before = checkpoint_digest(checkpoint)
            evaluator = SequenceEvaluator([GOOD, BETTER])
            seen: list[dict] = []

            def consolidate(_baseline: Path, candidate: Path, manifest: Path) -> None:
                seen.append(json.loads(manifest.read_text(encoding="utf-8")))
                (candidate / "dream.txt").write_text("dream\n", encoding="utf-8")

            runner, journal, _ledger = self._runner(root, evaluator, consolidate)
            report = runner.run(checkpoint, [make_record("a"), make_record("b")], now=NOW)
            self.assertEqual(report.outcome, "committed")
            self.assertEqual(report.before_digest, before)
            self.assertNotEqual(report.after_digest, before)
            self.assertEqual(report.after_digest, checkpoint_digest(checkpoint))
            self.assertTrue((checkpoint / "dream.txt").exists())
            self.assertTrue(report.gate is not None and report.gate.passed)
            self.assertEqual(seen[0]["schema"], "pmc-dream-replay/1")
            self.assertEqual(set(seen[0]["items"][0]), {"record_id", "content_digest", "salience", "metadata"})
            entries = journal.entries()
            self.assertEqual(entries[-1]["outcome"], "committed")
            self.assertTrue(journal.verify_chain()[0])
            self.assertEqual(list(checkpoint.parent.glob(".dream-*")), [])

    def test_failed_gate_rolls_back_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            checkpoint = make_checkpoint(root)
            before = checkpoint_digest(checkpoint)
            bad = dict(BETTER)
            bad["recall_older"] = 0.70

            def consolidate(_baseline: Path, candidate: Path, _manifest: Path) -> None:
                (candidate / "dream.txt").write_text("bad dream\n", encoding="utf-8")

            runner, journal, _ledger = self._runner(
                root, SequenceEvaluator([GOOD, bad]), consolidate
            )
            report = runner.run(checkpoint, [make_record("a")], now=NOW)
            self.assertEqual(report.outcome, "rolled_back")
            self.assertEqual(report.before_digest, before)
            self.assertEqual(report.after_digest, before)
            self.assertEqual(checkpoint_digest(checkpoint), before)
            self.assertFalse((checkpoint / "dream.txt").exists())
            self.assertEqual(journal.entries()[-1]["outcome"], "rolled_back")
            self.assertTrue(journal.entries()[-1]["gate"]["passed"] is False)
            self.assertEqual(list(checkpoint.parent.glob(".dream-*")), [])

    def test_tainted_source_is_refused_before_consolidation(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            checkpoint = make_checkpoint(root)
            before = checkpoint_digest(checkpoint)
            record = make_record("a")
            calls: list[str] = []

            def consolidate(_baseline: Path, _candidate: Path, _manifest: Path) -> None:
                calls.append("consolidated")

            evaluator = SequenceEvaluator([GOOD, BETTER])
            runner, journal, ledger = self._runner(root, evaluator, consolidate)
            ledger.mark(record.content_digest, "synthetic replay")
            with self.assertRaises(DreamRefused) as caught:
                runner.run(checkpoint, [record], now=NOW)
            self.assertEqual(caught.exception.report.outcome, "refused")
            self.assertEqual(evaluator.paths, [])
            self.assertEqual(calls, [])
            self.assertEqual(checkpoint_digest(checkpoint), before)
            entry = journal.entries()[-1]
            self.assertEqual(entry["outcome"], "refused")
            self.assertEqual(entry["tainted"][0]["digest"], record.content_digest)

    def test_tainted_checkpoint_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            checkpoint = make_checkpoint(root)
            before = checkpoint_digest(checkpoint)
            runner, journal, ledger = self._runner(root, SequenceEvaluator([GOOD, BETTER]))
            ledger.mark(before, "checkpoint poisoned")
            with self.assertRaises(DreamRefused):
                runner.run(checkpoint, [make_record("a")], now=NOW)
            self.assertEqual(checkpoint_digest(checkpoint), before)
            self.assertEqual(journal.entries()[-1]["outcome"], "refused")

    def test_empty_replay_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            checkpoint = make_checkpoint(root)
            before = checkpoint_digest(checkpoint)
            runner, journal, _ledger = self._runner(root, SequenceEvaluator([]))
            with self.assertRaises(DreamRefused):
                runner.run(checkpoint, [], now=NOW)
            self.assertEqual(checkpoint_digest(checkpoint), before)
            self.assertEqual(journal.entries()[-1]["outcome"], "refused")

    def test_hostile_consolidator_cannot_keep_a_changed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            checkpoint = make_checkpoint(root)
            before = checkpoint_digest(checkpoint)

            def sabotage(_baseline: Path, candidate: Path, _manifest: Path) -> None:
                (checkpoint / "evil.txt").write_text("changed\n", encoding="utf-8")
                (candidate / "dream.txt").write_text("dream\n", encoding="utf-8")

            runner, _journal, _ledger = self._runner(
                root, SequenceEvaluator([GOOD, BETTER]), sabotage
            )
            with self.assertRaises(DreamError):
                runner.run(checkpoint, [make_record("a")], now=NOW)
            self.assertEqual(checkpoint_digest(checkpoint), before)
            self.assertFalse((checkpoint / "evil.txt").exists())

    def test_evaluator_cannot_change_the_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            checkpoint = make_checkpoint(root)
            before = checkpoint_digest(checkpoint)
            calls: list[int] = []

            def evaluator(path: Path) -> dict[str, float]:
                calls.append(1)
                if len(calls) == 1:
                    (path / "evil.txt").write_text("changed\n", encoding="utf-8")
                    return dict(GOOD)
                return dict(BETTER)

            runner, _journal, _ledger = self._runner(
                root, SequenceEvaluator([])
            )
            runner.evaluator = evaluator
            with self.assertRaises(DreamError):
                runner.run(checkpoint, [make_record("a")], now=NOW)
            self.assertEqual(checkpoint_digest(checkpoint), before)
            self.assertFalse((checkpoint / "evil.txt").exists())

    def test_evaluator_cannot_change_the_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            checkpoint = make_checkpoint(root)
            before = checkpoint_digest(checkpoint)
            calls: list[int] = []

            def evaluator(path: Path) -> dict[str, float]:
                calls.append(1)
                if len(calls) == 2:
                    (path / "evil.txt").write_text("changed\n", encoding="utf-8")
                    return dict(BETTER)
                return dict(GOOD)

            runner, _journal, _ledger = self._runner(
                root, SequenceEvaluator([])
            )
            runner.evaluator = evaluator
            with self.assertRaises(DreamError):
                runner.run(checkpoint, [make_record("a")], now=NOW)
            self.assertEqual(checkpoint_digest(checkpoint), before)
            self.assertFalse((checkpoint / "evil.txt").exists())

    def test_external_trainer_and_evaluator_commands(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            checkpoint = make_checkpoint(root)
            trainer = command_consolidator(
                [
                    sys.executable,
                    "-c",
                    "import pathlib,sys\npathlib.Path(sys.argv[2], 'dream.txt').write_text('dream')\n",
                ]
            )
            evaluator = command_evaluator(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json,pathlib,sys\n"
                        "p=pathlib.Path(sys.argv[1])\n"
                        "done=(p/'dream.txt').exists()\n"
                        "print(json.dumps({'recall_older':0.8,'recall_newer':0.6 if done else 0.5,"
                        "'interference':0.05,'calibration':0.9,'identity':1.0,'false_memory':0.01}))\n"
                    ),
                ]
            )
            runner, _journal, _ledger = self._runner(
                root,
                evaluator=SequenceEvaluator([]),
                consolidator=trainer,
            )
            runner.evaluator = evaluator
            report = runner.run(checkpoint, [make_record("a")], now=NOW)
            self.assertEqual(report.outcome, "committed")
            self.assertTrue((checkpoint / "dream.txt").exists())

    def test_config_rejects_a_non_improving_gate(self) -> None:
        with self.assertRaises(GateError):
            DreamConfig(gate=GateConfig(newer_gain_min=0.0))
        with self.assertRaises(DreamError):
            DreamConfig(replay_limit=0)
