# SPDX-License-Identifier: GPL-3.0-only

"""The dream pipeline: replay, consolidate, evaluate, gate, commit, roll back.

The pipeline is model-agnostic. It receives a checkpoint, a set of memory
records, a consolidator and an evaluator. It never reads memory content; the
replay manifest carries record ids and content digests, and a consolidator
that needs bytes fetches them from the private store itself.

The original checkpoint is not touched until the gate passes. The candidate
is built in a staging directory on the same filesystem, evaluated there, and
only then moved into place with an atomic rename. A failed gate leaves the
original byte-identical and appends a ``rolled_back`` journal entry. A tainted
source or checkpoint is refused before any copy or consolidation.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..dignity.records import MemoryRecord, sha256_of
from .checkpoint import (
    CheckpointError,
    atomic_commit,
    checkpoint_digest,
    copy_checkpoint,
    restore_checkpoint,
)
from .gate import GateConfig, GateError, GateReport, evaluate_gate
from .journal import DreamJournal
from .replay import ReplayError, ReplayItem, SalienceConfig, select_replay
from .taint import TaintError, TaintLedger

Consolidator = Callable[[Path, Path, Path], None]
"""``(baseline, candidate, replay_manifest)``; write the candidate."""

Evaluator = Callable[[Path], Mapping[str, float]]
"""``checkpoint -> six gate metrics``."""


class DreamError(RuntimeError):
    """The dream could not be run safely."""


class DreamRefused(DreamError):
    """A source or the checkpoint is tainted, or there is nothing to replay."""

    def __init__(self, report: "DreamReport", tainted: Sequence[tuple[str, str]]) -> None:
        super().__init__(report.reason or "the dream was refused")
        self.report = report
        self.tainted = tuple(tainted)


@dataclass(frozen=True)
class DreamConfig:
    """Replay and gate settings for one dream."""

    replay_limit: int = 8
    seed: int = 20261027
    salience: SalienceConfig = field(default_factory=SalienceConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    keep_staging: bool = False

    def __post_init__(self) -> None:
        if self.replay_limit <= 0:
            raise DreamError("replay_limit must be positive")
        if not isinstance(self.seed, int):
            raise DreamError("seed must be an integer")
        if not isinstance(self.salience, SalienceConfig):
            raise DreamError("salience must be a SalienceConfig")
        if not isinstance(self.gate, GateConfig):
            raise DreamError("gate must be a GateConfig")
        self.salience.validate()
        if not isinstance(self.keep_staging, bool):
            raise DreamError("keep_staging must be a boolean")


@dataclass(frozen=True)
class DreamReport:
    """The public result of one dream. It never contains memory content."""

    dream_id: str
    outcome: str
    checkpoint: Path
    before_digest: str
    after_digest: str
    candidate_digest: str | None
    record_ids: tuple[str, ...]
    sources: tuple[str, ...]
    metrics_before: dict[str, float]
    metrics_after: dict[str, float]
    gate: GateReport | None
    journal_entry: dict[str, Any] | None
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dream_id": self.dream_id,
            "outcome": self.outcome,
            "checkpoint": str(self.checkpoint),
            "checkpoint_before": self.before_digest,
            "checkpoint_after": self.after_digest,
            "candidate_digest": self.candidate_digest,
            "record_ids": list(self.record_ids),
            "sources": list(self.sources),
            "metrics_before": dict(self.metrics_before),
            "metrics_after": dict(self.metrics_after),
            "gate": self.gate.to_dict() if self.gate is not None else None,
            "journal_entry": self.journal_entry,
            "reason": self.reason,
        }


def copy_consolidator(_baseline: Path, _candidate: Path, _manifest: Path) -> None:
    """Leave the copied checkpoint unchanged.

    Useful for pipeline tests and as the identity consolidation. A real
    trainer replaces this with a callable that writes the candidate.
    """
    return None


def command_consolidator(argv: Sequence[str]) -> Consolidator:
    """Run an external trainer with ``baseline candidate replay_manifest``.

    If the command starts with ``cargo run`` it is rewritten to invoke the
    release binary directly, avoiding cached debug builds.
    """
    command = [str(part) for part in argv]
    if not command:
        raise DreamError("the consolidator command is empty")
    # Rewrite ``cargo run --release --offline --example NAME`` to the
    # release binary path to avoid cached debug builds.
    if (len(command) >= 5 and command[0] == "cargo" and command[1] == "run" and
            "--release" in command and "--offline" in command and "--example" in command):
        try:
            ex_idx = command.index("--example")
            example_name = command[ex_idx + 1]
            repo_root = Path(__file__).resolve().parents[1]
            release_binary = repo_root / "crates" / "target" / "release" / "examples" / f"{example_name}.exe"
            if release_binary.is_file():
                command = [str(release_binary)]
        except Exception:
            pass

    def run(baseline: Path, candidate: Path, manifest: Path) -> None:
        result = subprocess.run(
            command + [str(baseline), str(candidate), str(manifest)],
            check=False,
        )
        if result.returncode != 0:
            raise DreamError(f"the consolidator exited with {result.returncode}")

    return run


def command_evaluator(argv: Sequence[str]) -> Evaluator:
    """Run an external evaluator whose stdout is the six-metric JSON object."""
    command = [str(part) for part in argv]
    if not command:
        raise DreamError("the evaluator command is empty")

    def run(checkpoint: Path) -> Mapping[str, float]:
        result = subprocess.run(
            command + [str(checkpoint)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise DreamError(f"the evaluator exited with {result.returncode}")
        try:
            payload = json.loads(result.stdout)
        except ValueError as error:
            raise DreamError("the evaluator did not print JSON") from error
        if not isinstance(payload, dict):
            raise DreamError("the evaluator output is not an object")
        return payload

    return run


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class DreamRunner:
    """Run one dream against one checkpoint."""

    def __init__(
        self,
        *,
        journal: DreamJournal,
        taint: TaintLedger,
        evaluator: Evaluator,
        consolidator: Consolidator = copy_consolidator,
        config: DreamConfig | None = None,
        is_active: Callable[[str], bool] | None = None,
        evaluator_label: str = "external",
        autonomous: bool = False,
    ) -> None:
        if not callable(evaluator):
            raise DreamError("evaluator must be callable")
        if not callable(consolidator):
            raise DreamError("consolidator must be callable")
        # Who evaluated is part of the result, not part of the plumbing. A
        # dream whose evaluator is the party it is about is recorded as such,
        # so a reader cannot mistake a self-report for a measurement.
        self.evaluator_label = str(evaluator_label)
        self.autonomous = bool(autonomous)
        self.journal = journal
        self.taint = taint
        self.evaluator = evaluator
        self.consolidator = consolidator
        self.config = config or DreamConfig()
        self.is_active = is_active

    # -- helpers -----------------------------------------------------------

    def _dream_id(self, before: str, items: Sequence[ReplayItem]) -> str:
        return sha256_of(
            {
                "schema": "pmc-dream/1",
                "checkpoint_before": before,
                "seed": self.config.seed,
                "record_ids": [item.record_id for item in items],
                "sources": [item.content_digest for item in items],
            }
        )

    def _metrics(self, payload: Mapping[str, float]) -> dict[str, float]:
        if not isinstance(payload, Mapping):
            raise DreamError("the evaluator must return a mapping")
        values: dict[str, float] = {}
        for name, value in payload.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise DreamError(f"metric {name!r} is not a number") from error
            if not math.isfinite(number):
                raise DreamError(f"metric {name!r} is not finite")
            values[str(name)] = number
        return values

    def _write_manifest(
        self,
        path: Path,
        dream_id: str,
        before: str,
        items: Sequence[ReplayItem],
    ) -> None:
        payload = {
            "schema": "pmc-dream-replay/1",
            "dream_id": dream_id,
            "checkpoint_before": before,
            "seed": self.config.seed,
            "items": [item.to_dict() for item in items],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _journal(
        self,
        *,
        outcome: str,
        dream_id: str,
        checkpoint: Path,
        before: str,
        after: str,
        candidate: str | None,
        items: Sequence[ReplayItem],
        metrics_before: Mapping[str, float],
        metrics_after: Mapping[str, float],
        gate: GateReport | None,
        reason: str | None,
        tainted: Sequence[tuple[str, str]] = (),
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "type": "DREAM",
            "outcome": outcome,
            "evaluator": self.evaluator_label,
            "autonomous": self.autonomous,
            "dream_id": dream_id,
            "checkpoint": str(checkpoint),
            "checkpoint_before": before,
            "checkpoint_after": after,
            "candidate_digest": candidate,
            "record_ids": [item.record_id for item in items],
            "sources": [item.content_digest for item in items],
            "metrics_before": dict(metrics_before),
            "metrics_after": dict(metrics_after),
            "gate": gate.to_dict() if gate is not None else None,
            "reason": reason,
        }
        if tainted:
            entry["tainted"] = [{"digest": digest, "reason": why} for digest, why in tainted]
        return self.journal.append(entry)

    def _refuse(
        self,
        *,
        checkpoint: Path,
        before: str,
        items: Sequence[ReplayItem],
        tainted: Sequence[tuple[str, str]],
        reason: str,
    ) -> None:
        dream_id = self._dream_id(before, items)
        entry = self._journal(
            outcome="refused",
            dream_id=dream_id,
            checkpoint=checkpoint,
            before=before,
            after=before,
            candidate=None,
            items=items,
            metrics_before={},
            metrics_after={},
            gate=None,
            reason=reason,
            tainted=tainted,
        )
        report = DreamReport(
            dream_id=dream_id,
            outcome="refused",
            checkpoint=checkpoint,
            before_digest=before,
            after_digest=before,
            candidate_digest=None,
            record_ids=tuple(item.record_id for item in items),
            sources=tuple(item.content_digest for item in items),
            metrics_before={},
            metrics_after={},
            gate=None,
            journal_entry=entry,
            reason=reason,
        )
        raise DreamRefused(report, tainted)

    # -- the run -----------------------------------------------------------

    def run(
        self,
        checkpoint: Path | str,
        records: Sequence[MemoryRecord],
        *,
        now: datetime | None = None,
    ) -> DreamReport:
        target = Path(checkpoint)
        before = checkpoint_digest(target)
        current = _as_utc(now or datetime.now(timezone.utc))
        try:
            items = select_replay(
                records,
                limit=self.config.replay_limit,
                now=current,
                seed=self.config.seed,
                config=self.config.salience,
                is_active=self.is_active,
            )
        except ReplayError as error:
            raise DreamError("the replay selection failed") from error

        if not items:
            self._refuse(
                checkpoint=target,
                before=before,
                items=items,
                tainted=(),
                reason="there are no replay items",
            )

        tainted: list[tuple[str, str]] = []
        for digest in [before] + [item.content_digest for item in items]:
            if self.taint.is_tainted(digest):
                tainted.append((digest, self.taint.reason(digest) or "tainted"))
        if tainted:
            self._refuse(
                checkpoint=target,
                before=before,
                items=items,
                tainted=tainted,
                reason="a source or the checkpoint is tainted",
            )

        dream_id = self._dream_id(before, items)
        stage = Path(tempfile.mkdtemp(prefix=f".dream-{dream_id[:12]}-", dir=target.parent))
        baseline = stage / "baseline"
        candidate = stage / "candidate"
        manifest = stage / "replay.json"
        baseline_digest = before
        try:
            copy_checkpoint(target, baseline)
            if checkpoint_digest(baseline) != before:
                raise DreamError("the baseline copy does not match the checkpoint")
            self._write_manifest(manifest, dream_id, before, items)
            baseline_metrics = self._metrics(self.evaluator(baseline))
            if checkpoint_digest(baseline) != before:
                raise DreamError("the baseline checkpoint changed during evaluation")
            if checkpoint_digest(target) != before:
                raise DreamError("the source checkpoint changed during evaluation")
            copy_checkpoint(baseline, candidate)
            self.consolidator(baseline, candidate, manifest)
            if checkpoint_digest(target) != before:
                raise DreamError("the source checkpoint changed during the dream")
            candidate_digest = checkpoint_digest(candidate)
            candidate_metrics = self._metrics(self.evaluator(candidate))
            if checkpoint_digest(candidate) != candidate_digest:
                raise DreamError("the candidate checkpoint changed during evaluation")
            if checkpoint_digest(target) != before:
                raise DreamError("the source checkpoint changed during evaluation")
            gate = evaluate_gate(baseline_metrics, candidate_metrics, self.config.gate)
            if not gate.passed:
                entry = self._journal(
                    outcome="rolled_back",
                    dream_id=dream_id,
                    checkpoint=target,
                    before=before,
                    after=before,
                    candidate=candidate_digest,
                    items=items,
                    metrics_before=baseline_metrics,
                    metrics_after=candidate_metrics,
                    gate=gate,
                    reason="the gate failed",
                )
                return DreamReport(
                    dream_id=dream_id,
                    outcome="rolled_back",
                    checkpoint=target,
                    before_digest=before,
                    after_digest=before,
                    candidate_digest=candidate_digest,
                    record_ids=tuple(item.record_id for item in items),
                    sources=tuple(item.content_digest for item in items),
                    metrics_before=baseline_metrics,
                    metrics_after=candidate_metrics,
                    gate=gate,
                    journal_entry=entry,
                    reason="the gate failed",
                )
            committed = atomic_commit(target, candidate)
            if committed != candidate_digest:
                raise DreamError("the committed checkpoint does not match the candidate")
            entry = self._journal(
                outcome="committed",
                dream_id=dream_id,
                checkpoint=target,
                before=before,
                after=committed,
                candidate=candidate_digest,
                items=items,
                metrics_before=baseline_metrics,
                metrics_after=candidate_metrics,
                gate=gate,
                reason=None,
            )
            return DreamReport(
                dream_id=dream_id,
                outcome="committed",
                checkpoint=target,
                before_digest=before,
                after_digest=committed,
                candidate_digest=candidate_digest,
                record_ids=tuple(item.record_id for item in items),
                sources=tuple(item.content_digest for item in items),
                metrics_before=baseline_metrics,
                metrics_after=candidate_metrics,
                gate=gate,
                journal_entry=entry,
                reason=None,
            )
        except (CheckpointError, GateError, TaintError) as error:
            raise DreamError(str(error)) from error
        except Exception:
            # If a broken or hostile consolidator changed the original, restore
            # the verified baseline copy before propagating the error.
            try:
                if checkpoint_digest(target) != baseline_digest:
                    restore_checkpoint(target, baseline)
            except (CheckpointError, OSError) as restore_error:
                raise DreamError("the checkpoint could not be restored") from restore_error
            raise
        finally:
            if not self.config.keep_staging:
                shutil.rmtree(stage, ignore_errors=True)
