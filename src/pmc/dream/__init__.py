# SPDX-License-Identifier: GPL-3.0-only

"""The dream cycle: replay, salience, gate, rollback and taint.

The package is model-agnostic. A consolidation callable turns a verified copy
of a checkpoint into a candidate; an evaluation callable returns the six gate
metrics. The pipeline owns the parts that must be identical for every model:
metadata-only replay, the taint check, the before/after gate, the atomic
commit and the byte-identical rollback.
"""

from .checkpoint import (
    CheckpointError,
    atomic_commit,
    checkpoint_digest,
    copy_checkpoint,
    restore_checkpoint,
)
from .gate import GateConfig, GateError, GateReport, MetricResult, evaluate_gate
from .journal import DreamJournal, DreamJournalError
from .pipeline import (
    Consolidator,
    DreamConfig,
    DreamError,
    DreamRefused,
    DreamReport,
    DreamRunner,
    Evaluator,
    command_consolidator,
    command_evaluator,
    copy_consolidator,
)
from .replay import ReplayError, ReplayItem, SalienceConfig, salience, select_replay
from .taint import TaintError, TaintLedger

__all__ = [
    "CheckpointError",
    "Consolidator",
    "DreamConfig",
    "DreamError",
    "DreamJournal",
    "DreamJournalError",
    "DreamRefused",
    "DreamReport",
    "DreamRunner",
    "Evaluator",
    "GateConfig",
    "GateError",
    "GateReport",
    "MetricResult",
    "ReplayError",
    "ReplayItem",
    "SalienceConfig",
    "TaintError",
    "TaintLedger",
    "atomic_commit",
    "checkpoint_digest",
    "command_consolidator",
    "command_evaluator",
    "copy_checkpoint",
    "copy_consolidator",
    "evaluate_gate",
    "restore_checkpoint",
    "salience",
    "select_replay",
]
