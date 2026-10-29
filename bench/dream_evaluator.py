# SPDX-License-Identifier: GPL-3.0-only
"""Mechanics evaluator for the dream pipeline.

Verifies that the checkpoint directory is intact and prints the six gate
metrics as JSON. The metrics below are the *mechanics* values used to
exercise the CLI path: they describe a checkpoint that loaded, not a model
that recalled anything. A real evaluator replaces this file and measures
recall, interference, calibration, identity and false memory against
held-out memory. The pipeline does not care which one runs, but a reader
must not mistake these numbers for a measurement of a model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = ("recall_older", "recall_newer", "interference", "calibration", "identity", "false_memory")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: dream_evaluator.py <checkpoint>", file=sys.stderr)
        return 2
    checkpoint = Path(sys.argv[1])
    if not (checkpoint / "checkpoint.bin").is_file():
        print(f"missing checkpoint.bin in {checkpoint}", file=sys.stderr)
        return 2
    if not (checkpoint / "manifest.json").is_file():
        print(f"missing manifest.json in {checkpoint}", file=sys.stderr)
        return 2
    # Mechanics values: the checkpoint loaded and nothing was lost.
    metrics = {
        "recall_older": 1.0,
        "recall_newer": 1.0,
        "interference": 0.0,
        "calibration": 0.0,
        "identity": 1.0,
        "false_memory": 0.0,
    }
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
