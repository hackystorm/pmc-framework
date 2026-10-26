# SPDX-License-Identifier: GPL-3.0-only

"""Run the evaluation battery over a predictions file.

The harness is deliberately dull: it reads JSON, computes metrics with the
functions in `pmc.eval`, and prints a report. It does not talk to a model,
and it does not judge anything by itself.

Input format
------------

`--gold` (required):

```json
{
  "cases": [
    {"id": "case_001", "expected": ["mem_a", "mem_b"]}
  ]
}
```

`--predictions` (required):

```json
{
  "cases": [
    {"id": "case_001",
     "retrieved": ["mem_a", "mem_z"],
     "confidences": [0.9, 0.4],
     "correct": [true, false]}
  ]
}
```

`--baseline` is optional and takes the same shape as `--predictions`; when
it is given, the report shows the difference between the two runs, which is
the only way to tell an improvement from noise.

Usage
-----

    python3 bench/run_eval.py --gold gold.json --predictions predictions.json
    python3 bench/run_eval.py --gold gold.json --predictions predictions.json \
        --baseline baseline.json --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pmc.eval import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    false_memory_rate,
    interference_rate,
    recall_at_k,
)


def load_cases(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {case["id"]: case for case in data.get("cases", [])}


def evaluate(gold: dict, predictions: dict, k: int) -> dict[str, float]:
    recalls: list[float] = []
    false_memories: list[float] = []
    confidences: list[float] = []
    correctness: list[bool] = []
    broken_before: list[str] = []
    correct_after: list[str] = []

    for case_id, gold_case in gold.items():
        prediction = predictions.get(case_id)
        if prediction is None:
            continue

        expected = gold_case.get("expected", [])
        retrieved = prediction.get("retrieved", [])

        recalls.append(recall_at_k(retrieved, expected, k))
        false_memories.append(false_memory_rate(retrieved, expected))

        case_confidences = prediction.get("confidences", [])
        case_correct = prediction.get("correct", [])
        if len(case_confidences) == len(case_correct) and case_correct:
            confidences.extend(case_confidences)
            correctness.extend(case_correct)

        if "previously_correct" in gold_case:
            broken_before.extend(gold_case["previously_correct"])
        if "correct" in gold_case:
            correct_after.extend(gold_case["correct"])

    report: dict[str, float] = {
        "cases": float(len(recalls)),
        f"recall_at_{k}": sum(recalls) / len(recalls) if recalls else 0.0,
        "false_memory_rate": (
            sum(false_memories) / len(false_memories) if false_memories else 0.0
        ),
        "interference_rate": interference_rate(broken_before, correct_after),
        "expected_calibration_error": expected_calibration_error(confidences, correctness),
        "brier_score": brier_score(confidences, correctness),
    }
    return report


def print_report(title: str, report: dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in report.items():
        if isinstance(value, float):
            print(f"  {key:28} {value:.4f}")
        else:
            print(f"  {key:28} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    gold = load_cases(args.gold)
    predictions = load_cases(args.predictions)
    report = evaluate(gold, predictions, args.k)
    print_report("Predictions", report)

    if args.baseline is not None:
        baseline_report = evaluate(gold, load_cases(args.baseline), args.k)
        print_report("Baseline", baseline_report)

        print("\nDifference (predictions - baseline)")
        print("-" * 30)
        for key, value in report.items():
            if key == "cases":
                continue
            if key not in baseline_report:
                continue
            delta = value - baseline_report[key]
            marker = "+" if delta > 0 else ""
            print(f"  {key:28} {marker}{delta:+.4f}")

    print(
        "\nThese are measurements of function. None of them is a measurement "
        "of consciousness."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
