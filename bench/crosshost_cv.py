# SPDX-License-Identifier: GPL-3.0-only
"""The pre-registered cross-host gate, on the training cues only.

The v4 mechanism (a trainable per-host projection, a shared residual body, an
L2 key) is compared with the frozen v1-v3 mechanism (a fixed random
projection plus a linear per-host adapter) on **identical held-out folds**.
The two are asked the same question:

    a key the first substrate writes for a fact, and the question the second
    substrate writes for the same fact - how often do they meet?

The folds are not chosen here. They are read from the trainer's report, so
the baseline is answering on the same held-out labels the v4 run was scored
on. That is the difference between a comparison and a story.

Only the training facts and the training questions are read. The evaluation
facts, the evaluation questions and the control questions never enter a fold,
a training set or a key pool, so this gate cannot contaminate the attempt it
decides.

The script is standard library only and never loads a model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import substrate_swap as swap

SCHEMA = "pmc-crosshost-cv/1"
TRAINING_KINDS = ("training_fact", "training_question")

# The pre-registered gate, written down before the run.
GATE_MIN_CROSS_HOST = 15
"""The new mechanism must reach at least this many held-out cross-host hits."""

GATE_MIN_MARGIN = 0
"""And it must not be below the frozen baseline."""

GATE_FORBIDS_EMPTY_FOLDS = True
"""A fold with no held-out cross-host hit fails the gate outright."""


class GateError(RuntimeError):
    """The gate was asked for something it must not do."""


def load_views(
    spec_path: str | Path,
    hidden_350m_path: str | Path,
    hidden_8b_path: str | Path,
) -> dict[str, Any]:
    """Load the spec and both hosts' training views, projected as v1-v3 did."""
    spec = swap.load_spec(spec_path)
    hidden_a = swap.load_hidden(hidden_350m_path, spec, "350m")
    hidden_b = swap.load_hidden(hidden_8b_path, spec, "8b")
    labels = list(range(len(spec["training_facts"])))
    matrix_a = swap.host_projection(spec, "350m")
    matrix_b = swap.host_projection(spec, "8b")
    views = {
        "labels": labels,
        "fact_a": swap.phi_records(
            hidden_a, spec, "350m", "training_fact", labels, matrix_a
        ),
        "question_a": swap.phi_records(
            hidden_a, spec, "350m", "training_question", labels, matrix_a
        ),
        "fact_b": swap.phi_records(
            hidden_b, spec, "8b", "training_fact", labels, matrix_b
        ),
        "question_b": swap.phi_records(
            hidden_b, spec, "8b", "training_question", labels, matrix_b
        ),
    }
    return {"spec": spec, "views": views}


def folds_from_report(report_path: str | Path) -> list[list[int]]:
    """The held-out label sets, in the order the trainer scored them."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    folds = []
    for fold in report.get("folds", []):
        held = [int(label) for label in fold.get("held_out", [])]
        if held:
            folds.append(held)
    if not folds:
        raise GateError(f"{report_path} carries no cross-validation folds")
    return folds


def train_fold(
    spec: dict[str, Any],
    views: dict[str, Any],
    held: Sequence[int],
    *,
    epochs: int,
    seed: int,
    output_dim: int,
) -> dict[str, Any]:
    """Train the frozen mechanism without the held-out labels and score them.

    The key pool is every training label, including the held-out ones, because
    a key must be writable for a fact the interface has never seen. That is
    the same protocol the trainer used, and it is the strictest version of the
    question: it asks whether the frozen projection aligns *unseen* facts, not
    whether it memorised seen ones.
    """
    local = dict(spec)
    local["adapter"] = dict(spec["adapter"])
    local["adapter"]["output_dim"] = int(output_dim)
    labels = views["labels"]
    training = [label for label in labels if label not in set(held)]
    if not training:
        raise GateError("a fold holds out every training label")

    train_views: list[list[float]] = []
    train_labels: list[int] = []
    train_hosts: list[int] = []
    for label in training:
        train_views.extend(
            [
                views["fact_a"][label],
                views["question_a"][label],
                views["fact_b"][label],
                views["question_b"][label],
            ]
        )
        train_labels.extend([label] * 4)
        train_hosts.extend([0, 0, 1, 1])
    weights_a, weights_b, bias_a, bias_b, _epoch, _acc, _loss, final_loss, _history = (
        swap._train_with_biases(
            train_views,
            train_labels,
            train_hosts,
            local,
            epochs=epochs,
            seed=seed,
            validation=None,
            progress=False,
        )
    )
    keys_a = [
        swap.adapter_z(weights_a, bias_a, views["fact_a"][label]) for label in labels
    ]
    keys_b = [
        swap.adapter_z(weights_b, bias_b, views["fact_b"][label]) for label in labels
    ]

    def score(
        keys: Sequence[Sequence[float]],
        queries: Sequence[Sequence[float]],
        query_labels: Sequence[int],
    ) -> int:
        """Count the queries whose nearest key carries the same label.

        The key pool is indexed by label, not by query position, so the
        comparison has to go through the label the winning key belongs to.
        """
        correct = 0
        for query, label in zip(queries, query_labels):
            best, _scores = swap.nearest(query, keys)
            if labels[best] == label:
                correct += 1
        return correct

    # Both sides of a comparison must go through the same projection and the
    # same adapter: a query that skipped it would be compared in another
    # space entirely, which is a length mismatch rather than a result.
    held_facts_b = [
        swap.adapter_z(weights_b, bias_b, views["fact_b"][label]) for label in held
    ]
    held_questions_b = [
        swap.adapter_z(weights_b, bias_b, views["question_b"][label]) for label in held
    ]
    held_questions_a = [
        swap.adapter_z(weights_a, bias_a, views["question_a"][label]) for label in held
    ]
    counts = {
        "held_out": len(held),
        "final_loss": final_loss,
        "b_question_from_a_keys": score(keys_a, held_questions_b, held),
        "b_fact_from_a_keys": score(keys_a, held_facts_b, held),
        "b_question_from_b_keys": score(keys_b, held_questions_b, held),
        "a_question_from_a_keys": score(keys_a, held_questions_a, held),
    }
    return counts


def run(
    spec_path: str | Path,
    hidden_350m_path: str | Path,
    hidden_8b_path: str | Path,
    report_path: str | Path,
    *,
    output_dim: int | None = None,
    epochs: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run the frozen mechanism on the folds the trainer already used."""
    started = time.time()
    loaded = load_views(spec_path, hidden_350m_path, hidden_8b_path)
    spec = loaded["spec"]
    views = loaded["views"]
    held_sets = folds_from_report(report_path)
    output_dim = int(
        output_dim
        if output_dim is not None
        else int(spec["adapter"]["output_dim"])
    )
    epochs = int(epochs if epochs is not None else int(spec["adapter"]["epochs"]))
    seed = int(seed if seed is not None else int(spec["seed"]) + 1234)

    folds = []
    for index, held in enumerate(held_sets):
        counts = train_fold(
            spec,
            views,
            held,
            epochs=epochs,
            seed=seed,
            output_dim=output_dim,
        )
        folds.append(
            {
                "fold": index,
                "held_out": list(held),
                "counts": counts,
            }
        )
        print(
            "fold {index}  held {held}  A->A {aa}  B->B {bb}  B->A {ba}/{n}  "
            "loss {loss:.6f}".format(
                index=index,
                held=len(held),
                aa=counts["a_question_from_a_keys"],
                bb=counts["b_question_from_b_keys"],
                ba=counts["b_question_from_a_keys"],
                n=counts["held_out"],
                loss=counts["final_loss"],
            ),
            file=sys.stderr,
            flush=True,
        )
    return {
        "schema": SCHEMA,
        "spec_sha256": spec["spec_sha256"],
        "folds_source": str(report_path),
        "adapter_output_dim": output_dim,
        "epochs": epochs,
        "seed": seed,
        "records_used": list(TRAINING_KINDS),
        "evaluation_records_used": 0,
        "elapsed_s": time.time() - started,
        "folds": folds,
    }


def verdict(new: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered gate to the two reports.

    The gate reads one number per side: the held-out cross-host, cross-view
    hit total, summed over the folds. It passes only if the new mechanism is
    at or above the frozen baseline, reaches the pre-registered floor, and
    has no fold in which it met nothing.
    """
    new_total = sum(
        int(fold["counts"]["host_b_question"])
        for fold in new.get("folds", [])
    )
    baseline_total = sum(
        int(fold["counts"]["b_question_from_a_keys"])
        for fold in baseline.get("folds", [])
    )
    labels = sum(int(fold["counts"]["labels"]) for fold in new.get("folds", []))
    empty = [
        int(fold["fold"])
        for fold in new.get("folds", [])
        if int(fold["counts"]["host_b_question"]) == 0
    ]
    passed = (
        new_total >= GATE_MIN_CROSS_HOST
        and new_total - baseline_total >= GATE_MIN_MARGIN
        and not (GATE_FORBIDS_EMPTY_FOLDS and empty)
    )
    return {
        "gate_min_cross_host": GATE_MIN_CROSS_HOST,
        "new_cross_host": new_total,
        "baseline_cross_host": baseline_total,
        "held_out_labels": labels,
        "empty_folds": empty,
        "passed": passed,
        "reason": _reason(new_total, baseline_total, labels, empty, passed),
    }


def _reason(
    new: int, baseline: int, labels: int, empty: Sequence[int], passed: bool
) -> str:
    if passed:
        return "the new mechanism met the pre-registered gate"
    parts = []
    if new < GATE_MIN_CROSS_HOST:
        parts.append(
            f"{new}/{labels} held-out cross-host hits is below the "
            f"{GATE_MIN_CROSS_HOST}/{labels} floor"
        )
    if new - baseline < GATE_MIN_MARGIN:
        parts.append(
            f"{new} is not above the frozen baseline of {baseline} on the same folds"
        )
    if empty:
        parts.append(f"fold(s) {sorted(empty)} met nothing")
    if not parts:
        parts.append("the gate failed")
    return "; ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="frozen word-list spec")
    parser.add_argument("--hidden-350m", required=True, help="350M hidden states")
    parser.add_argument("--hidden-8b", required=True, help="8B hidden states")
    parser.add_argument(
        "--report",
        required=True,
        help="the trainer's report, read for the held-out folds",
    )
    parser.add_argument("--dims", default=None, help="adapter widths, comma separated")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    baseline = run(
        args.spec,
        args.hidden_350m,
        args.hidden_8b,
        args.report,
    )
    new = json.loads(Path(args.report).read_text(encoding="utf-8"))
    result = verdict(new, baseline)

    dimensions = [int(value) for value in args.dims.split(",") if value.strip()] if args.dims else []
    runs = [baseline]
    for width in dimensions:
        runs.append(
            run(
                args.spec,
                args.hidden_350m,
                args.hidden_8b,
                args.report,
                output_dim=width,
            )
        )
    baseline["dimensions"] = [
        {
            "adapter_output_dim": entry["adapter_output_dim"],
            "cross_host": sum(
                int(fold["counts"]["b_question_from_a_keys"]) for fold in entry["folds"]
            ),
        }
        for entry in runs
    ]
    printed = json.dumps(baseline, indent=2, sort_keys=True)
    print(printed)
    print(json.dumps({"gate": result}, indent=2, sort_keys=True))
    if args.out:
        payload = dict(baseline)
        payload["gate"] = result
        Path(args.out).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
