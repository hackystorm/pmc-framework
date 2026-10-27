# SPDX-License-Identifier: GPL-3.0-only
"""Cross-host in-sample diagnostic on training facts only.

The PMC-mini swap runs (v1, v2, v3) are frozen and their evaluation facts
must not be used to design a later attempt.  This diagnostic therefore stays
inside the training split: it trains the same per-host linear interface on the
same training views and asks one question.

    Can the interface place a key written by one substrate next to the query
    that the *other* substrate produces, for cues it has already seen?

The answer is an in-sample capacity check, not a swap result.  It uses no
evaluation fact, no control question and no held-out cue, so it cannot
contaminate a later attempt.  What it decides is the shape of the next
attempt: how wide the shared state has to be, and whether the cross-host
signal exists at all in the extracted views.

The script is standard library only and never loads a model.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import substrate_swap as swap

TRAINING_KINDS = ("training_fact", "training_question")
"""The only record kinds this diagnostic is allowed to read."""


class DiagnosticError(RuntimeError):
    """The diagnostic was asked for something it must not do."""


def load_training_views(
    spec: dict[str, Any], hidden_a: dict[str, Any], hidden_b: dict[str, Any]
) -> dict[str, list[list[float]]]:
    """Return the four training views per fact, for both hosts.

    The views are the same ones the frozen bench trains on: the 350M fact
    statement, the 350M question, the 8B fact statement and the 8B question.
    Evaluation and control records are never selected.
    """
    for payload, host in ((hidden_a, "350m"), (hidden_b, "8b")):
        kinds = {record["kind"] for record in payload["records"]}
        unknown = kinds - set(TRAINING_KINDS) - {
            "eval_fact",
            "eval_question",
            "control_question",
        }
        if unknown:
            raise DiagnosticError(f"{host}: unknown record kinds {sorted(unknown)}")
    indices = list(range(len(spec["training_facts"])))
    matrix_a = swap.host_projection(spec, "350m")
    matrix_b = swap.host_projection(spec, "8b")
    return {
        "fact_a": swap.phi_records(
            hidden_a, spec, "350m", "training_fact", indices, matrix_a
        ),
        "question_a": swap.phi_records(
            hidden_a, spec, "350m", "training_question", indices, matrix_a
        ),
        "fact_b": swap.phi_records(
            hidden_b, spec, "8b", "training_fact", indices, matrix_b
        ),
        "question_b": swap.phi_records(
            hidden_b, spec, "8b", "training_question", indices, matrix_b
        ),
    }


def retrieval_accuracy(
    keys: Sequence[Sequence[float]], queries: Sequence[Sequence[float]]
) -> tuple[int, float]:
    """Return the number of correct nearest-key hits and the mean margin.

    The margin is the cosine of the correct key minus the best wrong key, so a
    positive mean means the correct key is ahead on average.
    """
    if len(keys) != len(queries):
        raise DiagnosticError("the key and query counts differ")
    correct = 0
    margins: list[float] = []
    for position, query in enumerate(queries):
        best, scores = swap.nearest(query, keys)
        if best == position:
            correct += 1
        wrong = [score for index, score in enumerate(scores) if index != position]
        if wrong:
            margins.append(scores[position] - max(wrong))
    mean_margin = sum(margins) / len(margins) if margins else 0.0
    return correct, mean_margin


def train_dimension(
    spec: dict[str, Any],
    views: dict[str, list[list[float]]],
    dimension: int,
    *,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    """Train the frozen per-host linear interface at one state width."""
    local = copy.deepcopy(spec)
    local["adapter"] = dict(local["adapter"])
    local["adapter"]["output_dim"] = int(dimension)
    facts = len(local["training_facts"])
    train_views: list[list[float]] = []
    train_labels: list[int] = []
    train_hosts: list[int] = []
    for index in range(facts):
        train_views.extend(
            [
                views["fact_a"][index],
                views["question_a"][index],
                views["fact_b"][index],
                views["question_b"][index],
            ]
        )
        train_labels.extend([index] * 4)
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
        swap.adapter_z(weights_a, bias_a, views["fact_a"][index])
        for index in range(facts)
    ]
    keys_b = [
        swap.adapter_z(weights_b, bias_b, views["fact_b"][index])
        for index in range(facts)
    ]
    queries_a_fact = keys_a
    queries_a_question = [
        swap.adapter_z(weights_a, bias_a, views["question_a"][index])
        for index in range(facts)
    ]
    queries_b_fact = [
        swap.adapter_z(weights_b, bias_b, views["fact_b"][index])
        for index in range(facts)
    ]
    queries_b_question = [
        swap.adapter_z(weights_b, bias_b, views["question_b"][index])
        for index in range(facts)
    ]
    report: dict[str, Any] = {
        "state_dim": int(dimension),
        "epochs": epochs,
        "final_loss": final_loss,
    }
    for name, keys, queries in (
        ("a_fact_from_a_keys", keys_a, queries_a_fact),
        ("a_question_from_a_keys", keys_a, queries_a_question),
        ("b_fact_from_a_keys", keys_a, queries_b_fact),
        ("b_question_from_a_keys", keys_a, queries_b_question),
        ("b_question_from_b_keys", keys_b, queries_b_question),
        ("a_question_from_b_keys", keys_b, queries_a_question),
    ):
        correct, margin = retrieval_accuracy(keys, queries)
        report[name] = {
            "correct": correct,
            "total": facts,
            "accuracy": correct / facts if facts else 0.0,
            "mean_margin": margin,
        }
    report["cross_host"] = report["b_question_from_a_keys"]["accuracy"]
    return report


def run(
    spec_path: str | Path,
    hidden_a_path: str | Path,
    hidden_b_path: str | Path,
    *,
    dimensions: Sequence[int] = (16, 32, 64),
    epochs: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run the diagnostic over the requested state widths."""
    spec = swap.load_spec(spec_path)
    hidden_a = swap.load_hidden(hidden_a_path, spec, "350m")
    hidden_b = swap.load_hidden(hidden_b_path, spec, "8b")
    views = load_training_views(spec, hidden_a, hidden_b)
    epochs = int(epochs if epochs is not None else spec["adapter"]["epochs"])
    seed = int(seed if seed is not None else int(spec["seed"]) + 1234)
    started = time.time()
    results = []
    for dimension in dimensions:
        result = train_dimension(
            spec, views, int(dimension), epochs=epochs, seed=seed
        )
        results.append(result)
        print(
            "dim {dim:>4}  loss {loss:.6f}  A->A {aa:>2}/{n}  A->B {ab:>2}/{n}  "
            "B->B {bb:>2}/{n}  cross-host {cross:>2}/{n}  margin {margin:+.4f}".format(
                dim=result["state_dim"],
                loss=result["final_loss"],
                aa=result["a_question_from_a_keys"]["correct"],
                ab=result["b_fact_from_a_keys"]["correct"],
                bb=result["b_question_from_b_keys"]["correct"],
                cross=result["b_question_from_a_keys"]["correct"],
                n=result["b_question_from_a_keys"]["total"],
                margin=result["b_question_from_a_keys"]["mean_margin"],
            ),
            file=sys.stderr,
            flush=True,
        )
    return {
        "schema": "pmc-crosshost-diagnostic/1",
        "spec_sha256": spec["spec_sha256"],
        "anchor": spec.get("anchor", "last"),
        "projection_seed": int(spec["projection"]["seed"]),
        "projection_output_dim": int(spec["projection"]["output_dim"]),
        "training_facts": len(spec["training_facts"]),
        "records_used": list(TRAINING_KINDS),
        "evaluation_records_used": 0,
        "epochs": epochs,
        "seed": seed,
        "elapsed_s": time.time() - started,
        "dimensions": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="frozen word-list spec")
    parser.add_argument("--hidden-350m", required=True, help="350M hidden states")
    parser.add_argument("--hidden-8b", required=True, help="8B hidden states")
    parser.add_argument(
        "--dims",
        default="16,32,64",
        help="comma-separated state widths to try",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)
    dimensions = [int(value) for value in args.dims.split(",") if value.strip()]
    if not dimensions:
        raise SystemExit("no state widths requested")
    report = run(
        args.spec,
        args.hidden_350m,
        args.hidden_8b,
        dimensions=dimensions,
        epochs=args.epochs,
        seed=args.seed,
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
