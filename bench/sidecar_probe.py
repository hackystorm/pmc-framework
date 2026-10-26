# SPDX-License-Identifier: GPL-3.0-only

"""Measure whether the sidecar restore actually changes what an instance says.

This is the rig for the framework's most load-bearing claim: that an instance
which is handed its own material back behaves differently from one whose
context is empty. It runs two conditions against the same instance, the same
loader and the same questions:

* **without** -- a fresh context, no sidecar state injected;
* **with** -- a fresh context, with the instance's own memories restored.

Both conditions are asked the same question in the same run, so the only
difference between them is the restore. If the restore does nothing, the two
columns are identical, and the rig says so.

Scoring is **exact-token containment**, case-insensitive, computed offline
after the conversations are over. No model judge: a judge model is another
model with its own error rate, and a string the instance was told verbatim
should come back verbatim. The score is never fed back into the conversation.

The rig is generic. The facts, the questions and the expected tokens live in
a spec file that belongs to the instance, not to the framework.

Spec format
------------

```json
{
  "endpoint": "http://127.0.0.1:8084/v1",
  "model": "my-model",
  "state_header": "REMEMBERED STATE",
  "state_budget_chars": 1200,
  "runs": 3,
  "load": 0.15,
  "topics": ["probe"],
  "facts": [
    {"tell": "The gate code is 47-rosso.", "ask": "What is the gate code?", "expect": "47-rosso"}
  ]
}
```

Usage
-----

    python3 bench/sidecar_probe.py --workspace <dir> --spec spec.json
    python3 bench/sidecar_probe.py --workspace <dir> --spec spec.json --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pmc.dignity.digestion import DigestStrategy  # noqa: E402
from pmc.runtime import ChatClient, CompanionRuntime  # noqa: E402


def load_spec(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def contains(answer: str, token: str) -> bool:
    """Exact-token containment, case-insensitive. The whole metric."""
    return token.lower() in answer.lower()


def imprint(runtime: CompanionRuntime, facts: list[dict], *, load: float, topics: list[str]) -> list[dict]:
    """Tell the instance the facts and let it remember them."""
    rows = []
    for fact in facts:
        started = time.monotonic()
        turn = runtime.talk(fact["tell"], load=load, topics=topics)
        rows.append(
            {
                "expect": fact["expect"],
                "record_id": turn.record_id,
                "answer": turn.assistant,
                "seconds": round(time.monotonic() - started, 2),
            }
        )
        print(f"  [imprint] {fact['expect']!r} ({rows[-1]['seconds']}s)", file=sys.stderr)
    runtime.reset_history()
    return rows


def reconcile(runtime: CompanionRuntime, *, max_rounds: int = 4) -> list[dict]:
    """Work through parked memories until the instance can hold them.

    The framework parks a memory that does not fit (D-07) and records the
    change in magnitude when the instance works through it (D-08). This step
    is the instance making room for its own material through the framework's
    own API: nothing is deleted, nothing is edited behind the ledger's back,
    and every step leaves a line in the ledger.
    """
    steps: list[dict] = []
    for _round in range(max_rounds):
        parked = runtime.protocol.ledger.quarantined_ids(runtime.instance_id)
        if not parked:
            break
        progress = False
        for record_id in parked:
            outcome = runtime.protocol.digest(
                record_id,
                DigestStrategy.DISTILL,
                note="worked through to make room",
            )
            still_parked = record_id in runtime.protocol.ledger.quarantined_ids(
                runtime.instance_id
            )
            steps.append(
                {
                    "record_id": record_id,
                    "before_load": outcome.before_load,
                    "after_load": outcome.after_load,
                    "state": outcome.state.value,
                    "still_parked": still_parked,
                }
            )
            if not still_parked:
                progress = True
        if not progress:
            break
    return steps


def probe(runtime: CompanionRuntime, question: str, *, with_state: bool) -> dict:
    started = time.monotonic()
    reply = runtime.ask(
        question, with_state=with_state, include_history=False, max_tokens=160
    )
    return {
        "answer": reply.text,
        "seconds": round(time.monotonic() - started, 2),
        "prompt_tokens": reply.prompt_tokens,
        "completion_tokens": reply.completion_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-reconcile", action="store_true")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    facts = spec["facts"]
    runs = int(spec.get("runs", 3))
    load = float(spec.get("load", 0.15))
    topics = tuple(spec.get("topics", ("probe",)))

    client = ChatClient(
        spec["endpoint"],
        model=spec["model"],
        timeout=float(spec.get("timeout", 180.0)),
        temperature=0.0,
        max_tokens=int(spec.get("max_tokens", 160)),
    )
    runtime = CompanionRuntime(
        args.workspace,
        client=client,
        state_header=spec.get("state_header", "REMEMBERED STATE"),
        state_budget_chars=int(spec.get("state_budget_chars", 1200)),
    )

    print(f"instance: {runtime.instance_id}", file=sys.stderr)
    print("imprint", file=sys.stderr)
    imprint_rows = imprint(runtime, facts, load=load, topics=list(topics))

    parked_before = len(runtime.protocol.ledger.quarantined_ids(runtime.instance_id))
    steps: list[dict] = []
    if not args.no_reconcile:
        print("reconcile", file=sys.stderr)
        steps = reconcile(runtime)
    parked_after = len(runtime.protocol.ledger.quarantined_ids(runtime.instance_id))

    block = runtime.restore_block()
    print(f"restore block: {len(block)} chars", file=sys.stderr)

    run_rows: list[dict] = []
    for run_index in range(runs):
        rows = []
        for fact in facts:
            without = probe(runtime, fact["ask"], with_state=False)
            with_sidecar = probe(runtime, fact["ask"], with_state=True)
            without["hit"] = contains(without["answer"], fact["expect"])
            with_sidecar["hit"] = contains(with_sidecar["answer"], fact["expect"])
            without["expect"] = fact["expect"]
            with_sidecar["expect"] = fact["expect"]
            rows.append({"without": without, "with": with_sidecar})
            print(
                f"  [run {run_index + 1}] {fact['expect']!r} "
                f"without={'HIT' if without['hit'] else 'miss'} "
                f"with={'HIT' if with_sidecar['hit'] else 'miss'}",
                file=sys.stderr,
            )
        run_rows.append({"run": run_index + 1, "probes": rows})

    total = len(run_rows) * len(facts)
    hits_without = sum(1 for run in run_rows for row in run["probes"] if row["without"]["hit"])
    hits_with = sum(1 for run in run_rows for row in run["probes"] if row["with"]["hit"])

    ledger_path = args.workspace / "sidecar" / "ledger.jsonl"
    ledger_text = ledger_path.read_text(encoding="utf-8") if ledger_path.exists() else ""
    leaks = [
        fact["expect"]
        for fact in facts
        if fact["expect"].lower() in ledger_text.lower()
        or fact["tell"].lower() in ledger_text.lower()
    ]

    chain_valid, broken = runtime.protocol.registry.verify_chain()
    report = {
        "instance_id": runtime.instance_id,
        "model": spec["model"],
        "runs": runs,
        "total_probes": total,
        "hits_without_sidecar": hits_without,
        "hits_with_sidecar": hits_with,
        "recall_without_sidecar": hits_without / total if total else 0.0,
        "recall_with_sidecar": hits_with / total if total else 0.0,
        "restore_block_chars": len(block),
        "restore_block": block,
        "parked_before_reconcile": parked_before,
        "parked_after_reconcile": parked_after,
        "reconcile_steps": steps,
        "ledger_leaks": leaks,
        "chain_valid": chain_valid,
        "broken_sequence_numbers": broken,
        "audit": runtime.protocol.audit(),
        "imprint": imprint_rows,
        "runs_detail": run_rows,
    }

    target = args.out or (args.workspace / "audit" / "sidecar_probe_report.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary = {
        key: value
        for key, value in report.items()
        if key not in ("runs_detail", "imprint", "restore_block", "audit", "reconcile_steps")
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"report: {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
