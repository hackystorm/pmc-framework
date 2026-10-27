# SPDX-License-Identifier: GPL-3.0-only
"""Offer a saved KV state to a different model and record the refusal.

A KV state is runtime context state. It is bound to the model digest and the
context geometry that produced it, and it is not a substrate that can be moved
to another model. This probe makes that boundary executable: it loads one
model, loads a state saved by another, asks the engine to restore it, and
records whether the engine refused before the C API.

The probe is deliberately a *negative* test. It does not show that a trained
PMC state would swap; it shows that a KV cache does not. The positive
Substrate Swap Test needs a state that is not bound to one model's KV geometry.

Exit status: 0 when the engine refused on identity, 1 when the state was
accepted or an unrelated error occurred, 2 when the state matches this model
(so the invocation is not a cross-model probe).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pmc.runtime.live_engine import KvState, KvStateError, LiveEngine, LiveEngineError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kv_swap_probe",
        description=(
            "Load a model and a KV state saved by a different model, then "
            "record the engine's identity refusal."
        ),
    )
    parser.add_argument("--state", required=True, help="state file to offer")
    parser.add_argument("--model", required=True, help="GGUF of the target model")
    parser.add_argument("--lib", required=True, help="patched libllama shared library")
    parser.add_argument("--shim", required=True, help="compiled LlamaSpike C shim")
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--message",
        default="What is the code?",
        help="dummy prompt used only if the state passes the identity check",
    )
    parser.add_argument("--out", default=None, help="write the JSON result here")
    return parser


def probe(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    state = KvState.load(args.state)
    engine = LiveEngine(
        args.model,
        args.lib,
        args.shim,
        n_ctx=args.ctx,
        n_batch=args.batch,
        n_threads=args.threads,
        compute_model_digest=True,
    )
    try:
        record: dict[str, Any] = {
            "probe": "kv-swap",
            "state": state.summary(),
            "engine": {
                "model_digest": engine.model_digest,
                "n_embd": engine.n_embd,
                "n_ctx": engine.n_ctx,
                "n_vocab": engine.n_vocab,
            },
            "matches": state.matches(
                engine.model_digest, engine.n_embd, engine.n_ctx
            ),
        }
        if record["matches"]:
            record["outcome"] = "identity-match"
            record["note"] = (
                "the state matches this model; use a state saved by another model"
            )
            return record, 2
        try:
            engine.chat(
                [{"role": "user", "content": args.message}],
                max_tokens=1,
                restore=state,
            )
        except LiveEngineError as error:
            record["outcome"] = "refused-identity"
            record["error"] = str(error)
            return record, 0
        record["outcome"] = "restored-unexpectedly"
        return record, 1
    finally:
        engine.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        record, status = probe(args)
    except (KvStateError, LiveEngineError, OSError) as error:
        print(f"kv_swap_probe: {error}", file=sys.stderr)
        return 1
    text = json.dumps(record, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
