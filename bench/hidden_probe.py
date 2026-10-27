# SPDX-License-Identifier: GPL-3.0-only

"""Extract final-token hidden states with the offline llama-embedding CLI.

The 8B model is served without ``--embeddings``; the running server will not
return hidden states and must not be reconfigured.  The offline example
``llama-embedding`` is a separate, one-shot process: it loads the GGUF, runs a
single prompt, prints the vector, and exits.  No server is started.

What comes out
--------------

With ``--pooling last`` and ``--embd-normalize -1`` the CLI prints the hidden
state at the last prompt token after the model's final output norm, with no
extra L2 normalisation.  "Unnormalised" here means exactly that: the CLI did
not rescale the vector; the model's own final norm still applies.  If a
trainer needs the pre-final-norm residual stream, this CLI does not expose it.

The CLI writes the vector through its common log.  ``--log-disable`` pauses
that log and discards the vector too, so this probe deliberately does not
pass it; it uses ``--embd-output-format raw`` and parses stdout.

One vector is produced per prompt.  The vector is hashed as packed
little-endian float32, so the hash is stable across runs on the same machine
and across machines with the same endianness.

Output is JSON Lines: one object per prompt, with the prompt hash, dimension,
L2 norm, vector hash, pooling mode and model basename.  The full vector is not
written by default; pass ``--dump-vectors DIR`` to keep one raw vector per
prompt when a downstream trainer needs the numbers.

Usage
-----

    python3 bench/hidden_probe.py \
        --llama-embedding ~/llama.cpp/build/bin/llama-embedding \
        --model ~/dualbrain/models/LFM2.5-350M-Q4_K_M.gguf \
        --text "the gate code is 47-rosso" \
        --text "the sky is green" \
        --pooling last --normalize -1 --expected-dim 1024 \
        --json-out /tmp/opencode/hidden.jsonl

    python3 bench/hidden_probe.py ... --text-file prompts.txt

A text file holds one prompt per line.  Empty lines are skipped.  Prompts
are passed to the CLI as a single ``-p`` argument, so a line is one prompt;
embedded newlines are not possible in this mode.

Exit status is non-zero and the message names the prompt index if the CLI
fails, prints no vector, prints a non-finite value, or returns a dimension
other than ``--expected-dim`` (when given).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path


class ProbeError(RuntimeError):
    """A failure that must stop the run, not be averaged away."""


def sha256_hex(data: bytes, prefix: int = 12) -> str:
    return hashlib.sha256(data).hexdigest()[:prefix]


def parse_vectors(stdout: str) -> list[list[float]]:
    """Return every all-float line the CLI emitted.

    The CLI writes vectors through the common log, so stdout also carries
    model-load lines.  A vector line is the one whose tokens all parse as
    floats.  Log lines that start with a number also contain words, so they
    do not qualify.
    """
    candidates: list[list[float]] = []
    for line in stdout.splitlines():
        fields = line.strip().split()
        if len(fields) < 2:
            continue
        values: list[float] = []
        ok = True
        for field in fields:
            try:
                value = float(field)
            except ValueError:
                ok = False
                break
            if not math.isfinite(value):
                raise ProbeError("CLI printed a non-finite value: %r" % field)
            values.append(value)
        if ok:
            candidates.append(values)
    return candidates


def vector_bytes(vector: list[float]) -> bytes:
    return struct.pack("<%df" % len(vector), *vector)


def l2_norm(vector: list[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in vector))


def cosine(left: list[float], right: list[float]) -> float:
    dot = math.fsum(a * b for a, b in zip(left, right))
    scale = l2_norm(left) * l2_norm(right)
    if scale == 0.0:
        raise ProbeError("cannot compute cosine with a zero-length vector")
    return dot / scale


def run_cli(args: argparse.Namespace, prompt: str) -> list[float]:
    command = [
        args.llama_embedding,
        "-m",
        args.model,
        "-p",
        prompt,
        "--pooling",
        args.pooling,
        "--embd-normalize",
        str(args.normalize),
        "--embd-output-format",
        "raw",
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=args.timeout,
            check=False,
        )
    except OSError as exc:
        raise ProbeError("cannot run %s: %s" % (args.llama_embedding, exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise ProbeError("CLI timed out after %s seconds" % args.timeout) from exc

    if completed.returncode != 0:
        tail = completed.stderr.strip().splitlines()[-8:]
        detail = "\n".join(tail) if tail else "(empty stderr)"
        raise ProbeError(
            "CLI exited %d\n%s" % (completed.returncode, detail)
        )

    candidates = parse_vectors(completed.stdout)
    if not candidates:
        raise ProbeError("CLI printed no vector")
    if args.pooling == "none":
        # --pooling none emits one line per token.  The last line is the last
        # prompt token's hidden state, which is the vector this rig wants.
        dimensions = {len(vector) for vector in candidates}
        if len(dimensions) != 1:
            raise ProbeError(
                "CLI printed token vectors of differing dimensions: %s"
                % sorted(dimensions)
            )
        return candidates[-1]
    if len(candidates) > 1:
        raise ProbeError(
            "CLI printed %d candidate vector lines for pooling=%s; expected exactly one"
            % (len(candidates), args.pooling)
        )
    return candidates[0]


def read_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.text:
        prompts.extend(args.text)
    if args.text_file:
        path = Path(args.text_file)
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ProbeError("cannot read --text-file %s: %s" % (path, exc)) from exc
        prompts.extend(line for line in content.splitlines() if line.strip())
    if not prompts:
        raise ProbeError("no prompts: pass --text (repeatable) or --text-file")
    for index, prompt in enumerate(prompts):
        if "\n" in prompt or "\r" in prompt:
            raise ProbeError(
                "prompt %d contains a newline; llama-embedding splits on newlines, "
                "so pass one prompt per --text or one per line in --text-file" % index
            )
    return prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract final-token hidden states with llama-embedding (offline, no server)."
    )
    parser.add_argument("--llama-embedding", required=True, help="path to the llama-embedding binary")
    parser.add_argument("--model", required=True, help="path to the GGUF model")
    parser.add_argument("--text", action="append", default=[], help="prompt text; repeat for more prompts")
    parser.add_argument("--text-file", help="file with one prompt per line")
    parser.add_argument("--pooling", default="last", help="pooling mode passed to the CLI (default: last)")
    parser.add_argument(
        "--normalize",
        type=int,
        default=-1,
        help="--embd-normalize value (default: -1, no extra norm)",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=None,
        help="fail if the vector length differs from this value",
    )
    parser.add_argument("--json-out", required=True, help="write one JSON object per prompt here")
    parser.add_argument(
        "--cosine-out",
        default=None,
        help="optional JSON file with the pairwise cosine matrix",
    )
    parser.add_argument(
        "--dump-vectors",
        default=None,
        help="optional directory for raw little-endian float32 vectors, one file per prompt",
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="per-prompt CLI timeout in seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        prompts = read_prompts(args)
        if not os.path.isfile(args.llama_embedding):
            raise ProbeError("--llama-embedding is not a file: %s" % args.llama_embedding)
        if not os.path.isfile(args.model):
            raise ProbeError("--model is not a file: %s" % args.model)

        model_name = Path(args.model).name
        records: list[dict] = []
        vectors: list[list[float]] = []
        dimensions: list[int] = []
        dump_dir = Path(args.dump_vectors) if args.dump_vectors else None
        if dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)

        for index, prompt in enumerate(prompts):
            try:
                vector = run_cli(args, prompt)
            except ProbeError as exc:
                raise ProbeError("prompt %d: %s" % (index, exc)) from exc

            dim = len(vector)
            dimensions.append(dim)
            if args.expected_dim is not None and dim != args.expected_dim:
                raise ProbeError(
                    "prompt %d: dimension %d != expected %d" % (index, dim, args.expected_dim)
                )
            if args.expected_dim is None and dimensions and dim != dimensions[0]:
                raise ProbeError(
                    "prompt %d: dimension %d != first dimension %d"
                    % (index, dim, dimensions[0])
                )

            prompt_hash = sha256_hex(prompt.encode("utf-8"))
            vector_hash = sha256_hex(vector_bytes(vector))
            vectors.append(vector)
            record = {
                "prompt_sha256": prompt_hash,
                "dim": dim,
                "l2_norm": l2_norm(vector),
                "vector_sha256": vector_hash,
                "pooling": args.pooling,
                "model": model_name,
                "normalize": args.normalize,
            }
            records.append(record)

            if dump_dir is not None:
                (dump_dir / ("%s.f32" % prompt_hash)).write_bytes(vector_bytes(vector))

            print(
                "prompt %d: dim=%d l2=%.6f vector=%s prompt=%s"
                % (index, dim, record["l2_norm"], vector_hash, prompt_hash)
            )

        out_path = Path(args.json_out)
        if out_path.parent and str(out_path.parent) != ".":
            out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        if args.cosine_out:
            matrix = [
                [cosine(left, right) for right in vectors]
                for left in vectors
            ]
            cosine_path = Path(args.cosine_out)
            if cosine_path.parent and str(cosine_path.parent) != ".":
                cosine_path.parent.mkdir(parents=True, exist_ok=True)
            cosine_path.write_text(
                json.dumps(
                    {
                        "prompt_sha256": [record["prompt_sha256"] for record in records],
                        "cosine": matrix,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    except ProbeError as exc:
        print("hidden_probe: error: %s" % exc, file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
