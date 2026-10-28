# SPDX-License-Identifier: GPL-3.0-only
"""Build the views file the Rust state encoder trains on.

The v4 mechanism reads the raw hidden state of each substrate and projects it
with a trainable per-host matrix, so it needs the extracted hidden states
themselves, not the frozen 128-dimensional projection the bench applies on
top of them.  The output is the `PMCV` binary that `pmc-train` reads.

Only the training facts and the training questions are written.  The
evaluation facts, the evaluation questions and the control questions are
never selected, so a gate run on this file cannot contaminate the attempt the
gate decides: the held-out cues stay held-out.

The script is standard library only and never loads a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import substrate_swap as swap

MAGIC = b"PMCV"
VERSION = 1
SCHEMA = "pmc-state-views/1"

KIND_FACT = 0
KIND_QUESTION = 1

TRAINING_KINDS = ("training_fact", "training_question")
"""The only record kinds this builder is allowed to read."""

KIND_IDS = {"training_fact": KIND_FACT, "training_question": KIND_QUESTION}


class ViewsError(RuntimeError):
    """The builder was asked for views it must not produce."""


def load_host(hidden_path: str | Path, spec: dict[str, Any], host: str) -> dict[str, Any]:
    """Load and validate one host's hidden states through the frozen reader.

    Using the bench's own reader keeps every check the bench already performs:
    the spec digest, the anchor, the hidden dimension, the prompt digests and
    the cue identity of every record.
    """
    return swap.load_hidden(hidden_path, spec, host)


def training_labels(spec: dict[str, Any]) -> list[int]:
    """The label indices of the training facts, in spec order."""
    return list(range(len(spec["training_facts"])))


def raw_hidden(
    records: dict[tuple[str, int], dict[str, Any]],
    kind: str,
    label: int,
    host: str,
    dimension: int,
) -> list[float]:
    """One raw hidden vector, checked for the right length and finiteness."""
    try:
        record = records[(kind, label)]
    except KeyError as error:
        raise ViewsError(f"{host}: no {kind} record for label {label}") from error
    vector = record["h"]
    if len(vector) != dimension:
        raise ViewsError(
            f"{host}: {kind} label {label} has {len(vector)} values, expected {dimension}"
        )
    for value in vector:
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            raise ViewsError(f"{host}: {kind} label {label} has a non-finite value")
    return [swap.f32(value) for value in vector]


def build_entries(
    spec: dict[str, Any],
    hidden_a: dict[str, Any],
    hidden_b: dict[str, Any],
) -> list[dict[str, Any]]:
    """The view entries, ordered by host, then label, then kind."""
    labels = training_labels(spec)
    if not labels:
        raise ViewsError("the spec has no training fact")
    sources = (
        (0, "350m", hidden_a),
        (1, "8b", hidden_b),
    )
    entries: list[dict[str, Any]] = []
    for host, name, hidden in sources:
        dimension = int(hidden["n_embd"])
        by_key = {
            (record["kind"], int(record["index"])): record
            for record in hidden["records"]
        }
        for label in labels:
            for kind in TRAINING_KINDS:
                entries.append(
                    {
                        "host": host,
                        "label": label,
                        "kind": KIND_IDS[kind],
                        "data": raw_hidden(by_key, kind, label, name, dimension),
                    }
                )
    return entries


def encode_views(entries: Sequence[dict[str, Any]]) -> bytes:
    """The `PMCV` bytes for the entries.

    The layout matches `pmc_train::pmc_state::read_views`: magic, version,
    count, then per entry the host, the label, the kind with three padding
    bytes, the dimension and the little-endian float payload.
    """
    raw = bytearray()
    raw += MAGIC
    raw += struct.pack("<I", VERSION)
    raw += struct.pack("<I", len(entries))
    for entry in entries:
        raw += struct.pack("<II", int(entry["host"]), int(entry["label"]))
        raw += bytes((int(entry["kind"]), 0, 0, 0))
        data = [swap.f32(value) for value in entry["data"]]
        raw += struct.pack("<I", len(data))
        raw += b"".join(struct.pack("<f", value) for value in data)
    return bytes(raw)


def decode_views(raw: bytes) -> list[dict[str, Any]]:
    """Read back a `PMCV` file, so the round trip can be checked in Python."""
    if raw[:4] != MAGIC:
        raise ViewsError("the views file does not carry the views magic")
    version, count = struct.unpack_from("<II", raw, 4)
    if version != VERSION:
        raise ViewsError(f"unsupported views version {version}")
    offset = 12
    entries: list[dict[str, Any]] = []
    for _ in range(count):
        if offset + 16 > len(raw):
            raise ViewsError("the views file is truncated in a header")
        host, label = struct.unpack_from("<II", raw, offset)
        kind = raw[offset + 8]
        dim = struct.unpack_from("<I", raw, offset + 12)[0]
        offset += 16
        if offset + dim * 4 > len(raw):
            raise ViewsError("the views file is truncated in a vector")
        data = [
            struct.unpack_from("<f", raw, offset + index * 4)[0]
            for index in range(dim)
        ]
        offset += dim * 4
        entries.append({"host": host, "label": label, "kind": kind, "data": data})
    if offset != len(raw):
        raise ViewsError("the views file has trailing bytes")
    return entries


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    spec_path: str | Path,
    hidden_350m_path: str | Path,
    hidden_8b_path: str | Path,
    *,
    out_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write the views file and return the manifest describing it."""
    started = time.time()
    spec = swap.load_spec(spec_path)
    hidden_a = load_host(hidden_350m_path, spec, "350m")
    hidden_b = load_host(hidden_8b_path, spec, "8b")
    entries = build_entries(spec, hidden_a, hidden_b)
    raw = encode_views(entries)
    out = Path(out_path)
    if out.parent and str(out.parent) not in ("", "."):
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)

    labels = training_labels(spec)
    manifest = {
        "schema": SCHEMA,
        "created": int(started),
        "elapsed_s": time.time() - started,
        "spec_sha256": spec["spec_sha256"],
        "anchor": spec.get("anchor", "last"),
        "hidden_350m_sha256": sha256_file(hidden_350m_path),
        "hidden_8b_sha256": sha256_file(hidden_8b_path),
        "hidden_350m_model_sha256": hidden_a.get("model_sha256", ""),
        "hidden_8b_model_sha256": hidden_b.get("model_sha256", ""),
        "view_count": len(entries),
        "views_sha256": hashlib.sha256(raw).hexdigest(),
        "views_bytes": len(raw),
        "labels": labels,
        "label_count": len(labels),
        "kinds": {str(value): name for name, value in KIND_IDS.items()},
        "kinds_used": list(TRAINING_KINDS),
        "evaluation_records_used": 0,
        "control_records_used": 0,
        "hosts": [
            {
                "host": index,
                "model": name,
                "dim": int(hidden["n_embd"]),
                "records": len(hidden["records"]),
            }
            for index, name, hidden in ((0, "350m", hidden_a), (1, "8b", hidden_b))
        ],
    }
    if manifest_path:
        path = Path(manifest_path)
        if path.parent and str(path.parent) not in ("", "."):
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="frozen word-list spec")
    parser.add_argument("--hidden-350m", required=True, help="350M hidden states")
    parser.add_argument("--hidden-8b", required=True, help="8B hidden states")
    parser.add_argument("--out", required=True, help="the PMCV file to write")
    parser.add_argument("--manifest", default=None, help="write the JSON manifest here")
    args = parser.parse_args(argv)
    manifest = run(
        args.spec,
        args.hidden_350m,
        args.hidden_8b,
        out_path=args.out,
        manifest_path=args.manifest,
    )
    print(
        "wrote {views} digest {views_sha256} views {view_count} labels "
        "{label_count} dims {dim_a}/{dim_b}".format(
            views=args.out,
            views_sha256=manifest["views_sha256"],
            view_count=manifest["view_count"],
            label_count=manifest["label_count"],
            dim_a=manifest["hosts"][0]["dim"],
            dim_b=manifest["hosts"][1]["dim"],
        ),
        file=sys.stderr,
        flush=True,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
