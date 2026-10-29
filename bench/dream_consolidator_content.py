# SPDX-License-Identifier: GPL-3.0-only
"""Consolidator that trains on the replayed memory content.

Reads the replay manifest, fetches each record's content from the instance's
private store (the consolidator is the only party allowed to read it), and
fine-tunes the baseline checkpoint on that content. The candidate is written
by the Rust trainer, which also evaluates the held-out suffix.

The private store layout is an implementation detail of pmc.dignity; this
script mirrors it (blob path = sha256(key)[:32].bin) so it can run as an
external command without importing the framework.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

def _cargo() -> str:
    candidate = os.environ.get("CARGO")
    if candidate and Path(candidate).is_file():
        return candidate
    found = shutil.which("cargo")
    if found:
        return found
    default = Path(os.environ.get("USERPROFILE", "")) / ".cargo" / "bin" / "cargo.exe"
    if default.is_file():
        return str(default)
    return "cargo"


def _canonical_json(payload: Any) -> str:
    import json

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _sha256_of(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _blob_path(store_root: Path, key: str) -> Path:
    return store_root / f"{_sha256_of(key)[:32]}.bin"


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: dream_consolidator_content.py <baseline> <candidate> <manifest>", file=sys.stderr)
        return 2
    baseline = Path(sys.argv[1])
    candidate = Path(sys.argv[2])
    manifest_path = Path(sys.argv[3])

    if not baseline.is_dir():
        print(f"baseline is not a directory: {baseline}", file=sys.stderr)
        return 2

    # Infer the workspace from the baseline checkpoint location.
    workspace = baseline.resolve().parent.parent
    store_root = workspace / "sidecar" / "private"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    record_ids = [item["record_id"] for item in items]

    if len(record_ids) < 2:
        print("not enough memories to train", file=sys.stderr)
        return 2

    # Shuffle and split into train/val (80/20).
    rng = random.Random(20261027)
    shuffled_ids = record_ids[:]
    rng.shuffle(shuffled_ids)
    split_idx = max(1, int(len(shuffled_ids) * 0.8))
    train_ids = shuffled_ids[:split_idx]
    val_ids = shuffled_ids[split_idx:]

    def ids_to_tokens(ids: list[str]) -> list[int]:
        tokens: list[int] = []
        for record_id in ids:
            blob = _blob_path(store_root, record_id)
            if not blob.exists():
                print(f"missing content for {record_id}: {blob}", file=sys.stderr)
                return []
            tokens.extend(blob.read_bytes())
        return tokens

    train_tokens = ids_to_tokens(train_ids)
    val_tokens = ids_to_tokens(val_ids)

    if len(train_tokens) < 8 or len(val_tokens) < 2:
        print("not enough content to train", file=sys.stderr)
        return 2

    train_token_file = manifest_path.with_suffix(".train.tokens")
    val_token_file = manifest_path.with_suffix(".val.tokens")
    train_token_file.write_text("\n".join(str(t) for t in train_tokens) + "\n", encoding="utf-8")
    val_token_file.write_text("\n".join(str(t) for t in val_tokens) + "\n", encoding="utf-8")

    if candidate.exists():
        shutil.rmtree(candidate)
    candidate.mkdir(parents=True)

    env = dict(os.environ)
    env["PMC_TRAIN_TOKENS"] = str(train_token_file)
    env["PMC_VAL_TOKENS"] = str(val_token_file)
    env["PMC_CHECKPOINT"] = str(candidate)
    env["PMC_BASELINE"] = str(baseline)
    env["PMC_STEPS"] = os.environ.get("PMC_STEPS", "10")
    env["PMC_LR"] = os.environ.get("PMC_LR", "0.0005")
    env["PMC_N_LAYER"] = os.environ.get("PMC_N_LAYER", "12")
    env["PMC_D_MODEL"] = os.environ.get("PMC_D_MODEL", "2048")
    env["PMC_N_HEAD"] = os.environ.get("PMC_N_HEAD", "16")
    env["PMC_FFN"] = os.environ.get("PMC_FFN", "5632")
    env["PMC_VOCAB"] = os.environ.get("PMC_VOCAB", "256")
    env["PMC_N_POS"] = os.environ.get("PMC_N_POS", "4096")
    env["PMC_RANK"] = os.environ.get("PMC_RANK", "128")
    env["PMC_HOLDOUT"] = os.environ.get("PMC_HOLDOUT", "8")
    env["PMC_EARLY_STOPPING"] = os.environ.get("PMC_EARLY_STOPPING", "1")
    env["PMC_PATIENCE"] = os.environ.get("PMC_PATIENCE", "3")
    env["PMC_EMBEDDING_SEED"] = os.environ.get("PMC_EMBEDDING_SEED", "20261032")

    if candidate.exists():
        shutil.rmtree(candidate)
    candidate.mkdir(parents=True)

    result = subprocess.run(
        [_cargo(), "run", "--release", "--offline", "--example", "train_lm"],
        cwd=Path(__file__).resolve().parents[1] / "crates",
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())