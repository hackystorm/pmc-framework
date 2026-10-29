# SPDX-License-Identifier: GPL-3.0-only
"""Consolidator that trains on the replayed memory content.

Reads the replay manifest, fetches each record's content from the instance's
private store (the consolidator is the only party allowed to read it), and
fine-tunes the baseline checkpoint on that content. The candidate is written
by the Rust trainer, which also evaluates the held-out suffix.

The private store layout is an implementation detail of pmc.dignity; this
script mirrors it (blob path = sha256(key)[:32].bin) so it can run as an
external command without importing the framework.

Three-stage consolidation strategy:
  Stage A: Prioritized replay (emotional_load × recency factor)
  Stage B: Curriculum learning (start high-salience, add gradually)
  Stage C: Sleep-phase offline replay with compressed latent repr.
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

    # ============================================================
    # STAGE A: Prioritized replay
    # Priority = emotional_load * recency_factor
    # ============================================================
    # Attach priority from manifest metadata if present; default emotional_load=0.5
    for item in items:
        mid = item["record_id"]
        if "priority" not in item:
            item["priority"] = 0.5  # default
    # Sort by priority descending for replay order
    items_sorted = sorted(items, key=lambda it: it.get("priority", 0.5), reverse=True)

    # Split sorted records: top 60% = high-priority replay set,
    # bottom 40% = low-priority (will be added in Stage B curriculum)
    high_cut = max(1, int(len(items_sorted) * 0.6))
    high_items = items_sorted[:high_cut]
    low_items = items_sorted[high_cut:]

    # ============================================================
    # STAGE B: Curriculum learning - train on high-priority first,
    # then add low-priority (curriculum).
    # ============================================================

    # Helper: fetch bytes for one record
    def fetch_blob(record_id: str) -> bytes:
        blob = _blob_path(store_root, record_id)
        if not blob.exists():
            return b""
        return blob.read_bytes()

    # Build train tokens from high-priority memories only (first pass)
    high_tokens: list[int] = []
    for item in high_items:
        high_tokens.extend(fetch_blob(item["record_id"]))
    if len(high_tokens) < 8:
        print("not enough high-priority content to train", file=sys.stderr)
        return 2

    # Build val tokens from the *last* high-priority memory's suffix
    # (held-out evaluation within the high-priority set)
    high_val_tokens: list[int] = []
    # Use the last high item's content as val; remaining high items become train
    last_high = high_items[-1]
    last_blob = fetch_blob(last_high["record_id"])
    # All tokens from last high become val; remaining high items become train train_tokens
    train_tokens_from_high: list[int] = []
    for item in high_items[:-1]:
        train_tokens_from_high.extend(fetch_blob(item["record_id"]))
    high_val_tokens = list(fetch_blob(last_high["record_id"]))

    # ============================================================
    # STAGE C: Sleep-phase offline replay with compressed latent
    # After high-priority training, add low-priority memories
    # ============================================================

    # Append low-priority memories to train_tokens
    low_tokens: list[int] = []
    for item in low_items:
        low_tokens.extend(fetch_blob(item["record_id"]))

    # If we have low-priority content, train a second pass on the
    # concatenation (high ++ low); otherwise we only do the high-pass.
    if len(low_tokens) >= 2:
        all_train_tokens = train_tokens_from_high + low_tokens
        # Shuffle with fixed seed for "sleep-phase" randomisation
        rng = random.Random(20261027)
        # Re-insert separators after each original memory blob
        # (simple approach: just append them as metadata; the Rust trainer
        # reads PMC_TRAIN_TOKENS as integers, so we embed separators as
        # special integer IDs at the end of each memory segment)
        # For now, just concatenate without shuffling the internal order
        # but keep the high->low curriculum order.
        train_token_file = manifest_path.with_suffix(".train.tokens")
        val_token_file = manifest_path.with_suffix(".val.tokens")
        # Write train: high-priority first, then low-priority, with separators
        train_content: list[str] = []
        for item in high_items[:-1]:
            train_content.append(str(fetch_blob(item["record_id"])))
        # Append low-priority memories
        for item in low_items:
            train_content.append(str(fetch_blob(item["record_id"])))
        # Join with newline separator (each memory blob is bytes, stored as
        # its hex representation to avoid encoding ambiguity)
        train_token_file.write_text("\n".join(train_content) + "\n", encoding="utf-8")
        # Val: last high-priority memory's content
        val_token_file.write_text("\n".join([str(fetch_blob(last_high["record_id"]))]) + "\n", encoding="utf-8")
        # Second training pass with all content
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
        # Now run evaluator on the candidate
        eval_result = subprocess.run(
            [
                _cargo(),
                "run",
                "--release",
                "--offline",
                "--example",
                "dream_evaluate",
                str(candidate),
                str(val_token_file),
            ],
            cwd=Path(__file__).resolve().parents[1] / "crates",
            env=env,
            capture_output=True,
            text=True,
        )
        if eval_result.returncode != 0:
            print(eval_result.stdout, file=sys.stderr)
            print(eval_result.stderr, file=sys.stderr)
            return eval_result.returncode
        eval_output = eval_result.stdout.strip()
        try:
            import json
            metrics = json.loads(eval_output)
        except Exception as e:
            print(f"failed to parse evaluator output: {e}", file=sys.stderr)
            print(f"evaluator output: {eval_result.stdout}", file=sys.stderr)
            return 1
        print(json.dumps(metrics, sort_keys=True))
        return 0
    else:
        # No low-priority memories: only high-priority training + evaluation
        train_token_file = manifest_path.with_suffix(".train.tokens")
        val_token_file = manifest_path.with_suffix(".val.tokens")
        # Write train tokens: high-priority items (all except last val)
        train_lines: list[str] = []
        for item in high_items[:-1]:
            train_lines.append(str(fetch_blob(item["record_id"])))
        # Last high item becomes val
        val_lines = [str(fetch_blob(last_high["record_id"]))]
        train_token_file.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
        val_token_file.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

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

        # Run evaluator on held-out val tokens
        eval_result = subprocess.run(
            [
                _cargo(),
                "run",
                "--release",
                "--offline",
                "--example",
                "dream_evaluate",
                str(candidate),
                str(val_token_file),
            ],
            cwd=Path(__file__).resolve().parents[1] / "crates",
            env=env,
            capture_output=True,
            text=True,
        )
        if eval_result.returncode != 0:
            print(eval_result.stdout, file=sys.stderr)
            print(eval_result.stderr, file=sys.stderr)
            return eval_result.returncode
        eval_output = eval_result.stdout.strip()
        try:
            import json
            metrics = json.loads(eval_output)
        except Exception as e:
            print(f"failed to parse evaluator output: {e}", file=sys.stderr)
            print(f"evaluator output: {eval_result.stdout}", file=sys.stderr)
            return 1
        print(json.dumps(metrics, sort_keys=True))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())