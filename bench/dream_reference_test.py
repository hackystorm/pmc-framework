# SPDX-License-Identifier: GPL-3.0-only

"""The dream reference check, exercised on a small opaque checkpoint."""

from __future__ import annotations

import json

import dream_reference


def test_reference_check_rolls_back_and_refuses(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payload = b"reference checkpoint"
    (checkpoint / "checkpoint.bin").write_bytes(payload)
    cache = tmp_path / "cache"

    result = dream_reference.run(checkpoint, cache)

    assert result["ok"] is True
    assert all(result["checks"].values())
    assert result["dream"]["outcome"] == "rolled_back"
    assert result["dream"]["gate"]["reasons"] == ["recall_newer"]
    assert result["refusal"]["outcome"] == "refused"
    assert result["refusal"]["tainted"] == [result["checkpoint_digest"]]
    assert (checkpoint / "checkpoint.bin").read_bytes() == payload

    entries = [
        json.loads(line)
        for line in (cache / "dreams.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["outcome"] for entry in entries] == ["rolled_back", "refused"]
