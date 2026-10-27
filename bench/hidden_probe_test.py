# SPDX-License-Identifier: GPL-3.0-only

"""Unit tests for the offline hidden-state probe."""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "bench" / "hidden_probe.py"

_spec = importlib.util.spec_from_file_location("hidden_probe", PROBE_PATH)
assert _spec is not None and _spec.loader is not None
hidden_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hidden_probe)


def test_parse_vectors_ignores_log_lines():
    stdout = (
        "load time = 1.23 ms\n"
        "0.1 0.2 -0.3\n"
        "llama_perf_context_print: total time = 2.00 ms\n"
    )
    assert hidden_probe.parse_vectors(stdout) == [[0.1, 0.2, -0.3]]


def test_parse_vectors_reads_every_token_vector():
    assert hidden_probe.parse_vectors("0.1 0.2\n0.3 0.4\n") == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]


def test_parse_vectors_rejects_non_finite_values():
    with pytest.raises(hidden_probe.ProbeError):
        hidden_probe.parse_vectors("0.1 nan\n")


def test_vector_hash_is_stable_and_prefixed():
    vector = [1.0, -2.0, 3.5]
    packed = hidden_probe.vector_bytes(vector)
    assert len(packed) == 12
    assert struct.unpack("<3f", packed) == (1.0, -2.0, 3.5)
    digest = hidden_probe.sha256_hex(packed)
    assert len(digest) == 12
    assert digest == hidden_probe.sha256_hex(packed)


def test_cosine_identity_and_orthogonal():
    assert hidden_probe.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert hidden_probe.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def _stub_cli(tmp_path: Path, body: str) -> Path:
    stub = tmp_path / "fake-llama-embedding"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        + body,
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def _model_file(tmp_path: Path) -> Path:
    model = tmp_path / "toy.gguf"
    model.write_bytes(b"not a real gguf")
    return model


def test_main_writes_jsonl_and_fails_on_wrong_dimension(tmp_path: Path):
    stub = _stub_cli(tmp_path, "print('load time = 1.0')\nprint('0.1 0.2 0.3')\n")
    model = _model_file(tmp_path)
    out = tmp_path / "out.jsonl"

    argv = [
        "--llama-embedding",
        str(stub),
        "--model",
        str(model),
        "--text",
        "a prompt",
        "--expected-dim",
        "3",
        "--json-out",
        str(out),
    ]
    assert hidden_probe.main(argv) == 0
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert record["dim"] == 3
    assert record["pooling"] == "last"
    assert record["vector_sha256"] == hidden_probe.sha256_hex(
        hidden_probe.vector_bytes([0.1, 0.2, 0.3])
    )

    bad_out = tmp_path / "bad.jsonl"
    argv[argv.index("--expected-dim") + 1] = "4"
    argv[argv.index("--json-out") + 1] = str(bad_out)
    assert hidden_probe.main(argv) == 2
    assert not bad_out.exists()


def test_main_pooling_none_uses_last_token_vector(tmp_path: Path):
    stub = _stub_cli(tmp_path, "print('0.1 0.2')\nprint('0.3 0.4')\n")
    model = _model_file(tmp_path)
    out = tmp_path / "none.jsonl"

    argv = [
        "--llama-embedding",
        str(stub),
        "--model",
        str(model),
        "--text",
        "a prompt",
        "--pooling",
        "none",
        "--expected-dim",
        "2",
        "--json-out",
        str(out),
    ]
    assert hidden_probe.main(argv) == 0
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert record["pooling"] == "none"
    assert record["vector_sha256"] == hidden_probe.sha256_hex(
        hidden_probe.vector_bytes([0.3, 0.4])
    )
