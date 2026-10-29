# SPDX-License-Identifier: GPL-3.0-only

"""The Python ``pmc fuse`` facade over the Rust binary."""

from __future__ import annotations

import json
import os
import sys

from pmc.cli import EXIT_INVALID, EXIT_OK, main


def run(capsys, argv):
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


FAKE_PAYLOAD = (
    "import json, sys\n"
    "print(json.dumps({{'argv': sys.argv[1:]}}))\n"
    "if {stderr!r}:\n"
    "    print({stderr!r}, file=sys.stderr)\n"
    "raise SystemExit({exit_code})\n"
)


def fake_binary(tmp_path, *, exit_code=0, stderr=""):
    """A stand-in for the Rust binary that reports what it was called with.

    On POSIX a shebang script is enough. On Windows ``subprocess`` cannot
    execute shebang scripts at all, so the same payload is written as a
    ``.py`` file and a ``.cmd`` shim invokes it with the current interpreter.
    Either way the facade under test receives one executable path.
    """
    payload = FAKE_PAYLOAD.format(stderr=stderr, exit_code=exit_code)
    if os.name == "nt":
        script = tmp_path / "pmc_fake_payload.py"
        script.write_text(payload, encoding="utf-8")
        path = tmp_path / "pmc.cmd"
        path.write_text(
            '@echo off\r\n'
            f'"{sys.executable}" "{script}" %*\r\n'
            "exit /b %ERRORLEVEL%\r\n",
            encoding="utf-8",
        )
        return path
    path = tmp_path / "pmc"
    path.write_text(f"#!{sys.executable}\n{payload}", encoding="utf-8")
    path.chmod(0o755)
    return path


def base_arguments(tmp_path):
    binary = fake_binary(tmp_path)
    return [
        "--json",
        "fuse",
        "--base",
        "base.gguf",
        "--rows",
        "rows.bin",
        "--meta",
        "meta.json",
        "--out",
        "out.gguf",
        "--manifest",
        "manifest.json",
        "--pmc-cli",
        str(binary),
    ]


def test_fuse_forwards_every_argument(capsys, tmp_path):
    code, out, _ = run(capsys, base_arguments(tmp_path))
    assert code == EXIT_OK
    payload = json.loads(out)
    assert payload["argv"] == [
        "fuse",
        "--base",
        "base.gguf",
        "--rows",
        "rows.bin",
        "--meta",
        "meta.json",
        "--out",
        "out.gguf",
        "--manifest",
        "manifest.json",
    ]


def test_fuse_uses_the_environment_binary(capsys, tmp_path, monkeypatch):
    binary = fake_binary(tmp_path)
    monkeypatch.setenv("PMC_CLI", str(binary))
    code, out, _ = run(
        capsys,
        [
            "fuse",
            "--base",
            "base.gguf",
            "--rows",
            "rows.bin",
            "--meta",
            "meta.json",
            "--out",
            "out.gguf",
        ],
    )
    assert code == EXIT_OK
    assert json.loads(out)["argv"][:2] == ["fuse", "--base"]


def test_fuse_propagates_the_rust_exit_code_and_stderr(capsys, tmp_path):
    binary = fake_binary(tmp_path, exit_code=3, stderr="fuse failed: bad delta")
    code, _, err = run(
        capsys,
        [
            "fuse",
            "--base",
            "base.gguf",
            "--rows",
            "rows.bin",
            "--meta",
            "meta.json",
            "--out",
            "out.gguf",
            "--pmc-cli",
            str(binary),
        ],
    )
    assert code == 3
    assert "fuse failed: bad delta" in err


def test_fuse_reports_a_missing_binary(capsys, tmp_path):
    code, _, err = run(
        capsys,
        [
            "fuse",
            "--base",
            "base.gguf",
            "--rows",
            "rows.bin",
            "--meta",
            "meta.json",
            "--out",
            "out.gguf",
            "--pmc-cli",
            str(tmp_path / "missing-pmc"),
        ],
    )
    assert code == EXIT_INVALID
    assert err.startswith("error:")
