# SPDX-License-Identifier: GPL-3.0-only

"""Where the Python CLI finds the Rust binary it is a facade over.

The point of these tests is a collision, not a lookup. ``pip install``-ing this
package puts a console script called ``pmc`` on ``PATH``, and the Rust crate
used to build a binary by the same name. Resolving one by the other's name
re-entered ``pmc fuse`` forever, because the Python facade found *itself*.

The Rust binary is therefore called ``pmc-rust``. These tests hold the line:
the resolver looks for that name, and it never claims a plain ``pmc``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from pmc.cli import _resolve_pmc_cli


def test_the_resolver_finds_a_built_binary_in_this_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv("PMC_CLI", raising=False)
    monkeypatch.setattr(Path, "is_file", lambda self: True, raising=False)
    # With nothing on PATH, the checkout target directory is the answer.
    monkeypatch.setattr("shutil.which", lambda _name: None)

    resolved = _resolve_pmc_cli(None)
    assert "pmc-rust" in Path(resolved).name


def test_the_python_console_script_is_never_resolved(tmp_path, monkeypatch):
    """The regression itself.

    A resolver that called ``shutil.which("pmc")`` would find this very CLI
    and recurse; a resolver that looks for ``pmc-rust`` cannot.
    """
    monkeypatch.delenv("PMC_CLI", raising=False)
    monkeypatch.setattr(Path, "is_file", lambda self: False, raising=False)

    # Both are on PATH, the way a pip-installed tree looks: the Python console
    # script answers to `pmc`, and the Rust binary is absent.
    def which(name):  # noqa: ANN001 - shutil.which's signature
        if name == "pmc-rust" or name == "pmc-rust.exe":
            return None
        return "C:/scripts/pmc"

    monkeypatch.setattr("shutil.which", which)
    with pytest.raises(FileNotFoundError) as error:
        _resolve_pmc_cli(None)
    assert "pmc-rust" in str(error.value)


def test_an_explicit_path_is_used_as_given(tmp_path):
    explicit = tmp_path / "pmc-rust"
    assert _resolve_pmc_cli(str(explicit)) == str(explicit)


def test_the_environment_variable_wins_over_the_checkout(tmp_path, monkeypatch):
    from_env = tmp_path / "from-env"
    monkeypatch.setenv("PMC_CLI", str(from_env))
    monkeypatch.setattr(Path, "is_file", lambda self: True, raising=False)

    assert _resolve_pmc_cli(None) == str(from_env)


def test_the_binary_name_is_not_the_console_script_name():
    """Stated so it cannot drift again: the Python entry point is `pmc`."""
    entry_points = _python_console_scripts()
    assert "pmc" in entry_points
    # And the Rust binary the resolver searches for is a different name.
    searched = _resolve_pmc_cli.__doc__ or ""
    assert "pmc-rust" in searched


def _python_console_scripts() -> set[str]:
    """The names pip installs, read from the packaging metadata."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    return set(project.get("project", {}).get("scripts", {}).keys())
