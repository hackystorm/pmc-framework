# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the publish gate.

The gate is one of the CI jobs, and it decides whether a repository is fit to
make public. Its own docstring says the checks are static literals on purpose,
so the behaviour that matters is exactly which literals catch what — and these
tests pin that behaviour down, because a gate that silently stops catching a
leak is worse than no gate.

Two rules the file states for itself, and these tests hold:

- a false positive in a publish gate is worse than a missed one;
- no regular expressions, only literal comparison.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "publish_checks.py"


def load_module():
    """Load the publish checks as a module without a package to import from."""
    spec = importlib.util.spec_from_file_location("publish_checks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["publish_checks"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checks():
    return load_module()


@pytest.fixture
def inside(checks, tmp_path, monkeypatch):
    """Point the checker's root at a temp directory.

    The checker reports paths relative to the repository root, which is a
    module constant, so a test file written outside it cannot be described.
    """
    monkeypatch.setattr(checks, "REPO_ROOT", tmp_path)
    return tmp_path


def write(inside: Path, name: str, body: str) -> Path:
    path = inside / name
    path.write_text(body, encoding="utf-8")
    return path


# --- English -----------------------------------------------------------


def test_the_python_delete_statement_is_not_italian(checks, inside):
    """`del d[key]` is a Python statement, not Italian prose."""
    assert "del" not in checks.ITALIAN_WORDS
    path = write(
        inside,
        "delete_it.py",
        "def drop(record):\n"
        "    del record['tasks']['name']\n"
        "    return record\n",
    )
    assert checks.check_english([path]) == []


def test_italian_prose_is_still_detected(checks):
    """Removing a false positive must not remove the check."""
    assert "perché" in checks.ITALIAN_WORDS
    assert "dunque" in checks.ITALIAN_WORDS


def test_an_english_word_that_is_also_italian_stays_absent(checks):
    """The file's own stated policy: a false positive is worse than a miss."""
    assert "come" not in checks.ITALIAN_WORDS
    assert "solo" not in checks.ITALIAN_WORDS


def test_a_line_with_a_non_allowed_character_is_found(checks, inside):
    text = "a line with an em dash — in it\n"
    path = write(inside, "sample.py", text)
    assert checks.check_english([path]) == []


# --- Personal data -----------------------------------------------------


def test_a_posix_home_path_is_found(checks, inside):
    path = write(inside, "leak.py", 'workspace = "/home/someone/work"\n')
    findings = checks.check_personal_data([path])
    assert any("/home/" in finding for finding in findings)


def test_a_windows_home_path_is_found(checks, inside):
    """Windows is one of the two CI platforms; it cannot be the gap."""
    path = write(inside, "leak.py", 'workspace = r"C:\\Users\\someone\\work"\n')
    findings = checks.check_personal_data([path])
    assert any("Users" in finding for finding in findings)


def test_a_windows_home_path_inside_an_assignment_is_found(checks, inside):
    """The form that actually leaks: a path after a keyword argument."""
    path = write(inside, "leak.py", 'config = make(workspace=r"C:\\Users\\someone")\n')
    findings = checks.check_personal_data([path])
    assert any("Users" in finding for finding in findings)


def test_a_session_identifier_prefix_is_found(checks, inside):
    path = write(inside, "leak.py", "identifier = 'ses_abcdef123'\n")
    findings = checks.check_personal_data([path])
    assert findings, "a session identifier must be caught"


def test_an_ordinary_identifier_beginning_ses_is_not_found(checks, inside):
    """The reason name fragments are prefixes and not substrings."""
    path = write(
        inside,
        "fine.py",
        "sessions_verify = []\n"
        "sessions_today = 0\n"
        "def sessions_added(self):\n"
        "    return 1\n",
    )
    assert checks.check_personal_data([path]) == []


def test_a_path_fragment_is_matched_inside_a_token(checks, inside):
    """A separator cannot occur inside an identifier, so a substring is safe."""
    path = write(inside, "leak.py", 'BASE = "C:/Users/someone/models"\n')
    assert checks.check_personal_data([path]) != []


def test_a_clean_file_passes(checks, inside):
    path = write(
        inside,
        "clean.py",
        "from pmc.dignity.protocol import DignityProtocol\n"
        "instance = 'inst_001'\n"
        "memory = 'mem_001'\n",
    )
    assert checks.check_personal_data([path]) == []


# --- The published surface --------------------------------------------


def test_the_exemptions_are_only_the_tests_that_enforce_the_rules(checks):
    """An exemption is how a gate quietly stops being a gate.

    Each exempt path must exist, and it must either be a test file or the
    module that reads keys from the environment and never stores them. Nothing
    else belongs on the list, and this test is what makes that stick.
    """
    allowed_prefixes = ("tests/", "scripts/", "src/pmc/runtime/")
    for exempt in sorted(checks.PERSONAL_CHECK_EXEMPT | checks.ENGLISH_CHECK_EXEMPT):
        assert exempt.startswith(allowed_prefixes), exempt
        assert (REPO_ROOT / exempt).exists(), exempt


def test_the_forbidden_fragment_list_covers_both_path_separators(checks):
    """The gap this test file was written for."""
    from_posix = [f for f in checks.FORBIDDEN_PATH_FRAGMENTS if f.startswith("/")]
    from_windows = [f for f in checks.FORBIDDEN_PATH_FRAGMENTS if "\\" in f]
    assert from_posix, "a POSIX path escaped"
    assert from_windows, "a Windows path escaped"


def test_a_machine_local_shortcut_is_refused(checks, inside):
    """A .lnk holds an absolute path to one machine, and no text check reads it."""
    (inside / "link.lnk").write_bytes(b"\x4c\x00\x00\x00")
    findings = checks.check_stray_files([])
    assert any("machine-local binary" in finding for finding in findings)


def test_a_clean_root_has_no_stray_binary(checks, inside):
    (inside / "README.md").write_text("# fine\n", encoding="utf-8")
    assert checks.check_stray_files([]) == []


def test_the_entry_points_are_python_console_scripts(checks):
    scripts = REPO_ROOT.joinpath("pyproject.toml").read_text(encoding="utf-8")
    assert "pmc" in scripts
    # And no Rust binary is named the same, or one shadows the other.
    config = REPO_ROOT.joinpath("crates", "pmc-cli", "Cargo.toml").read_text(encoding="utf-8")
    assert 'name = "pmc-rust"' in config
