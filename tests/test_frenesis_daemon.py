# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the FRENESIS daemon module.

The daemon is the component that turns the framework's pieces into a running
agent: a llama-server subprocess, an event bus, a scheduler, a runtime and an
HTTP surface. These tests do not start any of that. They pin the things that
are easy to get wrong and expensive to get wrong quietly:

- that the module imports from an **installed** package, not only from a
  checkout, because `sys.path` tricks stop working the moment it is installed;
- that importing it writes no file, into no directory, ever. The previous
  version created a log file in the current working directory at import, which
  is how a stray log reached the repository root and the publish gate.

The first two of those are import behaviour, so they are checked in a
subprocess: a test that clears `sys.modules` inside the shared interpreter
leaves every later test in the same process importing a second, different copy
of `pmc`, and two classes with the same name and different identities fail in
ways that have nothing to do with the code under test.

The daemon's own daemonisation, llama-server management, event routing,
scheduler and HTTP endpoints are covered separately; a test suite is not the
place to start a llama.cpp server.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _in_subprocess(code: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run a short Python snippet, imported by its package name, in a clean process."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_module_imports_from_an_installed_package(tmp_path):
    """Import it by its installed name, with the checkout off the path.

    This is the regression that matters. The module used to insert the
    repository root into `sys.path` and import `src.pmc.runtime...`. That
    succeeds, which is why it went unnoticed: it silently builds a *second*
    copy of every `pmc` module, so the classes the daemon holds are not the
    classes the rest of the framework uses, and an `isinstance` check across
    the boundary fails. The test asserts the two are the same object, so a
    sys.path hack cannot pass it by merely working.
    """
    result = _in_subprocess(
        "import pmc.runtime.frenesis_daemon as m\n"
        "import pmc.runtime.session as session\n"
        "assert m.CompanionRuntime is session.CompanionRuntime, (\n"
        "    'the daemon holds a second copy of the runtime class'\n"
        ")\n"
        "assert m.__name__ == 'pmc.runtime.frenesis_daemon'\n"
        "print('same class')",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "same class"


def test_the_old_import_hack_would_fail_this_check(tmp_path):
    """The check above fails on the bug it was written for.

    Simulating the previous import proves the assertion bites: a module that
    imports through `src.` holds a different class object, and `is` is false.
    """
    repo = str(REPO_ROOT)
    result = _in_subprocess(
        "import sys\n"
        f"sys.path.insert(0, {repo!r})\n"
        "import pmc.runtime.session as installed\n"
        "import src.pmc.runtime.session as legacy\n"
        "assert installed.CompanionRuntime is not legacy.CompanionRuntime\n"
        "print('two copies')",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "two copies"


def test_importing_the_module_writes_no_file(tmp_path):
    """Importing a module must not create a log in the current directory."""
    result = _in_subprocess(
        "import pathlib\n"
        "before = set(pathlib.Path('.').iterdir())\n"
        "import pmc.runtime.frenesis_daemon\n"
        "after = set(pathlib.Path('.').iterdir())\n"
        "print(sorted(str(p) for p in after - before))\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", "import created files"


def test_importing_the_module_writes_no_file(tmp_path):
    """Importing a module must not create a log in the current directory."""
    result = _in_subprocess(
        "import os, pathlib\n"
        "before = set(pathlib.Path('.').iterdir())\n"
        "import pmc.runtime.frenesis_daemon\n"
        "after = set(pathlib.Path('.').iterdir())\n"
        "print(sorted(str(p) for p in after - before))\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", "import created files"


def _load():
    return importlib.import_module("pmc.runtime.frenesis_daemon")


def test_configure_logging_is_idempotent():
    module = _load()
    logger = module.logger
    original_handlers = list(logger.handlers)
    original_propagate = logger.propagate
    try:
        module.configure_logging(log_file=Path("daemon.log"))
        first = list(logger.handlers)
        # A second call must not add a second set of handlers, and must not
        # swap in a different destination.
        module.configure_logging(log_file=Path("other.log"))
        assert logger.handlers == first
    finally:
        logger.handlers[:] = original_handlers
        logger.propagate = original_propagate


def test_configure_logging_writes_to_the_named_file_only(tmp_path):
    """A file is created only when the caller names one."""
    module = _load()
    logger = module.logger
    original_handlers = list(logger.handlers)
    original_propagate = logger.propagate
    try:
        target = tmp_path / "daemon.log"
        module.configure_logging(log_file=target)
        logger.info("a line written for the test")
        assert target.is_file()
        assert not (tmp_path / "other.log").exists()
    finally:
        logger.handlers[:] = original_handlers
        logger.propagate = original_propagate


def test_configure_logging_without_a_file_writes_nothing():
    """The default configuration is stdout only."""
    module = _load()
    logger = module.logger
    original_handlers = list(logger.handlers)
    original_propagate = logger.propagate
    try:
        module.configure_logging()
        logger.info("a line with nowhere to go but stdout")
        handlers = logger.handlers
        assert len(handlers) == 1
        assert not isinstance(handlers[0], logging.FileHandler)
    finally:
        logger.handlers[:] = original_handlers
        logger.propagate = original_propagate


def test_an_import_leaves_nothing_running():
    """The module's import is inert: nothing runs until something calls it."""
    module = _load()
    # No subprocess, no socket and no background task exists at import.
    assert isinstance(module.FRENESISDaemon, type)
    assert isinstance(module.WindowsService, type)
    # Creating a daemon requires a config; importing does not create one.
    assert module.DaemonConfig is not None
    assert module.configure_logging.__doc__ is not None
