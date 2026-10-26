# SPDX-License-Identifier: GPL-3.0-only

"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pmc.dignity.protocol import DignityProtocol  # noqa: E402
from pmc.dignity.records import Instance  # noqa: E402


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


@pytest.fixture()
def protocol(workspace: Path) -> DignityProtocol:
    return DignityProtocol(workspace)


@pytest.fixture()
def instance() -> Instance:
    return Instance(instance_id="inst_test_001", name="TestInstance", substrate="test")


@pytest.fixture()
def source_instance() -> Instance:
    return Instance(instance_id="inst_test_src", name="SourceInstance")


@pytest.fixture()
def target_instance() -> Instance:
    return Instance(instance_id="inst_test_dst", name="TargetInstance")
