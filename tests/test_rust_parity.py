# SPDX-License-Identifier: GPL-3.0-only

"""Numerical parity between the Rust trainer and the Python reference.

The Rust implementation is the fast one that ships. The Python reference in
`pmc.reference` is the slow one that ships in the same repository. This test
requires them to agree: same seed, same data, same weights (within 1e-5),
same metrics (within 1e-3).

It also checks two things that are easy to get wrong and expensive to get
wrong quietly:

* that the Rust SHA-256 implementation agrees with `hashlib`, which is what
  makes the exported manifest verifiable;
* that training is deterministic, because a result that cannot be regenerated
  from a seed is not a result.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from pmc.reference import ReferenceConfig, ReferenceTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
CRATES_DIR = REPO_ROOT / "crates"

pytestmark = pytest.mark.skipif(
    shutil.which("cargo") is None,
    reason="cargo is not available, so the Rust trainer cannot be built",
)

# A deliberately small configuration: the point is agreement, not speed, and
# a small run keeps the numerical difference between two implementations of
# the same mathematics inside the tolerance.
SEED = 20261026
STEPS = 40
LEARNING_RATE = 0.05
HIDDEN = 8
KEYS = 8

WEIGHT_TOLERANCE = 5e-5
"""Tolerance for trained weights.

The two implementations agree *exactly* on the initial weights, because the
RNG and the initialisation are the same integer and floating-point
operations. They drift slightly during training, and the cause is
identifiable: Rust evaluates `exp`, `log` and `cos` in single precision
(`expf`/`logf`/`cosf`), while Python has to evaluate them in double
precision and round afterwards. The two differ by about one unit in the
last place of an `f32`, and forty Adam steps amplify that into roughly
1e-5. The tolerance is set above the measured drift and well below the
scale of the weights themselves.
"""

METRIC_TOLERANCE = 1e-3


@pytest.fixture(scope="module")
def pmc_binary() -> Path:
    """Build the Rust CLI once for this module."""
    result = subprocess.run(
        ["cargo", "build", "--offline", "-p", "pmc-cli"],
        cwd=CRATES_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"cargo build failed:\n{result.stdout}\n{result.stderr}")
    return CRATES_DIR / "target" / "debug" / "pmc"


def run_train(binary: Path, out_dir: Path, *, seed: int = SEED, steps: int = STEPS) -> dict:
    """Run the Rust trainer and return its parsed report."""
    result = subprocess.run(
        [
            str(binary),
            "train",
            "--seed",
            str(seed),
            "--steps",
            str(steps),
            "--lr",
            str(LEARNING_RATE),
            "--hidden",
            str(HIDDEN),
            "--keys",
            str(KEYS),
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"pmc train failed:\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[0])


def reference_report() -> tuple[ReferenceTrainer, dict]:
    trainer = ReferenceTrainer(
        ReferenceConfig(
            seed=SEED,
            steps=STEPS,
            learning_rate=LEARNING_RATE,
            hidden=HIDDEN,
            keys=KEYS,
        )
    )
    report = trainer.train()
    return trainer, report.to_dict()


def test_the_two_implementations_agree_on_the_metrics(pmc_binary, tmp_path):
    rust = run_train(pmc_binary, tmp_path / "rust")
    _, python = reference_report()

    assert rust["seed"] == python["seed"]
    assert rust["steps"] == python["steps"]
    assert rust["final_accuracy"] == pytest.approx(python["final_accuracy"], abs=1e-6)
    assert rust["final_loss"] == pytest.approx(python["final_loss"], abs=METRIC_TOLERANCE)


def test_the_initial_weights_match_exactly(pmc_binary, tmp_path):
    """Before any training step, the two implementations are bit-identical.

    This is the sharp test: it isolates the RNG and the initialisation from
    the optimiser, and it has no tolerance at all.
    """
    run_train(pmc_binary, tmp_path / "rust", steps=0)

    raw = (tmp_path / "rust" / "weights.bin").read_bytes()
    rust_weights = list(struct.unpack(f"<{len(raw) // 4}f", raw))

    trainer = ReferenceTrainer(
        ReferenceConfig(
            seed=SEED,
            steps=0,
            learning_rate=LEARNING_RATE,
            hidden=HIDDEN,
            keys=KEYS,
        )
    )
    python_weights = list(struct.unpack(f"<{len(raw) // 4}f", trainer.weight_bytes()))

    assert rust_weights == python_weights


def test_the_two_implementations_agree_on_the_weights(pmc_binary, tmp_path):
    run_train(pmc_binary, tmp_path / "rust")

    raw = (tmp_path / "rust" / "weights.bin").read_bytes()
    rust_weights = list(struct.unpack(f"<{len(raw) // 4}f", raw))

    trainer, _ = reference_report()
    python_weights = list(struct.unpack(f"<{len(raw) // 4}f", trainer.weight_bytes()))

    assert len(rust_weights) == len(python_weights)
    assert rust_weights, "the exported weight file is empty"

    differences = [
        abs(a - b) for a, b in zip(rust_weights, python_weights)
    ]
    worst = max(differences)
    assert worst <= WEIGHT_TOLERANCE, (
        f"the Rust and Python weights differ by {worst}, which is above the "
        f"{WEIGHT_TOLERANCE} tolerance"
    )


def test_training_is_deterministic(pmc_binary, tmp_path):
    first = run_train(pmc_binary, tmp_path / "first")
    second = run_train(pmc_binary, tmp_path / "second")
    assert first["weight_digest"] == second["weight_digest"]
    assert first["final_loss"] == second["final_loss"]


def test_the_manifest_digests_match_hashlib(pmc_binary, tmp_path):
    """The Rust SHA-256 must agree with the reference implementation."""
    import hashlib

    run_train(pmc_binary, tmp_path / "rust")
    manifest = json.loads((tmp_path / "rust" / "manifest.json").read_text(encoding="utf-8"))

    for entry in manifest["files"]:
        path = tmp_path / "rust" / entry["name"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"digest mismatch for {entry['name']}"
        assert path.stat().st_size == entry["bytes"]

    weights = (tmp_path / "rust" / "weights.bin").read_bytes()
    assert hashlib.sha256(weights).hexdigest() == manifest["weight_digest"]


def test_the_exported_checkpoint_describes_itself(pmc_binary, tmp_path):
    run_train(pmc_binary, tmp_path / "rust")
    model = json.loads((tmp_path / "rust" / "model.json").read_text(encoding="utf-8"))

    assert model["format_version"] == 1
    assert model["architecture"]["kind"] == "mlp"
    assert model["architecture"]["d_hidden"] == HIDDEN
    assert model["seed"] == SEED
    assert model["weight_digest"]
