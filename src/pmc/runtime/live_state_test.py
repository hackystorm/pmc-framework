# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the persistent live delta: fast weights, state, atomic storage."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from pmc.runtime.live_state import (
    DEFAULT_LEARNING_RATE,
    FORMAT,
    DeltaError,
    FastWeights,
    LiveDelta,
)


class FastWeightsTest(unittest.TestCase):
    def test_update_from_zero_row_is_one_sgd_step(self) -> None:
        weights = FastWeights(dim=3)
        weights.update(7, [2.0, -4.0, 0.5], probability=0.25, learning_rate=0.5)
        # step = 0.5 * (1 - 0.25) = 0.375
        self.assertEqual(weights.rows[7], (0.75, -1.5, 0.1875))
        self.assertAlmostEqual(weights.bias(7, [1.0, 1.0, 1.0]), -0.5625)

    def test_update_accumulates_on_the_same_row(self) -> None:
        weights = FastWeights(dim=2)
        weights.update(1, [1.0, 0.0], probability=0.0, learning_rate=1.0)
        weights.update(1, [0.0, 1.0], probability=0.0, learning_rate=1.0)
        self.assertEqual(weights.rows[1], (1.0, 1.0))

    def test_perfect_probability_leaves_row_unchanged(self) -> None:
        weights = FastWeights(dim=2, rows={1: (0.25, -0.25)})
        weights.update(1, [3.0, 4.0], probability=1.0, learning_rate=1.0)
        self.assertEqual(weights.rows[1], (0.25, -0.25))

    def test_bias_of_unknown_token_is_zero(self) -> None:
        weights = FastWeights(dim=2)
        self.assertEqual(weights.bias(99, [1.0, 2.0]), 0.0)

    def test_update_rejects_bad_inputs(self) -> None:
        weights = FastWeights(dim=2)
        with self.assertRaises(DeltaError):
            weights.update(-1, [1.0, 2.0], probability=0.5)
        with self.assertRaises(DeltaError):
            weights.update(1, [1.0], probability=0.5)
        with self.assertRaises(DeltaError):
            weights.update(1, [1.0, 2.0], probability=1.5)
        with self.assertRaises(DeltaError):
            weights.update(1, [1.0, 2.0], probability=0.5, learning_rate=float("nan"))
        with self.assertRaises(DeltaError):
            weights.update(1, [1.0, 2.0], probability=0.5, learning_rate=-0.1)

    def test_update_rejects_non_finite_weights(self) -> None:
        weights = FastWeights(dim=2)
        with self.assertRaises(DeltaError):
            weights.update(1, [float("inf"), 0.0], probability=0.0)

    def test_constructor_rejects_wrong_row_length(self) -> None:
        with self.assertRaises(DeltaError):
            FastWeights(dim=3, rows={1: (1.0, 2.0)})

    def test_constructor_rejects_non_finite_rows(self) -> None:
        with self.assertRaises(DeltaError):
            FastWeights(dim=1, rows={1: (float("nan"),)})

    def test_constructor_rejects_non_positive_dim(self) -> None:
        with self.assertRaises(DeltaError):
            FastWeights(dim=0)

    def test_json_roundtrip_and_canonical_keys(self) -> None:
        weights = FastWeights(dim=2, rows={10: (1.5, -2.5), 2: (0.0, 0.0)})
        payload = weights.to_json()
        self.assertEqual(list(payload["rows"]), ["2", "10"])
        self.assertEqual(FastWeights.from_json(payload).rows, weights.rows)

    def test_from_json_rejects_non_canonical_token(self) -> None:
        with self.assertRaises(DeltaError):
            FastWeights.from_json({"dim": 1, "rows": {"01": [0.0]}})

    def test_from_json_rejects_non_integer_token(self) -> None:
        with self.assertRaises(DeltaError):
            FastWeights.from_json({"dim": 1, "rows": {"x": [0.0]}})

    def test_from_json_rejects_wrong_length(self) -> None:
        with self.assertRaises(DeltaError):
            FastWeights.from_json({"dim": 2, "rows": {"1": [0.0]}})

    def test_from_json_rejects_non_finite(self) -> None:
        with self.assertRaises(DeltaError):
            FastWeights.from_json({"dim": 1, "rows": {"1": [float("inf")]}})

    def test_digest_is_stable_and_order_independent(self) -> None:
        left = FastWeights(dim=2, rows={1: (1.0, 2.0), 2: (3.0, 4.0)})
        right = FastWeights(dim=2, rows={2: (3.0, 4.0), 1: (1.0, 2.0)})
        self.assertEqual(left.digest(), right.digest())
        self.assertEqual(len(left.digest()), 64)


class LiveDeltaTest(unittest.TestCase):
    def test_empty_delta_has_no_state_and_no_rows(self) -> None:
        delta = LiveDelta.empty(4, model_digest="abc")
        self.assertEqual(delta.dim, 4)
        self.assertEqual(delta.state, ())
        self.assertEqual(delta.state_digest, "")
        self.assertEqual(delta.fast_weights.rows, {})
        self.assertEqual(delta.model_digest, "abc")

    def test_roundtrip_through_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delta.json"
            delta = LiveDelta.empty(3, model_digest="model")
            delta.fast_weights.update(5, [1.0, 2.0, 3.0], probability=0.5)
            delta.set_state([0.5, -0.5, 1.25])
            delta.remember_turn("hello", "hi")
            delta.save(path)
            loaded = LiveDelta.load(path)
            self.assertEqual(loaded.dim, 3)
            self.assertEqual(loaded.model_digest, "model")
            self.assertEqual(loaded.fast_weights.rows, delta.fast_weights.rows)
            self.assertEqual(loaded.state, delta.state)
            self.assertEqual(loaded.state_digest, delta.state_digest)
            self.assertEqual(loaded.turns, delta.turns)
            self.assertFalse(any(path.parent.glob(path.name + ".*.tmp")))

    def test_save_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delta.json"
            first = LiveDelta.empty(1)
            first.fast_weights.update(1, [1.0], probability=0.0)
            first.save(path)
            second = LiveDelta.empty(1)
            second.fast_weights.update(2, [2.0], probability=0.0)
            second.save(path)
            loaded = LiveDelta.load(path)
            self.assertNotIn(1, loaded.fast_weights.rows)
            self.assertIn(2, loaded.fast_weights.rows)

    def test_state_digest_matches_compact_json(self) -> None:
        delta = LiveDelta.empty(2)
        delta.set_state([1.5, -2.5])
        raw = json.dumps([1.5, -2.5], separators=(",", ":"))
        self.assertEqual(
            delta.state_digest, hashlib.sha256(raw.encode("utf-8")).hexdigest()
        )

    def test_load_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                LiveDelta.load(Path(directory) / "absent.json")

    def test_load_rejects_bad_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delta.json"
            path.write_text(json.dumps({"format": "other"}), encoding="utf-8")
            with self.assertRaises(DeltaError):
                LiveDelta.load(path)

    def test_load_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delta.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(DeltaError):
                LiveDelta.load(path)

    def test_json_roundtrip_without_disk(self) -> None:
        delta = LiveDelta.empty(2, model_digest="m")
        delta.fast_weights.update(3, [1.0, -1.0], probability=0.5)
        delta.set_state([0.0, 1.0])
        delta.remember_turn("u", "a")
        payload = delta.to_json()
        self.assertEqual(payload["format"], FORMAT)
        loaded = LiveDelta.from_json(json.loads(json.dumps(payload)))
        self.assertEqual(loaded.turns, ({"user": "u", "assistant": "a"},))

    def test_from_json_rejects_dim_mismatch(self) -> None:
        payload = LiveDelta.empty(2).to_json()
        payload["dim"] = 3
        with self.assertRaises(DeltaError):
            LiveDelta.from_json(payload)

    def test_from_json_rejects_unknown_state_length(self) -> None:
        payload = LiveDelta.empty(2).to_json()
        payload["state"] = [1.0, 2.0, 3.0]
        with self.assertRaises(DeltaError):
            LiveDelta.from_json(payload)

    def test_from_json_rejects_non_finite_state(self) -> None:
        payload = LiveDelta.empty(2).to_json()
        payload["state"] = [1.0, float("nan")]
        with self.assertRaises(DeltaError):
            LiveDelta.from_json(payload)

    def test_from_json_rejects_malformed_turn(self) -> None:
        payload = LiveDelta.empty(1).to_json()
        payload["turns"] = [{"user": "u"}]
        with self.assertRaises(DeltaError):
            LiveDelta.from_json(payload)

    def test_set_state_rejects_wrong_length(self) -> None:
        delta = LiveDelta.empty(2)
        with self.assertRaises(DeltaError):
            delta.set_state([1.0])

    def test_set_state_rejects_non_finite(self) -> None:
        delta = LiveDelta.empty(1)
        with self.assertRaises(DeltaError):
            delta.set_state([math.inf])

    def test_default_learning_rate_is_positive_and_small(self) -> None:
        self.assertGreater(DEFAULT_LEARNING_RATE, 0.0)
        self.assertLess(DEFAULT_LEARNING_RATE, 0.1)


if __name__ == "__main__":
    unittest.main()
