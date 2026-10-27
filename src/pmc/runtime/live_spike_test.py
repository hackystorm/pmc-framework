# SPDX-License-Identifier: GPL-3.0-only

"""Unit tests for the pure helpers in the WP7a live-runtime spike.

No model is loaded here. The ctypes ABI is exercised by the spike command
itself, against a real GGUF; these tests only pin down the arithmetic and the
JSON contract.
"""

from __future__ import annotations

import hashlib
import json
import struct
import unittest

from pmc.runtime.live_spike import (
    REQUIRED_SUMMARY_KEYS,
    argmax_with_bias,
    build_summary,
    count_nonzero,
    dot,
    fast_weight_update,
    finite_vector,
    is_informative,
    runner_up_token,
    sha256_prefix,
    softmax_probability,
    token_argmax,
    vector_norm,
)


class ArgmaxTests(unittest.TestCase):
    def test_argmax_returns_highest_index(self) -> None:
        self.assertEqual(token_argmax([0.1, 2.0, 1.9]), 1)

    def test_argmax_ties_go_to_the_lowest_index(self) -> None:
        self.assertEqual(token_argmax([1.0, 1.0, 0.5]), 0)

    def test_argmax_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            token_argmax([])

    def test_runner_up_excludes_top1(self) -> None:
        self.assertEqual(runner_up_token([0.1, 2.0, 1.9, 0.2], 1), 2)

    def test_runner_up_needs_two_values(self) -> None:
        with self.assertRaises(ValueError):
            runner_up_token([1.0], 0)

    def test_argmax_with_bias_changes_the_winner(self) -> None:
        self.assertEqual(argmax_with_bias([5.0, 4.5, 1.0], 1, 1.0), 1)

    def test_argmax_with_bias_keeps_the_winner_when_bias_is_small(self) -> None:
        self.assertEqual(argmax_with_bias([5.0, 4.5, 1.0], 1, 0.1), 0)

    def test_argmax_with_bias_rejects_bad_token(self) -> None:
        with self.assertRaises(ValueError):
            argmax_with_bias([1.0, 2.0], 2, 1.0)


class VectorTests(unittest.TestCase):
    def test_finite_vector_checks_length_and_finiteness(self) -> None:
        self.assertTrue(finite_vector([1.0, 2.0], 2))
        self.assertFalse(finite_vector([1.0, 2.0], 3))
        self.assertFalse(finite_vector([1.0, float("nan")], 2))
        self.assertFalse(finite_vector([1.0, float("inf")], 2))

    def test_count_nonzero(self) -> None:
        self.assertEqual(count_nonzero([0.0, 1.0, -0.0, 2.0]), 2)
        self.assertEqual(count_nonzero([0.0] * 8), 0)

    def test_zero_logits_are_not_informative(self) -> None:
        self.assertFalse(is_informative([0.0] * 65536))

    def test_varied_logits_are_informative(self) -> None:
        self.assertTrue(is_informative([0.0, 1.0, -1.0]))

    def test_constant_nonzero_logits_are_not_informative(self) -> None:
        self.assertFalse(is_informative([1.0, 1.0, 1.0]))

    def test_non_positive_logits_can_be_informative(self) -> None:
        self.assertTrue(is_informative([-2.0, -1.0, 0.0]))

    def test_non_finite_logits_are_not_informative(self) -> None:
        self.assertFalse(is_informative([0.0, float("inf")]))

    def test_norm(self) -> None:
        self.assertAlmostEqual(vector_norm([3.0, 4.0]), 5.0)

    def test_dot(self) -> None:
        self.assertAlmostEqual(dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0)

    def test_dot_rejects_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            dot([1.0], [1.0, 2.0])


class HashTests(unittest.TestCase):
    def test_hash_is_canonical_and_truncated(self) -> None:
        expected = hashlib.sha256(struct.pack("<2f", 1.0, 2.0)).hexdigest()[:12]
        self.assertEqual(sha256_prefix([1.0, 2.0]), expected)
        self.assertEqual(len(sha256_prefix([1.0, 2.0])), 12)

    def test_hash_changes_with_the_vector(self) -> None:
        self.assertNotEqual(sha256_prefix([1.0, 2.0]), sha256_prefix([1.0, 3.0]))

    def test_hash_rejects_bad_length(self) -> None:
        with self.assertRaises(ValueError):
            sha256_prefix([1.0], 0)


class SoftmaxTests(unittest.TestCase):
    def test_uniform_logits_give_uniform_probability(self) -> None:
        self.assertAlmostEqual(softmax_probability([0.0, 0.0, 0.0], 1), 1.0 / 3.0)

    def test_dominant_logit_gives_probability_near_one(self) -> None:
        self.assertAlmostEqual(softmax_probability([1000.0, 0.0], 0), 1.0)

    def test_probabilities_sum_to_one(self) -> None:
        logits = [0.5, -1.0, 2.0, 0.0]
        total = sum(softmax_probability(logits, index) for index in range(4))
        self.assertAlmostEqual(total, 1.0)

    def test_bad_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            softmax_probability([0.0, 1.0], 2)

    def test_non_finite_logits_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            softmax_probability([0.0, float("nan")], 0)


class FastWeightTests(unittest.TestCase):
    def test_one_gradient_step_increases_the_target_readout(self) -> None:
        hidden = [1.0, 2.0, 3.0]
        row = [0.0, 0.0, 0.0]
        updated = fast_weight_update(row, hidden, 0.25, 0.5)
        self.assertEqual(updated, [0.375, 0.75, 1.125])
        self.assertAlmostEqual(vector_norm(updated), 1.4031215200402267)
        self.assertAlmostEqual(dot(updated, hidden), 5.25)
        self.assertGreater(dot(updated, hidden), 0.0)

    def test_probability_one_makes_no_change(self) -> None:
        self.assertEqual(fast_weight_update([1.0, 2.0], [3.0, 4.0], 1.0, 0.5), [1.0, 2.0])

    def test_zero_learning_rate_makes_no_change(self) -> None:
        self.assertEqual(fast_weight_update([1.0, 2.0], [3.0, 4.0], 0.0, 0.0), [1.0, 2.0])

    def test_input_row_is_not_mutated(self) -> None:
        row = [0.0, 0.0]
        fast_weight_update(row, [1.0, 1.0], 0.0, 1.0)
        self.assertEqual(row, [0.0, 0.0])

    def test_bad_shapes_and_ranges_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fast_weight_update([0.0], [0.0, 1.0], 0.5, 0.1)
        with self.assertRaises(ValueError):
            fast_weight_update([0.0], [0.0], 1.5, 0.1)
        with self.assertRaises(ValueError):
            fast_weight_update([0.0], [0.0], 0.5, -0.1)


class SummaryTests(unittest.TestCase):
    def _required(self) -> dict[str, object]:
        return {
            "h_norm": 81.25730913593492,
            "h_sha256": "489c41836649aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "logits_top1_before": 2910,
            "logits_top1_after": 14933,
            "biased_token": 14933,
            "finite_h": 1,
            "finite_logits": 0,
            "decode_ms": 27.881234,
            "extract_ms": 1.234567,
        }

    def test_required_keys_are_present(self) -> None:
        summary = build_summary(self._required())
        for key in REQUIRED_SUMMARY_KEYS:
            self.assertIn(key, summary)

    def test_fields_are_normalized(self) -> None:
        summary = build_summary(self._required())
        self.assertEqual(summary["h_sha256"], "489c41836649")
        self.assertEqual(summary["h_norm"], 81.257309)
        self.assertEqual(summary["decode_ms"], 27.881234)
        self.assertIs(summary["finite_h"], True)
        self.assertIs(summary["finite_logits"], False)
        self.assertIsInstance(summary["logits_top1_before"], int)

    def test_extra_fields_are_preserved(self) -> None:
        values = self._required()
        values["mode"] = "paired"
        summary = build_summary(values)
        self.assertEqual(summary["mode"], "paired")

    def test_summary_is_strict_json(self) -> None:
        payload = json.dumps(build_summary(self._required()), allow_nan=False)
        self.assertIn('"h_sha256": "489c41836649"', payload)

    def test_missing_key_is_rejected(self) -> None:
        values = self._required()
        values.pop("h_norm")
        with self.assertRaises(KeyError):
            build_summary(values)


if __name__ == "__main__":
    unittest.main()
