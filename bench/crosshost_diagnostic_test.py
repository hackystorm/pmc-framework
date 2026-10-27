# SPDX-License-Identifier: GPL-3.0-only
"""Checks for the cross-host training-only diagnostic."""

from __future__ import annotations

import math
import random
import unittest

import crosshost_diagnostic as diagnostic


def _unit(rng: random.Random, width: int) -> list[float]:
    values = [rng.gauss(0.0, 1.0) for _ in range(width)]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


class RetrievalAccuracyTest(unittest.TestCase):
    def test_counts_hits_and_margin(self) -> None:
        keys = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        queries = [[0.9, 0.1], [0.1, 0.9], [0.0, 1.0]]
        correct, margin = diagnostic.retrieval_accuracy(keys, queries)
        self.assertEqual(correct, 2)
        self.assertAlmostEqual(margin, 0.2, places=6)

    def test_mismatched_counts_are_refused(self) -> None:
        with self.assertRaises(diagnostic.DiagnosticError):
            diagnostic.retrieval_accuracy([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])


class TrainingOnlyTest(unittest.TestCase):
    def test_only_training_kinds_are_allowed(self) -> None:
        self.assertEqual(
            set(diagnostic.TRAINING_KINDS), {"training_fact", "training_question"}
        )
        for kind in ("eval_fact", "eval_question", "control_question"):
            self.assertNotIn(kind, diagnostic.TRAINING_KINDS)

    def test_unknown_record_kind_is_refused(self) -> None:
        payload = {"records": [{"kind": "mystery_record", "index": 0}]}
        with self.assertRaises(diagnostic.DiagnosticError):
            diagnostic.load_training_views({}, payload, payload)


class TrainDimensionTest(unittest.TestCase):
    def test_synthetic_views_train_and_report(self) -> None:
        rng = random.Random(7)
        facts = 4
        width = 8
        views = {
            name: [_unit(rng, width) for _ in range(facts)]
            for name in ("fact_a", "question_a", "fact_b", "question_b")
        }
        spec = {
            "training_facts": [{"cue": str(index)} for index in range(facts)],
            "adapter": {
                "output_dim": 4,
                "temperature": 0.1,
                "epochs": 5,
                "learning_rate": 0.01,
                "weight_decay": 0.0001,
            },
            "projection": {"output_dim": width},
        }
        result = diagnostic.train_dimension(
            spec, views, 4, epochs=5, seed=11
        )
        self.assertEqual(result["state_dim"], 4)
        self.assertEqual(result["epochs"], 5)
        self.assertIn("cross_host", result)
        self.assertEqual(result["b_question_from_a_keys"]["total"], facts)
        self.assertTrue(math.isfinite(result["final_loss"]))


if __name__ == "__main__":
    unittest.main()
