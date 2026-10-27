# SPDX-License-Identifier: GPL-3.0-only
"""Unit checks for the PMC-mini substrate swap.

The model-dependent arms are exercised by ``substrate_swap.py``; these tests
cover the frozen spec, the vector identities, the InfoNCE objective, the
adapter training on synthetic hidden states, and the pass/fail arithmetic.
They never load a model.
"""

from __future__ import annotations

import copy
import random
import unittest
from pathlib import Path

import substrate_swap as swap

V2_SPEC = Path(__file__).with_name("swap_words_v2.json")
V3_SPEC = Path(__file__).with_name("swap_words_v3.json")


def synthetic_hidden(spec: dict, host: str, seed: int) -> dict:
    rng = random.Random(seed)
    width = int(spec["projection"]["input_dim"][host])
    records = []
    for kind, index, _cue, _value, _prompt in swap.prompt_plan(spec):
        records.append(
            {
                "kind": kind,
                "index": index,
                "h": [rng.gauss(0.0, 1.0) for _ in range(width)],
            }
        )
    return {"host": host, "n_embd": width, "records": records}


def synthetic_result(spec: dict, *, pass_all: bool = True) -> dict:
    state = "a" * 64
    adapters = {"350m": "b" * 64, "8b": "c" * 64}
    counts = {
        "pmc": 6,
        "none": 0,
        "fresh": 1,
        "reinjection": 8,
        "control_false_positive": 0,
        "total": 8,
    }
    if not pass_all:
        counts = dict(counts)
        counts["pmc"] = 3
    host = {
        "state_sha256": state,
        "adapter_sha256": adapters["350m"],
        "counts": counts,
    }
    host_b = {
        "state_sha256": state,
        "adapter_sha256": adapters["8b"],
        "counts": counts,
    }
    return {
        "expected_state_sha256": state,
        "expected_adapters": adapters,
        "hosts": {"350m": host, "8b": host_b},
    }


class SpecTest(unittest.TestCase):
    def test_frozen_spec(self) -> None:
        spec = swap.load_spec()
        self.assertEqual(len(spec["training_facts"]), 22)
        self.assertEqual(len(spec["evaluation_facts"]), 8)
        self.assertEqual(len(spec["control_questions"]), 8)
        self.assertEqual(len(spec["words"]), 68)
        self.assertTrue(swap.no_target_training(spec))
        self.assertEqual(len(list(swap.prompt_plan(spec))), 68)
        self.assertGreater(spec["bias"], 0.0)

    def test_bad_split_is_rejected(self) -> None:
        spec = copy.deepcopy(swap.load_spec())
        spec["evaluation_facts"][0]["cue"] = spec["training_facts"][0]["cue"]
        with self.assertRaises(swap.SwapError):
            swap.validate_spec(spec)

    def test_v2_spec_still_validates(self) -> None:
        spec = swap.load_spec(V2_SPEC)
        self.assertEqual(len(spec["training_facts"]), 22)
        self.assertEqual(len(spec["words"]), 68)
        self.assertEqual(spec["anchor"], "cue")
        self.assertEqual(spec["adapter"]["output_dim"], 64)
        self.assertTrue(swap.no_target_training(spec))

    def test_v3_spec(self) -> None:
        spec = swap.load_spec(V3_SPEC)
        self.assertEqual(len(spec["training_facts"]), 30)
        self.assertEqual(len(spec["evaluation_facts"]), 8)
        self.assertEqual(len(spec["control_questions"]), 8)
        self.assertEqual(len(spec["words"]), 84)
        self.assertEqual(spec["anchor"], "cue_delta")
        self.assertEqual(spec["neutral_cue"], "thing")
        self.assertEqual(spec["adapter"]["output_dim"], 16)
        self.assertNotIn(spec["neutral_cue"], spec["words"])
        self.assertTrue(swap.no_target_training(spec))
        self.assertEqual(len(list(swap.prompt_plan(spec))), 84)

    def test_cue_and_neutral_prompt(self) -> None:
        prompt = "The code word for sun is salt."
        self.assertEqual(swap.cue_start(prompt, "sun"), 18)
        self.assertEqual(
            swap.neutral_prompt(prompt, "sun", "thing"),
            "The code word for thing is salt.",
        )
        with self.assertRaises(swap.SwapError):
            swap.cue_start("sun and sun", "sun")
        with self.assertRaises(swap.SwapError):
            swap.cue_start("no cue here", "sun")

    def test_prompts_are_identical_across_hosts(self) -> None:
        spec = swap.load_spec()
        fact = spec["training_facts"][0]
        prompt = swap.fact_prompt(spec, fact["cue"], fact["value"])
        self.assertIn(fact["cue"], prompt)
        self.assertIn(fact["value"], prompt)
        self.assertIn("{cue}", spec["templates"]["question"])


class VectorTest(unittest.TestCase):
    def test_projection_is_deterministic_and_normalized(self) -> None:
        spec = swap.load_spec()
        left = swap.host_projection(spec, "350m")
        right = swap.host_projection(spec, "350m")
        other = swap.host_projection(spec, "8b")
        self.assertEqual(left, right)
        self.assertNotEqual(len(left[0]), len(other[0]))
        hidden = [0.1] * int(spec["projection"]["input_dim"]["350m"])
        projected = swap.project_hidden(hidden, left)
        self.assertEqual(len(projected), 128)
        self.assertAlmostEqual(swap.dot(projected, projected), 1.0, places=12)

    def test_nearest_and_answer(self) -> None:
        keys = [[1.0, 0.0], [0.0, 1.0]]
        index, scores = swap.nearest([0.9, 0.1], keys)
        self.assertEqual(index, 0)
        self.assertGreater(scores[0], scores[1])
        self.assertTrue(swap.answer_ok("  River\n", "river"))
        self.assertFalse(swap.answer_ok("cloud", "river"))

    def test_digests_are_stable_and_sensitive(self) -> None:
        keys = [[1.0, 0.0], [0.0, 1.0]]
        values = [0.0, 1.0]
        first = swap.state_digest(keys, values)
        second = swap.state_digest(keys, values)
        changed = swap.state_digest(keys, [0.0, 2.0])
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        adapter = swap.adapter_digest([[1.0, 0.0]], [0.0])
        self.assertEqual(adapter, swap.adapter_digest([[1.0, 0.0]], [0.0]))
        self.assertNotEqual(adapter, swap.adapter_digest([[0.0, 1.0]], [0.0]))


class InfoNCETest(unittest.TestCase):
    def test_loss_and_gradient_shape(self) -> None:
        z = [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
        loss, gradient = swap.infonce_loss_and_vector_grad(z, [0, 0, 1, 1], 0.1)
        self.assertTrue(0.0 <= loss < 10.0)
        self.assertEqual(len(gradient), 4)
        self.assertEqual(len(gradient[0]), 2)
        self.assertTrue(all(isinstance(value, float) for value in gradient[0]))

    def test_validation_accuracy_with_identity_adapter(self) -> None:
        weights = [
            [1.0 if column == row else 0.0 for column in range(128)]
            for row in range(64)
        ]
        biases = [0.0] * 64
        views = []
        for fact in range(4):
            vector = [0.0] * 128
            vector[fact] = 1.0
            views.extend([vector, vector, vector, vector])
        accuracy = swap.validation_accuracy(
            (weights, weights),
            (biases, biases),
            views,
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            [0, 0, 1, 1] * 4,
            [0, 1, 2, 3],
        )
        self.assertEqual(accuracy, 1.0)


class TrainingTest(unittest.TestCase):
    def test_synthetic_training_produces_a_shared_state(self) -> None:
        spec = swap.load_spec()
        hidden_350m = synthetic_hidden(spec, "350m", 11)
        hidden_8b = synthetic_hidden(spec, "8b", 12)
        trained = swap.train_full_pair(
            spec,
            hidden_350m,
            hidden_8b,
            epochs=2,
            seed=7,
            progress=False,
        )
        self.assertEqual(len(trained["keys"]), 8)
        self.assertTrue(all(len(row) == 64 for row in trained["keys"]))
        self.assertEqual(trained["values"], [float(index) for index in range(8)])
        self.assertEqual(
            trained["state_sha256"], swap.state_digest(trained["keys"], trained["values"])
        )
        for host in ("350m", "8b"):
            self.assertEqual(
                trained["adapter_sha256"][host],
                swap.adapter_digest(trained["weights"][host], trained["biases"][host]),
            )
        payload = swap.adapters_payload(spec, trained)
        self.assertEqual(payload["state"]["state_sha256"], trained["state_sha256"])
        self.assertEqual(payload["state"]["value_words"], [fact["value"] for fact in spec["evaluation_facts"]])
        self.assertIn("best_validation_loss", payload["training"])
        self.assertEqual(payload["training"]["facts"], 22)

    def test_v3_training_produces_a_sixteen_dim_state(self) -> None:
        spec = swap.load_spec(V3_SPEC)
        hidden_350m = synthetic_hidden(spec, "350m", 21)
        hidden_8b = synthetic_hidden(spec, "8b", 22)
        trained = swap.train_full_pair(
            spec,
            hidden_350m,
            hidden_8b,
            epochs=2,
            seed=9,
            progress=False,
        )
        self.assertEqual(len(trained["keys"]), 8)
        self.assertTrue(all(len(row) == 16 for row in trained["keys"]))
        self.assertEqual(len(trained["weights"]["350m"]), 16)
        self.assertEqual(len(trained["weights"]["350m"][0]), 128)
        payload = swap.adapters_payload(spec, trained)
        self.assertEqual(payload["training"]["facts"], 30)
        self.assertEqual(
            payload["state"]["value_words"],
            [fact["value"] for fact in spec["evaluation_facts"]],
        )


class ResultTest(unittest.TestCase):
    def test_pass_and_counts(self) -> None:
        spec = swap.load_spec()
        summary = swap.evaluate_result(spec, synthetic_result(spec))
        self.assertTrue(summary["pass"])
        self.assertTrue(summary["checks"]["host_a_min"])
        self.assertTrue(summary["checks"]["host_b_margin"])
        failed = swap.evaluate_result(spec, synthetic_result(spec, pass_all=False))
        self.assertFalse(failed["pass"])

    def test_counts_from_rows(self) -> None:
        rows = []
        for arm in ("pmc", "none", "fresh", "reinjection"):
            for index in range(8):
                rows.append({"arm": arm, "correct": index == 0})
        for index in range(8):
            rows.append(
                {"arm": "control", "false_positive": index == 0}
            )
        counts = swap.counts_from_rows(rows)
        self.assertEqual(counts["pmc"], 1)
        self.assertEqual(counts["none"], 1)
        self.assertEqual(counts["fresh"], 1)
        self.assertEqual(counts["reinjection"], 1)
        self.assertEqual(counts["control_false_positive"], 1)
        self.assertEqual(counts["total"], 8)


if __name__ == "__main__":
    unittest.main()
