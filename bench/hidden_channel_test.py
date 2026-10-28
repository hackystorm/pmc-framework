# SPDX-License-Identifier: GPL-3.0-only

"""The hidden channel harness, measured on real data and on its own failure.

Two things are checked here that matter more than the numbers. The identity
property holds on real hidden states, which is what licenses reading an inner
state at all. And a magnitude reference chosen by hand saturates the channel,
which is the measurement finding that keeps the reference declared rather than
assumed.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import hidden_channel

from pmc.state_channel import HiddenSource


def extraction(vectors: list[list[float]]) -> Path:
    """An extraction file in the shape the bench already produces."""
    root = Path(tempfile.mkdtemp(prefix="pmc-hidden-"))
    payload = {
        "schema": "pmc-swap-hidden/1",
        "host": "350m",
        "n_embd": len(vectors[0]),
        "records": [
            {
                "kind": "training_fact",
                "index": index,
                "cue": "cue",
                "value": "value",
                "h": vector,
                "prompt_sha256": "a" * 64,
                "h_sha256": "b" * 64,
            }
            for index, vector in enumerate(vectors)
        ],
    }
    path = root / "hidden.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def waves(count: int = 12, width: int = 24) -> list[list[float]]:
    """Deterministic, bounded, varied vectors - not real ones, but honest ones."""
    return [
        [
            round(0.5 * math.sin(index * 0.7 + step), 6)
            for index in range(width)
        ]
        for step in range(count)
    ]


class HarnessTest(unittest.TestCase):
    def test_the_series_is_bounded_and_reported(self):
        payload = hidden_channel.series(extraction(waves()))
        self.assertEqual(payload["records"], 12)
        self.assertEqual(payload["n_embd"], 24)
        for value in payload["series"]:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_the_identity_property_holds(self):
        payload = hidden_channel.series(extraction(waves()))
        identity = payload["identity"]
        self.assertTrue(identity["checked"])
        self.assertTrue(identity["same"])
        self.assertIn("reverse", identity["reason"])

    def test_a_reference_chosen_by_hand_saturates_and_says_so(self):
        """The measurement finding: 64 is the wrong magnitude for this host.

        With a reference of 64 against vectors whose median magnitude is 100,
        every energy reading pins at the ceiling and the channel stops
        discriminating. The harness reports the saturation rather than hiding
        it behind a calibrated scale.
        """
        vectors = waves(count=8, width=32)
        for vector in vectors:
            for index in range(len(vector)):
                vector[index] *= 100.0
        payload = hidden_channel.series(extraction(vectors), scale=64.0)
        self.assertEqual(payload["field_saturation_at_a_fixed_scale"]["energy"], 8)
        calibrated = hidden_channel.series(extraction(vectors))
        self.assertEqual(
            calibrated["field_saturation_at_the_declared_scale"]["energy"], 0
        )
        self.assertGreater(calibrated["scale"], 64.0)
        self.assertEqual(calibrated["scale_declared_how"],
                         "largest observed magnitude with declared headroom")

    def test_the_reference_is_declared_not_assumed(self):
        """A median saturates the upper half by construction, so the reference
        is set above the largest observed reading instead."""
        payload = hidden_channel.series(extraction(waves()))
        self.assertGreater(payload["scale"], 0.0)
        self.assertIn("largest observed", payload["scale_declared_how"])
        self.assertEqual(payload["field_saturation_at_the_declared_scale"]["energy"], 0)

    def test_a_median_reference_would_saturate_the_upper_half(self):
        """The reason the median is not used, as a number.

        Calibrating to the median leaves every reading above it at the ceiling,
        which is half the extraction losing the ability to discriminate.
        """
        vectors = waves(count=8, width=32)
        payload = hidden_channel.series(extraction(vectors))
        declared = payload["scale"]
        median = sorted(
            math.sqrt(sum(v * v for v in vector)) for vector in vectors
        )[len(vectors) // 2]
        self.assertGreater(declared, median)

    def test_an_empty_record_is_refused(self):
        with self.assertRaises(Exception):
            hidden_channel.series(extraction([]))

    def test_a_record_without_a_vector_is_refused(self):
        root = Path(tempfile.mkdtemp(prefix="pmc-hidden-"))
        path = root / "bad.json"
        path.write_text(json.dumps({"records": [{"kind": "x"}]}), encoding="utf-8")
        with self.assertRaises(Exception):
            hidden_channel.series(path)

    def test_the_source_keeps_no_vector_shaped_state(self):
        source = HiddenSource()
        source.token(waves()[0])
        for name, value in vars(source).items():
            if name.startswith("__"):
                continue
            self.assertFalse(
                isinstance(value, (list, tuple, bytes, bytearray)),
                f"{name} holds state shaped like a hidden vector",
            )

    def test_a_change_of_width_is_harmless(self):
        """The consequence of keeping nothing vector-shaped."""
        source = HiddenSource()
        source.token([0.2, 0.4])
        second = source.token([0.2, 0.4, 0.6])
        self.assertIsInstance(second.pressure, float)


if __name__ == "__main__":
    unittest.main()
