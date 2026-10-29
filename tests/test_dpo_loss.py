# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the DPO loss reference implementation."""

from __future__ import annotations

import math

from pmc.reference import dpo_loss, f32


def test_dpo_loss_zero_at_equality():
    loss = dpo_loss(0.0, 0.0, 0.0, 0.0, 0.1)
    assert abs(loss - 0.693147) < 1e-5


def test_dpo_loss_falls_when_chosen_preferred():
    loss = dpo_loss(10.0, 0.0, 0.0, 0.0, 1.0)
    assert loss < 1e-4


def test_dpo_loss_rises_when_rejected_preferred():
    loss = dpo_loss(0.0, 10.0, 0.0, 0.0, 1.0)
    assert loss > 9.0


def test_dpo_loss_matches_rust_stable_form():
    # Cross-check the softplus identity used by both implementations.
    for x in (-5.0, -1.0, 0.0, 1.0, 5.0):
        expected = math.log(1.0 + math.exp(-x)) if -20.0 < x < 20.0 else (0.0 if x >= 20.0 else -x)
        # delta = x / beta with beta=1, ref terms zero
        got = dpo_loss(x, 0.0, 0.0, 0.0, 1.0)
        assert abs(got - expected) < 1e-5, (x, got, expected)