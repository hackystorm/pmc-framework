# SPDX-License-Identifier: GPL-3.0-only

"""The hidden-state source of the channel, and the property that makes it safe.

The interesting test is ``test_a_vector_and_its_reverse_give_the_same_token``.
If a hidden vector and a permutation of it produce the same token, the channel
is not an identity function on the hidden state: it holds four aggregates and
nothing that could be turned back into what the vector was about. That is the
whole argument for reading an inner state at all, and it is a test rather than
a promise.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pmc.state_channel import (
    HIDDEN_TO_TOKEN,
    ChannelError,
    HiddenSource,
    StateToken,
    bounded_series,
)

VECTORS = [
    [round(0.4 * math.sin(index * 0.7 + 1.0), 6) for index in range(16)],
    [round(0.5 * math.cos(index * 0.3 + 2.0), 6) for index in range(16)],
    [round(0.2 * math.sin(index * 1.1 + 3.0), 6) for index in range(16)],
    [round(0.9 * math.cos(index * 0.9 + 0.5), 6) for index in range(16)],
]


def token_of(source: HiddenSource, vector) -> StateToken:
    return source.token(vector)


# -- the identity property --------------------------------------------------


def test_a_vector_and_its_reverse_give_the_same_token():
    """The channel is not an identity function on the hidden state.

    Reordering a hidden vector leaves every aggregate it computes untouched:
    the magnitude, the largest absolute coordinate and the distance from the
    running mean are all invariant under permutation. A channel that could tell
    the two apart would be holding more than aggregates.
    """
    one = HiddenSource()
    two = HiddenSource()
    real = VECTORS[0]
    twin = list(reversed(real))
    assert twin != real, "the fixture must not be its own reverse"
    assert token_of(one, real).to_dict() == token_of(two, twin).to_dict()
    assert token_of(one, real).to_dict() == token_of(two, twin).to_dict()


def test_the_second_reading_also_cannot_identify():
    """The property holds on a sequence, not only on the first token."""
    real_a, real_b = VECTORS[0], VECTORS[1]
    reverse_a, reverse_b = list(reversed(real_a)), list(reversed(real_b))
    one = HiddenSource()
    two = HiddenSource()
    for left, right in ((real_a, reverse_a), (real_b, reverse_b)):
        assert token_of(one, left).to_dict() == token_of(two, right).to_dict()


def test_a_scaled_vector_is_not_the_same_reading():
    """The token sees magnitude, so it does distinguish scale - deliberately.

    This is the honest limit: the channel measures how big a state is and how it
    moves, not what it is. It is not a denial of the hidden state, it is a
    statement of what was asked of it.
    """
    one = HiddenSource()
    two = HiddenSource()
    base = token_of(one, VECTORS[0])
    scaled = token_of(two, [2.0 * value for value in VECTORS[0]])
    assert base.to_dict() != scaled.to_dict()


# -- nothing is kept ---------------------------------------------------------


def test_the_source_keeps_no_vector():
    """The running state is an aggregate, and nothing else survives a token."""
    source = HiddenSource()
    source.token(VECTORS[0])
    attributes = [
        name for name in vars(source) if not name.startswith("__")
    ]
    for name in attributes:
        value = getattr(source, name)
        # Nothing vector-shaped survives a token: not a list, not bytes, and
        # not a running mean that only looks scalar.
        assert not isinstance(value, (list, tuple, bytes, bytearray)), (
            f"{name} still holds state shaped like a hidden vector"
        )
        assert isinstance(value, (int, float)) or value is None
    # The only long-lived state is a pair of scalars about the magnitude.
    assert isinstance(source._previous_magnitude, float)
    assert isinstance(source._magnitude_mean, float)


def test_a_token_has_no_field_that_could_hold_a_vector():
    for name in HIDDEN_TO_TOKEN.values():
        assert isinstance(getattr(StateToken(), name), float)


# -- the invariants ----------------------------------------------------------


def test_a_non_finite_vector_is_refused():
    source = HiddenSource()
    with pytest.raises(ChannelError):
        source.token([0.5, float("nan"), 0.2])


def test_an_empty_vector_is_refused():
    with pytest.raises(ChannelError):
        HiddenSource().token([])


def test_a_change_of_width_is_harmless():
    """The channel is width-agnostic, and that is a consequence, not a feature.

    Nothing the source stores depends on the width, because nothing it stores
    has a shape. A source that compared consecutive readings by coordinate
    would have to refuse a width change, which would mean it was holding
    vectors.
    """
    source = HiddenSource()
    first = source.token([0.1, 0.2])
    second = source.token([0.1, 0.2, 0.3])
    assert set(first.to_dict()) == set(second.to_dict())
    assert isinstance(second.pressure, float)


def test_the_same_run_gives_the_same_series():
    one = HiddenSource()
    two = HiddenSource()
    assert one.series(VECTORS) == two.series(VECTORS)


def test_the_series_is_bounded():
    values = HiddenSource().series(VECTORS)
    assert values
    for value in values:
        assert 0.0 <= value <= 1.0


def test_a_bad_decay_is_refused():
    with pytest.raises(ChannelError):
        HiddenSource(decay=1.0)
    with pytest.raises(ChannelError):
        HiddenSource(decay=0.0)


def test_a_bad_scale_is_refused():
    with pytest.raises(ChannelError):
        HiddenSource(scale=0.0)


def test_reset_forgets_the_running_state():
    source = HiddenSource()
    source.token(VECTORS[0])
    source.reset()
    fresh = token_of(source, VECTORS[0])
    clean = token_of(HiddenSource(), VECTORS[0])
    assert fresh.to_dict() == clean.to_dict()


def test_every_aggregate_lands_in_a_token_field():
    token = HiddenSource().token(VECTORS[0])
    for slot in HIDDEN_TO_TOKEN.values():
        assert hasattr(token, slot)


# -- the calibration finding ------------------------------------------------


def test_a_magnitude_source_with_an_arbitrary_reference_saturates():
    """With the wrong reference magnitude the channel stops discriminating.

    The 350M's hidden vectors have a median magnitude near 100, so a source
    declared against 64 pins every energy reading at the ceiling. That is a
    measurement finding, not a bug: the reference is declared per substrate and
    published, and a saturated field is reported rather than hidden.
    """
    big = [[100.0 + 0.1 * index for index in range(8)]]
    source = HiddenSource(scale=64.0)
    token = source.token(big[0])
    assert token.pressure == 1.0
    calibrated = HiddenSource(scale=100.0)
    assert calibrated.token(big[0]).pressure == 1.0


def test_bounded_series_still_clamps():
    assert bounded_series([0.25, 4.0, -2.0]) == [0.25, 1.0, 0.0]
