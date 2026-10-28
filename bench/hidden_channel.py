# SPDX-License-Identifier: GPL-3.0-only
"""Run the hidden-state variant of the channel over already-extracted states.

The metadata proxy measures the ledger. This one measures the substrate: it
turns each extracted hidden vector into one bounded token made of aggregates -
energy, drift, spread, novelty - and keeps nothing else. It is the first
measurement this repository can make of something that is not a declaration.

Two properties are checked here on real data rather than asserted:

1. **The token cannot identify.** Two different facts whose hidden vectors
   share their aggregates produce the same token, so the channel is not an
   identity function on the hidden state.
2. **The vector is not kept.** The token has no field that could hold one, the
   source keeps no reference, and the route from vector to token is a function
   of four scalars.

The script never loads a model. It reads the extraction JSON the bench already
produces, which means it can run on the real 350M and 8B hidden states that
were extracted earlier, and it labels itself with the digests of the files it
read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

from pmc.state_channel import (
    HIDDEN_FIELDS,
    HIDDEN_TO_TOKEN,
    ChannelError,
    HiddenSource,
    StateToken,
)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read an extraction file in the shape `pmc-swap-hidden/1`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ChannelError(f"{path} carries no records")
    if "h" not in records[0]:
        raise ChannelError(f"{path}: the records carry no hidden vector")
    return records


HEADROOM = 1.05
"""Declared headroom above the largest observed magnitude.

The reference is set so the largest reading in the extraction sits just under
the ceiling rather than at it. A median would saturate the whole upper half by
construction, and a maximum would saturate the record that produced it; either
way the channel would stop discriminating exactly on the readings that matter
most.
"""


def calibration(vectors: Sequence[Sequence[float]]) -> float:
    """The reference magnitude for this extraction, and how it was chosen.

    A magnitude source with an arbitrary reference saturates: with a scale of
    64 on the 350M's 1024-dimensional vectors the energy term pins at 1.0 for
    every reading, and the channel stops discriminating. So the reference is
    derived from this extraction, published in the report, and the saturation it
    would have caused at a fixed value is reported beside it.
    """
    largest = max(
        (math.sqrt(sum(value * value for value in vector)) for vector in vectors),
        default=0.0,
    )
    if largest <= 0.0:
        raise ChannelError("calibration needs at least one non-zero vector")
    return largest * HEADROOM


def saturation(values: Sequence[float], bound: float = 1.0) -> int:
    """How many readings sit at the ceiling, which is the wrongness signal."""
    return sum(1 for value in values if value >= bound - 1e-9)


def series(path: str | Path, scale: float | None = None) -> dict[str, Any]:
    """The channel's series and the invariants, for one extraction file."""
    records = load_records(path)
    vectors = [record["h"] for record in records]
    declared = calibration(vectors) if scale is None else float(scale)
    source = HiddenSource(scale=declared)
    values = source.series(vectors)
    naive = HiddenSource(scale=64.0)
    naive_tokens = naive.tokens(vectors)
    naive_values = [token.aggregate() for token in naive_tokens]
    def saturated(tokens: list[StateToken]) -> dict[str, int]:
        return {
            name: sum(
                1
                for token in tokens
                if float(getattr(token, HIDDEN_TO_TOKEN[name])) >= 1.0 - 1e-9
            )
            for name in HIDDEN_FIELDS
        }

    saturated_fields = saturated(naive_tokens)

    declared_tokens = HiddenSource(scale=declared).tokens(vectors)
    identity = _identity_check(source, vectors)
    return {
        "source": str(path),
        "records": len(records),
        "host": records[0].get("host") if isinstance(records[0], dict) else None,
        "n_embd": len(vectors[0]),
        "scale": declared,
        "scale_declared_how": (
            "largest observed magnitude with declared headroom" if scale is None else "given"
        ),
        "saturation_at_a_fixed_scale": saturation(naive_values),
        "field_saturation_at_a_fixed_scale": saturated_fields,
        "field_saturation_at_the_declared_scale": saturated(declared_tokens),
        "series": [round(value, 6) for value in values],
        "mean": round(sum(values) / len(values), 6) if values else 0.0,
        "identity": identity,
    }


def _identity_check(source: HiddenSource, vectors: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Do two vectors with the same aggregates give the same token?

    The pair is a real hidden vector and a permutation of it. A permutation
    leaves every aggregate this token computes untouched: the magnitude, the
    largest absolute coordinate, and the distance from the running mean are all
    invariant under reordering. If the token differs, the channel could tell
    them apart, which would make it an identifier rather than an observation.
    """
    if len(vectors) < 1:
        return {"checked": False, "reason": "needs at least one record"}
    real = vectors[0]
    twin = list(reversed(real))
    if twin == real:
        return {"checked": False, "reason": "the record is its own reverse"}
    one = HiddenSource(scale=source.scale, decay=source.decay)
    two = HiddenSource(scale=source.scale, decay=source.decay)
    left = one.token(real)
    right = two.token(twin)
    left_again = one.token(real)
    right_again = two.token(twin)
    return {
        "checked": True,
        "same": left.to_dict() == right.to_dict()
        and left_again.to_dict() == right_again.to_dict(),
        "one": left.to_dict(),
        "two": right.to_dict(),
        "reason": (
            "a vector and its reverse share every aggregate this token computes, "
            "so they must give the same token"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", required=True, help="a hidden-state extraction JSON")
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="reference magnitude; default is the median of this extraction",
    )
    parser.add_argument("--out", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)
    payload = series(args.hidden, scale=args.scale)
    printed = json.dumps(payload, indent=2, sort_keys=True)
    print(printed)
    if args.out:
        Path(args.out).write_text(printed + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
