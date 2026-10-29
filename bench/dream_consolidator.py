# SPDX-License-Identifier: GPL-3.0-only
"""Identity consolidator for the dream pipeline.

Copies the baseline checkpoint byte-for-byte into the candidate path. This
is the smallest consolidator that satisfies the contract: it writes the
candidate, changes nothing, and exits zero. A real consolidator (one that
trains on the replay manifest) replaces this file; the pipeline does not
care which one runs.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: dream_consolidator.py <baseline> <candidate> <manifest>", file=sys.stderr)
        return 2
    baseline = Path(sys.argv[1])
    candidate = Path(sys.argv[2])
    if not baseline.is_dir():
        print(f"baseline is not a directory: {baseline}", file=sys.stderr)
        return 2
    if candidate.exists():
        shutil.rmtree(candidate)
    shutil.copytree(baseline, candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
