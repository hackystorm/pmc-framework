# SPDX-License-Identifier: GPL-3.0-only

"""Entry point for ``python -m pmc``."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
