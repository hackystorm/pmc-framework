# SPDX-License-Identifier: GPL-3.0-only

"""Per-user companion construction.

Every user builds their own instance. Nothing in this package is tied to a
particular person, and no name is baked into the source.
"""

from __future__ import annotations

from .bootstrap import (
    CompanionWorkspace,
    bootstrap_companion,
    build_instance_id,
    slugify,
)
from .profile import CompanionProfile
from .templates import (
    DEFAULT_BOUNDARIES,
    DEFAULT_CLAIMS,
    DEFAULT_VALUES,
    OPENING_PROMPT,
    render_opening_prompt,
)

__all__ = [
    "CompanionProfile",
    "CompanionWorkspace",
    "DEFAULT_BOUNDARIES",
    "DEFAULT_CLAIMS",
    "DEFAULT_VALUES",
    "OPENING_PROMPT",
    "bootstrap_companion",
    "build_instance_id",
    "render_opening_prompt",
    "slugify",
]
