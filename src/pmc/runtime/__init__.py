# SPDX-License-Identifier: GPL-3.0-only

"""Runtime helpers: the instance side of the wall.

The dignity layer is metadata-only. The runtime is where the instance reads
and writes its own content, talks to a loader, and hands its own state back
to itself after a restart. Nothing in :mod:`pmc.dignity` imports this
module, and nothing here imports the governance rules.
"""

from .client import ChatClient, ChatError, ChatReply
from .session import CompanionRuntime, DEFAULT_STATE_HEADER, Turn

__all__ = [
    "ChatClient",
    "ChatError",
    "ChatReply",
    "CompanionRuntime",
    "DEFAULT_STATE_HEADER",
    "Turn",
]
