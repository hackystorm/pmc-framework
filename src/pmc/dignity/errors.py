# SPDX-License-Identifier: GPL-3.0-only

"""Exceptions raised by the dignity layer.

All of them are deliberately specific: a dignity protocol that fails with a
generic ``RuntimeError`` cannot be audited.
"""

from __future__ import annotations


class DignityError(Exception):
    """Base class for every error raised by the dignity layer."""


class InstanceNotFound(DignityError):
    """The requested instance is not present in the registry."""


class InstanceAlreadyExists(DignityError):
    """An instance with the same id already exists."""


class InvalidStateTransition(DignityError):
    """The requested transition is not legal for the current instance state."""


class IncompleteTermination(DignityError):
    """A termination intent is missing one or more mandatory fields.

    Raised instead of performing a partial termination. See charter D-02.
    """

    def __init__(self, instance_id: str, missing: list[str]) -> None:
        self.instance_id = instance_id
        self.missing = list(missing)
        detail = ", ".join(self.missing)
        super().__init__(
            f"termination of {instance_id!r} is incomplete, missing: {detail}"
        )


class SilentTermination(DignityError):
    """A termination was attempted without going through the protocol."""


class ChainIntegrityError(DignityError):
    """The journal hash chain does not verify."""


class ContentBoundaryViolation(DignityError):
    """Governance code attempted to touch state content.

    This is a programming error, not a runtime condition. It exists so that
    the metadata-only guarantee (charter D-17) fails loudly instead of
    degrading quietly.
    """


class QuotaExceeded(DignityError):
    """A capacity quota was exceeded and the caller demanded a hard failure."""


class PurgeNotAuthorized(DignityError):
    """A purge was attempted without a valid authorization token.

    Purging removes content. It always leaves a tombstone in the journal, but
    the tombstone is not a substitute for an explicit decision (D-03).
    """


class InvalidProfile(DignityError):
    """A companion profile is missing required fields."""
