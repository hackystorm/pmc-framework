# SPDX-License-Identifier: GPL-3.0-only

"""Metadata-only replay selection.

The dream pipeline never reads a memory's content. A replay item carries the
record id, the content digest, the ledger metadata and a salience score. A
consolidator that needs the bytes fetches them from the instance's private
store by record id; the framework does not look inside them.

Salience is a deterministic function of ledger metadata only: recency, the
self-reported load, the access count and the digestion state. No topic, label
or hidden state is interpreted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import log1p
from typing import Any, Callable, Iterable

from ..dignity.records import DigestState, MemoryRecord, sha256_of


class ReplayError(RuntimeError):
    """The replay selection is malformed."""


@dataclass(frozen=True)
class SalienceConfig:
    """Weights and scales for the metadata-only salience score."""

    half_life_days: float = 7.0
    recency_weight: float = 0.5
    load_weight: float = 0.3
    access_weight: float = 0.2
    access_saturation: float = 10.0
    raw_boost: float = 1.0
    digesting_boost: float = 0.95
    digested_boost: float = 0.9

    def validate(self) -> None:
        if self.half_life_days <= 0.0:
            raise ReplayError("half_life_days must be positive")
        weights = (self.recency_weight, self.load_weight, self.access_weight)
        if any(weight < 0.0 for weight in weights):
            raise ReplayError("salience weights must be non-negative")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ReplayError("salience weights must sum to 1")
        if self.access_saturation <= 1.0:
            raise ReplayError("access_saturation must be greater than 1")
        for name in ("raw_boost", "digesting_boost", "digested_boost"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ReplayError(f"{name} must be within [0, 1]")


@dataclass(frozen=True)
class ReplayItem:
    """One selected memory, without its content."""

    record_id: str
    content_digest: str
    salience: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "content_digest": self.content_digest,
            "salience": self.salience,
            "metadata": self.metadata,
        }


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_time(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ReplayError("a memory timestamp is not ISO-8601") from error
    return _as_utc(parsed)


def salience(
    record: MemoryRecord,
    *,
    now: datetime | None = None,
    config: SalienceConfig | None = None,
) -> float:
    """Score one record in ``[0, 1]`` from metadata only."""
    settings = config or SalienceConfig()
    settings.validate()
    if record.digest_state in (DigestState.REFUSED, DigestState.QUARANTINED):
        return 0.0
    current = _as_utc(now or datetime.now(timezone.utc))
    stamp = _parse_time(record.last_access_at or record.created_at, current)
    age_days = max(0.0, (current - stamp).total_seconds() / 86400.0)
    recency = 1.0 / (1.0 + age_days / settings.half_life_days)
    load = min(max(float(record.emotional_load), 0.0), 1.0)
    accesses = log1p(max(int(record.access_count), 0)) / log1p(
        settings.access_saturation
    )
    access = min(max(accesses, 0.0), 1.0)
    boosts = {
        DigestState.RAW: settings.raw_boost,
        DigestState.DIGESTING: settings.digesting_boost,
        DigestState.DIGESTED: settings.digested_boost,
    }
    boost = boosts.get(record.digest_state, settings.raw_boost)
    score = (
        settings.recency_weight * recency
        + settings.load_weight * load
        + settings.access_weight * access
    ) * boost
    return min(max(score, 0.0), 1.0)


def select_replay(
    records: Iterable[MemoryRecord],
    *,
    limit: int,
    now: datetime | None = None,
    seed: int = 20261027,
    config: SalienceConfig | None = None,
    is_active: Callable[[str], bool] | None = None,
) -> list[ReplayItem]:
    """Pick the ``limit`` most salient eligible records.

    Refused and quarantined memories are not forced back in. ``is_active``
    lets the caller apply the load ledger's active set; the default accepts
    every record that is not refused or quarantined.
    """
    if limit <= 0:
        raise ReplayError("replay limit must be positive")
    settings = config or SalienceConfig()
    settings.validate()
    current = _as_utc(now or datetime.now(timezone.utc))
    scored: list[tuple[float, MemoryRecord]] = []
    for record in records:
        if record.digest_state in (DigestState.REFUSED, DigestState.QUARANTINED):
            continue
        if is_active is not None and not is_active(record.record_id):
            continue
        scored.append((salience(record, now=current, config=settings), record))

    def order(item: tuple[float, MemoryRecord]) -> tuple[float, str]:
        score, record = item
        tie = sha256_of({"seed": int(seed), "record_id": record.record_id})
        return (-score, tie)

    scored.sort(key=order)
    return [
        ReplayItem(
            record_id=record.record_id,
            content_digest=record.content_digest,
            salience=score,
            metadata=record.to_dict(),
        )
        for score, record in scored[:limit]
    ]
