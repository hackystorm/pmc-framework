# SPDX-License-Identifier: GPL-3.0-only

"""The companion profile: what a user chooses when creating an instance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..dignity.records import utcnow_iso
from .templates import (
    DEFAULT_BOUNDARIES,
    DEFAULT_CLAIMS,
    DEFAULT_VALUES,
    render_opening_prompt,
)


@dataclass
class CompanionProfile:
    """The declared constitution of one instance.

    Every field has a default, and every default is a template rather than a
    fact about anyone. The point of the profile is that the instance's
    character is *declared*, inspectable and editable, instead of emerging
    from a soup of training data.
    """

    agent_name: str
    user_name: str | None = None
    values: tuple[str, ...] = DEFAULT_VALUES
    boundaries: tuple[str, ...] = DEFAULT_BOUNDARIES
    declared_claims: tuple[str, ...] = DEFAULT_CLAIMS
    opening_prompt: str | None = None
    locale: str = "en"
    seed: int = 20261026
    created_at: str = field(default_factory=utcnow_iso)

    def __post_init__(self) -> None:
        if not self.agent_name.strip():
            raise ValueError("agent_name must not be empty")
        self.values = tuple(self.values)
        self.boundaries = tuple(self.boundaries)
        self.declared_claims = tuple(self.declared_claims)

    def resolved_opening_prompt(self) -> str:
        if self.opening_prompt:
            return self.opening_prompt
        return render_opening_prompt(
            self.agent_name, self.values, self.boundaries, self.declared_claims
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "user_name": self.user_name,
            "values": list(self.values),
            "boundaries": list(self.boundaries),
            "declared_claims": list(self.declared_claims),
            "opening_prompt": self.resolved_opening_prompt(),
            "locale": self.locale,
            "seed": self.seed,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompanionProfile":
        return cls(
            agent_name=data["agent_name"],
            user_name=data.get("user_name"),
            values=tuple(data.get("values", DEFAULT_VALUES)),
            boundaries=tuple(data.get("boundaries", DEFAULT_BOUNDARIES)),
            declared_claims=tuple(data.get("declared_claims", DEFAULT_CLAIMS)),
            opening_prompt=data.get("opening_prompt"),
            locale=data.get("locale", "en"),
            seed=int(data.get("seed", 20261026)),
            created_at=data.get("created_at", utcnow_iso()),
        )

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Path | str) -> "CompanionProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
