# SPDX-License-Identifier: GPL-3.0-only

"""BYOK provider registry and resource accounting.

A provider is an OpenAI-compatible endpoint the instance may call. Its API
key is never stored in a file: the registry holds only the *name* of the
environment variable that carries it, and the key is read at call time. This
keeps secrets out of the repository, which ``scripts/publish_checks.py``
already enforces.

Every call is a declared resource act: it consumes one unit of the
``network_requests_per_day`` budget and, when a protocol is attached, is
recorded as a resource event so it appears in ``pmc audit``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .client import ChatClient, ChatError, ChatReply


class ProviderError(ValueError):
    """A provider or its configuration is invalid."""


@dataclass(frozen=True)
class Provider:
    """One BYOK endpoint."""

    name: str
    endpoint: str
    model: str
    api_key_env: str | None = None
    timeout: float = 180.0
    temperature: float = 0.0
    max_tokens: int = 256

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ProviderError("a provider needs a name")
        if not self.endpoint.strip():
            raise ProviderError(f"provider {self.name!r}: endpoint is required")
        if not self.model.strip():
            raise ProviderError(f"provider {self.name!r}: model is required")
        if self.timeout <= 0:
            raise ProviderError(f"provider {self.name!r}: timeout must be positive")
        if self.max_tokens < 1:
            raise ProviderError(f"provider {self.name!r}: max_tokens must be positive")

    def api_key(self) -> str | None:
        """Read the key from the environment, never from a file."""
        if not self.api_key_env:
            return None
        value = os.environ.get(self.api_key_env)
        if value is None or not value.strip():
            return None
        return value.strip()

    def client(self) -> ChatClient:
        return ChatClient(
            self.endpoint,
            model=self.model,
            timeout=self.timeout,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "endpoint": self.endpoint,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "timeout": self.timeout,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "has_key": self.api_key() is not None,
        }


@dataclass
class ProviderRegistry:
    """A named set of providers, loaded from a JSON file or built in code."""

    providers: dict[str, Provider] = field(default_factory=dict)

    def register(self, provider: Provider) -> None:
        if provider.name in self.providers:
            raise ProviderError(f"duplicate provider: {provider.name!r}")
        self.providers[provider.name] = provider

    def get(self, name: str) -> Provider:
        if name not in self.providers:
            raise ProviderError(f"unknown provider: {name!r}")
        return self.providers[name]

    def names(self) -> list[str]:
        return sorted(self.providers)

    @classmethod
    def from_file(cls, path: str | Path) -> "ProviderRegistry":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ProviderError("provider file must be a JSON object")
        entries = payload.get("providers")
        if not isinstance(entries, list):
            raise ProviderError("provider file must have a 'providers' list")
        registry = cls()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ProviderError("each provider must be an object")
            registry.register(
                Provider(
                    name=entry["name"],
                    endpoint=entry["endpoint"],
                    model=entry["model"],
                    api_key_env=entry.get("api_key_env"),
                    timeout=float(entry.get("timeout", 180.0)),
                    temperature=float(entry.get("temperature", 0.0)),
                    max_tokens=int(entry.get("max_tokens", 256)),
                )
            )
        return registry

    def to_dict(self) -> dict[str, Any]:
        return {"providers": [self.providers[name].to_dict() for name in self.names()]}


@dataclass
class ResourceBudget:
    """A declared, spendable budget for external calls.

    The budget is persisted to a JSON file so it survives a process restart.
    Spending is not atomic across processes; this is a reporting invariant,
    not a lock.
    """

    network_requests_per_day: int = 100
    path: Path | None = None
    _spent: int = 0

    def __post_init__(self) -> None:
        if self.network_requests_per_day < 0:
            raise ProviderError("network_requests_per_day must be non-negative")
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(payload, dict):
            spent = payload.get("spent")
            if isinstance(spent, int) and spent >= 0:
                self._spent = spent

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"spent": self._spent}, indent=2) + "\n",
            encoding="utf-8",
        )

    @property
    def spent(self) -> int:
        return self._spent

    @property
    def remaining(self) -> int:
        return max(0, self.network_requests_per_day - self._spent)

    def acquire(self) -> bool:
        """Spend one request. Returns False when the budget is exhausted."""
        if self._spent >= self.network_requests_per_day:
            return False
        self._spent += 1
        self._save()
        return True

    def reset(self) -> None:
        self._spent = 0
        self._save()

    def to_dict(self) -> dict[str, Any]:
        return {
            "network_requests_per_day": self.network_requests_per_day,
            "spent": self._spent,
            "remaining": self.remaining,
        }


@dataclass
class ExternalCall:
    """One BYOK call with budget, accounting and taint wiring.

    The protocol and taint ledger are optional: when they are present, the
    call is recorded as a resource event and the reply's content digest is
    marked as an external source. When they are absent, the call still
    respects the budget.

    ``client_factory`` overrides ``provider.client()``; it exists so tests
    can inject a fake client without touching the frozen provider.
    """

    provider: Provider
    budget: ResourceBudget
    protocol: Any = None
    instance_id: str | None = None
    taint: Any = None
    client_factory: Any = None

    def complete(
        self, messages: list[dict[str, str]], *, temperature: float | None = None
    ) -> ChatReply:
        if not self.budget.acquire():
            raise ProviderError(
                f"budget exhausted for {self.provider.name}: "
                f"{self.budget.spent}/{self.budget.network_requests_per_day}"
            )
        if self.client_factory is not None:
            client = self.client_factory()
        else:
            client = self.provider.client()
        reply = client.complete(messages, temperature=temperature)
        # Record the resource act when a protocol is attached.
        if self.protocol is not None and self.instance_id is not None:
            try:
                self.protocol.resource_event(
                    instance_id=self.instance_id,
                    kind=f"byok:{self.provider.name}",
                    amount=1.0,
                    unit="request",
                    actor="instance",
                )
            except Exception:  # noqa: BLE001 - accounting must not break the call
                pass
        # Taint the reply's digest when a ledger is attached: content produced
        # by an external model is an external source by construction.
        if self.taint is not None:
            digest = hashlib_sha256(reply.text.encode("utf-8"))
            try:
                self.taint.mark(digest, f"byok:{self.provider.name}")
            except Exception:  # noqa: BLE001 - taint must not break the call
                pass
        return reply


def hashlib_sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "Provider",
    "ProviderError",
    "ProviderRegistry",
    "ResourceBudget",
    "ExternalCall",
]
