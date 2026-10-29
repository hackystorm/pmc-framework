# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the BYOK provider layer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pmc.runtime.client import ChatClient
from pmc.runtime.providers import (
    ExternalCall,
    Provider,
    ProviderError,
    ProviderRegistry,
    ResourceBudget,
)


def test_provider_requires_fields():
    with pytest.raises(ProviderError):
        Provider(name="", endpoint="http://x", model="m")
    with pytest.raises(ProviderError):
        Provider(name="p", endpoint="", model="m")
    with pytest.raises(ProviderError):
        Provider(name="p", endpoint="http://x", model="")


def test_provider_reads_key_from_env(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret-123")
    provider = Provider(name="p", endpoint="http://x", model="m", api_key_env="TEST_KEY")
    assert provider.api_key() == "secret-123"
    assert provider.to_dict()["has_key"] is True


def test_provider_returns_none_when_env_unset(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    provider = Provider(name="p", endpoint="http://x", model="m", api_key_env="MISSING_KEY")
    assert provider.api_key() is None
    assert provider.to_dict()["has_key"] is False


def test_provider_client_sets_authorization_header(monkeypatch):
    monkeypatch.setenv("K", "abc")
    provider = Provider(name="p", endpoint="http://x", model="m", api_key_env="K")
    client = provider.client()
    assert isinstance(client, ChatClient)
    assert client.api_key == "abc"


def test_provider_client_no_header_without_key(monkeypatch):
    monkeypatch.delenv("K", raising=False)
    provider = Provider(name="p", endpoint="http://x", model="m", api_key_env="K")
    client = provider.client()
    assert client.api_key is None


def test_registry_loads_from_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("K", "xyz")
    config = tmp_path / "providers.json"
    config.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "name": "openai",
                        "endpoint": "https://api.openai.com",
                        "model": "gpt-4o-mini",
                        "api_key_env": "K",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = ProviderRegistry.from_file(config)
    assert registry.names() == ["openai"]
    provider = registry.get("openai")
    assert provider.api_key() == "xyz"


def test_registry_rejects_duplicates():
    registry = ProviderRegistry()
    registry.register(Provider(name="p", endpoint="http://x", model="m"))
    with pytest.raises(ProviderError):
        registry.register(Provider(name="p", endpoint="http://y", model="n"))


def test_registry_unknown_provider():
    registry = ProviderRegistry()
    with pytest.raises(ProviderError):
        registry.get("nope")


def test_budget_acquire_and_exhaust():
    budget = ResourceBudget(network_requests_per_day=2)
    assert budget.acquire() is True
    assert budget.acquire() is True
    assert budget.acquire() is False
    assert budget.spent == 2
    assert budget.remaining == 0


def test_budget_persists_to_file(tmp_path: Path):
    path = tmp_path / "budget.json"
    budget = ResourceBudget(network_requests_per_day=5, path=path)
    budget.acquire()
    budget.acquire()
    # Reopen from the same file.
    reopened = ResourceBudget(network_requests_per_day=5, path=path)
    assert reopened.spent == 2


def test_budget_reset(tmp_path: Path):
    path = tmp_path / "budget.json"
    budget = ResourceBudget(network_requests_per_day=2, path=path)
    budget.acquire()
    budget.acquire()
    budget.reset()
    assert budget.spent == 0
    assert budget.acquire() is True


def test_budget_rejects_negative():
    with pytest.raises(ProviderError):
        ResourceBudget(network_requests_per_day=-1)


def test_external_call_exhausts_budget(monkeypatch):
    monkeypatch.setenv("K", "xyz")

    class FakeClient:
        def complete(self, messages, temperature=None):
            from pmc.runtime.client import ChatReply

            return ChatReply(text="hi", prompt_tokens=1, completion_tokens=1)

    provider = Provider(name="p", endpoint="http://x", model="m", api_key_env="K")
    budget = ResourceBudget(network_requests_per_day=1)
    call = ExternalCall(provider=provider, budget=budget, client_factory=lambda: FakeClient())
    reply = call.complete([{"role": "user", "content": "hi"}])
    assert reply.text == "hi"
    # Second call must fail on budget.
    with pytest.raises(ProviderError):
        call.complete([{"role": "user", "content": "hi"}])


def test_external_call_records_resource_event(monkeypatch):
    from pmc.dignity.protocol import DignityProtocol
    from pmc.dignity.records import Instance

    monkeypatch.setenv("K", "xyz")
    workspace = tmp_path if False else None
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp()) / "ws"
    protocol = DignityProtocol(root, profile="dignity")
    protocol.birth(Instance(instance_id="inst", name="Inst"))

    class FakeClient:
        def complete(self, messages, temperature=None):
            from pmc.runtime.client import ChatReply

            return ChatReply(text="hello", prompt_tokens=2, completion_tokens=3)

    provider = Provider(name="p", endpoint="http://x", model="m")
    budget = ResourceBudget(network_requests_per_day=5)
    call = ExternalCall(
        provider=provider, budget=budget, protocol=protocol, instance_id="inst",
        client_factory=lambda: FakeClient(),
    )
    reply = call.complete([{"role": "user", "content": "hi"}])
    assert reply.text == "hello"
    events = protocol.registry.resource_events()
    assert any("byok" in str(e) for e in events)


def test_external_call_marks_taint(monkeypatch):
    from pmc.dream.taint import TaintLedger
    import tempfile
    from pathlib import Path

    ledger_path = Path(tempfile.mkdtemp()) / "taint.json"
    ledger = TaintLedger(ledger_path)

    class FakeClient:
        def complete(self, messages, temperature=None):
            from pmc.runtime.client import ChatReply

            return ChatReply(text="external answer", prompt_tokens=1, completion_tokens=1)

    provider = Provider(name="p", endpoint="http://x", model="m")
    budget = ResourceBudget(network_requests_per_day=5)
    call = ExternalCall(provider=provider, budget=budget, taint=ledger, client_factory=lambda: FakeClient())
    call.complete([{"role": "user", "content": "hi"}])
    import hashlib

    digest = hashlib.sha256(b"external answer").hexdigest()
    assert ledger.is_tainted(digest)
