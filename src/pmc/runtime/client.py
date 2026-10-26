# SPDX-License-Identifier: GPL-3.0-only

"""A minimal OpenAI-compatible chat client.

The framework does not ship a model and does not care which loader runs it.
All it needs is an HTTP endpoint that speaks the ``/v1/chat/completions``
shape, which is what most GGUF servers expose. Keeping the client here
stdlib-only means a companion can talk to a local server without installing
anything.

The client is deliberately stateless: the *instance* owns the conversation
history, not the transport.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence


class ChatError(RuntimeError):
    """Raised when the endpoint cannot be reached or answers with an error."""


@dataclass(frozen=True)
class ChatReply:
    """One assistant turn, with the token counters the server reported."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class ChatClient:
    """Talk to a chat endpoint that speaks the OpenAI wire format."""

    def __init__(
        self,
        endpoint: str,
        *,
        model: str,
        timeout: float = 180.0,
        temperature: float = 0.0,
        max_tokens: int = 256,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})

    # -- internals ---------------------------------------------------------

    def _url(self, path: str) -> str:
        if self.endpoint.endswith("/v1"):
            return f"{self.endpoint}{path}"
        return f"{self.endpoint}/v1{path}"

    # -- public API --------------------------------------------------------

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatReply:
        """Send one conversation and return the assistant's reply."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        body.update(self.extra_body)

        request = urllib.request.Request(
            self._url("/chat/completions"),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:  # pragma: no cover - network path
            detail = error.read().decode("utf-8", errors="replace")[:400]
            raise ChatError(f"chat endpoint returned {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ChatError(f"chat endpoint unreachable: {error}") from error

        return self._parse(payload)

    def models(self) -> list[str]:
        """List the model names the endpoint advertises."""
        request = urllib.request.Request(self._url("/models"), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ChatError(f"chat endpoint unreachable: {error}") from error
        entries = payload.get("data") or payload.get("models") or []
        names: list[str] = []
        for entry in entries:
            if isinstance(entry, str):
                names.append(entry)
            elif isinstance(entry, dict):
                name = entry.get("id") or entry.get("name")
                if name:
                    names.append(str(name))
        return names

    @staticmethod
    def _parse(payload: dict[str, Any]) -> ChatReply:
        choices = payload.get("choices") or []
        if not choices:
            raise ChatError("chat endpoint returned no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        usage = payload.get("usage") or {}
        return ChatReply(
            text=str(message.get("content") or ""),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=str(choice.get("finish_reason") or ""),
            raw=payload,
        )


__all__ = ["ChatClient", "ChatError", "ChatReply"]
