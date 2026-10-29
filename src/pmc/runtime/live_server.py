# SPDX-License-Identifier: GPL-3.0-only
"""OpenAI-compatible HTTP face for the live runtime.

This module is deliberately thin: it validates a request, calls
:class:`pmc.runtime.live_engine.LiveEngine` in the same process, and serialises
the answer. It never loads a model, never starts a second server, and never
touches the state banks directly.

Two limitations are explicit rather than hidden:

* **Greedy only.** ``temperature`` must be 0. A positive temperature is a 400,
  not a silent downgrade.
* **No streaming.** ``stream: true`` is a 400. The completion arrives whole.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .live_engine import (
    DEFAULT_MAX_TOKENS,
    MAX_REQUEST_TOKENS,
    LiveEngine,
    LiveEngineError,
)
from .live_state import DEFAULT_LEARNING_RATE

MAX_BODY_BYTES = 1 << 20

_ROLES = ("system", "user", "assistant", "tool")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: "LiveHTTPServer"

    # The instance's text is never logged, and the test output stays readable.
    def log_message(self, _format: str, *_args: object) -> None:
        return

    # -- responses --------------------------------------------------------

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(
        self, status: int, message: str, error_type: str = "invalid_request_error"
    ) -> None:
        self._send_json(
            status,
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "param": None,
                    "code": None,
                }
            },
        )

    # -- request parsing --------------------------------------------------

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("Content-Length must be an integer") from error
        if length <= 0:
            raise ValueError("request body is empty")
        if length > MAX_BODY_BYTES:
            raise ValueError(f"request body exceeds {MAX_BODY_BYTES} bytes")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError("request body is truncated")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"request body is not valid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    @staticmethod
    def _stops(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("stop must be a string or an array of strings")
        stops: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("stop must contain only strings")
            if item:
                stops.append(item)
        return stops

    def _chat_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("stream"):
            raise ValueError("streaming is not supported; use stream=false")
        if payload.get("n", 1) != 1:
            raise ValueError("only n=1 is supported")

        raw_temperature = payload.get("temperature", 0.0)
        if isinstance(raw_temperature, bool) or not isinstance(
            raw_temperature, (int, float)
        ):
            raise ValueError("temperature must be a number")
        temperature = float(raw_temperature)
        if temperature != 0.0:
            raise ValueError(
                "the live runtime samples greedily; temperature must be 0"
            )

        raw_max_tokens = payload.get("max_tokens")
        if raw_max_tokens is None:
            max_tokens = None
        else:
            if isinstance(raw_max_tokens, bool) or not isinstance(
                raw_max_tokens, int
            ):
                raise ValueError("max_tokens must be an integer")
            if raw_max_tokens <= 0 or raw_max_tokens > MAX_REQUEST_TOKENS:
                raise ValueError(
                    f"max_tokens must be in [1, {MAX_REQUEST_TOKENS}]"
                )
            max_tokens = raw_max_tokens

        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("messages must be a non-empty array")
        messages: list[dict[str, str]] = []
        for index, message in enumerate(raw_messages):
            if not isinstance(message, dict):
                raise ValueError(f"messages[{index}] must be an object")
            role = message.get("role")
            content = message.get("content")
            if role not in _ROLES:
                raise ValueError(
                    f"messages[{index}].role must be one of {_ROLES}"
                )
            if role == "tool":
                if not isinstance(content, str):
                    raise ValueError(
                        f"messages[{index}].content must be a string"
                    )
            else:
                if not isinstance(content, str) or not content:
                    raise ValueError(
                        f"messages[{index}].content must be a non-empty string"
                    )
            messages.append({"role": role, "content": content if isinstance(content, str) else ""})

        raw_tools = payload.get("tools")
        tools: list[dict[str, Any]] | None = None
        if raw_tools is not None:
            if not isinstance(raw_tools, list):
                raise ValueError("tools must be an array")
            validated: list[dict[str, Any]] = []
            for index, tool in enumerate(raw_tools):
                if not isinstance(tool, dict):
                    raise ValueError(f"tools[{index}] must be an object")
                if tool.get("type") != "function":
                    raise ValueError(f"tools[{index}].type must be 'function'")
                function = tool.get("function")
                if not isinstance(function, dict) or not function.get("name"):
                    raise ValueError(f"tools[{index}].function.name is required")
                validated.append(tool)
            tools = validated

        return {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stops": self._stops(payload.get("stop")),
            "tools": tools,
        }

    @staticmethod
    def _apply_stops(text: str, stops: list[str]) -> tuple[str, str | None]:
        if not stops:
            return text, None
        cut = len(text)
        for stop in stops:
            index = text.find(stop)
            if index >= 0:
                cut = min(cut, index)
        if cut < len(text):
            return text[:cut], "stop"
        return text, None

    # -- endpoints --------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if path == "/v1/models":
            engine = self.server.engine
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": engine.model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "pmc",
                        }
                    ],
                },
            )
            return
        if path == "/v1/pmc/state":
            self._send_json(200, self.server.engine.state_summary())
            return
        self._send_error(404, f"no route for {path}", "not_found_error")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path != "/v1/chat/completions":
            self._send_error(404, f"no route for {path}", "not_found_error")
            return
        try:
            payload = self._read_json()
            request = self._chat_request(payload)
        except ValueError as error:
            self._send_error(400, str(error))
            return
        try:
            result = self.server.engine.chat(
                request["messages"],
                max_tokens=request["max_tokens"],
                temperature=request["temperature"],
            )
        except (LiveEngineError, ValueError) as error:
            self._send_error(400, str(error))
            return
        except Exception:  # noqa: BLE001 - the HTTP boundary must not leak
            traceback.print_exc()
            self._send_error(500, "internal runtime error", "server_error")
            return

        text, stopped = self._apply_stops(result.text, request["stops"])
        finish_reason = stopped or result.finish_reason
        self._send_json(
            200,
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.server.engine.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text, "tool_calls": None},
                        "finish_reason": finish_reason,
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
            },
        )


class LiveHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server that owns one in-process engine."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, address: tuple[str, int], engine: Any, tool_registry: Any = None
    ) -> None:
        self.engine = engine
        self.tool_registry = tool_registry
        super().__init__(address, _Handler)

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one declared tool through the registry, or raise ToolError."""
        if self.tool_registry is None:
            raise ValueError("no tool registry attached to this server")
        context = {"engine": self.engine}
        return self.tool_registry.call(name, arguments, context)


def make_server(
    engine: Any,
    host: str = "127.0.0.1",
    port: int = 8080,
    tool_registry: Any = None,
) -> LiveHTTPServer:
    """Bind an engine to a local HTTP port (port 0 picks a free port)."""
    return LiveHTTPServer((host, port), engine, tool_registry=tool_registry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pmc-live",
        description=(
            "Serve one PMC live model with an OpenAI-compatible endpoint. "
            "The model, the fast weights and the HTTP server share one process."
        ),
    )
    parser.add_argument("--model", required=True, help="fused or base GGUF file")
    parser.add_argument("--lib", required=True, help="patched libllama shared library")
    parser.add_argument("--shim", required=True, help="compiled LlamaSpike C shim")
    parser.add_argument("--delta", default=None, help="delta JSON path (optional)")
    parser.add_argument("--ctx", type=int, default=2048, help="context size")
    parser.add_argument("--batch", type=int, default=256, help="decode batch size")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads")
    parser.add_argument("--model-name", default="pmc-live", help="API model id")
    parser.add_argument("--system", default="", help="system prompt text")
    parser.add_argument("--system-file", default=None, help="system prompt file")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="fast-weight step size",
    )
    parser.add_argument(
        "--learn-from-prompt",
        action="store_true",
        help="also update fast weights from the prompt tokens",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="default completion budget",
    )
    parser.add_argument(
        "--no-digest",
        action="store_true",
        help="skip the model content digest (faster startup, weaker delta check)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8080, help="bind port")
    args = parser.parse_args(argv)

    system_prompt = args.system
    if args.system_file:
        try:
            system_prompt = Path(args.system_file).read_text(encoding="utf-8")
        except OSError as error:
            parser.error(f"cannot read --system-file: {error}")

    try:
        engine = LiveEngine(
            args.model,
            args.lib,
            args.shim,
            n_ctx=args.ctx,
            n_batch=args.batch,
            n_threads=args.threads,
            delta_path=args.delta,
            system_prompt=system_prompt,
            model_name=args.model_name,
            learning_rate=args.learning_rate,
            learn_from_prompt=args.learn_from_prompt,
            default_max_tokens=args.max_tokens,
            compute_model_digest=not args.no_digest,
        )
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"pmc-live: cannot start: {error}", file=sys.stderr)
        return 1

    try:
        server = make_server(engine, args.host, args.port)
    except OSError as error:
        engine.close()
        print(f"pmc-live: cannot bind {args.host}:{args.port}: {error}", file=sys.stderr)
        return 1

    bound_host, bound_port = server.server_address[:2]
    print(
        f"pmc-live listening on http://{bound_host}:{bound_port} "
        f"(model {engine.model_name})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("pmc-live: stopping", file=sys.stderr)
    finally:
        server.server_close()
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
