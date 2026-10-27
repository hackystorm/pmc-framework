# SPDX-License-Identifier: GPL-3.0-only

"""HTTP tests for the live server, using a fake engine (no model needed)."""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from typing import Any

from pmc.runtime.live_engine import ChatResult
from pmc.runtime.live_server import make_server


class FakeEngine:
    model_name = "fake-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.text = "hello"
        self.finish_reason = "stop"

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> ChatResult:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return ChatResult(
            text=self.text,
            finish_reason=self.finish_reason,
            prompt_tokens=7,
            completion_tokens=3,
            fast_weight_rows=2,
            fast_weight_digest="a" * 64,
            state_digest="b" * 64,
        )

    def state_summary(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "turns": 0,
            "fast_weight_rows": 2,
            "state_banks": [{"name": "bank", "dims": [2, 4], "elements": 8}],
        }

    def close(self) -> None:
        return


class LiveServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FakeEngine()
        self.server = make_server(self.engine, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address[:2]
        self.base = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.engine.close()

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        raw: bytes | None = None,
    ) -> tuple[int, dict[str, Any]]:
        data = raw
        if data is None and payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))

    # -- GET --------------------------------------------------------------

    def test_health(self) -> None:
        status, payload = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok"})

    def test_models_lists_the_engine_name(self) -> None:
        status, payload = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["data"][0]["id"], "fake-model")
        self.assertEqual(payload["data"][0]["object"], "model")

    def test_state_reports_metadata_only(self) -> None:
        status, payload = self.request("GET", "/v1/pmc/state")
        self.assertEqual(status, 200)
        self.assertEqual(payload["model"], "fake-model")
        self.assertEqual(payload["state_banks"][0]["dims"], [2, 4])
        self.assertNotIn("data", payload["state_banks"][0])

    def test_unknown_route_is_404(self) -> None:
        status, payload = self.request("GET", "/v1/nope")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["type"], "not_found_error")

    # -- chat completions -------------------------------------------------

    def test_chat_completion_shape(self) -> None:
        status, payload = self.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["model"], "fake-model")
        choice = payload["choices"][0]
        self.assertEqual(choice["message"], {"role": "assistant", "content": "hello"})
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(payload["usage"]["prompt_tokens"], 7)
        self.assertEqual(payload["usage"]["completion_tokens"], 3)
        self.assertEqual(payload["usage"]["total_tokens"], 10)
        self.assertTrue(payload["id"].startswith("chatcmpl-"))
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(self.engine.calls[0]["max_tokens"], 5)
        self.assertEqual(self.engine.calls[0]["temperature"], 0.0)
        self.assertEqual(
            self.engine.calls[0]["messages"],
            [{"role": "user", "content": "hi"}],
        )

    def test_stop_truncates_text(self) -> None:
        self.engine.text = "hello world"
        status, payload = self.request(
            "POST",
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "stop": [" world"],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["choices"][0]["message"]["content"], "hello")
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

    def test_max_tokens_default_is_none(self) -> None:
        status, _ = self.request(
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(status, 200)
        self.assertIsNone(self.engine.calls[0]["max_tokens"])

    def test_positive_temperature_is_400(self) -> None:
        status, payload = self.request(
            "POST",
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.5,
            },
        )
        self.assertEqual(status, 400)
        self.assertIn("temperature", payload["error"]["message"])
        self.assertEqual(self.engine.calls, [])

    def test_streaming_is_400(self) -> None:
        status, payload = self.request(
            "POST",
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        self.assertEqual(status, 400)
        self.assertIn("streaming", payload["error"]["message"])

    def test_multiple_choices_are_400(self) -> None:
        status, payload = self.request(
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "n": 2},
        )
        self.assertEqual(status, 400)
        self.assertIn("n=1", payload["error"]["message"])

    def test_malformed_json_is_400(self) -> None:
        status, payload = self.request("POST", "/v1/chat/completions", raw=b"{")
        self.assertEqual(status, 400)
        self.assertIn("JSON", payload["error"]["message"])

    def test_empty_messages_is_400(self) -> None:
        status, payload = self.request(
            "POST", "/v1/chat/completions", {"messages": []}
        )
        self.assertEqual(status, 400)
        self.assertIn("messages", payload["error"]["message"])

    def test_unknown_role_is_400(self) -> None:
        status, payload = self.request(
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "tool", "content": "x"}]},
        )
        self.assertEqual(status, 400)
        self.assertIn("role", payload["error"]["message"])

    def test_zero_max_tokens_is_400(self) -> None:
        status, payload = self.request(
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0},
        )
        self.assertEqual(status, 400)
        self.assertIn("max_tokens", payload["error"]["message"])

    def test_unknown_post_route_is_404(self) -> None:
        status, payload = self.request("POST", "/v1/other", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["type"], "not_found_error")


if __name__ == "__main__":
    unittest.main()
