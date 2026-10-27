# SPDX-License-Identifier: GPL-3.0-only

"""Argument tests for the KV swap probe."""

from __future__ import annotations

import unittest

from kv_swap_probe import build_parser


class ParserTest(unittest.TestCase):
    def test_the_probe_needs_a_state_and_a_model(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])

        args = build_parser().parse_args(
            [
                "--state",
                "state.pmcstate",
                "--model",
                "model.gguf",
                "--lib",
                "libllama.so",
                "--shim",
                "shim.so",
            ]
        )
        self.assertEqual(args.state, "state.pmcstate")
        self.assertEqual(args.ctx, 128)
        self.assertEqual(args.batch, 32)

    def test_the_dummy_message_is_parameterised(self) -> None:
        args = build_parser().parse_args(
            [
                "--state",
                "state.pmcstate",
                "--model",
                "model.gguf",
                "--lib",
                "libllama.so",
                "--shim",
                "shim.so",
                "--message",
                "hello",
            ]
        )
        self.assertEqual(args.message, "hello")


if __name__ == "__main__":
    unittest.main()
