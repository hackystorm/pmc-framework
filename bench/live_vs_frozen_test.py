# SPDX-License-Identifier: GPL-3.0-only

"""Pure-helper tests for the same-memory benchmark."""

from __future__ import annotations

import unittest

from live_vs_frozen import (
    answer_record,
    build_parser,
    code_present,
    code_present_digits,
    digits_only,
    fact_message,
    first_reply_echo_free,
    main,
    message_text,
)


class CodeTest(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertTrue(code_present("The code is 4821.", "4821"))
        self.assertFalse(code_present("The code is 4822.", "4821"))

    def test_digits_match_ignores_punctuation(self) -> None:
        self.assertTrue(code_present_digits("The code is 4,821.", "4821"))
        self.assertFalse(code_present("The code is 4,821.", "4821"))
        self.assertFalse(code_present_digits("The code is 4812.", "4821"))

    def test_digits_only(self) -> None:
        self.assertEqual(digits_only("a4b8c2d1"), "4821")
        self.assertEqual(digits_only("no numbers"), "")

    def test_fact_message(self) -> None:
        self.assertEqual(fact_message("4821"), "Remember the code: 4821.")

    def test_quiet_fact_asks_for_a_non_echoing_reply(self) -> None:
        quiet = fact_message("4821", "quiet")
        self.assertIn("4821", quiet)
        self.assertIn("noted", quiet)

    def test_unknown_first_style_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fact_message("4821", "loud")

    def test_echo_free_flag(self) -> None:
        self.assertFalse(first_reply_echo_free("The code is 4821.", "4821"))
        self.assertTrue(first_reply_echo_free("Noted.", "4821"))

    def test_message_text_strips(self) -> None:
        self.assertEqual(message_text("  hi  "), "hi")

    def test_answer_record_flags_both_matches(self) -> None:
        record = answer_record("The code is 4,821.", "4821")
        self.assertFalse(record["code_found"])
        self.assertTrue(record["code_found_digits"])
        self.assertEqual(record["digits"], "4821")

    def test_kv_is_a_condition(self) -> None:
        args = build_parser().parse_args(["--condition", "kv"])
        self.assertEqual(args.condition, "kv")

    def test_replay_and_quiet_parse(self) -> None:
        args = build_parser().parse_args(
            ["--condition", "replay", "--first-style", "quiet"]
        )
        self.assertEqual(args.condition, "replay")
        self.assertEqual(args.first_style, "quiet")

    def test_kv_requires_the_live_paths(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main(["--condition", "kv"])
        self.assertEqual(caught.exception.code, 2)

    def test_kv_restart_requires_a_state_file(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main(
                [
                    "--condition",
                    "kv-restart",
                    "--model",
                    "model.gguf",
                    "--lib",
                    "libllama.so",
                    "--shim",
                    "shim.so",
                ]
            )
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
