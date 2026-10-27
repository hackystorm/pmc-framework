# SPDX-License-Identifier: GPL-3.0-only

"""Unit tests for the live engine's pure helpers, plus an opt-in model test.

The model test is skipped unless ``PMC_LIVE_MODEL``, ``PMC_LIVE_LIB`` and
``PMC_LIVE_SHIM`` point at a working patched build. It never hard-codes a path.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import unittest
from pathlib import Path

from pmc.runtime.live_engine import (
    KvState,
    KvStateError,
    LiveEngine,
    LiveEngineError,
    file_sha256,
)
from pmc.runtime.live_state import FastWeights, LiveDelta


class FileDigestTest(unittest.TestCase):
    def test_sha256_matches_hashlib(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            payload = b"abc" * 1000
            path.write_bytes(payload)
            self.assertEqual(
                file_sha256(path, chunk_size=7),
                hashlib.sha256(payload).hexdigest(),
            )

    def test_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.bin"
            path.write_bytes(b"")
            self.assertEqual(
                file_sha256(path), hashlib.sha256(b"").hexdigest()
            )


class StepSizeTest(unittest.TestCase):
    def test_step_is_scaled_by_the_squared_norm(self) -> None:
        hidden = [3.0, 4.0]
        step = LiveEngine._step_size(hidden, 0.5)
        self.assertAlmostEqual(step, 0.02)
        weights = FastWeights(dim=2)
        weights.update(1, hidden, probability=0.5, learning_rate=step)
        self.assertAlmostEqual(weights.bias(1, hidden), 0.25)

    def test_zero_hidden_is_rejected(self) -> None:
        with self.assertRaises(LiveEngineError):
            LiveEngine._step_size([0.0, 0.0], 0.01)

    def test_non_finite_hidden_is_rejected(self) -> None:
        with self.assertRaises(LiveEngineError):
            LiveEngine._step_size([1.0, math.inf], 0.01)

    def test_bad_learning_rate_is_rejected(self) -> None:
        with self.assertRaises(LiveEngineError):
            LiveEngine._step_size([1.0], -0.1)
        with self.assertRaises(LiveEngineError):
            LiveEngine._step_size([1.0], math.nan)


class LastUserTest(unittest.TestCase):
    def test_last_user_wins(self) -> None:
        pairs = [
            ("system", "s"),
            ("user", "first"),
            ("assistant", "a"),
            ("user", "second"),
        ]
        self.assertEqual(LiveEngine._last_user(pairs), "second")

    def test_missing_user_is_empty(self) -> None:
        self.assertEqual(LiveEngine._last_user([("system", "s")]), "")


class KvStateTest(unittest.TestCase):
    def test_valid_state(self) -> None:
        data = b"state-bytes"
        state = KvState(
            data=data,
            tokens=(1, 2, 3),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        self.assertEqual(state.tokens, (1, 2, 3))
        self.assertNotIn("state-bytes", repr(state))

    def test_empty_data_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            KvState(data=b"", tokens=(1,), sha256=hashlib.sha256(b"").hexdigest())

    def test_digest_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            KvState(data=b"abc", tokens=(1,), sha256="0" * 64)

    def test_non_hex_digest_is_rejected(self) -> None:
        data = b"abc"
        with self.assertRaises(ValueError):
            KvState(data=data, tokens=(1,), sha256="z" * 64)

    def test_empty_prefix_is_rejected(self) -> None:
        data = b"abc"
        with self.assertRaises(ValueError):
            KvState(data=data, tokens=(), sha256=hashlib.sha256(data).hexdigest())

    def test_negative_token_is_rejected(self) -> None:
        data = b"abc"
        with self.assertRaises(ValueError):
            KvState(data=data, tokens=(1, -1), sha256=hashlib.sha256(data).hexdigest())

    def test_identity_matches_only_the_saved_model_and_geometry(self) -> None:
        data = b"state-bytes"
        state = KvState(
            data=data,
            tokens=(1, 2, 3),
            sha256=hashlib.sha256(data).hexdigest(),
            model_digest="a" * 64,
            n_embd=2048,
            n_ctx=128,
        )
        self.assertTrue(state.matches("a" * 64, 2048, 128))
        self.assertFalse(state.matches("b" * 64, 2048, 128))
        self.assertFalse(state.matches("a" * 64, 1024, 128))
        self.assertFalse(state.matches("a" * 64, 2048, 256))
        self.assertTrue(state.matches("", 2048, 128))

    def test_unknown_identity_is_accepted(self) -> None:
        data = b"state-bytes"
        state = KvState(
            data=data,
            tokens=(1,),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        self.assertTrue(state.matches("b" * 64, 1024, 4096))

    def test_bad_model_digest_is_rejected(self) -> None:
        data = b"abc"
        with self.assertRaises(ValueError):
            KvState(
                data=data,
                tokens=(1,),
                sha256=hashlib.sha256(data).hexdigest(),
                model_digest="z" * 64,
                n_embd=2048,
                n_ctx=128,
            )

    def test_half_geometry_is_rejected(self) -> None:
        data = b"abc"
        with self.assertRaises(ValueError):
            KvState(
                data=data,
                tokens=(1,),
                sha256=hashlib.sha256(data).hexdigest(),
                n_embd=2048,
                n_ctx=0,
            )

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = b"state-bytes" * 100
            state = KvState(
                data=data,
                tokens=(1, 2, 3),
                sha256=hashlib.sha256(data).hexdigest(),
                model_digest="a" * 64,
                n_embd=2048,
                n_ctx=128,
            )
            path = Path(directory) / "state.pmcstate"
            state.save(path)
            self.assertTrue(path.read_bytes().startswith(b"PMCKV1\x00"))
            self.assertEqual(KvState.load(path), state)

    def test_load_rejects_a_tampered_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = b"state-bytes"
            state = KvState(
                data=data,
                tokens=(1,),
                sha256=hashlib.sha256(data).hexdigest(),
                n_embd=2048,
                n_ctx=128,
            )
            path = Path(directory) / "state.pmcstate"
            state.save(path)
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 0x01
            path.write_bytes(bytes(raw))
            with self.assertRaises(KvStateError):
                KvState.load(path)

    def test_load_rejects_a_different_expected_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = b"state-bytes"
            state = KvState(
                data=data,
                tokens=(1,),
                sha256=hashlib.sha256(data).hexdigest(),
                model_digest="a" * 64,
                n_embd=2048,
                n_ctx=128,
            )
            path = Path(directory) / "state.pmcstate"
            state.save(path)
            with self.assertRaises(KvStateError):
                KvState.load(path, expected_model_digest="b" * 64)
            with self.assertRaises(KvStateError):
                KvState.load(path, expected_n_embd=1024)
            with self.assertRaises(KvStateError):
                KvState.load(path, expected_n_ctx=256)

    def test_load_rejects_a_truncated_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pmcstate"
            path.write_bytes(b"PMCKV1\x00")
            with self.assertRaises(KvStateError):
                KvState.load(path)


@unittest.skipUnless(
    os.environ.get("PMC_LIVE_MODEL")
    and os.environ.get("PMC_LIVE_LIB")
    and os.environ.get("PMC_LIVE_SHIM"),
    "set PMC_LIVE_MODEL, PMC_LIVE_LIB and PMC_LIVE_SHIM for a model test",
)
class LiveEngineModelTest(unittest.TestCase):
    def test_chat_persists_state_and_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            delta_path = Path(directory) / "delta.json"
            engine = LiveEngine(
                os.environ["PMC_LIVE_MODEL"],
                os.environ["PMC_LIVE_LIB"],
                os.environ["PMC_LIVE_SHIM"],
                n_ctx=128,
                n_batch=64,
                n_threads=2,
                delta_path=delta_path,
                system_prompt="You are a terse assistant.",
                model_name="test-model",
                compute_model_digest=False,
            )
            try:
                result = engine.chat(
                    [{"role": "user", "content": "Say the word apple."}],
                    max_tokens=4,
                )
                self.assertGreater(result.prompt_tokens, 0)
                self.assertIn(result.finish_reason, ("stop", "length"))
                self.assertEqual(result.fast_weight_rows, len(engine.delta.fast_weights.rows))
                self.assertEqual(result.state_digest, engine.delta.state_digest)
                self.assertTrue(delta_path.is_file())
                summary = engine.state_summary()
                self.assertEqual(summary["embedding_dim"], engine.n_embd)
                self.assertEqual(
                    len(summary["state_banks"]), len(engine.state_banks)
                )
            finally:
                engine.close()

            loaded = LiveDelta.load(delta_path)
            self.assertEqual(loaded.dim, engine.n_embd)
            self.assertEqual(len(loaded.turns), 1)
            self.assertEqual(loaded.state_digest, result.state_digest)

            with self.assertRaises(LiveEngineError):
                engine.chat(
                    [{"role": "user", "content": "x"}],
                    max_tokens=1,
                    temperature=0.7,
                )

    def test_kv_state_restore(self) -> None:
        engine = LiveEngine(
            os.environ["PMC_LIVE_MODEL"],
            os.environ["PMC_LIVE_LIB"],
            os.environ["PMC_LIVE_SHIM"],
            n_ctx=128,
            n_batch=64,
            n_threads=2,
            system_prompt="You are a terse assistant.",
            model_name="test-model",
            compute_model_digest=False,
        )
        try:
            fact = "Remember the code: 4821."
            first = engine.chat(
                [{"role": "user", "content": fact}],
                max_tokens=8,
                save_state=True,
            )
            state = first.kv_state
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(
                len(state.tokens), first.prompt_tokens + first.completion_tokens
            )
            self.assertGreater(len(state.data), 0)
            self.assertEqual(len(state.sha256), 64)
            self.assertEqual(state.n_embd, engine.n_embd)
            self.assertEqual(state.n_ctx, 128)

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "state.pmcstate"
                state.save(path)
                loaded = KvState.load(path)
                self.assertEqual(loaded, state)

            transcript = [
                {"role": "user", "content": fact},
                {"role": "assistant", "content": first.text},
                {
                    "role": "user",
                    "content": "What is the code? Reply with the four digits only.",
                },
            ]
            forged = KvState(
                data=state.data,
                tokens=state.tokens,
                sha256=state.sha256,
                model_digest=state.model_digest,
                n_embd=state.n_embd,
                n_ctx=state.n_ctx + 1,
            )
            with self.assertRaises(LiveEngineError):
                engine.chat(transcript, max_tokens=1, restore=forged)

            second = engine.chat(
                transcript,
                max_tokens=4,
                restore=loaded,
            )
            self.assertGreater(second.prompt_tokens, len(state.tokens))
            self.assertIsNone(second.kv_state)
            self.assertIn(second.finish_reason, ("stop", "length"))
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
