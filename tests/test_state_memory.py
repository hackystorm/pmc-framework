# SPDX-License-Identifier: GPL-3.0-only

"""The engine consumes a self-contained ``pmc.state.memory.*`` bank."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pmc.runtime.live_engine import LiveEngine
from pmc.runtime.state_banks import StateBank
from pmc.runtime.state_memory import StateMemory, StateMemoryError

_TYPE_ARRAY = 9
_TYPE_UINT32 = 4
_TYPE_FLOAT32 = 6


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _u32_array(values: list[int]) -> bytes:
    return struct.pack("<IQ", _TYPE_UINT32, len(values)) + struct.pack(
        f"<{len(values)}I", *values
    )


def _f32_array(values: list[float]) -> bytes:
    return struct.pack("<IQ", _TYPE_FLOAT32, len(values)) + struct.pack(
        f"<{len(values)}f", *values
    )


def _gguf(entries: list[tuple[str, int, bytes]]) -> bytes:
    out = bytearray(b"GGUF")
    out += struct.pack("<IQQ", 3, 0, len(entries))
    for key, value_type, raw in entries:
        out += _string(key) + struct.pack("<I", value_type) + raw
    return bytes(out)


def _memory_entries() -> list[tuple[str, int, bytes]]:
    banks = _valid_banks()
    return [
        (f"pmc.state.{name}.dims", _TYPE_ARRAY, _u32_array(list(bank.dims)))
        for name, bank in banks.items()
    ] + [
        (f"pmc.state.{name}.data", _TYPE_ARRAY, _f32_array(list(bank.data)))
        for name, bank in banks.items()
    ]


def _write_model(path: Path, entries: list[tuple[str, int, bytes]]) -> None:
    path.write_bytes(_gguf(entries))


def _valid_banks() -> dict[str, StateBank]:
    return {
        "memory.keys": StateBank(
            "memory.keys", (2, 3), (0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
        ),
        "memory.values": StateBank("memory.values", (2,), (4.0, 7.0)),
        "memory.projection": StateBank(
            "memory.projection",
            (3, 4),
            (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        ),
        "memory.projection_bias": StateBank(
            "memory.projection_bias", (3,), (0.0, 0.0, 0.0)
        ),
        "memory.logit_bias": StateBank("memory.logit_bias", (1,), (10.0,)),
        "memory.min_similarity": StateBank(
            "memory.min_similarity", (1,), (0.5,)
        ),
    }


class StateMemoryTest(unittest.TestCase):
    def test_readout_uses_the_nearest_key(self) -> None:
        memory = StateMemory.from_banks(_valid_banks(), n_embd=4, n_vocab=16)
        assert memory is not None
        self.assertEqual(memory.readout([1.0, 0.0, 0.0, 0.0]), (7, 10.0))
        self.assertEqual(memory.readout([0.0, 1.0, 0.0, 0.0]), (4, 10.0))
        self.assertIsNone(memory.readout([0.0, 0.0, 0.0, 0.0]))
        self.assertIsNone(memory.readout([-1.0, 0.0, 0.0, 0.0]))

    def test_summary_has_no_payload(self) -> None:
        memory = StateMemory.from_banks(_valid_banks(), n_embd=4, n_vocab=16)
        assert memory is not None
        summary = memory.summary()
        self.assertEqual(summary["keys"], 2)
        self.assertEqual(summary["key_dim"], 3)
        self.assertNotIn("values", summary)
        self.assertNotIn("projection", summary)

    def test_no_memory_is_none(self) -> None:
        self.assertIsNone(StateMemory.from_banks({}, n_embd=4, n_vocab=16))

    def test_partial_or_unknown_memory_is_rejected(self) -> None:
        banks = _valid_banks()
        banks.pop("memory.logit_bias")
        with self.assertRaises(StateMemoryError):
            StateMemory.from_banks(banks, n_embd=4, n_vocab=16)
        banks = _valid_banks()
        banks["memory.extra"] = StateBank("memory.extra", (1,), (0.0,))
        with self.assertRaises(StateMemoryError):
            StateMemory.from_banks(banks, n_embd=4, n_vocab=16)

    def test_non_unit_key_is_rejected(self) -> None:
        banks = _valid_banks()
        banks["memory.keys"] = StateBank(
            "memory.keys", (2, 3), (2.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        )
        with self.assertRaises(StateMemoryError):
            StateMemory.from_banks(banks, n_embd=4, n_vocab=16)

    def test_non_integer_or_out_of_range_token_is_rejected(self) -> None:
        banks = _valid_banks()
        banks["memory.values"] = StateBank("memory.values", (2,), (4.5, 7.0))
        with self.assertRaises(StateMemoryError):
            StateMemory.from_banks(banks, n_embd=4, n_vocab=16)
        banks["memory.values"] = StateBank("memory.values", (2,), (4.0, 99.0))
        with self.assertRaises(StateMemoryError):
            StateMemory.from_banks(banks, n_embd=4, n_vocab=16)

    def test_wrong_projection_shape_is_rejected(self) -> None:
        banks = _valid_banks()
        banks["memory.projection"] = StateBank(
            "memory.projection", (3, 3), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        )
        with self.assertRaises(StateMemoryError):
            StateMemory.from_banks(banks, n_embd=4, n_vocab=16)


class FakeSpike:
    """The smallest patched-lib stand-in the engine needs for one token."""

    n_embd = 4
    n_vocab = 16

    def __init__(self, _lib: str, _shim: str, _model: str) -> None:
        self.last_hidden = [0.0] * self.n_embd
        self.last_logits = [0.0] * self.n_vocab

    def new_context(self, **_kwargs: object) -> object:
        return object()

    def chat_prompt(self, _pairs: object, add_assistant: bool = True) -> str:
        self.assert_assistant = add_assistant
        return "prompt"

    def tokenize(self, text: str, parse_special: bool = False) -> list[int]:
        return [1, 2] if text == "prompt" else [1]

    def clear_context(self, _context: object) -> None:
        pass

    def decode(
        self,
        _context: object,
        _tokens: list[int],
        all_outputs: bool = False,
        start_pos: int = 0,
    ) -> None:
        self.last_hidden = [1.0, 0.0, 0.0, 0.0]
        self.last_logits = [0.0] * self.n_vocab
        self.last_logits[15] = 5.0

    def read_embeddings(self, _context: object, _index: int) -> list[float]:
        return list(self.last_hidden)

    def read_logits(self, _context: object, _index: int) -> list[float]:
        return list(self.last_logits)

    def sample_biased(
        self, _context: object, biases: list[tuple[int, float]]
    ) -> int:
        scores = list(self.last_logits)
        for token, bias in biases:
            scores[token] += bias
        return max(range(self.n_vocab), key=scores.__getitem__)

    def token_is_eog(self, _token: int) -> bool:
        return False

    def detokenize(self, tokens: list[int]) -> str:
        return "memory" if tokens == [7] else "base"

    def close(self) -> None:
        pass


class LiveEngineStateMemoryTest(unittest.TestCase):
    def _engine(self, model: Path) -> LiveEngine:
        return LiveEngine(
            str(model),
            "unused-lib",
            "unused-shim",
            n_ctx=8,
            n_batch=4,
            n_threads=1,
            compute_model_digest=False,
        )

    def test_engine_consumes_the_bank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            _write_model(model, _memory_entries())
            with mock.patch("pmc.runtime.live_engine.LlamaSpike", FakeSpike):
                engine = self._engine(model)
                try:
                    result = engine.chat(
                        [{"role": "user", "content": "remember"}],
                        max_tokens=1,
                        learn=False,
                    )
                    self.assertEqual(result.text, "memory")
                    self.assertIsNotNone(engine.state_memory)
                    summary = engine.state_summary()
                    assert isinstance(summary["state_memory"], dict)
                    self.assertEqual(summary["state_memory"]["keys"], 2)
                    self.assertEqual(summary["state_memory"]["logit_bias"], 10.0)
                    self.assertNotIn("values", summary["state_memory"])
                finally:
                    engine.close()

    def test_engine_without_the_bank_uses_the_base_logits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            _write_model(model, [])
            with mock.patch("pmc.runtime.live_engine.LlamaSpike", FakeSpike):
                engine = self._engine(model)
                try:
                    result = engine.chat(
                        [{"role": "user", "content": "remember"}],
                        max_tokens=1,
                        learn=False,
                    )
                    self.assertEqual(result.text, "base")
                    self.assertIsNone(engine.state_memory)
                    self.assertIsNone(engine.state_summary()["state_memory"])
                finally:
                    engine.close()


if __name__ == "__main__":
    unittest.main()
