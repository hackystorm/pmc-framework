# SPDX-License-Identifier: GPL-3.0-only
"""Tests for the fused-GGUF state-bank reader."""

from __future__ import annotations

import math
import os
import struct
import tempfile
import unittest
from pathlib import Path

from pmc.runtime.state_banks import GgufFormatError, read_state_banks

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


def _build(entries: list[tuple[str, int, bytes]], *, version: int = 3) -> bytes:
    out = bytearray(b"GGUF")
    out += struct.pack("<IQQ", version, 0, len(entries))
    for key, value_type, raw in entries:
        out += _string(key)
        out += struct.pack("<I", value_type)
        out += raw
    return bytes(out)


def _state_entries(
    name: str = "bank",
    dims: list[int] | None = None,
    data: list[float] | None = None,
) -> list[tuple[str, int, bytes]]:
    dims = [2, 4] if dims is None else dims
    data = [0.5, -1.0, 2.25, 3.0, -0.125, 4.5, -5.0, 0.75] if data is None else data
    return [
        (f"pmc.state.{name}.dims", _TYPE_ARRAY, _u32_array(dims)),
        (f"pmc.state.{name}.data", _TYPE_ARRAY, _f32_array(data)),
    ]


class StateBankReaderTest(unittest.TestCase):
    def _read(self, raw: bytes):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            path.write_bytes(raw)
            return read_state_banks(path)

    def test_reads_one_bank(self) -> None:
        banks = self._read(_build(_state_entries()))
        self.assertEqual(list(banks), ["bank"])
        bank = banks["bank"]
        self.assertEqual(bank.dims, (2, 4))
        self.assertEqual(bank.elements, 8)
        self.assertEqual(bank.data[0], 0.5)
        self.assertEqual(bank.data[-1], 0.75)
        self.assertEqual(list(bank.to_array()), list(bank.data))

    def test_no_state_metadata_is_empty(self) -> None:
        raw = _build([("general.architecture", 8, _string("lfm2"))])
        self.assertEqual(self._read(raw), {})

    def test_missing_data_is_rejected(self) -> None:
        entries = _state_entries()
        with self.assertRaises(GgufFormatError):
            self._read(_build(entries[:1]))

    def test_wrong_element_count_is_rejected(self) -> None:
        entries = _state_entries(data=[0.5, -1.0])
        with self.assertRaises(GgufFormatError):
            self._read(_build(entries))

    def test_non_positive_dimension_is_rejected(self) -> None:
        with self.assertRaises(GgufFormatError):
            self._read(_build(_state_entries(dims=[0, 8])))

    def test_non_finite_value_is_rejected(self) -> None:
        with self.assertRaises(GgufFormatError):
            self._read(_build(_state_entries(data=[math.nan] * 8)))

    def test_unknown_state_key_suffix_is_rejected(self) -> None:
        entries = _state_entries()
        entries.append(("pmc.state.bank.extra", _TYPE_UINT32, struct.pack("<I", 1)))
        with self.assertRaises(GgufFormatError):
            self._read(_build(entries))

    def test_bad_magic_is_rejected(self) -> None:
        with self.assertRaises(GgufFormatError):
            self._read(b"NOPE" + _build([])[4:])

    def test_truncated_metadata_is_rejected(self) -> None:
        with self.assertRaises(GgufFormatError):
            self._read(_build([])[:-1])

    @unittest.skipUnless(
        os.environ.get("PMC_STATE_GGUF"),
        "set PMC_STATE_GGUF to a fused file to run the integration check",
    )
    def test_real_fused_file(self) -> None:
        banks = read_state_banks(os.environ["PMC_STATE_GGUF"])
        self.assertIn("bank", banks)
        bank = banks["bank"]
        self.assertEqual(bank.dims, (2, 4))
        self.assertEqual(bank.elements, 8)
        self.assertTrue(all(math.isfinite(value) for value in bank.data))


if __name__ == "__main__":
    unittest.main()
