# SPDX-License-Identifier: GPL-3.0-only
"""Read PMC state banks from a fused GGUF file.

The fusion CLI stores state banks as two metadata arrays per bank:

    pmc.state.<name>.dims   ARRAY[UINT32]
    pmc.state.<name>.data   ARRAY[FLOAT32]

This module reads only the GGUF header and the metadata key/value section. It
never reads the tensor data, so opening a multi-gigabyte fused model costs the
metadata section, not the model. The reader is intentionally strict: a bank
with missing fields, a non-positive dimension, a wrong element count, or a
non-finite value is rejected instead of being passed to the runtime.
"""

from __future__ import annotations

import math
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_GGUF_MAGIC = b"GGUF"
_STATE_PREFIX = "pmc.state."
_SUPPORTED_VERSIONS = (2, 3)
_MAX_METADATA_KEYS = 1_000_000
_MAX_ARRAY_ELEMENTS = 100_000_000

# GGUF metadata value types.
_TYPE_UINT8 = 0
_TYPE_INT8 = 1
_TYPE_UINT16 = 2
_TYPE_INT16 = 3
_TYPE_UINT32 = 4
_TYPE_INT32 = 5
_TYPE_FLOAT32 = 6
_TYPE_BOOL = 7
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_TYPE_UINT64 = 10
_TYPE_INT64 = 11
_TYPE_FLOAT64 = 12

_SCALAR_FORMATS = {
    _TYPE_UINT8: "<B",
    _TYPE_INT8: "<b",
    _TYPE_UINT16: "<H",
    _TYPE_INT16: "<h",
    _TYPE_UINT32: "<I",
    _TYPE_INT32: "<i",
    _TYPE_FLOAT32: "<f",
    _TYPE_BOOL: "<?",
    _TYPE_UINT64: "<Q",
    _TYPE_INT64: "<q",
    _TYPE_FLOAT64: "<d",
}

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")


class GgufFormatError(ValueError):
    """The file is not a readable GGUF metadata section."""


@dataclass(frozen=True)
class StateBank:
    """One state bank read from ``pmc.state.<name>`` metadata."""

    name: str
    dims: tuple[int, ...]
    data: tuple[float, ...]

    @property
    def elements(self) -> int:
        return len(self.data)

    def to_array(self) -> array:
        """Return the bank as a float32 array for the runtime."""
        return array("f", self.data)


def _read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    if size < 0:
        raise GgufFormatError(f"{label}: negative size")
    raw = stream.read(size)
    if len(raw) != size:
        raise GgufFormatError(f"{label}: truncated file")
    return raw


def _read_u32(stream: BinaryIO, label: str) -> int:
    return _U32.unpack(_read_exact(stream, _U32.size, label))[0]


def _read_u64(stream: BinaryIO, label: str) -> int:
    return _U64.unpack(_read_exact(stream, _U64.size, label))[0]


def _read_string(stream: BinaryIO, label: str) -> str:
    length = _read_u64(stream, f"{label} length")
    if length > _MAX_ARRAY_ELEMENTS:
        raise GgufFormatError(f"{label}: string is too long")
    raw = _read_exact(stream, length, label)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GgufFormatError(f"{label}: invalid UTF-8") from error


def _read_array(stream: BinaryIO, label: str) -> list[object]:
    item_type = _read_u32(stream, f"{label} item type")
    count = _read_u64(stream, f"{label} count")
    if count > _MAX_ARRAY_ELEMENTS:
        raise GgufFormatError(f"{label}: array is too long")

    scalar_format = _SCALAR_FORMATS.get(item_type)
    if scalar_format is not None:
        item_size = struct.calcsize(scalar_format)
        raw = _read_exact(stream, item_size * count, label)
        if count == 0:
            return []
        return list(struct.unpack(f"<{count}{scalar_format[1:]}", raw))

    return [_read_value(stream, item_type, f"{label}[{index}]") for index in range(count)]


def _read_value(stream: BinaryIO, value_type: int, label: str) -> object:
    if value_type == _TYPE_STRING:
        return _read_string(stream, label)
    if value_type == _TYPE_ARRAY:
        return _read_array(stream, label)
    scalar_format = _SCALAR_FORMATS.get(value_type)
    if scalar_format is None:
        raise GgufFormatError(f"{label}: unknown value type {value_type}")
    raw = _read_exact(stream, struct.calcsize(scalar_format), label)
    return struct.unpack(scalar_format, raw)[0]


def _as_dims(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise GgufFormatError(f"{name}: dims must be a non-empty array")
    dims: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise GgufFormatError(f"{name}: dims must contain positive integers")
        dims.append(item)
    return tuple(dims)


def _as_data(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise GgufFormatError(f"{name}: data must be an array")
    data: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise GgufFormatError(f"{name}: data must contain numbers")
        number = float(item)
        if not math.isfinite(number):
            raise GgufFormatError(f"{name}: data contains a non-finite value")
        data.append(number)
    return tuple(data)


def read_state_banks(path: str | Path) -> dict[str, StateBank]:
    """Read every ``pmc.state.*`` bank from a fused GGUF file.

    The returned mapping is ordered by the metadata order in the file. Missing
    or malformed banks raise :class:`GgufFormatError`; a file with no state
    metadata returns an empty mapping. Tensor data is never read.
    """
    path = Path(path)
    with path.open("rb") as stream:
        magic = _read_exact(stream, 4, "magic")
        if magic != _GGUF_MAGIC:
            raise GgufFormatError("magic: not a GGUF file")
        version = _read_u32(stream, "version")
        if version not in _SUPPORTED_VERSIONS:
            raise GgufFormatError(f"version: unsupported GGUF version {version}")
        _read_u64(stream, "tensor count")
        metadata_count = _read_u64(stream, "metadata count")
        if metadata_count > _MAX_METADATA_KEYS:
            raise GgufFormatError("metadata count: too large")

        values: dict[str, object] = {}
        for index in range(metadata_count):
            key = _read_string(stream, f"metadata[{index}] key")
            value_type = _read_u32(stream, f"metadata[{index}] type")
            value = _read_value(stream, value_type, f"metadata[{index}] value")
            if key.startswith(_STATE_PREFIX):
                values[key] = value

    dims: dict[str, tuple[int, ...]] = {}
    data: dict[str, tuple[float, ...]] = {}
    for key, value in values.items():
        suffix = key[len(_STATE_PREFIX) :]
        if suffix.endswith(".dims"):
            dims[suffix[: -len(".dims")]] = _as_dims(value, key)
        elif suffix.endswith(".data"):
            data[suffix[: -len(".data")]] = _as_data(value, key)
        else:
            raise GgufFormatError(f"{key}: state key must end in .dims or .data")

    if set(dims) != set(data):
        missing_dims = sorted(set(data) - set(dims))
        missing_data = sorted(set(dims) - set(data))
        raise GgufFormatError(
            f"state banks are incomplete: missing dims {missing_dims}, "
            f"missing data {missing_data}"
        )

    banks: dict[str, StateBank] = {}
    for name in dims:
        bank_dims = dims[name]
        bank_data = data[name]
        expected = math.prod(bank_dims)
        if len(bank_data) != expected:
            raise GgufFormatError(
                f"pmc.state.{name}: {len(bank_data)} values do not match dims {bank_dims}"
            )
        banks[name] = StateBank(name=name, dims=bank_dims, data=bank_data)
    return banks
