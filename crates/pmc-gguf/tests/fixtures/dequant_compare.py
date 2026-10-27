#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Dump dequantized GGUF rows as raw little-endian float32.

Helper for the ignored Rust test `dequant_matches_gguf_py`. It is never used
by the normal test suite and deliberately has no hard-coded paths: both the
model and the directory containing the `gguf` package are passed on the
command line.

stdout: exactly `rows * row_len` float32 values, row-major.
stderr: diagnostics.
"""

import argparse
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-py", required=True,
                        help="directory that contains the gguf package")
    parser.add_argument("--model", required=True, help="GGUF file")
    parser.add_argument("--tensor", required=True, help="tensor name")
    parser.add_argument("--rows", type=int, default=4, help="rows to dump")
    args = parser.parse_args()

    sys.path.insert(0, args.gguf_py)
    from gguf import GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(args.model)
    matches = [t for t in reader.tensors if t.name == args.tensor]
    if not matches:
        print(f"tensor not found: {args.tensor}", file=sys.stderr)
        return 2
    tensor = matches[0]
    if args.rows > tensor.data.shape[0]:
        print(f"only {tensor.data.shape[0]} rows available", file=sys.stderr)
        return 2

    block = np.ascontiguousarray(tensor.data[: args.rows])
    values = dequantize(block, tensor.tensor_type)
    values = np.asarray(values, dtype="<f4").reshape(-1)
    print(
        f"tensor={tensor.name} type={tensor.tensor_type.name} "
        f"rows={args.rows} values={values.size}",
        file=sys.stderr,
    )
    sys.stdout.buffer.write(values.tobytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
