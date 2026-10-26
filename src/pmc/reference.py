# SPDX-License-Identifier: GPL-3.0-only

"""Reference implementation of the trainer, in pure Python.

This module exists to make the Rust trainer trustworthy. It is a second,
independent implementation of the same mathematics, written to be read
rather than to be fast, and the test suite requires the two to agree on the
weights (within 1e-5) and on the metrics (within 1e-3).

It has no dependencies beyond the standard library, and it rounds every
arithmetic operation to 32-bit precision so that it tracks the Rust
implementation rather than merely approximating it.

The reference is not throwaway work. It is what makes the Rust code
auditable: a reader who distrusts the fast implementation can run the slow
one and check the claim.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Sequence

__all__ = [
    "ReferenceConfig",
    "ReferenceReport",
    "ReferenceTrainer",
    "f32",
]


def f32(value: float) -> float:
    """Round a Python float to the nearest 32-bit float.

    The Rust implementation works in ``f32`` throughout. Python works in
    ``f64``, so every operation is rounded explicitly to keep the two
    implementations comparable step by step rather than merely at the end.
    """
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _mul(a: float, b: float) -> float:
    return f32(f32(a) * f32(b))


def _add(a: float, b: float) -> float:
    return f32(f32(a) + f32(b))


def _div(a: float, b: float) -> float:
    return f32(f32(a) / f32(b))


class ReferenceRng:
    """xorshift64*, identical to the Rust implementation."""

    _MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed if seed != 0 else 0x9E3779B97F4A7C15

    def next_u64(self) -> int:
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & self._MASK
        x ^= x >> 27
        self.state = x & self._MASK
        return (x * 0x2545F4914F6CDD1D) & self._MASK

    def next_f32(self) -> float:
        return (self.next_u64() >> 40) / float(1 << 24)

    def normal(self) -> float:
        u1 = max(self.next_f32(), 1e-7)
        u2 = self.next_f32()
        return f32(
            f32(math.sqrt(f32(f32(-2.0) * f32(math.log(u1)))))
            * f32(math.cos(f32(f32(math.tau) * f32(u2))))
        )


@dataclass
class ReferenceConfig:
    """The training configuration, mirroring the Rust `TrainConfig`."""

    seed: int = 20261026
    steps: int = 200
    learning_rate: float = 0.05
    hidden: int = 32
    keys: int = 16


@dataclass
class ReferenceReport:
    """What the reference run measured."""

    seed: int
    steps: int
    learning_rate: float
    hidden: int
    keys: int
    initial_accuracy: float
    final_accuracy: float
    final_loss: float
    weight_digest: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "seed": self.seed,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "hidden": self.hidden,
            "keys": self.keys,
            "initial_accuracy": self.initial_accuracy,
            "final_accuracy": self.final_accuracy,
            "final_loss": self.final_loss,
            "weight_digest": self.weight_digest,
        }


def _matrix(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    inner = len(a[0])
    cols = len(b[0])
    out = _matrix(rows, cols)
    for i in range(rows):
        for p in range(inner):
            left = a[i][p]
            if left == 0.0:
                continue
            for j in range(cols):
                out[i][j] = _add(out[i][j], _mul(left, b[p][j]))
    return out


def _transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*a)]


def _add_bias(a: list[list[float]], bias: list[list[float]]) -> list[list[float]]:
    out = [list(row) for row in a]
    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] = _add(out[i][j], bias[0][j])
    return out


def _softmax_rows(logits: list[list[float]]) -> list[list[float]]:
    out = _matrix(len(logits), len(logits[0]))
    for i, row in enumerate(logits):
        maximum = max(row)
        if not math.isfinite(maximum):
            for j in range(len(row)):
                out[i][j] = f32(1.0 / f32(len(row)))
            continue
        total = 0.0
        for j, value in enumerate(row):
            exponent = f32(math.exp(f32(value - maximum)))
            out[i][j] = exponent
            total = _add(total, exponent)
        for j in range(len(row)):
            out[i][j] = _div(out[i][j], total)
    return out


def _target_index(target: list[float]) -> int:
    best = 0
    best_value = -math.inf
    for index, value in enumerate(target):
        if value > best_value:
            best_value = value
            best = index
    return best


class ReferenceTrainer:
    """A two-layer perceptron trained with Adam on a synthetic memory task."""

    def __init__(self, config: ReferenceConfig) -> None:
        self.config = config
        rng = ReferenceRng(config.seed)

        scale1 = f32(math.sqrt(f32(f32(2.0) / f32(config.keys))))
        scale2 = f32(math.sqrt(f32(f32(2.0) / f32(config.hidden))))

        self.w1 = [[f32(rng.normal() * scale1) for _ in range(config.hidden)] for _ in range(config.keys)]
        self.b1 = [[0.0] * config.hidden]
        self.w2 = [[f32(rng.normal() * scale2) for _ in range(config.keys)] for _ in range(config.hidden)]
        self.b2 = [[0.0] * config.keys]

        self.task = self._build_task(config.keys, config.seed)

        self.m = [_matrix(*shape) for shape in self._param_shapes()]
        self.v = [_matrix(*shape) for shape in self._param_shapes()]
        self.step_count = 0

    # -- task --------------------------------------------------------------

    @staticmethod
    def _build_task(keys: int, seed: int) -> list[int]:
        mapping = list(range(keys))
        rng = ReferenceRng((seed ^ 0x5DEECE66D5A41B0D) & ((1 << 64) - 1))
        for i in range(keys - 1, 0, -1):
            j = rng.next_u64() % (i + 1)
            mapping[i], mapping[j] = mapping[j], mapping[i]
        return mapping

    def _batch(self, keys: Sequence[int]) -> tuple[list[list[float]], list[list[float]]]:
        inputs = _matrix(len(keys), self.config.keys)
        targets = _matrix(len(keys), self.config.keys)
        for index, key in enumerate(keys):
            inputs[index][key % self.config.keys] = 1.0
            targets[index][self.task[key % self.config.keys]] = 1.0
        return inputs, targets

    # -- parameters --------------------------------------------------------

    def _param_shapes(self) -> list[tuple[int, int]]:
        return [
            (self.config.keys, self.config.hidden),
            (1, self.config.hidden),
            (self.config.hidden, self.config.keys),
            (1, self.config.keys),
        ]

    def _params(self) -> list[list[list[float]]]:
        return [self.w1, self.b1, self.w2, self.b2]

    # -- forward and backward ---------------------------------------------

    def forward(self, x: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        z1 = _add_bias(_matmul(x, self.w1), self.b1)
        h = [[value if value > 0.0 else 0.0 for value in row] for row in z1]
        logits = _add_bias(_matmul(h, self.w2), self.b2)
        probs = _softmax_rows(logits)
        return z1, h, probs

    def backward(
        self,
        x: list[list[float]],
        z1: list[list[float]],
        h: list[list[float]],
        probs: list[list[float]],
        targets: list[list[float]],
    ) -> list[list[list[float]]]:
        rows = len(probs)
        cols = len(probs[0])

        d_logits = [list(row) for row in probs]
        for i in range(rows):
            target = _target_index(targets[i])
            d_logits[i][target] = f32(d_logits[i][target] - 1.0)
        scale = _div(1.0, f32(rows))
        d_logits = [[_mul(value, scale) for value in row] for row in d_logits]

        dw2 = _matmul(_transpose(h), d_logits)
        db2 = [[0.0] * cols]
        for i in range(rows):
            for j in range(cols):
                db2[0][j] = _add(db2[0][j], d_logits[i][j])

        d_h = _matmul(d_logits, _transpose(self.w2))
        d_z1 = [list(row) for row in d_h]
        for i in range(len(d_z1)):
            for j in range(len(d_z1[0])):
                if z1[i][j] <= 0.0:
                    d_z1[i][j] = 0.0

        dw1 = _matmul(_transpose(x), d_z1)
        db1 = [[0.0] * self.config.hidden]
        for i in range(rows):
            for j in range(self.config.hidden):
                db1[0][j] = _add(db1[0][j], d_z1[i][j])

        return [dw1, db1, dw2, db2]

    # -- optimiser ---------------------------------------------------------

    def _adam_step(self, grads: list[list[list[float]]]) -> None:
        learning_rate = f32(self.config.learning_rate)
        beta1 = f32(0.9)
        beta2 = f32(0.999)
        epsilon = f32(1e-8)

        self.step_count += 1
        # Adam's bias correction divides the moment estimates by
        # `1 - beta^step`, which for the first step is `1 - beta` itself.
        correction1 = f32(f32(1.0) - f32(beta1**self.step_count))
        correction2 = f32(f32(1.0) - f32(beta2**self.step_count))

        for index, param in enumerate(self._params()):
            grad = grads[index]
            for i in range(len(param)):
                for j in range(len(param[0])):
                    g = grad[i][j]
                    self.m[index][i][j] = _add(
                        _mul(beta1, self.m[index][i][j]),
                        _mul(f32(f32(1.0) - beta1), g),
                    )
                    self.v[index][i][j] = _add(
                        _mul(beta2, self.v[index][i][j]),
                        _mul(_mul(f32(f32(1.0) - beta2), g), g),
                    )
                    m_hat = _div(self.m[index][i][j], correction1)
                    v_hat = _div(self.v[index][i][j], correction2)
                    denominator = _add(f32(math.sqrt(v_hat)), epsilon)
                    param[i][j] = f32(
                        f32(param[i][j] - _div(_mul(learning_rate, m_hat), denominator))
                    )

    # -- measurement -------------------------------------------------------

    def accuracy(self) -> float:
        correct = 0
        for key in range(self.config.keys):
            x = _matrix(1, self.config.keys)
            x[0][key] = 1.0
            _, _, probs = self.forward(x)
            best = 0
            best_value = -math.inf
            for index, value in enumerate(probs[0]):
                if value > best_value:
                    best_value = value
                    best = index
            if best == self.task[key]:
                correct += 1
        return correct / self.config.keys

    def loss(self, keys: Sequence[int]) -> float:
        inputs, targets = self.batch_targets(keys)
        _, _, probs = self.forward(inputs)
        total = 0.0
        for i in range(len(probs)):
            target = _target_index(targets[i])
            total = _add(total, f32(-math.log(max(probs[i][target], 1e-9))))
        return _div(total, f32(len(probs)))

    def batch_targets(self, keys: Sequence[int]) -> tuple[list[list[float]], list[list[float]]]:
        return self._batch(keys)

    def weight_bytes(self) -> bytes:
        chunks: list[bytes] = []
        for param in self._params():
            for row in param:
                for value in row:
                    chunks.append(struct.pack("<f", value))
        return b"".join(chunks)

    def weight_digest(self) -> str:
        return hashlib.sha256(self.weight_bytes()).hexdigest()

    # -- training ----------------------------------------------------------

    def train(self) -> ReferenceReport:
        config = self.config
        initial_accuracy = self.accuracy()
        all_keys = list(range(config.keys))

        for _ in range(config.steps):
            inputs, targets = self._batch(all_keys)
            z1, h, probs = self.forward(inputs)
            grads = self.backward(inputs, z1, h, probs, targets)
            self._adam_step(grads)

        return ReferenceReport(
            seed=config.seed,
            steps=config.steps,
            learning_rate=config.learning_rate,
            hidden=config.hidden,
            keys=config.keys,
            initial_accuracy=initial_accuracy,
            final_accuracy=self.accuracy(),
            final_loss=self.loss(all_keys),
            weight_digest=self.weight_digest(),
        )
