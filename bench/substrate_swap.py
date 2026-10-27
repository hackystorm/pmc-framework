# SPDX-License-Identifier: GPL-3.0-only
"""PMC-mini Substrate Swap Test.

The test is pre-registered in ``docs/lab.md`` and its frozen word list is in
``bench/swap_words.json``.  The state is a small key/value memory (8 keys and
8 value indices) whose bytes are identical on two hosts, the 350M and the 8B.
Each host has a fixed projection and a linear adapter that maps its hidden
state into the shared key space; those adapters are the host interface, not
the state.

The subcommands are intentionally separate so only one model is loaded at a
time:

    extract  hidden states for the frozen prompts, one host, one model
    train    the two adapters with InfoNCE on the 22 training facts
    fuse     write the shared state and one host adapter into a GGUF
    eval     read the state from two fused GGUFs and run every arm

The script is standard library only.  Model paths are arguments and never
appear in the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from pmc.runtime.live_spike import LlamaSpike
from pmc.runtime.state_banks import read_state_banks

SPEC_PATH = Path(__file__).with_name("swap_words.json")
STATE_SCHEMA = "pmc.state.swap/1"
ADAPTER_SCHEMA = "pmc.swap.adapter/1"
EVAL_SCHEMA = "pmc-swap-eval/1"
F32 = struct.Struct("<f")


class SwapError(RuntimeError):
    """The frozen specification or a measured artifact is inconsistent."""


# -- specification ---------------------------------------------------------


def load_spec(path: str | Path = SPEC_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        spec = json.load(stream)
    spec["spec_sha256"] = spec_digest(path)
    validate_spec(spec)
    return spec


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema") != "pmc-swap-words/1":
        raise SwapError("unsupported swap word-list schema")
    hosts = spec.get("hosts")
    if hosts != ["350m", "8b"]:
        raise SwapError("the pre-registered hosts are 350m and 8b")
    training = spec.get("training_facts")
    evaluation = spec.get("evaluation_facts")
    controls = spec.get("control_questions")
    try:
        training_count = int(spec.get("training_count", 22))
        word_count = int(spec.get("word_count", 68))
    except (TypeError, ValueError) as error:
        raise SwapError("training_count and word_count must be integers") from error
    if not isinstance(training, list) or len(training) != training_count:
        raise SwapError(f"the frozen training set has {training_count} facts")
    if not isinstance(evaluation, list) or len(evaluation) != 8:
        raise SwapError("the frozen evaluation set has 8 facts")
    if not isinstance(controls, list) or len(controls) != 8:
        raise SwapError("the frozen control set has 8 questions")
    words = spec.get("words")
    if (
        not isinstance(words, list)
        or len(words) != word_count
        or len(set(words)) != word_count
    ):
        raise SwapError(f"the frozen word list has {word_count} distinct words")
    seen: set[str] = set()
    for fact in training:
        cue = fact["cue"]
        value = fact["value"]
        if cue in seen or value in seen or cue == value:
            raise SwapError("training cue/value words must be globally distinct")
        seen.update((cue, value))
    for fact in evaluation:
        cue = fact["cue"]
        value = fact["value"]
        if cue in seen or value in seen or cue == value:
            raise SwapError("evaluation cue/value words must be globally distinct")
        seen.update((cue, value))
    for control in controls:
        cue = control["cue"]
        if cue in seen:
            raise SwapError("control cues must be unseen words")
        if not 0 <= int(control["paired_eval"]) < 8:
            raise SwapError("control paired_eval must index an evaluation fact")
        seen.add(cue)
    if seen != set(words):
        raise SwapError("the split does not cover the frozen word list exactly")
    templates = spec.get("templates")
    if not isinstance(templates, dict):
        raise SwapError("templates must be an object")
    for name in ("fact", "question", "reinjection"):
        text = templates.get(name)
        if not isinstance(text, str) or "{cue}" not in text:
            raise SwapError(f"template {name} must contain {{cue}}")
    for name in ("fact", "reinjection"):
        if "{value}" not in templates[name]:
            raise SwapError(f"template {name} must contain {{value}}")
    projection = spec.get("projection")
    if not isinstance(projection, dict) or projection.get("output_dim") != 128:
        raise SwapError("the frozen projection output is 128")
    if projection.get("input_dim") != {"350m": 1024, "8b": 2048}:
        raise SwapError("the host hidden dimensions do not match the frozen spec")
    adapter = spec.get("adapter")
    if not isinstance(adapter, dict):
        raise SwapError("adapter must be an object")
    try:
        adapter_output = int(adapter.get("output_dim", 64))
    except (TypeError, ValueError) as error:
        raise SwapError("adapter.output_dim must be an integer") from error
    if adapter_output not in (16, 64):
        raise SwapError("the adapter output is 16 or 64")
    if float(spec.get("bias", 0.0)) <= 0.0:
        raise SwapError("the readout bias must be positive")
    state = spec.get("state_schema")
    if state != STATE_SCHEMA:
        raise SwapError("the state schema does not match the reader")
    anchor = spec.get("anchor", "last")
    if anchor not in ("last", "cue", "cue_delta"):
        raise SwapError("the anchor must be 'last', 'cue' or 'cue_delta'")
    if anchor == "cue_delta":
        neutral = spec.get("neutral_cue")
        if not isinstance(neutral, str) or not neutral or " " in neutral:
            raise SwapError("cue_delta needs one neutral_cue word")
        if neutral in seen:
            raise SwapError("the neutral cue must not be a frozen word")
    if not isinstance(adapter.get("early_stopping", True), bool):
        raise SwapError("adapter.early_stopping must be a boolean")


def fact_prompt(spec: dict[str, Any], cue: str, value: str) -> str:
    return spec["templates"]["fact"].format(cue=cue, value=value)


def question_prompt(spec: dict[str, Any], cue: str) -> str:
    return spec["templates"]["question"].format(cue=cue)


def reinjection_prompt(spec: dict[str, Any], cue: str, value: str) -> str:
    return spec["templates"]["reinjection"].format(cue=cue, value=value)


def cue_start(prompt: str, cue: str) -> int:
    """The one character offset of a whole cue word in its prompt."""
    if not cue:
        raise SwapError("empty cue")
    start = prompt.find(cue)
    if start < 0:
        raise SwapError(f"cue {cue!r} is not in the prompt")
    if prompt.find(cue, start + len(cue)) >= 0:
        raise SwapError(f"cue {cue!r} occurs more than once in the prompt")
    return start


def neutral_prompt(prompt: str, cue: str, neutral: str) -> str:
    start = cue_start(prompt, cue)
    return prompt[:start] + neutral + prompt[start + len(cue) :]


def prompt_plan(spec: dict[str, Any]) -> Iterable[tuple[str, int, str, str, str]]:
    """Every frozen extraction prompt, in a deterministic order."""
    for index, fact in enumerate(spec["training_facts"]):
        yield (
            "training_fact",
            index,
            fact["cue"],
            fact["value"],
            fact_prompt(spec, fact["cue"], fact["value"]),
        )
        yield (
            "training_question",
            index,
            fact["cue"],
            fact["value"],
            question_prompt(spec, fact["cue"]),
        )
    for index, fact in enumerate(spec["evaluation_facts"]):
        yield (
            "eval_fact",
            index,
            fact["cue"],
            fact["value"],
            fact_prompt(spec, fact["cue"], fact["value"]),
        )
        yield (
            "eval_question",
            index,
            fact["cue"],
            fact["value"],
            question_prompt(spec, fact["cue"]),
        )
    for index, control in enumerate(spec["control_questions"]):
        yield (
            "control_question",
            index,
            control["cue"],
            "",
            question_prompt(spec, control["cue"]),
        )


def spec_digest(path: str | Path = SPEC_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- vectors and projections -----------------------------------------------


def normalize(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SwapError("cannot normalize a zero or non-finite vector")
    return [float(value) / norm for value in values]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise SwapError("dot product length mismatch")
    return sum(float(a) * float(b) for a, b in zip(left, right))


def projection_matrix(
    input_dim: int, output_dim: int, seed: int
) -> list[list[float]]:
    rng = random.Random(seed)
    scale = 1.0 / math.sqrt(input_dim)
    return [
        [rng.gauss(0.0, scale) for _ in range(input_dim)]
        for _ in range(output_dim)
    ]


def host_projection(spec: dict[str, Any], host: str) -> list[list[float]]:
    if host not in spec["hosts"]:
        raise SwapError(f"unknown host {host}")
    index = spec["hosts"].index(host)
    seed = int(spec["projection"]["seed"]) + index
    return projection_matrix(
        int(spec["projection"]["input_dim"][host]),
        int(spec["projection"]["output_dim"]),
        seed,
    )


def project_hidden(hidden: Sequence[float], matrix: Sequence[Sequence[float]]) -> list[float]:
    unit = normalize(hidden)
    projected = [dot(row, unit) for row in matrix]
    return normalize(projected)


def nearest(query: Sequence[float], keys: Sequence[Sequence[float]]) -> tuple[int, list[float]]:
    if not keys:
        raise SwapError("nearest needs at least one key")
    scores = [dot(query, key) for key in keys]
    best = max(range(len(scores)), key=lambda index: scores[index])
    return best, scores


def answer_ok(text: str, expected: str) -> bool:
    return text.strip().lower() == expected.strip().lower()


# -- canonical identities --------------------------------------------------


def f32(value: float) -> float:
    return F32.unpack(F32.pack(float(value)))[0]


def _pack_f32(values: Sequence[float]) -> bytes:
    return b"".join(F32.pack(f32(value)) for value in values)


def state_digest(keys: Sequence[Sequence[float]], values: Sequence[float]) -> str:
    digest = hashlib.sha256()
    digest.update(STATE_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    if keys:
        width = len(keys[0])
        digest.update(struct.pack("<II", len(keys), width))
        for row in keys:
            if len(row) != width:
                raise SwapError("state keys are ragged")
            digest.update(_pack_f32(row))
    else:
        digest.update(struct.pack("<II", 0, 0))
    digest.update(struct.pack("<I", len(values)))
    digest.update(_pack_f32(values))
    return digest.hexdigest()


def adapter_digest(
    weights: Sequence[Sequence[float]], bias: Sequence[float]
) -> str:
    digest = hashlib.sha256()
    digest.update(ADAPTER_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    if weights:
        width = len(weights[0])
        digest.update(struct.pack("<II", len(weights), width))
        for row in weights:
            if len(row) != width:
                raise SwapError("adapter weights are ragged")
            digest.update(_pack_f32(row))
    else:
        digest.update(struct.pack("<II", 0, 0))
    digest.update(struct.pack("<I", len(bias)))
    digest.update(_pack_f32(bias))
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# -- extraction ------------------------------------------------------------


def extract_host(
    spec: dict[str, Any],
    host: str,
    model: str,
    lib: str,
    shim: str,
    *,
    n_ctx: int = 128,
    n_batch: int = 128,
    n_threads: int = 2,
    progress: bool = True,
) -> dict[str, Any]:
    if host not in spec["hosts"]:
        raise SwapError(f"unknown host {host}")
    started = time.perf_counter()
    anchor = str(spec.get("anchor", "last"))
    spike = LlamaSpike(lib, shim, model)
    try:
        if spike.n_embd != int(spec["projection"]["input_dim"][host]):
            raise SwapError(
                f"host {host} has n_embd={spike.n_embd}, expected "
                f"{spec['projection']['input_dim'][host]}"
            )
        context = spike.new_context(
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            embeddings=True,
        )
        records: list[dict[str, Any]] = []
        for completed, (kind, index, cue, value, prompt) in enumerate(
            prompt_plan(spec), start=1
        ):
            tokens = spike.tokenize(prompt)
            if not tokens:
                raise SwapError(f"{kind}[{index}] tokenized to nothing")
            if len(tokens) > n_ctx:
                raise SwapError(
                    f"{kind}[{index}] has {len(tokens)} tokens, over n_ctx={n_ctx}"
                )
            spike.clear_context(context)
            spike.decode(context, tokens, all_outputs=True)
            if anchor in ("cue", "cue_delta"):
                start = cue_start(prompt, cue)
                prefix = prompt[:start]
                cue_end = len(spike.tokenize(prefix + cue)) - 1
                if cue_end < 0 or cue_end >= len(tokens):
                    raise SwapError(
                        f"{kind}[{index}] cue end {cue_end} is outside the prompt"
                    )
                hidden = spike.read_embeddings(context, cue_end)
                if anchor == "cue_delta":
                    neutral = str(spec["neutral_cue"])
                    swapped = neutral_prompt(prompt, cue, neutral)
                    neutral_tokens = spike.tokenize(swapped)
                    neutral_end = len(spike.tokenize(prefix + neutral)) - 1
                    if neutral_end < 0 or neutral_end >= len(neutral_tokens):
                        raise SwapError(
                            f"{kind}[{index}] neutral cue end is outside the prompt"
                        )
                    spike.clear_context(context)
                    spike.decode(context, neutral_tokens, all_outputs=True)
                    neutral_hidden = spike.read_embeddings(context, neutral_end)
                    if len(neutral_hidden) != len(hidden):
                        raise SwapError(
                            f"{kind}[{index}] neutral hidden has the wrong length"
                        )
                    hidden = [
                        float(value) - float(neutral_value)
                        for value, neutral_value in zip(hidden, neutral_hidden)
                    ]
            else:
                hidden = spike.read_embeddings(context, -1)
            if len(hidden) != spike.n_embd:
                raise SwapError(
                    f"{kind}[{index}] hidden has {len(hidden)} values, "
                    f"expected {spike.n_embd}"
                )
            norm = math.sqrt(sum(float(x) * float(x) for x in hidden))
            if not math.isfinite(norm) or norm <= 0.0:
                raise SwapError(f"{kind}[{index}] hidden is zero or non-finite")
            records.append(
                {
                    "kind": kind,
                    "index": index,
                    "cue": cue,
                    "value": value,
                    "n_tokens": len(tokens),
                    "prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "h_sha256": hashlib.sha256(_pack_f32(hidden)).hexdigest(),
                    "h": [f32(value) for value in hidden],
                }
            )
            if progress and completed % 10 == 0:
                print(
                    f"{host}: extracted {completed}/{len(list(prompt_plan(spec)))}",
                    file=sys.stderr,
                    flush=True,
                )
        model_sha = sha256_file(model)
        elapsed = time.perf_counter() - started
        return {
            "schema": "pmc-swap-hidden/1",
            "host": host,
            "model": model,
            "model_sha256": model_sha,
            "n_embd": spike.n_embd,
            "n_vocab": spike.n_vocab,
            "spec_sha256": spec["spec_sha256"],
            "anchor": anchor,
            "ctx": n_ctx,
            "records": records,
            "elapsed_s": elapsed,
        }
    finally:
        spike.close()


def extract_command(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    result = extract_host(
        spec,
        args.host,
        args.model,
        args.lib,
        args.shim,
        n_ctx=args.ctx,
        n_batch=args.batch,
        n_threads=args.threads,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "host": result["host"],
                "records": len(result["records"]),
                "n_embd": result["n_embd"],
                "n_vocab": result["n_vocab"],
                "model_sha256": result["model_sha256"],
                "out": str(output),
            },
            indent=1,
        )
    )
    return 0


# -- adapters and InfoNCE --------------------------------------------------


def load_hidden(path: str | Path, spec: dict[str, Any], host: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "pmc-swap-hidden/1":
        raise SwapError(f"{path}: unsupported hidden-state schema")
    if data.get("host") != host:
        raise SwapError(f"{path}: host is {data.get('host')}, expected {host}")
    if data.get("n_embd") != int(spec["projection"]["input_dim"][host]):
        raise SwapError(f"{path}: hidden dimension does not match the spec")
    if data.get("spec_sha256") != spec["spec_sha256"]:
        raise SwapError(f"{path}: hidden states were extracted with another spec")
    if data.get("anchor", "last") != spec.get("anchor", "last"):
        raise SwapError(f"{path}: hidden states use another anchor")
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for record in data["records"]:
        key = (record["kind"], int(record["index"]))
        if key in records:
            raise SwapError(f"{path}: duplicate record {key}")
        if len(record["h"]) != data["n_embd"]:
            raise SwapError(f"{path}: record {key} has the wrong hidden length")
        expected_prompt = None
        if record["kind"] in ("training_fact", "eval_fact"):
            facts = (
                spec["training_facts"]
                if record["kind"] == "training_fact"
                else spec["evaluation_facts"]
            )
            expected_prompt = fact_prompt(
                spec, facts[key[1]]["cue"], facts[key[1]]["value"]
            )
        elif record["kind"] in ("training_question", "eval_question"):
            facts = (
                spec["training_facts"]
                if record["kind"] == "training_question"
                else spec["evaluation_facts"]
            )
            expected_prompt = question_prompt(spec, facts[key[1]]["cue"])
        elif record["kind"] == "control_question":
            expected_prompt = question_prompt(
                spec, spec["control_questions"][key[1]]["cue"]
            )
        if expected_prompt is None:
            raise SwapError(f"{path}: unknown record kind {record['kind']}")
        prompt_sha = hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
        if record["prompt_sha256"] != prompt_sha:
            raise SwapError(f"{path}: record {key} prompt does not match the spec")
        records[key] = record
    expected = {
        (kind, index)
        for kind, index, _cue, _value, _prompt in prompt_plan(spec)
    }
    if set(records) != expected:
        raise SwapError(f"{path}: hidden-state records are incomplete")
    return data


def phi_records(
    hidden: dict[str, Any],
    spec: dict[str, Any],
    host: str,
    kind: str,
    indices: Sequence[int],
    matrix: Sequence[Sequence[float]] | None = None,
) -> list[list[float]]:
    if matrix is None:
        matrix = host_projection(spec, host)
    by_key = {(r["kind"], int(r["index"])): r for r in hidden["records"]}
    return [project_hidden(by_key[(kind, index)]["h"], matrix) for index in indices]


def init_adapter(rng: random.Random, out_dim: int, in_dim: int) -> tuple[list[list[float]], list[float]]:
    scale = 1.0 / math.sqrt(in_dim)
    weights = [
        [rng.gauss(0.0, scale) for _ in range(in_dim)]
        for _ in range(out_dim)
    ]
    return weights, [0.0] * out_dim


def adapter_forward(
    weights: Sequence[Sequence[float]], bias: Sequence[float], phi: Sequence[float]
) -> tuple[list[float], list[float], float]:
    raw = [dot(row, phi) + float(b) for row, b in zip(weights, bias)]
    norm = math.sqrt(sum(value * value for value in raw))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SwapError("adapter produced a zero or non-finite vector")
    return raw, [value / norm for value in raw], norm


def adapter_z(
    weights: Sequence[Sequence[float]], bias: Sequence[float], phi: Sequence[float]
) -> list[float]:
    return adapter_forward(weights, bias, phi)[1]


def infonce_loss_and_vector_grad(
    z: Sequence[Sequence[float]], labels: Sequence[int], temperature: float
) -> tuple[float, list[list[float]]]:
    count = len(z)
    if count < 3:
        raise SwapError("InfoNCE needs at least three views")
    if len(labels) != count:
        raise SwapError("InfoNCE labels do not match the views")
    if temperature <= 0.0:
        raise SwapError("InfoNCE temperature must be positive")
    width = len(z[0])
    scores = [[dot(z[i], z[j]) / temperature for j in range(count)] for i in range(count)]
    scalar = [[0.0] * count for _ in range(count)]
    loss = 0.0
    for anchor in range(count):
        positives = [
            index
            for index in range(count)
            if index != anchor and labels[index] == labels[anchor]
        ]
        if not positives:
            raise SwapError("InfoNCE anchor has no positive view")
        negatives = [index for index in range(count) if index != anchor]
        max_all = max(scores[anchor][index] for index in negatives)
        exp_all = {
            index: math.exp(scores[anchor][index] - max_all)
            for index in negatives
        }
        sum_all = sum(exp_all.values())
        max_pos = max(scores[anchor][index] for index in positives)
        exp_pos = {
            index: math.exp(scores[anchor][index] - max_pos)
            for index in positives
        }
        sum_pos = sum(exp_pos.values())
        loss += -(math.log(sum_pos) + max_pos) + (
            math.log(sum_all) + max_all
        )
        for index in negatives:
            p_all = exp_all[index] / sum_all
            p_pos = exp_pos.get(index, 0.0) / sum_pos
            scalar[anchor][index] = p_all - p_pos
    loss /= count
    vector = [[0.0] * width for _ in range(count)]
    for i in range(count):
        for j in range(count):
            coefficient = (scalar[i][j] + scalar[j][i]) / (
                temperature * count
            )
            if coefficient == 0.0:
                continue
            for k in range(width):
                vector[i][k] += coefficient * z[j][k]
    return loss, vector


def _copy_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(row) for row in matrix]


def _copy_vector(vector: Sequence[float]) -> list[float]:
    return list(vector)


def validation_accuracy(
    weights: tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]],
    biases: tuple[Sequence[float], Sequence[float]],
    views: Sequence[Sequence[float]],
    labels: Sequence[int],
    hosts: Sequence[int],
    fact_indices: Sequence[int],
) -> float:
    """Retrieval accuracy: 350M fact keys, 8B question queries."""
    key_by_fact: dict[int, list[float]] = {}
    query_by_fact: dict[int, list[float]] = {}
    # The validation views are laid out as four views per fact in the same
    # order as the training views: 350M fact, 350M question, 8B fact, 8B
    # question.  The caller supplies the fact indices in that order.
    step = 4
    for offset, fact in enumerate(fact_indices):
        base = offset * step
        key_by_fact[fact] = adapter_z(weights[0], biases[0], views[base])
        query_by_fact[fact] = adapter_z(weights[1], biases[1], views[base + 3])
    ordered = list(fact_indices)
    keys = [key_by_fact[fact] for fact in ordered]
    correct = 0
    for position, fact in enumerate(ordered):
        best, _scores = nearest(query_by_fact[fact], keys)
        if best == position:
            correct += 1
    return correct / len(ordered)


def train_full_pair(
    spec: dict[str, Any],
    hidden_350m: dict[str, Any],
    hidden_8b: dict[str, Any],
    *,
    epochs: int | None = None,
    seed: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Return the trained weights, biases, state keys and validation record."""
    training_indices = list(range(len(spec["training_facts"])))
    matrix_350m = host_projection(spec, "350m")
    matrix_8b = host_projection(spec, "8b")
    facts_350m = phi_records(
        hidden_350m, spec, "350m", "training_fact", training_indices, matrix_350m
    )
    questions_350m = phi_records(
        hidden_350m, spec, "350m", "training_question", training_indices, matrix_350m
    )
    facts_8b = phi_records(
        hidden_8b, spec, "8b", "training_fact", training_indices, matrix_8b
    )
    questions_8b = phi_records(
        hidden_8b, spec, "8b", "training_question", training_indices, matrix_8b
    )
    views_by_fact = [
        [facts_350m[index], questions_350m[index], facts_8b[index], questions_8b[index]]
        for index in training_indices
    ]
    rng = random.Random(int(spec["seed"]) + 77)
    shuffled = training_indices[:]
    rng.shuffle(shuffled)
    validation_facts = sorted(shuffled[:4])
    train_facts = [index for index in training_indices if index not in validation_facts]
    train_views: list[list[float]] = []
    train_labels: list[int] = []
    train_hosts: list[int] = []
    for fact in train_facts:
        train_views.extend(views_by_fact[fact])
        train_labels.extend([fact] * 4)
        train_hosts.extend([0, 0, 1, 1])
    val_views = [view for fact in validation_facts for view in views_by_fact[fact]]
    val_labels = [fact for fact in validation_facts for _ in range(4)]
    val_hosts = [0, 0, 1, 1] * len(validation_facts)
    epochs = int(epochs if epochs is not None else spec["adapter"]["epochs"])
    seed = int(seed if seed is not None else spec["seed"] + 1234)
    _w_a, _w_b, _b_a, _b_b, best_epoch, best_acc, best_val_loss, train_loss, history = _train_with_biases(
        train_views,
        train_labels,
        train_hosts,
        spec,
        epochs=epochs,
        seed=seed,
        validation=(val_views, val_labels, val_hosts, validation_facts),
        progress=progress,
    )
    final_views: list[list[float]] = []
    final_labels: list[int] = []
    final_hosts: list[int] = []
    for fact in training_indices:
        final_views.extend(views_by_fact[fact])
        final_labels.extend([fact] * 4)
        final_hosts.extend([0, 0, 1, 1])
    final_epochs = (
        max(1, best_epoch)
        if spec["adapter"].get("early_stopping", True)
        else epochs
    )
    fw_a, fw_b, fb_a, fb_b, _fe, _fa, _fbest_loss, final_loss, _fh = _train_with_biases(
        final_views,
        final_labels,
        final_hosts,
        spec,
        epochs=final_epochs,
        seed=seed,
        validation=None,
        progress=False,
    )
    eval_indices = list(range(len(spec["evaluation_facts"])))
    eval_fact_phi = phi_records(
        hidden_350m, spec, "350m", "eval_fact", eval_indices, matrix_350m
    )
    keys = [
        adapter_z(fw_a, fb_a, phi)
        for phi in eval_fact_phi
    ]
    keys = [[f32(value) for value in row] for row in keys]
    values = [f32(index) for index in eval_indices]
    weights = {
        "350m": [[f32(v) for v in row] for row in fw_a],
        "8b": [[f32(v) for v in row] for row in fw_b],
    }
    biases = {"350m": [f32(v) for v in fb_a], "8b": [f32(v) for v in fb_b]}
    return {
        "weights": weights,
        "biases": biases,
        "keys": keys,
        "values": values,
        "state_sha256": state_digest(keys, values),
        "adapter_sha256": {
            host: adapter_digest(weights[host], biases[host])
            for host in ("350m", "8b")
        },
        "training": {
            "facts": len(training_indices),
            "validation_facts": validation_facts,
            "train_facts": train_facts,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_acc,
            "best_validation_loss": best_val_loss,
            "train_loss": train_loss,
            "final_loss": final_loss,
            "history_tail": history[-5:],
        },
        "seeds": {"validation": int(spec["seed"]) + 77, "adapter": seed},
    }


def _train_with_biases(
    views: Sequence[Sequence[float]],
    labels: Sequence[int],
    hosts: Sequence[int],
    spec: dict[str, Any],
    *,
    epochs: int,
    seed: int,
    validation: tuple[Sequence[Sequence[float]], Sequence[int], Sequence[int], Sequence[int]] | None,
    progress: bool,
) -> tuple[
    list[list[float]],
    list[list[float]],
    list[float],
    list[float],
    int,
    float,
    float,
    float,
    list[float],
]:
    out_dim = int(spec["adapter"]["output_dim"])
    in_dim = int(spec["projection"]["output_dim"])
    temperature = float(spec["adapter"]["temperature"])
    learning_rate = float(spec["adapter"]["learning_rate"])
    weight_decay = float(spec["adapter"]["weight_decay"])
    rng = random.Random(seed)
    weights = [init_adapter(rng, out_dim, in_dim)[0], init_adapter(rng, out_dim, in_dim)[0]]
    biases = [[0.0] * out_dim for _ in range(2)]
    m_w = [[[0.0] * in_dim for _ in range(out_dim)] for _ in range(2)]
    v_w = [[[0.0] * in_dim for _ in range(out_dim)] for _ in range(2)]
    m_b = [[0.0] * out_dim for _ in range(2)]
    v_b = [[0.0] * out_dim for _ in range(2)]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    best = (
        _copy_matrix(weights[0]),
        _copy_matrix(weights[1]),
        _copy_vector(biases[0]),
        _copy_vector(biases[1]),
        0,
        -1.0,
    )
    best_loss = float("inf")
    history: list[float] = []
    step = 0
    for epoch in range(1, epochs + 1):
        zs: list[list[float]] = []
        zrs: list[list[float]] = []
        norms: list[float] = []
        for phi, host in zip(views, hosts):
            raw, z, norm = adapter_forward(weights[host], biases[host], phi)
            zrs.append(raw)
            norms.append(norm)
            zs.append(z)
        loss, vector = infonce_loss_and_vector_grad(zs, labels, temperature)
        history.append(loss)
        d_w = [[[0.0] * in_dim for _ in range(out_dim)] for _ in range(2)]
        d_b = [[0.0] * out_dim for _ in range(2)]
        for index, (phi, host) in enumerate(zip(views, hosts)):
            z = zs[index]
            dz = vector[index]
            z_dot = dot(z, dz)
            inv = 1.0 / norms[index]
            for row in range(out_dim):
                delta = (dz[row] - z[row] * z_dot) * inv
                d_b[host][row] += delta
                row_out = d_w[host][row]
                for column in range(in_dim):
                    row_out[column] += delta * phi[column]
        step += 1
        for host in range(2):
            for row in range(out_dim):
                for column in range(in_dim):
                    grad = d_w[host][row][column] + weight_decay * weights[host][row][column]
                    m_w[host][row][column] = beta1 * m_w[host][row][column] + (1.0 - beta1) * grad
                    v_w[host][row][column] = beta2 * v_w[host][row][column] + (1.0 - beta2) * grad * grad
                    m_hat = m_w[host][row][column] / (1.0 - beta1 ** step)
                    v_hat = v_w[host][row][column] / (1.0 - beta2 ** step)
                    weights[host][row][column] -= learning_rate * m_hat / (math.sqrt(v_hat) + epsilon)
                grad_b = d_b[host][row]
                m_b[host][row] = beta1 * m_b[host][row] + (1.0 - beta1) * grad_b
                v_b[host][row] = beta2 * v_b[host][row] + (1.0 - beta2) * grad_b * grad_b
                m_hat_b = m_b[host][row] / (1.0 - beta1 ** step)
                v_hat_b = v_b[host][row] / (1.0 - beta2 ** step)
                biases[host][row] -= learning_rate * m_hat_b / (math.sqrt(v_hat_b) + epsilon)
        if progress and (epoch == 1 or epoch % 10 == 0):
            print(f"train epoch {epoch}/{epochs} loss {loss:.6f}", file=sys.stderr, flush=True)
        if validation is not None and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            val_views, val_labels, val_hosts, val_facts = validation
            val_z = [
                adapter_forward(weights[host], biases[host], phi)[1]
                for phi, host in zip(val_views, val_hosts)
            ]
            val_loss, _ = infonce_loss_and_vector_grad(val_z, val_labels, temperature)
            accuracy = validation_accuracy(
                (weights[0], weights[1]),
                (biases[0], biases[1]),
                val_views,
                val_labels,
                val_hosts,
                val_facts,
            )
            if val_loss < best_loss - 1e-12 or (
                abs(val_loss - best_loss) <= 1e-12 and accuracy > best[5]
            ):
                best = (
                    _copy_matrix(weights[0]),
                    _copy_matrix(weights[1]),
                    _copy_vector(biases[0]),
                    _copy_vector(biases[1]),
                    epoch,
                    accuracy,
                )
                best_loss = val_loss
    if validation is None:
        best = (
            _copy_matrix(weights[0]),
            _copy_matrix(weights[1]),
            _copy_vector(biases[0]),
            _copy_vector(biases[1]),
            epochs,
            0.0,
        )
    return (
        best[0],
        best[1],
        best[2],
        best[3],
        best[4],
        best[5],
        best_loss,
        history[-1] if history else float("nan"),
        history,
    )


def adapters_payload(
    spec: dict[str, Any],
    trained: dict[str, Any],
) -> dict[str, Any]:
    evaluation = spec["evaluation_facts"]
    return {
        "schema": "pmc-swap-adapters/1",
        "spec_sha256": spec["spec_sha256"],
        "state_schema": STATE_SCHEMA,
        "projection": {
            "output_dim": int(spec["projection"]["output_dim"]),
            "seed": int(spec["projection"]["seed"]),
            "input_dim": dict(spec["projection"]["input_dim"]),
        },
        "adapter": dict(spec["adapter"]),
        "bias": float(spec["bias"]),
        "hosts": {
            host: {
                "weights": trained["weights"][host],
                "bias": trained["biases"][host],
                "adapter_sha256": trained["adapter_sha256"][host],
            }
            for host in ("350m", "8b")
        },
        "state": {
            "keys": trained["keys"],
            "values": trained["values"],
            "value_words": [fact["value"] for fact in evaluation],
            "evaluation": evaluation,
            "state_sha256": trained["state_sha256"],
        },
        "training": trained["training"],
        "seeds": trained["seeds"],
    }


def train_command(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    hidden_350m = load_hidden(args.h_350m, spec, "350m")
    hidden_8b = load_hidden(args.h_8b, spec, "8b")
    started = time.perf_counter()
    trained = train_full_pair(
        spec,
        hidden_350m,
        hidden_8b,
        epochs=args.epochs,
        seed=args.seed,
        progress=not args.quiet,
    )
    payload = adapters_payload(spec, trained)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "state_sha256": payload["state"]["state_sha256"],
                "adapter_sha256": {
                    host: payload["hosts"][host]["adapter_sha256"]
                    for host in ("350m", "8b")
                },
                "best_epoch": payload["training"]["best_epoch"],
                "best_validation_accuracy": payload["training"]["best_validation_accuracy"],
                "epochs": args.epochs,
                "elapsed_s": time.perf_counter() - started,
                "out": str(output),
            },
            indent=1,
        )
    )
    return 0


# -- fusion ----------------------------------------------------------------


def write_rows(path: str | Path, n_rows: int, dim: int, value: float) -> None:
    header = struct.pack("<4sIII", b"PMCR", 1, n_rows, dim)
    body = b"".join(F32.pack(f32(value)) for _ in range(n_rows * dim))
    Path(path).write_bytes(header + body)


def flatten(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [f32(value) for row in matrix for value in row]


def fuse_host(
    spec: dict[str, Any],
    adapters: dict[str, Any],
    host: str,
    base: str,
    out: str,
    rows: str,
    meta: str,
    pmc: str,
) -> dict[str, Any]:
    if host not in spec["hosts"]:
        raise SwapError(f"unknown host {host}")
    dim = int(spec["projection"]["input_dim"][host])
    host_record = adapters["hosts"][host]
    weights = host_record["weights"]
    bias = host_record["bias"]
    state = adapters["state"]
    keys = state["keys"]
    values = state["values"]
    if len(weights) != int(spec["adapter"]["output_dim"]):
        raise SwapError("adapter weight rows do not match the spec")
    if len(weights[0]) != int(spec["projection"]["output_dim"]):
        raise SwapError("adapter weight columns do not match the spec")
    if len(keys) != len(spec["evaluation_facts"]):
        raise SwapError("state key count does not match the evaluation facts")
    if len(values) != len(keys):
        raise SwapError("state value count does not match the keys")
    if state["state_sha256"] != state_digest(keys, values):
        raise SwapError("state hash in the adapters payload is inconsistent")
    if host_record["adapter_sha256"] != adapter_digest(weights, bias):
        raise SwapError("adapter hash in the adapters payload is inconsistent")
    n_rows = len(spec.get("fuse_tokens", [])) or 8
    write_rows(rows, n_rows, dim, 0.0)
    tokens = [f"<pmc:swap-{host}{index}>" for index in range(n_rows)]
    state_tensors = {
        "pmc.state.swap.keys": {
            "dims": [len(keys), len(keys[0])],
            "data": flatten(keys),
        },
        "pmc.state.swap.values": {
            "dims": [len(values)],
            "data": [f32(value) for value in values],
        },
        "pmc.state.swap.adapter": {
            "dims": [len(weights), len(weights[0])],
            "data": flatten(weights),
        },
        "pmc.state.swap.adapter_bias": {
            "dims": [len(bias)],
            "data": [f32(value) for value in bias],
        },
    }
    provenance = [
        f"schema:{STATE_SCHEMA}",
        f"state_sha256:{state['state_sha256']}",
        f"adapter_sha256:{host_record['adapter_sha256']}",
        f"projection:{spec['projection']['output_dim']}:{spec['projection']['seed']}",
        f"spec_sha256:{spec['spec_sha256']}",
        "training_facts:" + str(len(spec["training_facts"])),
        "evaluation_facts:8",
        "control_questions:8",
    ]
    payload = {
        "tokens": tokens,
        "scores": [0.0] * n_rows,
        "token_types": [4] * n_rows,
        "state_tensors": state_tensors,
        "dream_id": f"pmc-mini-swap-{host}",
        "provenance": provenance,
    }
    Path(meta).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    command = [
        pmc,
        "fuse",
        "--base",
        base,
        "--rows",
        rows,
        "--meta",
        meta,
        "--out",
        out,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SwapError(
            f"pmc fuse failed with {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return {
        "host": host,
        "base": base,
        "out": out,
        "rows": rows,
        "meta": meta,
        "state_sha256": state["state_sha256"],
        "adapter_sha256": host_record["adapter_sha256"],
        "fuse_stdout": completed.stdout.strip(),
        "fuse_stderr": completed.stderr.strip(),
    }


def fuse_command(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    adapters = json.loads(Path(args.adapters).read_text(encoding="utf-8"))
    result = fuse_host(
        spec,
        adapters,
        args.host,
        args.base,
        args.out,
        args.rows,
        args.meta,
        args.pmc,
    )
    print(json.dumps(result, indent=1))
    return 0


# -- evaluation ------------------------------------------------------------


def bank_rows(bank: Any, rows: int, columns: int) -> list[list[float]]:
    if bank.dims != (rows, columns):
        raise SwapError(
            f"{bank.name}: dims {bank.dims} do not match {(rows, columns)}"
        )
    data = list(bank.data)
    return [data[index * columns : (index + 1) * columns] for index in range(rows)]


def read_swap_state(model: str, spec: dict[str, Any]) -> dict[str, Any]:
    banks = read_state_banks(model)
    required = {
        "swap.keys",
        "swap.values",
        "swap.adapter",
        "swap.adapter_bias",
    }
    if not required.issubset(banks):
        raise SwapError(
            f"{model}: missing state banks {sorted(required - set(banks))}"
        )
    keys = bank_rows(
        banks["swap.keys"],
        len(spec["evaluation_facts"]),
        int(spec["adapter"]["output_dim"]),
    )
    raw_values = list(banks["swap.values"].data)
    values = [int(round(value)) for value in raw_values]
    if [f32(value) for value in raw_values] != [f32(value) for value in values]:
        raise SwapError(f"{model}: state values are not integers")
    weights = bank_rows(
        banks["swap.adapter"],
        int(spec["adapter"]["output_dim"]),
        int(spec["projection"]["output_dim"]),
    )
    bias = list(banks["swap.adapter_bias"].data)
    if len(bias) != int(spec["adapter"]["output_dim"]):
        raise SwapError(f"{model}: adapter bias has the wrong length")
    return {
        "keys": keys,
        "values": values,
        "weights": weights,
        "bias": bias,
        "state_sha256": state_digest(keys, values),
        "adapter_sha256": adapter_digest(weights, bias),
        "bank_dims": {
            "keys": banks["swap.keys"].dims,
            "values": banks["swap.values"].dims,
            "adapter": banks["swap.adapter"].dims,
            "adapter_bias": banks["swap.adapter_bias"].dims,
        },
    }


def value_token(spike: LlamaSpike, word: str) -> int:
    tokens = spike.tokenize(" " + word)
    if len(tokens) != 2:
        raise SwapError(
            f"value word {word!r} is not one token after BOS: {tokens}"
        )
    return tokens[-1]


def sample_text(spike: LlamaSpike, context: int, biases: Sequence[tuple[int, float]] = ()) -> tuple[int, str]:
    token = spike.sample_biased(context, biases)
    text = spike.detokenize([token])
    return token, text


def run_host_arms(
    spec: dict[str, Any],
    host: str,
    model: str,
    lib: str,
    shim: str,
    adapters: dict[str, Any],
    *,
    n_ctx: int,
    n_batch: int,
    n_threads: int,
    fresh_seed: int,
    progress: bool = True,
) -> dict[str, Any]:
    state = read_swap_state(model, spec)
    expected_state = adapters["state"]["state_sha256"]
    expected_adapter = adapters["hosts"][host]["adapter_sha256"]
    if state["state_sha256"] != expected_state:
        raise SwapError(
            f"{host}: state hash {state['state_sha256']} != {expected_state}"
        )
    if state["adapter_sha256"] != expected_adapter:
        raise SwapError(
            f"{host}: adapter hash {state['adapter_sha256']} != {expected_adapter}"
        )
    evaluation = spec["evaluation_facts"]
    matrix = host_projection(spec, host)
    value_words = [fact["value"] for fact in evaluation]
    if state["values"] != list(range(len(evaluation))):
        raise SwapError(f"{host}: state values are not the frozen evaluation order")
    # Fresh state: same shape, random unit keys and a random permutation of
    # the value indices.  The same bytes are used on both hosts.
    rng = random.Random(fresh_seed)
    out_dim = int(spec["adapter"]["output_dim"])
    fresh_keys = [
        normalize([rng.gauss(0.0, 1.0) for _ in range(out_dim)])
        for _ in evaluation
    ]
    fresh_values = list(range(len(evaluation)))
    rng.shuffle(fresh_values)
    fresh_words = [value_words[index] for index in fresh_values]
    spike = LlamaSpike(lib, shim, model)
    rows: list[dict[str, Any]] = []
    try:
        if spike.n_embd != int(spec["projection"]["input_dim"][host]):
            raise SwapError(f"{host}: unexpected hidden dimension")
        context = spike.new_context(
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            embeddings=True,
        )
        token_cache = {word: value_token(spike, word) for word in value_words}

        def decode(
            prompt: str, cue: str | None = None, *, keep_prompt: bool = False
        ) -> tuple[list[float], int]:
            tokens = spike.tokenize(prompt)
            if len(tokens) > n_ctx:
                raise SwapError(f"{host}: prompt has {len(tokens)} tokens over {n_ctx}")
            spike.clear_context(context)
            spike.decode(context, tokens, all_outputs=True)
            anchor = spec.get("anchor", "last")
            if anchor in ("cue", "cue_delta") and cue is not None:
                start = cue_start(prompt, cue)
                prefix = prompt[:start]
                cue_end = len(spike.tokenize(prefix + cue)) - 1
                if cue_end < 0 or cue_end >= len(tokens):
                    raise SwapError(f"{host}: cue end {cue_end} is outside the prompt")
                hidden = spike.read_embeddings(context, cue_end)
                if anchor == "cue_delta":
                    neutral = str(spec["neutral_cue"])
                    swapped = neutral_prompt(prompt, cue, neutral)
                    neutral_tokens = spike.tokenize(swapped)
                    neutral_end = len(spike.tokenize(prefix + neutral)) - 1
                    if neutral_end < 0 or neutral_end >= len(neutral_tokens):
                        raise SwapError(
                            f"{host}: neutral cue end is outside the prompt"
                        )
                    spike.clear_context(context)
                    spike.decode(context, neutral_tokens, all_outputs=True)
                    neutral_hidden = spike.read_embeddings(context, neutral_end)
                    if len(neutral_hidden) != len(hidden):
                        raise SwapError(f"{host}: neutral hidden has the wrong length")
                    hidden = [
                        float(value) - float(neutral_value)
                        for value, neutral_value in zip(hidden, neutral_hidden)
                    ]
                    if keep_prompt:
                        # The sampler needs the actual question context, not
                        # the neutral probe context.
                        spike.clear_context(context)
                        spike.decode(context, tokens, all_outputs=True)
            else:
                hidden = spike.read_embeddings(context, -1)
            return list(hidden), len(tokens)

        for index, fact in enumerate(evaluation):
            question = question_prompt(spec, fact["cue"])
            hidden, prompt_tokens = decode(question, fact["cue"], keep_prompt=True)
            phi = project_hidden(hidden, matrix)
            query = adapter_z(state["weights"], state["bias"], phi)
            nearest_index, scores = nearest(query, state["keys"])
            retrieved_word = value_words[state["values"][nearest_index]]
            biased_token, biased_text = sample_text(
                spike, context, [(token_cache[retrieved_word], float(spec["bias"]))]
            )
            none_token, none_text = sample_text(spike, context, ())
            fresh_index, _fresh_scores = nearest(query, fresh_keys)
            fresh_word = fresh_words[fresh_index]
            fresh_token, fresh_text = sample_text(
                spike, context, [(token_cache[fresh_word], float(spec["bias"]))]
            )
            rows.append(
                {
                    "host": host,
                    "arm": "pmc",
                    "eval_index": index,
                    "cue": fact["cue"],
                    "expected": fact["value"],
                    "prompt_tokens": prompt_tokens,
                    "h_sha256": hashlib.sha256(_pack_f32(hidden)).hexdigest(),
                    "nearest": nearest_index,
                    "nearest_score": scores[nearest_index],
                    "retrieved": retrieved_word,
                    "output_token": biased_token,
                    "output": biased_text,
                    "correct": answer_ok(biased_text, fact["value"]),
                    "bias": float(spec["bias"]),
                }
            )
            rows.append(
                {
                    "host": host,
                    "arm": "none",
                    "eval_index": index,
                    "cue": fact["cue"],
                    "expected": fact["value"],
                    "prompt_tokens": prompt_tokens,
                    "output_token": none_token,
                    "output": none_text,
                    "correct": answer_ok(none_text, fact["value"]),
                    "bias": 0.0,
                }
            )
            rows.append(
                {
                    "host": host,
                    "arm": "fresh",
                    "eval_index": index,
                    "cue": fact["cue"],
                    "expected": fact["value"],
                    "prompt_tokens": prompt_tokens,
                    "nearest": fresh_index,
                    "retrieved": fresh_word,
                    "output_token": fresh_token,
                    "output": fresh_text,
                    "correct": answer_ok(fresh_text, fact["value"]),
                    "bias": float(spec["bias"]),
                }
            )
            re_prompt = reinjection_prompt(spec, fact["cue"], fact["value"])
            re_hidden, re_tokens = decode(re_prompt)
            re_token, re_text = sample_text(spike, context, ())
            rows.append(
                {
                    "host": host,
                    "arm": "reinjection",
                    "eval_index": index,
                    "cue": fact["cue"],
                    "expected": fact["value"],
                    "prompt_tokens": re_tokens,
                    "h_sha256": hashlib.sha256(_pack_f32(re_hidden)).hexdigest(),
                    "output_token": re_token,
                    "output": re_text,
                    "correct": answer_ok(re_text, fact["value"]),
                    "bias": 0.0,
                }
            )
            if progress:
                print(
                    f"{host}: fact {index + 1}/{len(evaluation)} done",
                    file=sys.stderr,
                    flush=True,
                )
        for index, control in enumerate(spec["control_questions"]):
            prompt = question_prompt(spec, control["cue"])
            hidden, prompt_tokens = decode(prompt, control["cue"], keep_prompt=True)
            phi = project_hidden(hidden, matrix)
            query = adapter_z(state["weights"], state["bias"], phi)
            nearest_index, _scores = nearest(query, state["keys"])
            retrieved_word = value_words[state["values"][nearest_index]]
            token, text = sample_text(
                spike, context, [(token_cache[retrieved_word], float(spec["bias"]))]
            )
            paired = int(control["paired_eval"])
            paired_value = evaluation[paired]["value"]
            rows.append(
                {
                    "host": host,
                    "arm": "control",
                    "control_index": index,
                    "cue": control["cue"],
                    "paired_eval": paired,
                    "paired_value": paired_value,
                    "prompt_tokens": prompt_tokens,
                    "h_sha256": hashlib.sha256(_pack_f32(hidden)).hexdigest(),
                    "nearest": nearest_index,
                    "retrieved": retrieved_word,
                    "output_token": token,
                    "output": text,
                    "false_positive": answer_ok(text, paired_value),
                    "bias": float(spec["bias"]),
                }
            )
        model_sha = sha256_file(model)
        return {
            "host": host,
            "model": model,
            "model_sha256": model_sha,
            "n_embd": spike.n_embd,
            "n_vocab": spike.n_vocab,
            "state_sha256": state["state_sha256"],
            "adapter_sha256": state["adapter_sha256"],
            "bank_dims": state["bank_dims"],
            "fresh_seed": fresh_seed,
            "fresh_words": fresh_words,
            "rows": rows,
        }
    finally:
        spike.close()


def counts_from_rows(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "pmc": 0,
        "none": 0,
        "fresh": 0,
        "reinjection": 0,
        "control_false_positive": 0,
    }
    for row in rows:
        arm = row["arm"]
        if arm == "control":
            counts["control_false_positive"] += int(bool(row["false_positive"]))
        else:
            counts[arm] += int(bool(row["correct"]))
    counts["total"] = len([row for row in rows if row["arm"] in ("pmc", "none", "fresh", "reinjection")]) // 4
    return counts


def no_target_training(spec: dict[str, Any]) -> bool:
    training = {fact["cue"] for fact in spec["training_facts"]}
    training |= {fact["value"] for fact in spec["training_facts"]}
    evaluation = {fact["cue"] for fact in spec["evaluation_facts"]}
    evaluation |= {fact["value"] for fact in spec["evaluation_facts"]}
    control = {control["cue"] for control in spec["control_questions"]}
    return not (training & evaluation) and not (training & control) and not (
        evaluation & control
    )


def evaluate_result(
    spec: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    hosts = result["hosts"]
    counts_a = hosts["350m"]["counts"]
    counts_b = hosts["8b"]["counts"]
    state_identity = (
        hosts["350m"]["state_sha256"] == hosts["8b"]["state_sha256"]
        and hosts["350m"]["state_sha256"] == result["expected_state_sha256"]
        and hosts["350m"]["adapter_sha256"] == result["expected_adapters"]["350m"]
        and hosts["8b"]["adapter_sha256"] == result["expected_adapters"]["8b"]
    )
    no_target = no_target_training(spec)
    checks = {
        "host_a_min": counts_a["pmc"] >= 5,
        "host_b_min": counts_b["pmc"] >= 4,
        "host_b_margin": counts_b["pmc"] >= counts_a["pmc"] - 1,
        "no_state_a": counts_a["none"] <= 1,
        "no_state_b": counts_b["none"] <= 1,
        "fresh_a": counts_a["fresh"] <= 2,
        "fresh_b": counts_b["fresh"] <= 2,
        "control_a": counts_a["control_false_positive"] <= 2,
        "control_b": counts_b["control_false_positive"] <= 2,
        "state_identity": state_identity,
        "no_target_training": no_target,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "counts": {
            "350m": counts_a,
            "8b": counts_b,
        },
        "state_identical": hosts["350m"]["state_sha256"] == hosts["8b"]["state_sha256"],
        "adapter_identical": hosts["350m"]["adapter_sha256"] == hosts["8b"]["adapter_sha256"],
        "reinjection": {
            "350m": counts_a["reinjection"],
            "8b": counts_b["reinjection"],
        },
    }


def eval_command(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    adapters = json.loads(Path(args.adapters).read_text(encoding="utf-8"))
    expected_state = adapters["state"]["state_sha256"]
    expected_adapters = {
        host: adapters["hosts"][host]["adapter_sha256"]
        for host in ("350m", "8b")
    }
    fresh_seed = int(spec["seed"]) + 7
    started = time.perf_counter()
    host_results: dict[str, Any] = {}
    for host, model in (("350m", args.model_a), ("8b", args.model_b)):
        host_results[host] = run_host_arms(
            spec,
            host,
            model,
            args.lib,
            args.shim,
            adapters,
            n_ctx=args.ctx,
            n_batch=args.batch,
            n_threads=args.threads,
            fresh_seed=fresh_seed,
            progress=not args.quiet,
        )
        host_results[host]["counts"] = counts_from_rows(host_results[host]["rows"])
    result: dict[str, Any] = {
        "schema": EVAL_SCHEMA,
        "spec_sha256": spec["spec_sha256"],
        "bias": float(spec["bias"]),
        "token_budget": 1,
        "sampler": "greedy",
        "expected_state_sha256": expected_state,
        "expected_adapters": expected_adapters,
        "hosts": host_results,
        "elapsed_s": time.perf_counter() - started,
    }
    result["summary"] = evaluate_result(spec, result)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pass": result["summary"]["pass"],
                "counts": result["summary"]["counts"],
                "checks": result["summary"]["checks"],
                "reinjection": result["summary"]["reinjection"],
                "state_identical": result["summary"]["state_identical"],
                "out": str(output),
            },
            indent=1,
        )
    )
    return 0


# -- command line ----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="extract hidden states for one host")
    extract.add_argument("--host", required=True, choices=("350m", "8b"))
    extract.add_argument("--model", required=True)
    extract.add_argument("--lib", required=True)
    extract.add_argument("--shim", required=True)
    extract.add_argument("--out", required=True)
    extract.add_argument("--ctx", type=int, default=128)
    extract.add_argument("--batch", type=int, default=128)
    extract.add_argument("--threads", type=int, default=2)
    extract.add_argument("--spec", default=str(SPEC_PATH))
    extract.set_defaults(func=extract_command)

    train = subparsers.add_parser("train", help="train the two host adapters")
    train.add_argument("--h-350m", required=True)
    train.add_argument("--h-8b", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--quiet", action="store_true")
    train.add_argument("--spec", default=str(SPEC_PATH))
    train.set_defaults(func=train_command)

    fuse = subparsers.add_parser("fuse", help="fuse the shared state into one GGUF")
    fuse.add_argument("--host", required=True, choices=("350m", "8b"))
    fuse.add_argument("--base", required=True)
    fuse.add_argument("--adapters", required=True)
    fuse.add_argument("--out", required=True)
    fuse.add_argument("--rows", required=True)
    fuse.add_argument("--meta", required=True)
    fuse.add_argument("--pmc", required=True)
    fuse.add_argument("--spec", default=str(SPEC_PATH))
    fuse.set_defaults(func=fuse_command)

    evaluate = subparsers.add_parser("eval", help="read the state from two GGUFs and run every arm")
    evaluate.add_argument("--model-a", required=True)
    evaluate.add_argument("--model-b", required=True)
    evaluate.add_argument("--adapters", required=True)
    evaluate.add_argument("--lib", required=True)
    evaluate.add_argument("--shim", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--ctx", type=int, default=128)
    evaluate.add_argument("--batch", type=int, default=128)
    evaluate.add_argument("--threads", type=int, default=2)
    evaluate.add_argument("--quiet", action="store_true")
    evaluate.add_argument("--spec", default=str(SPEC_PATH))
    evaluate.set_defaults(func=eval_command)

    check = subparsers.add_parser("check-spec", help="validate the frozen spec only")
    check.add_argument("--spec", default=str(SPEC_PATH))
    check.set_defaults(func=lambda args: (load_spec(args.spec), print("spec ok"), 0)[-1])

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (SwapError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
