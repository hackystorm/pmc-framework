# SPDX-License-Identifier: GPL-3.0-only
"""Same-memory comparison: live fast weights versus frozen context.

The question this script answers is narrow and testable:

* **frozen** — the fact is re-injected as text on every turn. The model sees
  the whole transcript and answers from attention over the context.
* **live** — the fact is not in the second prompt. The live engine learned
  fast-weight rows from the tokens it generated in the first turn (and, with
  ``--learn-from-prompt``, from the prompt tokens too); those rows bias the
  second turn in the same process.
* **kv** — the first turn's whole context state (KV cache and positions) is
  saved; the second prompt is the full transcript, but only the suffix after
  the saved token prefix is decoded. The fact and the first reply are restored
  as state, not re-read as text. This is the substrate-resident condition.
* **replay** — the whole transcript is re-decoded on the live graph with
  ``learn=False``. It is the in-process control for the kv condition: same
  model, template, sampler and first reply, without a second server.
* **kv-restart** — the state and the first turn are loaded from disk in a new
  process; only the suffix is decoded. It measures persistence across a
  restart on the same model and geometry.
* **control** — the question with no fact at all, on the frozen server. It is
  the model's own prior for this question.

``--first-style quiet`` asks the first reply for the single word "noted", so a
valid run does not repeat the code in the assistant turn. The output records
``first_reply_echo_free``; a quiet run that echoes is reported as such, not
silently treated as the echo-free control.

The first reply in the frozen, live and kv conditions is generated from the
same prompt, with the same greedy decode, on the same model file, so the three
texts should match. The memory comparison is the *second* answer: frozen sees
the fact in the context; live carries it in the fast weights; kv carries it in
a restored context state.

Run one condition per invocation so only one model copy exists at a time:

```sh
# with a frozen server already listening (llama-server --jinja)
PYTHONPATH=src python bench/live_vs_frozen.py --condition frozen \
    --frozen-url http://127.0.0.1:8099 --out frozen.json

# stop the server, then the live condition (one in-process model)
PYTHONPATH=src python bench/live_vs_frozen.py --condition live \
    --model /path/to/fused.gguf --lib /path/to/libllama.so \
    --shim /path/to/libpmc_spike.so --delta live-delta.json --out live.json

# the kv condition: the same first turn, then a restored context state
PYTHONPATH=src python bench/live_vs_frozen.py --condition kv \
    --model /path/to/fused.gguf --lib /path/to/libllama.so \
    --shim /path/to/libpmc_spike.so --out kv.json

# the echo-free control, and the restart of the saved state
PYTHONPATH=src python bench/live_vs_frozen.py --condition kv \
    --first-style quiet --kv-state /path/outside/the/repo/kv-quiet.pmcstate \
    --model /path/to/fused.gguf --lib /path/to/libllama.so \
    --shim /path/to/libpmc_spike.so --out kv-quiet.json
PYTHONPATH=src python bench/live_vs_frozen.py --condition kv-restart \
    --kv-state /path/outside/the/repo/kv-quiet.pmcstate \
    --model /path/to/fused.gguf --lib /path/to/libllama.so \
    --shim /path/to/libpmc_spike.so --out kv-quiet-restart.json
```

The control belongs to the frozen run because it only needs the frozen server.
The comparison is deliberately asymmetric and the output says so: the live
memory contains what the instance *said*, not what the user typed, unless
``--learn-from-prompt`` is set. The kv memory contains the same tokens, but in
the KV cache instead of the output rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from pmc.runtime.client import ChatClient, ChatError
from pmc.runtime.live_engine import KvState, KvStateError, LiveEngine, LiveEngineError

DEFAULT_CODE = "4821"
DEFAULT_SYSTEM = "You are a terse assistant."
DEFAULT_FACT = "Remember the code: {code}."
DEFAULT_QUIET_FACT = (
    "Remember the code: {code}. Reply with the single word: noted."
)
DEFAULT_QUESTION = "What is the code? Reply with the four digits only."
DEFAULT_MAX_TOKENS = 24
FIRST_STYLES = ("echo", "quiet")


def fact_message(code: str, style: str = "echo") -> str:
    """The fact prompt. ``quiet`` asks for a reply that does not repeat it."""
    if style == "echo":
        return DEFAULT_FACT.format(code=code)
    if style == "quiet":
        return DEFAULT_QUIET_FACT.format(code=code)
    raise ValueError(f"unknown first style {style!r}")


def message_text(text: str) -> str:
    """One user message in the wire shape the client and engine both accept."""
    return text.strip()


def digits_only(text: str) -> str:
    """The decimal digits of ``text`` in order, with nothing else."""
    return "".join(character for character in text if character.isdigit())


def code_present(text: str, code: str) -> bool:
    """True when the exact code appears in the text."""
    return code in text


def code_present_digits(text: str, code: str) -> bool:
    """True when the code appears after removing non-digits.

    Kept separate from :func:`code_present` so the output can show both: a
    match through punctuation ("4,821") is a different fact from an exact
    four-digit match.
    """
    return code in digits_only(text)


def first_reply_echo_free(text: str, code: str) -> bool:
    """True when the assistant's first reply does not repeat the code.

    The original KV run used a fact whose reply echoed the code, so the saved
    state contained the fact twice. The quiet first style is valid only when
    this flag is true; the output records it instead of assuming it.
    """
    return not code_present(text, code)


def timed(call: Any) -> tuple[Any, float]:
    started = time.monotonic()
    value = call()
    return value, round(time.monotonic() - started, 3)


def answer_record(text: str, code: str, **extra: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "text": text,
        "code_found": code_present(text, code),
        "code_found_digits": code_present_digits(text, code),
        "digits": digits_only(text),
    }
    record.update(extra)
    return record


def run_frozen(args: argparse.Namespace) -> dict[str, Any]:
    client = ChatClient(
        args.frozen_url,
        model=args.frozen_model,
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    system = args.system
    fact = fact_message(args.code, args.first_style)
    question = args.question
    transcript: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": fact},
    ]

    first, first_s = timed(
        lambda: client.complete(transcript, temperature=0.0, max_tokens=args.max_tokens)
    )
    transcript.append({"role": "assistant", "content": first.text})

    second, second_s = timed(
        lambda: client.complete(
            transcript + [{"role": "user", "content": question}],
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
    )
    control, control_s = timed(
        lambda: client.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
    )

    return {
        "condition": "frozen",
        "model": args.frozen_model,
        "endpoint": args.frozen_url,
        "code": args.code,
        "fact": fact,
        "question": question,
        "first_style": args.first_style,
        "first_reply": answer_record(
            first.text,
            args.code,
            prompt_tokens=first.prompt_tokens,
            completion_tokens=first.completion_tokens,
            finish_reason=first.finish_reason,
            elapsed_s=first_s,
            first_reply_echo_free=first_reply_echo_free(first.text, args.code),
        ),
        "answers": {
            "frozen_memory": answer_record(
                second.text,
                args.code,
                prompt_tokens=second.prompt_tokens,
                completion_tokens=second.completion_tokens,
                finish_reason=second.finish_reason,
                elapsed_s=second_s,
            ),
            "no_memory": answer_record(
                control.text,
                args.code,
                prompt_tokens=control.prompt_tokens,
                completion_tokens=control.completion_tokens,
                finish_reason=control.finish_reason,
                elapsed_s=control_s,
            ),
        },
        "server_timings": {
            "first": first.raw.get("timings"),
            "frozen_memory": second.raw.get("timings"),
            "no_memory": control.raw.get("timings"),
        },
    }


def open_engine(
    args: argparse.Namespace,
    *,
    use_delta: bool = True,
    system_prompt: str | None = None,
) -> LiveEngine:
    """The live configuration shared by the in-process conditions."""
    return LiveEngine(
        args.model,
        args.lib,
        args.shim,
        n_ctx=args.ctx,
        n_batch=args.batch,
        n_threads=args.threads,
        delta_path=args.delta if use_delta else None,
        system_prompt=args.system if system_prompt is None else system_prompt,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        learn_from_prompt=args.learn_from_prompt,
        default_max_tokens=args.max_tokens,
        compute_model_digest=not args.no_digest,
    )


def run_live(args: argparse.Namespace) -> dict[str, Any]:
    engine = open_engine(args)
    try:
        fact = fact_message(args.code, args.first_style)
        question = args.question
        first, first_s = timed(
            lambda: engine.chat(
                [{"role": "user", "content": fact}], max_tokens=args.max_tokens
            )
        )
        summary_after_first = engine.state_summary()
        second, second_s = timed(
            lambda: engine.chat(
                [{"role": "user", "content": question}], max_tokens=args.max_tokens
            )
        )
        summary = engine.state_summary()
    finally:
        engine.close()

    return {
        "condition": "live",
        "model": args.model_name,
        "model_digest": summary["model_digest"],
        "code": args.code,
        "fact": fact,
        "question": question,
        "first_style": args.first_style,
        "learning": not args.no_learn,
        "learn_from_prompt": args.learn_from_prompt,
        "first_reply": answer_record(
            first.text,
            args.code,
            prompt_tokens=first.prompt_tokens,
            completion_tokens=first.completion_tokens,
            finish_reason=first.finish_reason,
            elapsed_s=first_s,
            fast_weight_rows=first.fast_weight_rows,
            state_digest=first.state_digest,
            first_reply_echo_free=first_reply_echo_free(first.text, args.code),
        ),
        "answers": {
            "live_memory": answer_record(
                second.text,
                args.code,
                prompt_tokens=second.prompt_tokens,
                completion_tokens=second.completion_tokens,
                finish_reason=second.finish_reason,
                elapsed_s=second_s,
                fast_weight_rows=second.fast_weight_rows,
                fast_weight_digest=second.fast_weight_digest,
                state_digest=second.state_digest,
            )
        },
        "substrate": {
            "after_first_turn": summary_after_first,
            "after_second_turn": summary,
            "delta_path": str(args.delta) if args.delta else None,
        },
    }


def run_kv(args: argparse.Namespace) -> dict[str, Any]:
    """Keep the fact in a restored context state, not in re-decoded text.

    The first turn builds the fact prompt, generates a reply, and saves the
    whole context state. The second prompt is the full transcript, but only the
    suffix after the saved state's token prefix is decoded: the fact and the
    first reply come back as KV cache, not as text the model reads again. The
    engine refuses the restore unless the prompt prefix is token-identical to
    the saved one.
    """
    engine = open_engine(args)
    try:
        fact = fact_message(args.code, args.first_style)
        question = args.question
        first, first_s = timed(
            lambda: engine.chat(
                [{"role": "user", "content": fact}],
                max_tokens=args.max_tokens,
                save_state=True,
                learn=not args.no_learn,
            )
        )
        state = first.kv_state
        if state is None:
            raise LiveEngineError("the first turn did not save a KV state")
        summary_after_first = engine.state_summary()
        if args.kv_state:
            state.save(args.kv_state)
            turn = {
                "code": args.code,
                "system": args.system,
                "fact": fact,
                "first_reply": first.text,
                "question": question,
                "first_style": args.first_style,
                "model_digest": summary_after_first["model_digest"],
                "n_embd": summary_after_first["embedding_dim"],
                "n_ctx": state.n_ctx,
                "kv_state": state.summary(),
            }
            turn_path = Path(args.kv_state).with_suffix(".turn.json")
            turn_path.write_text(
                json.dumps(turn, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        transcript = [
            {"role": "user", "content": fact},
            {"role": "assistant", "content": first.text},
            {"role": "user", "content": question},
        ]
        second, second_s = timed(
            lambda: engine.chat(
                transcript,
                max_tokens=args.max_tokens,
                restore=state,
                learn=not args.no_learn,
            )
        )
        summary = engine.state_summary()
    finally:
        engine.close()

    return {
        "condition": "kv",
        "model": args.model_name,
        "model_digest": summary["model_digest"],
        "code": args.code,
        "fact": fact,
        "question": question,
        "first_style": args.first_style,
        "learning": not args.no_learn,
        "learn_from_prompt": args.learn_from_prompt,
        "first_reply": answer_record(
            first.text,
            args.code,
            prompt_tokens=first.prompt_tokens,
            completion_tokens=first.completion_tokens,
            finish_reason=first.finish_reason,
            elapsed_s=first_s,
            kv_bytes=len(state.data),
            kv_tokens=len(state.tokens),
            kv_sha256=state.sha256,
            kv_model_digest=state.model_digest,
            kv_n_embd=state.n_embd,
            kv_n_ctx=state.n_ctx,
            first_reply_echo_free=first_reply_echo_free(first.text, args.code),
        ),
        "answers": {
            "kv_memory": answer_record(
                second.text,
                args.code,
                prompt_tokens=second.prompt_tokens,
                completion_tokens=second.completion_tokens,
                finish_reason=second.finish_reason,
                elapsed_s=second_s,
                restored_tokens=len(state.tokens),
                decoded_suffix_tokens=second.prompt_tokens - len(state.tokens),
                fast_weight_rows=second.fast_weight_rows,
                fast_weight_digest=second.fast_weight_digest,
                state_digest=second.state_digest,
            )
        },
        "substrate": {
            "after_first_turn": summary_after_first,
            "after_second_turn": summary,
            "kv_state": state.summary(),
            "delta_path": str(args.delta) if args.delta else None,
        },
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    """Re-inject the whole transcript on the live graph, with no learning.

    This is the in-process control for the KV condition: the same model, the
    same chat template, the same greedy sampler and the same first reply, but
    the second turn re-decodes the fact and the reply as text. It is not the
    stock-provider frozen condition; it isolates *state restore* from
    *re-decoding* without starting a second server.
    """
    engine = open_engine(args, use_delta=False)
    try:
        fact = fact_message(args.code, args.first_style)
        question = args.question
        first, first_s = timed(
            lambda: engine.chat(
                [{"role": "user", "content": fact}],
                max_tokens=args.max_tokens,
                learn=False,
            )
        )
        summary_after_first = engine.state_summary()
        transcript = [
            {"role": "user", "content": fact},
            {"role": "assistant", "content": first.text},
            {"role": "user", "content": question},
        ]
        second, second_s = timed(
            lambda: engine.chat(
                transcript, max_tokens=args.max_tokens, learn=False
            )
        )
        control, control_s = timed(
            lambda: engine.chat(
                [{"role": "user", "content": question}],
                max_tokens=args.max_tokens,
                learn=False,
            )
        )
        summary = engine.state_summary()
    finally:
        engine.close()

    return {
        "condition": "replay",
        "model": args.model_name,
        "model_digest": summary["model_digest"],
        "code": args.code,
        "fact": fact,
        "question": question,
        "first_style": args.first_style,
        "learning": False,
        "first_reply": answer_record(
            first.text,
            args.code,
            prompt_tokens=first.prompt_tokens,
            completion_tokens=first.completion_tokens,
            finish_reason=first.finish_reason,
            elapsed_s=first_s,
            fast_weight_rows=first.fast_weight_rows,
            state_digest=first.state_digest,
            first_reply_echo_free=first_reply_echo_free(first.text, args.code),
        ),
        "answers": {
            "replay_memory": answer_record(
                second.text,
                args.code,
                prompt_tokens=second.prompt_tokens,
                completion_tokens=second.completion_tokens,
                finish_reason=second.finish_reason,
                elapsed_s=second_s,
                fast_weight_rows=second.fast_weight_rows,
                fast_weight_digest=second.fast_weight_digest,
                state_digest=second.state_digest,
            ),
            "no_memory": answer_record(
                control.text,
                args.code,
                prompt_tokens=control.prompt_tokens,
                completion_tokens=control.completion_tokens,
                finish_reason=control.finish_reason,
                elapsed_s=control_s,
                fast_weight_rows=control.fast_weight_rows,
                fast_weight_digest=control.fast_weight_digest,
                state_digest=control.state_digest,
            ),
        },
        "substrate": {
            "after_first_turn": summary_after_first,
            "after_second_turn": summary,
        },
    }


def run_kv_restart(args: argparse.Namespace) -> dict[str, Any]:
    """Restore a saved state in a new process and decode only the suffix.

    The state file was written by the ``kv`` condition. The turn file next to
    it holds the system prompt, the fact and the first reply, so the new
    process can rebuild the exact prompt prefix without regenerating anything.
    """
    if not args.kv_state:
        raise LiveEngineError("kv-restart needs --kv-state")
    state_path = Path(args.kv_state)
    turn_path = state_path.with_suffix(".turn.json")
    if not turn_path.is_file():
        raise LiveEngineError(f"missing turn file: {turn_path}")
    try:
        turn = json.loads(turn_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LiveEngineError(f"cannot read the turn file: {error}") from error
    if not isinstance(turn, dict):
        raise LiveEngineError("the turn file must contain an object")
    for key in ("code", "system", "fact", "first_reply", "question", "first_style"):
        if not isinstance(turn.get(key), str):
            raise LiveEngineError(f"the turn file has no string {key!r}")
    if turn["first_style"] not in FIRST_STYLES:
        raise LiveEngineError(
            f"the turn file has an unknown first style {turn['first_style']!r}"
        )

    engine = open_engine(
        args, use_delta=False, system_prompt=turn["system"]
    )
    try:
        state = KvState.load(
            state_path,
            expected_model_digest=engine.model_digest,
            expected_n_embd=engine.n_embd,
            expected_n_ctx=engine.n_ctx,
        )
        saved_state = turn.get("kv_state")
        if (
            not isinstance(saved_state, dict)
            or saved_state.get("sha256") != state.sha256
        ):
            raise LiveEngineError(
                "the state file does not match the turn file; refusing to mix them"
            )
        transcript = [
            {"role": "user", "content": turn["fact"]},
            {"role": "assistant", "content": turn["first_reply"]},
            {"role": "user", "content": turn["question"]},
        ]
        second, second_s = timed(
            lambda: engine.chat(
                transcript,
                max_tokens=args.max_tokens,
                restore=state,
                learn=False,
            )
        )
        summary = engine.state_summary()
    finally:
        engine.close()

    return {
        "condition": "kv-restart",
        "model": args.model_name,
        "model_digest": summary["model_digest"],
        "code": turn["code"],
        "fact": turn["fact"],
        "question": turn["question"],
        "first_style": turn["first_style"],
        "learning": False,
        "state_file": str(state_path),
        "first_reply": answer_record(
            turn["first_reply"],
            turn["code"],
            source="saved",
            first_reply_echo_free=first_reply_echo_free(
                turn["first_reply"], turn["code"]
            ),
        ),
        "answers": {
            "kv_restart_memory": answer_record(
                second.text,
                turn["code"],
                prompt_tokens=second.prompt_tokens,
                completion_tokens=second.completion_tokens,
                finish_reason=second.finish_reason,
                elapsed_s=second_s,
                restored_tokens=len(state.tokens),
                decoded_suffix_tokens=second.prompt_tokens - len(state.tokens),
                fast_weight_rows=second.fast_weight_rows,
                fast_weight_digest=second.fast_weight_digest,
                state_digest=second.state_digest,
            )
        },
        "substrate": {
            "after_restart_turn": summary,
            "kv_state": state.summary(),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="live_vs_frozen",
        description=(
            "Measure the same memory in the frozen mode (fact re-injected in "
            "the context), the live mode (fact carried by fast weights), and "
            "the kv mode (fact carried by a restored context state). Run one "
            "condition per invocation."
        ),
    )
    parser.add_argument(
        "--condition",
        choices=("frozen", "live", "kv", "replay", "kv-restart"),
        required=True,
    )
    parser.add_argument("--code", default=DEFAULT_CODE, help="the four digits to remember")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="system prompt")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="second-turn question")
    parser.add_argument(
        "--first-style",
        choices=FIRST_STYLES,
        default="echo",
        help=(
            "echo: the original fact; quiet: ask for a reply that does not "
            "repeat the code, so the saved state isolates the user sentence"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--out", default=None, help="write the JSON result to this path")
    parser.add_argument("--timeout", type=float, default=300.0, help="frozen HTTP timeout")

    frozen = parser.add_argument_group("frozen")
    frozen.add_argument("--frozen-url", default="http://127.0.0.1:8099")
    frozen.add_argument("--frozen-model", default="local")

    live = parser.add_argument_group("live")
    live.add_argument("--model", default=None, help="fused or base GGUF")
    live.add_argument("--lib", default=None, help="patched libllama shared library")
    live.add_argument("--shim", default=None, help="compiled LlamaSpike C shim")
    live.add_argument("--delta", default=None, help="delta JSON path")
    live.add_argument(
        "--kv-state",
        default=None,
        help=(
            "state file: write it in the kv condition, read it in kv-restart; "
            "the turn file is written next to it"
        ),
    )
    live.add_argument("--model-name", default="pmc-live")
    live.add_argument("--ctx", type=int, default=512)
    live.add_argument("--batch", type=int, default=64)
    live.add_argument("--threads", type=int, default=2)
    live.add_argument("--learning-rate", type=float, default=0.01)
    live.add_argument(
        "--learn-from-prompt",
        action="store_true",
        help="also learn the user's prompt tokens (off: only the instance's own speech)",
    )
    live.add_argument(
        "--no-learn",
        action="store_true",
        help="do not update the fast weights; use for the state and replay controls",
    )
    live.add_argument(
        "--no-digest",
        action="store_true",
        help="skip the model content digest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.condition in ("live", "kv", "replay", "kv-restart"):
        missing = [
            name
            for name, value in (
                ("--model", args.model),
                ("--lib", args.lib),
                ("--shim", args.shim),
            )
            if not value
        ]
        if missing:
            parser.error(f"{args.condition} needs " + ", ".join(missing))
    if args.condition == "kv-restart" and not args.kv_state:
        parser.error("kv-restart needs --kv-state")

    try:
        if args.condition == "frozen":
            result = run_frozen(args)
        elif args.condition == "live":
            result = run_live(args)
        elif args.condition == "kv":
            result = run_kv(args)
        elif args.condition == "replay":
            result = run_replay(args)
        else:
            result = run_kv_restart(args)
    except (ChatError, KvStateError, LiveEngineError, OSError) as error:
        print(f"live_vs_frozen: {error}", file=sys.stderr)
        return 1

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
