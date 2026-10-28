# SPDX-License-Identifier: GPL-3.0-only

# Benchmarks

The evaluation battery. Its job is to fail as easily as it passes.

## What is measured

| Metric | Question it answers |
|---|---|
| recall at k | can it retrieve what it consolidated? |
| forgetting rate | what did consolidation cost? |
| false-memory rate | does it confabulate? |
| interference rate | what did the new material break? |
| expected calibration error | does its confidence match its accuracy? |
| Brier score | the same question, squared |
| rumination pressure | is it looping rather than remembering? |
| sidecar overhead | how large is the public record next to the private interior? |

Every one of them measures **function**. None of them measures
consciousness, and the harness says so in its own output.

## Running it

```bash
python3 bench/run_eval.py --gold gold.json --predictions predictions.json
python3 bench/run_eval.py --gold gold.json --predictions predictions.json \
    --baseline baseline.json --k 5
```

The formats are documented in the header of `run_eval.py`.

## Stock-provider load probe

`stock_load_probe.c` links only against a stock `libllama` and loads a fused
GGUF to prove that an unmodified provider accepts it. It contains no PMC code
and reads no state: the frozen artifact is the model file, not the probe.

```sh
cc -O2 -I/path/to/llama.cpp/include -I/path/to/llama.cpp/ggml/include \
    bench/stock_load_probe.c \
    -L/path/to/llama.cpp/build/bin -lllama -lggml -lggml-base \
    -Wl,-rpath,/path/to/llama.cpp/build/bin -o stock_load_probe
./stock_load_probe MODEL.fused.gguf
```

Measured on the 8B fused file with llama.cpp `1248fd8` (no patch, stock
`libllama`): `n_vocab=128008`, `n_embd=2048`, `n_params=8467873216`,
`model_size=3326681856`, exit status 0. The loader logs the `pmc.state.*`,
`pmc.version`, `pmc.manifest` and `pmc.dream_id` metadata and then ignores it.
The publication file was re-fused with the portable manifest (base and output
recorded by basename, size, FNV identity and full-file SHA-256, no absolute
paths) and passes the same probe with the same numbers.

## The sidecar probe

`sidecar_probe.py` is a different kind of rig: it talks to a **real loader**
and asks the instance a question twice, once with its own state restored and
once with an empty context.

```bash
python3 bench/sidecar_probe.py --workspace <instance-dir> --spec spec.json
```

The spec holds the facts, the questions and the expected tokens. It belongs
to the instance, not to the framework: the rig ships, the data does not.

The two conditions are what make the result meaningful. A recall number on
its own says nothing; the same number with and without the restore says
whether the sidecar is doing anything at all. The rig also checks that the
public ledger never carries the content it points at, and that the journal's
hash chain still verifies.

Scoring is exact-token containment, computed offline. There is no model
judge: a judge is another model with its own error rate, and a string the
instance was told verbatim should come back verbatim.

## Live versus frozen memory

`live_vs_frozen.py` measures one fact in six conditions: the frozen mode (the
fact re-injected into the context every turn), the live mode (the fact carried
by the runtime's fast weights, absent from the second prompt), the kv mode (the
first turn's context state saved, restored into a cleared context, and only the
new suffix decoded), a replay mode (the full transcript re-decoded on the live
graph with no learning), a restart mode (the saved state loaded in a new
process), and a no-memory control. Run one condition per invocation so one
model copy exists at a time.

```sh
# A frozen server is listening with the model's Jinja template.
PYTHONPATH=src python3 bench/live_vs_frozen.py --condition frozen \
    --first-style quiet --frozen-url http://127.0.0.1:8099 \
    --out frozen-quiet.json

# Stop the server, then use the in-process live engine.
PYTHONPATH=src python3 bench/live_vs_frozen.py --condition live \
    --model /path/to/fused.gguf --lib /path/to/libllama.so \
    --shim /path/to/libpmc_spike.so \
    --delta /path/outside/the/repo/delta.json --out live.json

# Echo-free kv: no fast weights, save the state and the first turn.
PYTHONPATH=src python3 bench/live_vs_frozen.py --condition kv \
    --first-style quiet --no-learn \
    --kv-state /path/outside/the/repo/kv-quiet.pmcstate \
    --model /path/to/fused.gguf --lib /path/to/libllama.so \
    --shim /path/to/libpmc_spike.so --out kv-quiet.json

# Replay the same transcript on the live graph, without learning.
PYTHONPATH=src python3 bench/live_vs_frozen.py --condition replay \
    --first-style quiet \
    --model /path/to/fused.gguf --lib /path/to/libllama.so \
    --shim /path/to/libpmc_spike.so --out replay-quiet.json

# Restore the saved state in a new process; the turn file is next to it.
PYTHONPATH=src python3 bench/live_vs_frozen.py --condition kv-restart \
    --kv-state /path/outside/the/repo/kv-quiet.pmcstate \
    --model /path/to/fused.gguf --lib /path/to/libllama.so \
    --shim /path/to/libpmc_spike.so --out kv-quiet-restart.json

# The negative boundary: offer the state to a different model.
PYTHONPATH=src:bench python3 bench/kv_swap_probe.py \
    --state /path/outside/the/repo/kv-quiet.pmcstate \
    --model /path/to/other.gguf --lib /path/to/libllama.so \
    --shim /path/to/libpmc_spike.so --out kv-swap.json
```

The in-process commands used `--ctx 128 --batch 32 --threads 2
--max-tokens 24`. The frozen condition used the private `build3` server with
`--jinja`; the stock `1248fd8` build returns zero logits on LFM2, so the frozen
column is an independent provider baseline, not a stock-inference result.

Measured on the 8B fused artifact, code `4821`, greedy:

| condition | answer | second prompt tokens | second-turn work |
| --- | --- | --- | --- |
| frozen (fact in context) | `4821` | 69 | server: 23 new after 46 cached |
| control (no fact) | `1234` | 33 | server: 21 new after 12 cached |
| live, `lr=0.01` (own reply only) | `1234` | 33 | 33 decoded |
| live, `lr=2.0` (own reply only) | `1234` | 33 | 33 decoded |
| live, `lr=2.0` + prompt learning | `1234` | 33 | 33 decoded |
| kv (restored context state) | `4821` | 69 | 46 restored + 23 decoded |

With only the generated tokens learned, the first live reply is the frozen
first reply: `I'm sorry, but I don't have any information about the code 4821.`
The diagnostic on the question's hidden state (`h_norm 63.5252`) shows the
`482` row at bias `0.0` and the wrong `123` row at `+1.738`: the fact is
stored but not retrievable from a different context, and the wrong answer is
learned in the query context. The result is negative for output-row fast
weights as episodic memory; the frozen context wins.

The kv row is the contrast. The first turn's state was 861,339 bytes over 46
tokens (SHA-256
`b6698d22372bcd22493eb84fd76439123f49afeb8dcb76a0f113258e5eb8b11d`); the second
prompt was 69 tokens, and only the 23-token suffix was decoded after the state
was restored into a cleared context. The engine refuses a restore unless the
prompt prefix is token-identical to the saved one. The answer matched frozen
with the same fast-weight rows as the live condition, so the state, not the
rows, carried the retrieval.

### The echo-free control

The first kv run echoed the code in the assistant's refusal, so the state
contained `4821` twice. `--first-style quiet` asks for the single word
`noted`, and `--no-learn` leaves the fast weights at zero. On the same 8B the
first reply was `Noted.` (37 prompt, 3 completion; the frozen server used 4
completion tokens for the same text); `first_reply_echo_free` is `true`. The
saved state was **787,539 bytes over 40 tokens** (SHA-256
`36750dcb407ef18d00e0c15ff190282fc2a109b84608326fb8d0db9bfc15ae64`). The
second prompt was 63 tokens; 40 were restored, **23 were decoded**, and the
answer was `4821.`.

| quiet condition | first reply | answer | second prompt | second-turn work |
| --- | --- | --- | --- | --- |
| frozen | `Noted.` | `4821.` | 63 | server: 23 new after 40 cached |
| replay | `Noted.` | `4821.` | 63 | 63 decoded |
| kv | `Noted.` | `4821.` | 63 | 40 restored + 23 decoded |
| no memory | — | `1234` | 33 | frozen: 21 new after 12 cached; replay: 33 decoded |

The replay condition uses the same engine, template, sampler and first reply,
re-decodes all 63 tokens with `learn=False`, and answers `4821.`. Its no-memory
control answers `1234`. The frozen server with the same quiet first turn
answers `4821.` and its no-memory control answers `1234`. The state and the
re-decoded text agree with no echo and no fast weights: the answer comes from
the saved user sentence after a context reset.

### Restart and the model boundary

`KvState.save` writes one file: magic `PMCKV1\0`, a big-endian `u32` header
length, a compact JSON header (format, token count, SHA-256, model digest,
`n_embd`, `n_ctx`), then the raw state bytes. Loading verifies the magic,
header length, payload length, SHA-256, token list and identity. A new process
(`--condition kv-restart`) loads the same model, rebuilds the exact first turn
from the turn file, checks that the turn's state SHA matches the loaded state,
and restores it after the prefix check. Measured:
40 restored, 23 decoded, `4821.`, 2.955 s, the KV file's SHA-256 unchanged, and
the live delta's hidden-state digest
`3f8a01920faa29ec19a54eb60d7ee77ea15c1aa1d1c6f42576a813e059ab433e` identical
to the in-process kv run. This is runtime context state surviving a process
restart on the same model and geometry, not weight-resident memory.

`bench/kv_swap_probe.py` offers the 8B state to the 350M engine. The state
carries digest `4b3c8348…` and 2048 dimensions; the engine has digest
`7e6f7264…` and 1024 dimensions. `matches=false`, and the engine refuses with
`refused-identity` ("the restored state was saved for a different model or
context geometry") before the C restore. That is the negative boundary of the
Substrate Swap Test: a KV cache is model-bound. The positive test on a trained
PMC state is not yet measured.

The `--learn-from-prompt` option also changed the first reply (prompt learning
biases generation before the instance speaks), so it is not a free memory
improvement. The pure helpers and the probe parser are tested with:

```sh
PYTHONPATH=src:bench python3 -m pytest bench/live_vs_frozen_test.py \
    bench/kv_swap_probe_test.py -q
```

## Dream reference check

`dream_reference.py` runs the dream pipeline against a real 50M checkpoint. It
is a checkpoint-handling check, not a six-metric memory result: the candidate
is the identity copy of the checkpoint and the evaluator returns a fixed
metric set, so the gate must refuse the candidate on `recall_newer` (no
improvement). The checkpoint must stay byte-identical, and a tainted
checkpoint digest must be refused before the pipeline copies anything.

```sh
PYTHONPATH=src:bench python3 bench/dream_reference.py \
    --checkpoint <evidence-cache>/pmc-50m
```

The script exits non-zero unless all eight checks pass:

- the identity dream is rolled back;
- the gate names `recall_newer`;
- the dream leaves the checkpoint byte-identical;
- the dream journal chain verifies;
- the tainted dream is refused;
- the refusal names the checkpoint digest;
- the refusal leaves the checkpoint byte-identical;
- the refusal journal chain verifies.

Measured on the 50.4M reference core (checkpoint digest
`10ab040e…`, 201,574,436 bytes): all eight checks pass. This is E2 for
checkpoint handling. The metric values were supplied by the harness and the
candidate learned nothing, so it is not evidence that a real consolidation
would pass the gate.

## PMC-mini Substrate Swap Test

`substrate_swap.py` is a bench-level proof of the state-bank swap, not the
50–100M PMC. The pre-registration, the word lists, the thresholds and the
three negative results are in
[`docs/lab.md`](../docs/lab.md#the-substrate-swap-test-pre-registered).
The test trains two host adapters, writes one shared key/value state into both
fused GGUFs, and evaluates the same state bytes on the 350M and the 8B. The
state is `pmc.state.swap.*` GGUF metadata; the adapters are the host interface,
not the state. A stock provider loads the file and ignores the banks; the
runtime reader applies them. The command is intentionally split so only one
model is loaded at a time. The v1 and v2 state is `8×64`; the v3 state is
`8×16` with a cue-delta anchor.

```sh
# One host at a time.  v1 uses the last prompt token; v2 uses the last cue
# token; v3 uses h(cue)-h(thing) at the same position.  Every spec pins the
# split, templates and seed.
PYTHONPATH=src:bench python3 bench/substrate_swap.py extract \
    --host 350m --model /path/to/LFM2.5-350M.gguf \
    --lib /path/to/libllama.so --shim /path/to/libpmc_spike.so \
    --spec bench/swap_words_v3.json --out h350-v3.json
PYTHONPATH=src:bench python3 bench/substrate_swap.py extract \
    --host 8b --model /path/to/LFM2.5-8B-A1B.gguf \
    --lib /path/to/libllama.so --shim /path/to/libpmc_spike.so \
    --spec bench/swap_words_v3.json --out h8b-v3.json

PYTHONPATH=src:bench python3 bench/substrate_swap.py train \
    --h-350m h350-v3.json --h-8b h8b-v3.json \
    --spec bench/swap_words_v3.json --out adapters-v3.json

PYTHONPATH=src:bench python3 bench/substrate_swap.py fuse \
    --host 350m --spec bench/swap_words_v3.json \
    --base /path/to/LFM2.5-350M.gguf --adapters adapters-v3.json \
    --out fused-350m.gguf --rows rows-350m.bin --meta meta-350m.json \
    --pmc crates/target/release/pmc
# Repeat fuse with --host 8b and the 8B base and output.

PYTHONPATH=src:bench python3 bench/substrate_swap.py eval \
    --spec bench/swap_words_v3.json \
    --model-a fused-350m.gguf --model-b fused-8b.gguf \
    --adapters adapters-v3.json --lib /path/to/libllama.so \
    --shim /path/to/libpmc_spike.so --out eval-v3.json --quiet
```

The v1 and v2 runs use the same commands with their spec files and
`h350-v1`/`h8b-v1` or `h350-v2`/`h8b-v2`. The eval reads the state banks from
the fused files, checks the state and adapter digests, and runs every arm: PMC,
no state, fresh random state, text re-injection, and the unseen control cues.
It reports every fact and every miss.

Measured (v1: `bench/swap_words.json`, last token, early-stopped 15 epochs;
v2: `bench/swap_words_v2.json`, cue token, full 300 epochs; v3:
`bench/swap_words_v3.json`, cue minus neutral cue, 30 training cues, `8×16`
state, full 300 epochs):

| run | host A PMC | host B PMC | no state A/B | fresh A/B | control FP A/B | re-injection A/B | pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | 5/8 | 1/8 | 0/0 | 1/2 | 2/1 | 4/4 | no |
| v2 | 8/8 | 0/8 | 0/0 | 2/2 | 1/1 | 5/6 | no |
| v3 | 8/8 | 2/8 | 0/0 | 2/2 | 3/1 | 6/6 | no |

The v1 state is `4a43d78a…`; the v1 adapters are `a5d918e4…` (350M) and
`83ebfab2…` (8B). The v2 state is `b1ac7188…`; the v2 adapters are `681d62da…`
(350M) and `6a5d8b02…` (8B). The v3 state is `35c3430e…`; the v3 adapters are
`7a4db210…` (350M) and `fe3b635f…` (8B). All three fused pairs load in stock
`libllama` (`n_vocab=65544` on the 350M, `128008` on the 8B) with the banks
ignored. The v3 fused files were SHA-256 `db3970e3…` (350M, 229,591,968 bytes)
and `782d3ce4…` (8B, 3,335,440,736 bytes); they are regenerable from the bases
and the saved adapters and were removed after the stock-load and evaluation.
The state identity is identical across hosts; the adapters differ. The
no-state and fresh-state controls fail as intended. The v1/v2 control false
positives are at the threshold; the v3 host-A control is 3/8, over it. The
state identity and no-target-training conditions hold.

The negative is the cross-host interface, not the state bytes. In v1 host B is
at chance even in sample (0.045) because the early-stopped 15-epoch adapter is
underfit. In v2 host A is 8/8 and host B is 22/22 in sample, but host B is 0/8
on the eight unseen cues: the linear 8B adapter memorizes the 22 training cue
mappings and does not generalize. In v3 the cue delta and the 16-value state
make host A 8/8 and host B 15/30 in sample, but host B is 2/8 on the unseen
cues and the host-A control is 3/8. The lower capacity underfits host B
instead of generalizing; the host-A evaluation margins are 0.970–0.989, while
host B answers only `silver`→`basil` and `rocket`→`rose`. The v3 evaluation
facts were not used to tune the adapter. The text re-injection is the model's
own memory, not the swap; it is reported, not required.

Two implementation bugs were found during the runs. The added output rows
were first `-100.0` instead of `0.0`; the `<pmc:swap-*>` tokens then won on
several retrieved facts and host A scored 2/8 in the first pass. The v2
evaluation also first read the last prompt token while the v2 adapter was
trained on the cue token; host A scored 4/8 in the first pass. Both are fixed;
the table is after the fixes. The v3 path re-decodes the actual question after
the neutral-cue probe so the sampler reads the right context. The pure helpers,
the prompt swap, the projection, the adapter, the spec checks, the 16-dim
training and the pass arithmetic are tested with:

```sh
PYTHONPATH=src:bench python3 -m unittest bench.substrate_swap_test -v
PYTHONPATH=src:bench python3 bench/substrate_swap.py check-spec \
    --spec bench/swap_words_v3.json
```

## Hidden-state extraction for the dream

The dream trainer needs `h`, the final-token hidden state of a prompt. The
chat server cannot produce it today; the offline CLI can.

### Why not `--embeddings` on the server

`--embeddings` does not add a hidden-state field to chat completions. It
**restricts the server to the embedding use case**, and it is a startup flag.
Applying it to the process on `:8084` would mean restarting a server that is
running a long benchmark. The lab rule forbids that. The flag is for a
dedicated embedding server, not for a chat server that also returns vectors.

Upstream PR #30732 (`return_last_hidden_state` on chat completions) is open
and not merged; it was checked on 2026-10-27. The local build has no such
symbol, so even a fresh server would not return `h` through the chat API.

### The offline CLI

`llama-embedding` (source: `examples/embedding`) is a one-shot process. It
loads the GGUF, encodes one prompt, prints the vector and exits. It does not
bind a port and does not touch the running server.

The server build directory has `LLAMA_BUILD_EXAMPLES=OFF`, so
`cmake --build <server-build> --target llama-embedding` fails with
`No rule to make target`. Do not turn examples on in that directory while
the benchmark owns it. Build in a directory that already has examples
enabled, or in a separate one:

```bash
cmake -S <llama.cpp> -B <llama.cpp>/build-embed \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build <llama.cpp>/build-embed --target llama-embedding -j2
```

The binary is `<build-dir>/bin/llama-embedding`. On this host the existing
examples-enabled `build/` directory was used, so the binary is
`<llama.cpp>/build/bin/llama-embedding`.

Extraction command:

```bash
<llama-embedding> -m <model.gguf> -p "<prompt>" \
    --pooling last --embd-normalize -1 --embd-output-format raw
```

- `--pooling last` selects the last prompt token. The graph picks the
  highest position in the sequence, and `t_embd` is assigned after the
  model's final output norm. This is the vector the trainer needs.
- `--embd-normalize -1` leaves the vector at the model's scale; it only
  suppresses the CLI's extra L2 rescaling. The model's final norm still
  applies.
- `--embd-output-format raw` prints one whitespace-separated vector per
  line. Without it, one prompt still prints the full vector, but several
  prompts are truncated to 16 values.

Do **not** add `--log-disable`. The CLI writes the vector through the same
common log as the loader messages; pausing that log discards the vector too.
Measured on the 350M model: the exact `--log-disable` invocation exits 0 and
prints 0 bytes on both stdout and stderr. Parse stdout instead.

`--pooling none` returns one vector per token, not one per prompt; it does
not error. For the 9-token probe it printed 9 vectors of 1024 floats, and
the last line was bit-identical to the `--pooling last` vector. Use `last`
when one vector per prompt is wanted; use `none` only when every token is
wanted.

### The probe

`hidden_probe.py` runs the CLI once per prompt and writes JSON Lines:

```bash
python3 bench/hidden_probe.py \
    --llama-embedding <llama.cpp>/build/bin/llama-embedding \
    --model <model.gguf> \
    --text "the gate code is 47-rosso" \
    --text "the sky is green and the sea is dry" \
    --pooling last --normalize -1 --expected-dim 1024 \
    --json-out hidden.jsonl --cosine-out cosine.json \
    --dump-vectors vectors/
```

Each line holds the prompt SHA-256 (first 12 hex), dimension, L2 norm,
vector SHA-256 (first 12 hex, over packed little-endian float32), pooling
mode, model basename and normalize value. `--cosine-out` writes the pairwise
cosine matrix, and `--dump-vectors` keeps the raw float32 vectors for a
trainer. `--text-file` reads one prompt per line.

The probe exits non-zero and names the prompt index if the CLI fails, prints
no vector, prints a non-finite value, returns a dimension other than
`--expected-dim`, or returns inconsistent dimensions across prompts. Unit
coverage for the parser, hashing and failure path is in
`bench/hidden_probe_test.py` (`python3 -m pytest bench/hidden_probe_test.py`).

### Evidence status

These are extraction checks, not benchmark scores; the "no published
numbers" note below refers to the evaluation battery.

- **350M, measured (E2):** real GGUF, real CLI. Dimension 1024, all values
  finite, two runs produced the same vector hash, and two different prompts
  had cosine 0.3767 (below the 0.99 distinctness bound). `--pooling none`
  returned per-token vectors and its last vector matched `last`.
- **8B, measured via the patched live spike (E2):** the 8B server was stopped;
  `src/pmc/runtime` reads a 2048-dim hidden state (norm 49.708656, sha256
  06ced724d445) and 128000 logits in one decode. The bench's own 8B CLI probe
  is still unverified; the number above comes from the live runtime, not this
  benchmark.
- **Post-norm caveat:** the CLI exposes the hidden state after the model's
  final output norm. Whether the trainer wants that or the pre-norm residual
  stream is not yet measured; if it needs the pre-norm value, this CLI does
  not provide it without a code change.
- **Tokenizer caveat:** the 350M GGUF does not set
  `tokenizer.ggml.add_eos_token`; the CLI warns and the last position is the
  final text token, not an EOS token. Check this per model before training.

## Cross-host gate over the training cues

`pmc_state_views.py` builds the views file the v4 mechanism trains on, from
the already-extracted hidden states and the frozen word list. It writes only
the 30 training facts and the 30 training questions for both hosts, so a gate
run cannot read the evaluation or control cues that the attempt it decides is
scored against.

```sh
PYTHONPATH=src:bench python3 bench/pmc_state_views.py \
    --spec bench/swap_words_v3.json \
    --hidden-350m /path/to/h350-v3.json --hidden-8b /path/to/h8b-v3.json \
    --out state-views-v4.pmc --manifest state-views-v4.json
```

`train_pmc_state` trains the encoder from Rust and cross-validates it, then
`crosshost_cv.py` trains the frozen v1–v3 mechanism on the **same** held-out
folds and applies the pre-registered gate:

```sh
PMC_VIEWS=state-views-v4.pmc PMC_REPORT=report.json PMC_FOLDS=6 \
    cargo run --release --offline --example train_pmc_state

PYTHONPATH=src:bench python3 bench/crosshost_cv.py \
    --spec bench/swap_words_v3.json \
    --hidden-350m /path/to/h350-v3.json --hidden-8b /path/to/h8b-v3.json \
    --report report.json --dims 64 --out gate.json
```

The result on 2026-10-28: 706,112 parameters, loss 0.786 to 0.0014, 30/30 on
every in-sample view kind, and **11/30** held-out cross-host questions across
six folds with one fold at 0. The frozen mechanism on the same folds: 2/30.
The gate required 15/30 with no empty fold, so it failed. Full numbers and
artifacts are in [`docs/lab.md`](../docs/lab.md#the-v4-mechanism-pre-registered-2026-10-31).

## Rules for a result

1. A result without a baseline is not a result.
2. A result without a seed and a configuration is not reproducible.
3. A result that only holds for the model that produced it is a result
   about the model, not about the method. Judge it independently.
4. Negative results get published with the same care as positive ones.

## What does not exist yet

There is no trained PMC checkpoint with weight-resident recall, and no
Kaggle notebook. The 50.4M reference core is a token-level checkpoint trained
on one fixed sequence, not the PMC memory; it has no episodic state and no
recall measurement. The numbers above are infrastructure, same-memory,
runtime-context-state and dream-checkpoint-handling measurements, not the
recall/forgetting battery. When the first trained evaluation lands, it lands
here, with its configuration, its seed, its manifest and its ablations.
