<!-- SPDX-License-Identifier: GPL-3.0-only -->

# WP7 live runtime

This directory contains the WP7a feasibility spike and the WP7 runtime built on
top of it.

The spike loads one GGUF model in one process, reads the final-token hidden
state and the logits, updates one fast-weight row, applies a sampler logit bias,
and shows the sampled token before and after the bias. There is no HTTP, no
server, no second model process, and no framework code in the inference path.

The runtime (`live_state.py`, `live_engine.py`, `live_server.py`) keeps one
model and one context in one process, serves an OpenAI-compatible API from that
process, learns from the tokens the instance generates, and persists a delta.
It never starts a second model server. The frozen mode is still the artifact on
a stock provider; the live mode is the optional substrate-resident runtime.

The spike deliberately separates two claims:

* **same decode** — one `llama_decode` call produces both `h` and logits;
* **paired decode** — the same token sequence is decoded twice in two contexts
  in the same process, one context for `h` and one for logits.

On stock llama.cpp the LFM2.5-350M same-decode claim fails: `--mode single`
reports the failure and exits 2. `patches/llama-lfm2-logits-with-embeddings.patch`
fixes the LFM2 graph; with it, the 350M and the 8B `lfm2moe` both pass the gate.
The measured results are below. The stock failure remains a recorded negative.
`--mode paired` is an explicitly labelled fallback; it must not be cited as
evidence that `h` and logits coexist in one decode.

## Files

* `live_spike.py` — the ctypes caller, the sampler chain, and the fast-weight
  row arithmetic. Its `tokenize` takes `parse_special`, which the runtime sets
  for templated chat prompts.
* `live_spike_test.py` — unittest coverage for the pure helpers. It does not
  load a model.
* `state_banks.py` — metadata-only reader for `pmc.state.*` banks in a fused
  GGUF; it never reads tensor data.
* `state_banks_test.py` — synthetic and real-file coverage for the reader.
* `state_memory.py` — validates and consumes a self-contained
  `pmc.state.memory.*` bank: unit keys, token values, a hidden-to-key
  projection, a positive logit bias and an optional cosine floor.
* `tests/test_state_memory.py` — nine standard-library checks for the reader,
  the engine path and the base-model control, with a synthetic GGUF and a fake
  spike.
* `live_state.py` — `FastWeights` and `LiveDelta`: rows, last hidden state,
  state digest, turns, model digest, strict canonical JSON, atomic save.
* `live_state_test.py` — fast-weight, delta and persistence tests.
* `live_engine.py` — the in-process engine: chat template, prompt decode,
  hidden state, greedy sampling with fast-weight biases, learning, delta save,
  and `KvState` snapshots with model/geometry identity and a one-file format.
* `live_engine_test.py` — pure helpers plus an env-gated real-model test
  (`PMC_LIVE_MODEL`, `PMC_LIVE_LIB`, `PMC_LIVE_SHIM`).
* `live_server.py` — OpenAI-compatible HTTP face: `/v1/models`,
  `/v1/chat/completions`, `/v1/pmc/state`, `/health`. One process, one engine.
* `live_server_test.py` — HTTP tests with a fake engine (no model).
* `llama_spike_shim.c` — the C shim compiled against the llama.cpp headers in
  the repository; only C owns the ABI boundary, including the embedding and
  logits use; Python only handles opaque pointers and scalar arguments.

## State banks and the frozen 8B load

The fusion CLI writes state banks as two metadata arrays per bank,
`pmc.state.<name>.dims` and `pmc.state.<name>.data`. `state_banks.py` reads
only the GGUF header and the metadata section, never the tensor data. It
returns each bank as a `StateBank(name, dims, data)` and rejects a missing
half, a wrong element count, non-positive dimensions and non-finite values.
The real 8B fused file yields `bank` with dims `(2, 4)` and eight finite
values; the synthetic and real-file tests are in `state_banks_test.py`.

The same 8B file loads in a stock `libllama` built from `1248fd8` without this
patch: `bench/stock_load_probe.c` reports `n_vocab=128008`, `n_embd=2048`,
`n_params=8467873216` and `model_size=3326681856`. The stock loader logs the
`pmc.state.*`, `pmc.version`, `pmc.manifest` and `pmc.dream_id` metadata and
then ignores it. The CLI entry point in this upstream tree is the unified
`llama-app`, whose build needs the UI asset pipeline; the measured check uses
the stock C API. This is the frozen-provider claim for the 8B artifact, and it
is separate from the patched live graph below.

### The state-memory bank the engine consumes

The engine consumes one self-contained schema, stored with the same
`.dims`/`.data` convention:

```text
pmc.state.memory.keys             [n, d]       unit-length keys
pmc.state.memory.values           [n]          token ids
pmc.state.memory.projection       [d, n_embd]  hidden -> key space
pmc.state.memory.projection_bias  [d]
pmc.state.memory.logit_bias       [1]          positive readout bias
pmc.state.memory.min_similarity   [1]          optional cosine floor
```

At the first generated token the engine projects the current hidden state,
normalises it, chooses the nearest key and adds `logit_bias` to the stored
value token when the cosine is at least the floor. The bank is metadata only,
and the engine does not name, classify or gate the hidden state (D-17).

`tests/test_state_memory.py` drives the real `LiveEngine.chat` path with a
synthetic GGUF and a fake patched-lib spike. With the bank the engine emits the
stored token; without it the base token. Partial, unknown, non-unit,
non-integer or out-of-vocabulary banks are refused. This is a synthetic
mechanics check, not a trained 50–100M memory. The PMC-mini
`pmc.state.swap.*` bank needs the benchmark's host projection, so the engine
reports it as metadata and does not consume it.

## Live runtime

One process owns one model and one context. `LiveEngine.chat` applies the
model's chat template, tokenizes the templated prompt with
`parse_special=True`, decodes it in `n_batch` chunks, keeps the final hidden
state, samples greedily with the fast-weight readouts as logit biases, learns
from the generated tokens, and persists the delta. `LiveServer` wraps the same
engine in an HTTP layer; `/v1/pmc/state` returns metadata only (model digest,
embedding size, vocabulary size, row count, fast-weight digest, state digest,
turn count, bank names and dimensions, and the state-memory summary: counts,
projection dimensions, bias and cosine floor), never hidden values, keys or
value bytes.
Sampling is greedy and non-streaming; both are explicit 400 errors, not silent
fallbacks. A delta written for a different embedding size or model digest is
refused.

### The templated prompt must be tokenized with `parse_special=True`

The chat template writes control tokens such as `<|im_start|>` as text. With
`parse_special=False` the tokenizer splits those markers into characters. On
the 8B model the measured prompt grew from 25 to 43 tokens and the model
answered `<|im_start` instead of the requested word. The engine now tokenizes
the templated prompt with `parse_special=True`; the same model, prompt and
greedy decode then produced the same text as a frozen `llama-server --jinja`
run: `The word "apple` (25 prompt tokens, 4 completion tokens). This was a
live-versus-frozen mismatch in the runtime, not a model-quality observation.

### The fast-weight step is a logit-space step

The raw SGD row `W <- W + lr (1 - p) h` has readout `lr (1 - p) ||h||^2`. On
the measured 8B hidden state (`h_norm = 49.7`), `lr = 0.01` would make the first
bias `24.7` logits and collapse the sampler onto one token. The engine divides
the step by `||h||^2`: the configured learning rate is a step in logit space,
and the first bias is `lr (1 - p)`. The measured rows after the 8B turn have
norms between `1e-6` and `1.4e-4`.

### Measured 8B live turn

The fused 8B (`n_embd=2048`, `n_vocab=128008`, state bank `[2,4]`) with the
system "You are a terse assistant." and the user "Say the word apple." gave 25
prompt tokens, 4 completion tokens, `The word "apple`, 4 fast-weight rows, one
persisted turn, and the state bank read from the GGUF. The HTTP server returned
the same completion with `prompt_tokens=25`, `completion_tokens=4`, the model
digest and the state summary.

The 350M probe loads and runs the same engine, but its chat template leads to
an EOG first token: it is a hidden-state probe, not a chat model. The live
runtime is measured on the 8B target.

## Same-memory result: frozen context, live fast weights, restored KV state

`bench/live_vs_frozen.py` measures one fact in the conditions below on the same
8B model and greedy decode:

| condition | fact where | measured answer | second prompt |
| --- | --- | --- | --- |
| frozen | re-injected in the context | `4821` (correct) | 69 tokens; 23 new after 46 cached |
| control | no fact | `1234` (the model's prior) | 33 tokens |
| live, `lr=0.01` | fast weights from the first reply | `1234` | 33 tokens, no fact |
| live, `lr=2.0` | fast weights from the first reply | `1234` | 33 tokens, no fact |
| live, `lr=2.0`, prompt learning | fast weights from prompt and reply | `1234` | 33 tokens, no fact |
| kv | restored context state | `4821` (correct) | 69 tokens; 46 restored + 23 decoded |

The first reply in the frozen, live and kv conditions was identical:
`I'm sorry, but I don't have any information about the code 4821.` The model
repeats the code in its refusal, so the fast weights do get a `482` row to
learn; the problem is retrieval, not storage.

The diagnostic explains the fast-weight miss on the question's hidden state
(`h_norm = 63.5252`):

* the fact token `482` is present in the delta, but its bias at this hidden
  state is `0.0`; the row is stored but not retrievable from a different
  context;
* the wrong answer's token `123` has bias `+1.738`, because the live engine
  generated `1234` in this very context and learned it; a later turn would
  reinforce the error;
* the first reply's function words (`I`, `'m`, ` information`, ` don`, ` any`)
  carry small biases, while the fact does not.

The first `kv` condition is the substrate-resident contrast. It saves the first
turn's whole context state, clears the context, restores the state into it, and
decodes only the suffix of the second prompt. The measured state was
**861,339 bytes over 46 tokens** (SHA-256
`b6698d22372bcd22493eb84fd76439123f49afeb8dcb76a0f113258e5eb8b11d`); the second
prompt was 69 tokens, so **23 tokens were decoded** instead of 69. The engine
checks that the new prompt's prefix is token-identical to the saved prefix and
refuses the restore otherwise; it passed. The answer matched the frozen
condition (`4821`) with the same 18 fast-weight rows as the live condition.

The caveat of that run was the echo: the first reply repeated `4821`, so the
saved state contained the code twice, in the user turn and in the assistant's
refusal. The live fast-weight condition saw the same echo tokens and still
answered `1234`, and the frozen condition re-decodes both sentences, so the
contrast was between *state* and *rows*, not between two different transcripts.
The echo-free control below closes that caveat.

### The echo-free control

`--first-style quiet` asks the first reply for the single word "noted";
`--no-learn` keeps the fast weights at zero, so the answer cannot come from
the rows. On the same 8B, the first reply was `Noted.` (37 prompt, 3
completion). It contains no `4821`; the output's
`first_reply_echo_free` is `true`. The saved state was **787,539 bytes over 40
tokens** (SHA-256
`36750dcb407ef18d00e0c15ff190282fc2a109b84608326fb8d0db9bfc15ae64`). The
second prompt was 63 tokens; 40 were restored, **23 were decoded**, and the
answer was `4821.`.

The replay control uses the same engine, template, sampler and first reply,
re-decodes all 63 tokens with `learn=False`, and answers `4821.`. Its
no-memory control (33 prompt tokens) answers `1234`. The frozen server
(`build3`, `--jinja`) with the same quiet first turn answers `4821.` and its
no-memory control answers `1234`. The state and the re-decoded text therefore
agree, with no echo and no fast weights. The saved state carries the user
sentence, and the answer is retrieved from it after a context reset.

### Persistence across a restart

`KvState.save` writes one file: the magic `PMCKV1\0`, a big-endian `u32`
header length, a compact JSON header (format, token count, SHA-256, model
digest, `n_embd`, `n_ctx`), then the raw state bytes. Loading verifies the
magic, header length, payload length, SHA-256, token list and identity.

A new process (`--condition kv-restart --kv-state ...`) loads the same model,
loads the state, rebuilds the exact first turn from the turn file next to the
state, checks that the turn's state SHA matches the loaded state, and restores
it after the prefix check. Measured: 40 restored, 23
decoded, `4821.`, 2.955 s, the KV file's SHA-256 unchanged, and the live
delta's hidden-state digest
`3f8a01920faa29ec19a54eb60d7ee77ea15c1aa1d1c6f42576a813e059ab433e` identical
to the in-process kv run. This is persistence of runtime context
state across a process restart on the same model and geometry. It is not
weight-resident memory, and it is not a model swap.

### The model boundary

`KvState` carries the model digest, embedding size and context size.
`LiveEngine.chat` rejects a state whose identity does not match before it calls
`llama_state_set_data`. The real cross-model probe offers the 8B state to the
350M engine: state digest
`4b3c83481216c05a81d466cd0e19f1e43f6732d52ee899de8355a9a840a28a50` / 2048
dimensions against engine digest
`7e6f72643caafc9a68256686638c4d7916f2cec76d1df478d4c3ddcd95a6aed4` / 1024
dimensions; `matches=false`; outcome `refused-identity` with "the restored
state was saved for a different model or context geometry". The probe is
`bench/kv_swap_probe.py`. This is the negative boundary of the Substrate Swap
Test: a KV cache is model-bound. The positive test on a trained PMC state is
still not measured.

This is a positive result for **runtime context state** and a negative result
for **output-row fast weights as episodic memory**. The rows are exact per
token and keyed by the hidden state at learning time; a fact learned in one
sentence does not transfer to a different question. The KV state is not
weight-resident memory: it lives in the live process unless its bytes are
saved, is bound to the same model and context geometry, and is not written
into the GGUF. It now survives a restart when the bytes and the first turn are
saved, but it does not survive a model swap. The `--learn-from-prompt` variant
also biases the first reply before the instance speaks; that is a real
behaviour change, not just a memory improvement. The next gate is the
Substrate Swap Test on the trained PMC state, which is not bound to one
model's KV geometry.

## Exact C API sequence

The calls below are the sequence the spike uses. Everything is one process.

1. Load the model:
   `llama_model_default_params()` → `llama_model_load_from_file(path, params)`.
2. Create a context:
   `llama_context_default_params()`; set `n_ctx=128`, `n_batch=32`,
   `n_ubatch=32`, `n_threads=2`, `n_threads_batch=2`, `embeddings=true`,
   `no_perf=true`; call `llama_init_from_model(model, params)`.
3. Get the vocabulary and tokenize:
   `llama_model_get_vocab(model)` → `llama_tokenize(vocab, text, len, ...)`.
4. Allocate and fill a batch:
   `llama_batch_init(n_tokens, 0, 1)`; set `token`, `pos`, `n_seq_id`,
   `seq_id`, and the `logits` output flags. The batch arrays are allocated by
   llama.cpp but are **not initialized**.
5. Decode: `llama_decode(ctx, batch)`.
6. Extract both tensors for the last output row:
   `llama_get_embeddings_ith(ctx, -1)` and `llama_get_logits_ith(ctx, -1)`.
7. Sample:
   `llama_sampler_chain_default_params()` → `llama_sampler_chain_init(params)`
   → `llama_sampler_init_logit_bias(n_vocab, 1, &bias)` →
   `llama_sampler_chain_add(chain, bias_sampler)` →
   `llama_sampler_init_greedy()` → `llama_sampler_chain_add(chain, greedy)` →
   `llama_sampler_sample(chain, ctx, -1)`.

The order of the chain matters: the logit-bias stage runs before the greedy
stage. `llama_sampler_chain_add` takes ownership of each added sampler; the
chain itself is released with `llama_sampler_free(chain)` only.

### Saving and restoring the context state

The state API is three scalar/pointer calls, so the spike binds it directly:
`llama_state_get_size(ctx)`, `llama_state_get_data(ctx, dst, size)` and
`llama_state_set_data(ctx, src, size)`. The measured fact that matters:
`llama_state_get_size` on a freshly cleared context returns only the empty-state
header (**463 bytes** here), not the buffer needed for a restored state. The
echoing first-turn state was **861,339 bytes over 46 tokens**; the echo-free
state was **787,539 bytes over 40 tokens**. The load therefore passes the saved
buffer and its own size; it must not compare that size to a fresh context's
`state_size`. The C call returns the number of bytes read, and the spike
requires it to equal the saved size.

The engine's restore rule is strict: the new prompt must start with exactly the
saved token prefix. The engine tokenizes the full prompt, compares the prefix,
clears the context, loads the state, and decodes only the suffix at
`start_pos = len(prefix)`. A prefix mismatch is a hard error, never a silent
fallback. The state is bound to the model and context geometry; a 350M state
cannot be loaded into an 8B context. The engine's save rule is also explicit:
if the completion budget ended before the last sampled token was decoded, it
decodes that token so the saved state covers the whole reply. The state is not
written into the GGUF and not persisted by the delta; the benchmark holds the
bytes in memory for the second turn.

## Why the parameter structs live in C

`llama_context_params` and `llama_model_params` are not stable ABI. This build
carries extra fields (`n_rs_seq`, `n_outputs_max`, `ctx_type`, `samplers`,
`ctx_other`, and several trailing booleans). A ctypes mirror compiled against a
different `llama.h` can silently misalign the struct and corrupt the context.
The spike therefore uses a tiny C shim compiled against the exact header of
the build under test. Python only handles opaque pointers and scalar
arguments. The shim exposes:

```c
void *pmc_model_load(const char *path, char *err, int32_t errlen);
void *pmc_ctx_new(void *model, uint32_t n_ctx, uint32_t n_batch,
                  int32_t n_threads, int32_t embeddings, int32_t no_perf,
                  char *err, int32_t errlen);
void  pmc_ctx_free(void *ctx);
void  pmc_model_free(void *model);
```

The implementation is deliberately small: call the default-params function,
set the fields above, call the load/init function, check for NULL, and report
the error string. The shim sets `n_gpu_layers=0` and `use_mmap=true`; this
spike is CPU-only. The shim source is `llama_spike_shim.c` in this directory.

Build it against the same headers and library as the spike:

```sh
LLAMA=/path/to/llama.cpp
cc -O2 -fPIC -shared -o libpmc_spike.so src/pmc/runtime/llama_spike_shim.c \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  -L"$LLAMA/build/bin" -lllama \
  -Wl,-rpath,"$LLAMA/build/bin"
```

## Batch pitfall

`llama_batch_init` allocates the arrays but leaves their contents
uninitialized. A first probe that filled only `token` and `logits` failed
exactly like this:

```text
init: invalid seq_id[0][0] = 489646581 >= 1
decode: failed to initialize batch
llama_decode: failed to decode, ret = -1
```

The spike fills `pos[i] = i`, `n_seq_id[i] = 1`, and `seq_id[i][0] = 0` for
every token before decoding. It also sets `logits[i] = 1` for every token when
embeddings are enabled, and only for the last token otherwise.

## The hidden state is post-output-norm

In `src/models/lfm2.cpp` the graph does:

```c
cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
res->t_embd = cur;
```

before it would build the output projection. The tensor returned by
`llama_get_embeddings_ith(ctx, -1)` is therefore the **post-final-RMSNorm**
hidden state, and it is exactly the input to the output row. That is the
correct `h` for the output-row gradient

```text
d(-log p_t) / dW_t = (p_t - y_t) * h
```

A pre-norm hidden state would make that gradient wrong. The spike uses the
post-norm tensor and does not reconstruct or renormalize it.

A GGUF tensor check on the 350M model shows `token_embd.weight` of shape
`[1024, 65536]` and `token_embd_norm.weight` of shape `[1024]`, with no
`output.scale` tensor. The output row is the tied embedding row, so the
single-row form `logit_t = W_t . h` applies without an extra scale factor.

## Same-decode finding on LFM2

The header says embeddings are extracted "together with logits", but the LFM2
graph in the stock build does not build the output projection when embeddings
are enabled (line 278 of `1248fd8`; line 606 in the private fork's working
tree):

```c
if (!cparams.embeddings) {
    cur = build_lora_mm(model.output, cur, model.output_s);
    res->t_logits = cur;
}
```

`llama_get_logits_ith` still returns a non-NULL pointer, but every value is
`0.0`. A NULL check is therefore insufficient. The spike checks that the
logits vector is finite **and** informative (more than one distinct value, and
at least one non-zero value). On LFM2.5-350M with `embeddings=true` the
measured `logits_nonzero` is 0 out of 65536.

`--mode single` reports:

```text
single-decode coexistence gate failed: llama_get_embeddings_ith returned 1024
floats (expected 1024, finite=True), but llama_get_logits_ith returned a
non-NULL 65536-float buffer (expected 65536) with 0 non-zero values. ...
Architecture 'lfm2' builds the output projection only when cparams.embeddings
is false (src/models/lfm2.cpp:606 in this llama.cpp tree), so h and logits
cannot coexist in one decode.
```

and exits 2. This is a real failed gate, not a setup error.

`--mode paired` decodes the same token sequence twice in the same process:
once with `embeddings=true` to get `h`, once with `embeddings=false` to get
logits. The two contexts share the model and the token sequence, and the CPU
result is deterministic across runs. It is still two decodes, so the JSON
reports `"gate_ok": false`, `"same_decode": false`, `"fallback_used": true`.

### Cross-architecture control

To separate an API mistake from an LFM2 graph limitation, the same script was
run in `--mode single` on a non-LFM2 generative model,
`bitnet-2b-tq1_0.gguf` (architecture `bitnet`, also in the local model
directory). It returned `gate_ok: true`, `same_decode: true`,
`h_length: 2560`, `logits_length: 128256`, `logits_nonzero: 128256`, and
flipped the greedy token from 539 to 264 with a +1000 bias. That is a real
single-decode success: the ctypes ABI, batch setup, extraction, and sampler
chain are correct. The failure is specific to the LFM2 output-projection
guard, not to the Python caller.

The control is not the requested 350M model and is not a substitute for it.

## Patched result: 350M and 8B

`patches/llama-lfm2-logits-with-embeddings.patch` removes the embedding guard
from the LFM2 base graph. `res->t_embd` is still set before the output
projection, so `h` is unchanged. The patch is for live mode; the frozen GGUF
needs no patch.

Measured with the patched stock worktree (`1248fd8` plus the patch), the shim
in this directory, and greedy sampling:

| field | 350M `lfm2` | 8B `lfm2moe` |
| --- | --- | --- |
| `gate_ok` | true | true |
| `same_decode` | true | true |
| `n_embd` | 1024 | 2048 |
| `n_vocab` | 65536 | 128000 |
| `h_norm` | 81.257309 | 49.708656 |
| `h_sha256` | 489c41836649 | 06ced724d445 |
| `logits_nonzero` | 65536 | 128000 |
| `logits_top1_before` | 2910 | 5884 |
| `logits_top1_after` | 14933 | 2409 |
| `biased_token` | 14933 | 2409 |
| `fw_caused_flip` | true | true |
| `decode_ms` | 35.15 | 791.56 |
| `extract_ms` | 5.08 | 16.92 |

The 350M `h_sha256` matches the stock embeddings-only decode, so the patch did
not change `h`; only the output projection was added. The 8B `h_sha256` is the
first measured 8B hidden state. Both runs used the same `libllama.so` and the
same shim. The 350M is the probe; the 8B is the target. The 8B run used
`--ctx 128 --batch 32 --threads 2`; the 350M used the defaults. Neither run
used the paired fallback.

## Fast-weight row

The spike keeps one side row `W` of length `n_embd` for the chosen target
token. It starts at zero and is updated with one gradient-descent step on
`-log p_target`:

```text
W <- W + lr * (1 - p_target) * h
readout = W . h
```

The readout is a change in that token's logit. The spike feeds it to
`llama_sampler_init_logit_bias` and samples again. This is a surrogate for the
live runtime's fast-weight adapter: the frozen GGUF and the model's output
weight are not modified. `W` is an ordinary `n_embd`-float side row (1024
floats for the 350M model), and the update uses the same post-norm `h` that
the output projection would use.

With the measured 350M paired-run vectors, `lr=1e-4` and
`p_target=0.033592821`, the row norm is `0.007852765` and the readout is
`0.638094528`. The top1/top2 gap is `0.526865`, so the row readout alone is
large enough to flip the greedy sample; the spike reports `fw_caused_flip:
true`. This is a numerical observation for this prompt and seed-free greedy
decode, not a general claim about the model.

The live engine uses the same raw update with a step divided by `||h||^2`, so
the configured learning rate is the first readout rather than a multiplier of
the hidden norm (see "The fast-weight step is a logit-space step" above). The
same-memory run on the 8B separates the two questions: a readout can be large
enough to flip a sample inside one context and still be `0.0` on the hidden
state of a different question. The `482` row had bias `0.0` at the question's
hidden state even with `lr=2.0`, while the wrong `123` row had `+1.738` because
it had been generated in that context. A large bias is not a memory that
transfers; the context/KV is.

## How to run

```sh
LLAMA=/path/to/llama.cpp
cc -O2 -fPIC -shared -o libpmc_spike.so src/pmc/runtime/llama_spike_shim.c \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  -L"$LLAMA/build/bin" -lllama \
  -Wl,-rpath,"$LLAMA/build/bin"

# Strict same-decode gate. Stock LFM2 fails with exit 2; the patch makes it pass.
python3 src/pmc/runtime/live_spike.py \
  --lib "$LLAMA/build/bin/libllama.so" \
  --model /path/to/model.gguf \
  --shim ./libpmc_spike.so \
  --mode single

# 8B target: same gate, two threads, after the 8B server releases the RAM.
python3 src/pmc/runtime/live_spike.py \
  --lib "$LLAMA/build/bin/libllama.so" \
  --model /path/to/LFM2.5-8B-A1B.gguf \
  --shim ./libpmc_spike.so \
  --mode single --ctx 128 --batch 32 --threads 2

# Explicit two-context fallback. Exits 0 when the fallback completes.
python3 src/pmc/runtime/live_spike.py \
  --lib "$LLAMA/build/bin/libllama.so" \
  --model /path/to/model.gguf \
  --shim ./libpmc_spike.so \
  --mode paired

# Live server: one process, one model, one OpenAI-compatible endpoint.
LD_LIBRARY_PATH="$LLAMA/build/bin" PYTHONPATH=src \
python3 -m pmc.runtime.live_server \
  --model /path/to/fused.gguf \
  --lib "$LLAMA/build/bin/libllama.so" \
  --shim ./libpmc_spike.so \
  --delta /path/outside/the/repo/live-delta.json \
  --ctx 512 --batch 64 --threads 2 \
  --system "You are a terse assistant."

# Same-memory benchmark: one condition per invocation, one model at a time.
PYTHONPATH=src python3 bench/live_vs_frozen.py --condition frozen \
  --frozen-url http://127.0.0.1:8099 --out frozen.json
PYTHONPATH=src python3 bench/live_vs_frozen.py --condition live \
  --model /path/to/fused.gguf --lib "$LLAMA/build/bin/libllama.so" \
  --shim ./libpmc_spike.so --delta /path/outside/the/repo/live-delta.json \
  --out live.json
```

The live server serves `/v1/models`, `/v1/chat/completions`,
`/v1/pmc/state` and `/health`. It is greedy-only and non-streaming, and it
returns 400 rather than silently changing the request. The benchmark takes
`--learning-rate` (default `0.01`) and an optional `--learn-from-prompt`; the
measured table is in "Same-memory result" above.

Both modes print one JSON line on stdout. llama.cpp's own logs go to stderr.
The default prompt is `Hello, my name is`; the text is used only for
tokenization, is never decoded back, and is not included in the JSON.

Optional flags: `--ctx` (default 128), `--batch` (default 32), `--threads`
(default 2), `--bias` (default 1000.0), `--lr` (default 1e-4), and `--prompt`.

## JSON contract

Required fields, exactly as requested by the WP7a deliverable:

| field | meaning |
| --- | --- |
| `h_norm` | L2 norm of the `n_embd`-float hidden state (1024 for the 350M model) |
| `h_sha256` | first 12 hex characters of SHA-256 over canonical little-endian float32 `h` |
| `logits_top1_before` | greedy top-1 token id from the raw logits; `-1` if the gate failed before sampling |
| `logits_top1_after` | top-1 token id after adding the bias; `-1` if the gate failed |
| `biased_token` | token the bias was attached to; `-1` if the gate failed |
| `finite_h` | all `n_embd` hidden values are finite (1024 for the 350M model) |
| `finite_logits` | all `n_vocab` logits are finite (true for an all-zero buffer; 65536 for the 350M model) |
| `decode_ms` | wall time of the logits-producing decode |
| `extract_ms` | wall time to copy and validate the tensors, compute the norm/hash, and find top-1 |

Additional fields expose the evidence behind the claims: `gate`, `gate_ok`,
`same_decode`, `fallback_used`, `fallback_ok`, `arch`, `n_embd`, `n_vocab`,
`prompt_tokens`, `logits_nonzero`, `emb_ctx_logits_nonzero`,
`logits_informative`, `top1_logit_before`, `biased_logit_before`,
`logit_gap`, `sampler_bias`, `sampled_before`, `sampled_after`, and the
`fw_*` row fields. `ok` is true only when the strict same-decode gate passed;
a successful paired fallback sets `fallback_ok: true` and leaves `ok: false`.

Exit codes: `0` means the requested mode completed (for `paired`, even when
the same-decode gate is false); `2` means the strict `single` gate failed;
`1` means a load/decode/extraction error.

## Tests

```sh
python3 -m pytest src/pmc/runtime/ -q
python3 -m pytest -q
```

The pure tests import no model and load no native library. They cover argmax,
runner-up selection, biased argmax, finite/informative checks, norms, dot
products, canonical hashing, softmax, the fast-weight step, the JSON summary
contract, the state-bank reader, the delta format and atomic save, the
fast-weight normalization, the last-user helper, and the HTTP request and
response shapes. The server tests use a fake engine.

`live_engine_test.py` has two opt-in real-model tests. They are skipped unless
`PMC_LIVE_MODEL`, `PMC_LIVE_LIB` and `PMC_LIVE_SHIM` are set; with them the
first runs a chat, checks the prompt/completion counts, the fast-weight rows,
the state digest, the persisted delta and the temperature rejection; the second
saves the context state, writes and reloads it through the one-file format,
checks the identity, rejects a forged context size, restores it into a cleared
context after checking the token prefix, decodes only the suffix, and checks
the counters. Both pass on the 8B fused artifact. The 350M chat ends at EOG;
the 8B chat produces text. The benchmark helper tests run with:

```sh
PYTHONPATH=src:bench python3 -m pytest bench/live_vs_frozen_test.py \
    bench/kv_swap_probe_test.py -q
```

## Evidence level

The C ABI sequence, the batch failure, the zero-filled embeddings-mode logits,
the paired token flip, the BitNet control, and the patched 350M/8B same-decode
results were measured on the real models and a real `libllama.so` build (E2:
integration on real artifacts). The 350M `h_sha256` matches the stock
embeddings-only decode; the 8B `h_sha256` is the first 8B hidden-state
measurement. The arithmetic helpers are covered by unit tests (E1). The claim
that post-norm `h` is the correct gradient input is established from the graph
source, not from a training run (inferred from the model graph, not measured).

The live engine's 8B turn and the HTTP server were measured on the fused 8B
artifact (E2). The live-versus-frozen text parity after fixing
`parse_special=True` was measured against a `llama-server --jinja` run of the
same artifact (E2 for parity). The same-memory result is E3: the frozen
baseline, the no-memory control and three live variants ran with the exact
command in `bench/live_vs_frozen.py`; the frozen answered `4821`, the control
and all live variants answered `1234`, and the diagnostic measured the `482`
row bias at `0.0` and the wrong `123` row at `+1.738`. The result is negative
for output-row fast weights as episodic memory and is recorded as such.

The KV condition is E3 as well. The first run saved a 861,339-byte state over
46 tokens, restored it after a prefix check, decoded 23 of 69 prompt tokens, and
answered `4821`; its first reply echoed the code. The echo-free follow-up used
`--first-style quiet --no-learn`: the first reply was `Noted.` (no `4821`), the
state was 787,539 bytes over 40 tokens, 23 of 63 prompt tokens were decoded,
and the answer was `4821.`. The in-process replay control decoded all 63 tokens
and answered `4821.`; the frozen server and the no-memory control answered
`4821.` and `1234` respectively. A new process loaded the same state file and
reproduced the answer with 40 restored and 23 decoded tokens; the live delta's
hidden-state digest `3f8a0192…` matched the in-process run. The 8B state offered to the 350M engine
was refused on identity (`matches=false`) before the C API. The KV state is
runtime context state, not weight-resident memory and not a GGUF artifact. The
Substrate Swap Test on a trained PMC state is not yet measured; a KV cache is
model-bound and cannot be swapped across the 350M and 8B geometries.

The patched same-decode result is a live-mode result, not a claim about stock
llama.cpp. The 8B artifact is stock-loadable; inference on an unpatched
LFM2.5 graph returns zero logits, so a provider needs the output-projection
fix (this patch or an equivalent upstream fix). The 350M is a hidden-state
probe, not a chat model.
