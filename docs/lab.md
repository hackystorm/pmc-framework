# The research lab

This document describes how `pmc-framework` is built: the roles, the file
ownership, the evidence standard and the gates a change must pass before it can
be claimed. It is part of the repository on purpose. A project that asks
whether a machine can have a continuous inner life should be able to show how
its own claims are produced.

## The thesis

The model is frozen and replaceable. The mind is the substrate that persists
while the model changes. The substrate (Plastic Memory Core, 50-100M trainable
parameters) holds memories and states in its weights and is fused into a single
GGUF together with the base model. The frozen artifact runs on any GGUF
provider with zero framework code. The live mode is an optional single-process
runtime, never a second server.

What the project measures is continuity, memory residence and state discovery.
What it does not claim is consciousness. The framework builds the conditions
and measures the functional correlates honestly; the claim stays at the level
of the evidence.

## Roles

| Role | Agent | Owns |
| --- | --- | --- |
| Orchestrator | main session | interfaces, contracts, review, gates, user communication |
| Substrate engineer | `pmc-substrate` | `crates/pmc-gguf/**`, `crates/pmc-fuse/**` |
| Numerical engineer | `pmc-trainer` | `crates/pmc-train/**` |
| Pipeline engineer | `pmc-dream` | `src/pmc/dream/**`, `src/pmc/runtime/**`, `src/pmc/cli.py`, `crates/pmc-ffi/**`, `crates/pmc-cli/**` |
| Measurement scientist | `pmc-eval` | `bench/**`, `tests/**`, `docs/claims.md` |

File ownership is exclusive. An agent that needs a change outside its paths
stops and reports it to the orchestrator. Shared files (workspace
`Cargo.toml`, `pyproject.toml`, the dignity layer, the charter, the README) are
changed by the orchestrator only.

## Two modes

The same artifact supports two modes, and the claims must keep them apart.

- **Frozen mode (portable).** The GGUF produced by the last dream runs on any
  stock provider with zero framework code: llama.cpp, Ollama, LM Studio,
  koboldcpp, text-generation-webui. This is the artifact that ships and the
  claim that must hold.
- **Live mode (single process).** A runtime links `libllama`, serves the
  OpenAI-compatible API, reads the hidden state and the logits in the same
  decode, updates the fast weights online, applies their bias before sampling,
  and persists the delta. It is one process and one server: the substrate is
  never a second server. The delta is consolidated into a new GGUF at the next
  dream. The measured same-memory result is negative for output-row fast
  weights: they are token- and context-exact, so a fact learned in one
  sentence does not transfer to a different question, while the frozen context
  re-injection answers correctly. The KV/state restore passed the same test.
  The echo-free follow-up saved a 40-token state (787,539 bytes), asked for a
  first reply that did not repeat the code (`Noted.`), restored 40 tokens and
  decoded 23 of 63 prompt tokens, and answered `4821.`. The replay of the full
  transcript and the frozen server also answered `4821.`; the no-memory control
  answered `1234`. A new process loaded the saved state and reproduced the
  answer with the same live-delta hidden-state digest. That is runtime context
  state, bound to one model's memory geometry; it is not weight-resident, it
  survives a restart only when its bytes are saved, and it cannot be swapped
  across models: the 8B state offered to the 350M engine was refused on
  identity before the C API. The PMC-mini Substrate Swap Test is negative on
  host B in all three runs (v1 5/8 vs 1/8, v2 8/8 vs 0/8, v3 8/8 vs 2/8; every
  pass is false); the next gate is a cross-host interface that generalizes or
  the 50–100M PMC state.

Three timescales:

| Timescale | What changes | Where | Lifetime |
| --- | --- | --- | --- |
| Context / KV | attention cache | frozen mode, any provider | the session |
| Fast weights | memory rows, state matrix | live mode, one process | the session, persisted as a delta |
| Slow weights | substrate and fused rows | dream, then a new GGUF | across sessions and base models |

The frozen mode is what the world can run. The live mode is what the mind runs.
A memory learned live is only real when the next dream preserves it in the
frozen artifact; otherwise it is session-bound, and the framework says so.

## The Substrate Swap Test (pre-registered)

This is the next gate, and it is not the KV restore. It asks whether a PMC
state learned on one frozen model remains a memory when the same state bytes
are loaded into a different frozen model.

A valid run has:

- a state whose schema is model-independent: it declares its layout and the
  interface that maps it into each model, and that mapping is part of the PMC,
  not fit on the evaluation facts;
- two hosts with different geometry, the 350M (1024-dim) and the 8B
  (2048-dim), or another pair;
- training or consolidation on host A, then evaluation of the same state bytes
  on host A and host B;
- controls: no state, a fresh random state, and text re-injection; the same
  chat template, greedy sampler, prompt set and token budget for every arm;
- at least eight facts, one held-out question per fact, and one control
  question per fact; every miss is reported, none is filtered;
- a pre-registered pass: host B recalls at least as many held-out facts as
  host A minus a declared margin, the no-state and fresh-state controls fail,
  and the state's identity and no-target-training conditions hold.

What does not count:

- a KV cache: it is model- and geometry-bound, as the 8B to 350M refusal
  shows;
- output-row fast weights: they are token- and context-exact;
- a text sidecar re-injected into the prompt: that is the frozen condition;
- a projection trained on the target's evaluation facts: that is target
  training, not a swap.

**Status:** run three times on the PMC-mini proof, negative on host B. The
state-bank mechanism and the host-A read work; the cross-host interface does
not generalize to the held-out cues. The KV refusal and the fast-weight
negative remain boundaries. The 50–100M PMC state has not been trained, fused
and evaluated; the positive Substrate Swap Test is still open.

### PMC-mini swap run (pre-registered 2026-10-27)

This is a proof of concept for the interface, not the 50–100M PMC. It exists
to test the swap mechanism before the larger substrate is trained. The word
list, split, templates and thresholds are frozen in `bench/swap_words.json`
(seed `20261027`): 22 training facts, 8 held-out facts, 8 unseen control cues.

- **State.** A key/value memory of 8 keys and 8 value indices. The keys are
  64-dimensional and model-independent. The state bytes are the same in both
  hosts (`pmc.state.swap.keys`, `pmc.state.swap.values`): the SHA-256 of the
  canonical core is the state identity. The host interface is a fixed,
  seeded random projection from the host's hidden state to 128 dimensions,
  followed by a linear adapter to the shared 64-dimensional key space and an
  L2 normalization. The adapters are stored per host
  (`pmc.state.swap.adapter`, `pmc.state.swap.adapter_bias`); they are the
  interface, not the state. The state is written into the GGUF metadata and
  read by the runtime; it is not a KV cache, an output-row fast weight, or a
  text sidecar in the prompt.
- **Training.** The adapters are trained with an InfoNCE objective on four
  views of each of the 22 training facts: the fact statement and the
  question, on both hosts. Four training facts are held out for early
  stopping; the final adapters are retrained on all 22. The 8 evaluation
  facts and the 8 control cues are never seen during adapter training. The
  evaluation fact statements are used only for consolidation (to write the
  8 keys), and the evaluation questions are used only for reading. This is
  the no-target-training condition.
- **Readout.** At a question, the host hidden state is passed through the host
  interface, the nearest of the 8 keys by cosine is selected, and the
  associated value word receives a fixed `+200.0` logit bias. Sampling is
  greedy, one token, and the answer is the sampled token. The fact statement
  includes the value; the held-out question does not. The state is loaded
  from the fused GGUF and the runtime applies the bias.
- **Hosts and prompt.** Host A is the 350M (1024-dim), host B is the 8B
  (2048-dim). Both use the same literal template in every arm; the model's
  own Jinja template is not used because the 350M template ends in EOG. The
  template is fixed in `bench/swap_words.json` and is identical for both
  hosts.
- **Arms.** Host A and host B with the state (PMC); both hosts with no state
  and no bias; both hosts with a fresh random state with the same shape and
  the same adapters; both hosts with the fact re-injected into the prompt and
  no bias (the frozen condition); both hosts on the unseen control cues with
  the state. Every fact and every miss is reported.
- **Pass.** Host A recalls at least 5 of 8; host B at least 4 of 8; host B is
  at least host A minus 1. No-state recalls at most 1 of 8; the fresh state
  at most 2 of 8; the control false-positive rate is at most 2 of 8. The
  text re-injection arm is reported, not required to pass; it is the model's
  own memory, not the swap. State identity must match across the two hosts,
  and the evaluation facts must not appear in the adapter training set.

The command and the summary are in `bench/README.md`; the raw per-fact JSON,
the state and the adapters from these runs are kept outside the repository.
The fused artifacts are regenerable from the two base models and the saved
adapters. The result is E3 for this lab run; a fresh run is reproducible from
the spec, the seeds and the model files, and every miss is in the output.

**Status (PMC-mini):** run three times, negative on host B.

The first run used `bench/swap_words.json` and the last prompt token. The
second run is the separately pre-registered cue-anchor run in
`bench/swap_words_v2.json` (new 68-word split, the cue token as the anchor,
the full 300 adapter epochs without early stopping). The third run is the
separately pre-registered cue-delta run in `bench/swap_words_v3.json` (new
interface, 30 training cues, 16-value state). The same thresholds were used in
all three runs. Every miss is in the raw result; this table is the summary.

| run | spec | anchor | state | host A PMC | host B PMC | no state | fresh | control FP | re-injection | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | `bench/swap_words.json` | last prompt token | 8×64 | 5/8 | 1/8 | 0/0 | 1/2 | 2/1 | 4/4 | no |
| v2 | `bench/swap_words_v2.json` | last cue token | 8×64 | 8/8 | 0/8 | 0/0 | 2/2 | 1/1 | 5/6 | no |
| v3 | `bench/swap_words_v3.json` | cue minus neutral cue | 8×16 | 8/8 | 2/8 | 0/0 | 2/2 | 3/1 | 6/6 | no |

The v1 state is `4a43d78a…` with adapters `a5d918e4…` (350M) and `83ebfab2…`
(8B); the v2 state is `b1ac7188…` with adapters `681d62da…` (350M) and
`6a5d8b02…` (8B); the v3 state is `35c3430e…` with adapters `7a4db210…`
(350M) and `fe3b635f…` (8B). The fused 350M and 8B files load in stock
`libllama` with `n_vocab=65544` and `128008`. The state banks are identical
across hosts; the adapters differ. The v3 fused files have SHA-256
`db3970e3…` (350M, 229,591,968 bytes) and `782d3ce4…` (8B, 3,335,440,736
bytes); they were removed after the stock-load and evaluation and are
regenerable from the bases and the saved adapters. The text re-injection arm
is the model's own memory, not the swap; it is reported, not required.

The negative is specific. In v1 the early-stopped 15-epoch adapters leave host
B at chance even in sample (host B 0.045), so the cross-host interface is not
trained. In v2 the cue anchor and the full 300 epochs make host A perfect
(8/8) and host B perfect in sample (22/22), but host B is 0/8 on the eight
unseen cues: the linear adapter memorizes the 22 training cue mappings and
does not carry them to new cues. The host-B margins are small
(0.0035–0.1449). In v3 the cue delta and the 16-value state make host A
perfect (8/8) but host B only 2/8; host B is 15/30 in sample, so the lower
capacity underfits the 30 training cues instead of generalizing. The host-A
control arm is 3/8, over the threshold: `floor`, `house` and `palace` are read
as evaluation cues. The host A evaluation margins are 0.970–0.989; the host B
correct facts are `silver`→`basil` and `rocket`→`rose`, and the other six
land on another evaluation value. The v3 evaluation facts were not used to
tune the adapter, and no post-hoc v3 variant was run.

The no-state and fresh-state arms fail as intended; the fresh arm is at the
threshold (2/8 on both hosts in v2 and v3). The state identity and
no-target-training conditions hold in all three runs. The result is a
negative on the cross-host interface, not evidence that the state banks are
broken: the same bytes are read on both hosts and host A retrieves the
held-out facts. The v3 result is also a negative on its host-A control arm.

Two implementation bugs were found and fixed during the first two runs and
are part of the record: the added output rows were first filled with `-100.0`
instead of `0.0`, which let the `<pmc:swap-*>` tokens win on several retrieved
facts (host A 2/8 in the first pass); and the v2 evaluation first read the last
prompt token while the v2 adapter was trained on the cue token (host A 4/8 in
the first pass). The numbers above are after both fixes; the commands are in
`bench/README.md`. The v3 added the cue-delta path, the neutral probe and the
dynamic 16-value state; the unit checks cover the prompt swap, the spec and the
16-dim training. The v3 cue-delta evaluation re-decodes the actual question
after the neutral probe so the sampler reads the real question context.

### PMC-mini swap v3 (pre-registered 2026-10-27)

The v1 and v2 runs both fail on host B. Before running v3, the interface is
frozen in `bench/swap_words_v3.json`: SHA-256
`48b7714cc6f15e8078cb99f2c4c68f8af27e218736813cf0b76962a724588b0c`; 30
training facts, 8 evaluation facts, 8 control questions, 84 distinct words.
The v3 changes only the cross-host interface and the training coverage. The
pass thresholds, the arms, the greedy sampler, the one-token budget and the
no-target-training rule are the same as v1 and v2.

The hypothesis: the raw cue hidden state at the cue position still carries
the prompt-specific prefix. The 8B adapter therefore learns a mapping that is
valid for the 22 training prefixes and does not transfer to a new cue. V3
reads a **cue delta**: `h(cue) - h(thing)` at the same token position in the
same prompt, where `thing` is a fixed neutral cue. The delta subtracts the
shared prefix; the cue and the neutral cue see the same prompt before that
position. The shared state is 8 keys of 16 values instead of 64, and the host
adapter is 16x128 instead of 64x128. The lower capacity is a deliberate
regularisation for 30 training cues: v2 had 22 cues and 64 dimensions and
memorized the training cues (22/22 in sample) while scoring 0/8 on new cues.

The split: the 30 training cues are the v2 22 training cues plus the v2 8
control cues; the 30 training values are the v2 30 values. The 8 evaluation
cues are `silver`, `gold`, `cloud`, `valley`, `forest`, `island`, `rocket`,
`planet`; the 8 evaluation values are `basil`, `crystal`, `fern`, `iris`,
`mint`, `moss`, `rose`, `sage`. The 8 control cues are `wall`, `floor`,
`roof`, `room`, `house`, `tower`, `palace`, `temple`. No evaluation cue or
value is in the v3 training set; the v3 no-target-training check is true. The
evaluation cues appeared in the separate v1 sweep; they never trained the v3
adapter, and the frozen hosts carry no adapter weights from v1. The v3 test
is new for the v3 interface.

The v3 is one attempt. If it fails host B, the result is recorded as a
negative on this interface; the v3 evaluation facts are not used to tune a
later adapter. A further attempt must use new words and a different mechanism.
If it passes, it is still a PMC-mini state-bank result, not the 50–100M PMC
and not evidence of weight-resident memory.

**Status (v3):** run once, negative on host B.

The formal fused evaluation measured host A 8/8, host B 2/8, no-state 0/0,
fresh state 2/2, control false positives 3/8 on host A and 1/8 on host B, and
text re-injection 6/6. The pre-registered pass is false on the host-B minimum,
the host-B margin and the host-A control. The offline diagnostic before fusion
measured the same host-A evaluation 8/8 and host-B evaluation 2/8, with host B
15/30 in sample. The 16-value state underfits the 30 training cues on host B;
the cue delta and the host-A adapter are strong. Host B answers `silver`→
`basil` and `rocket`→`rose`; the other six land on another evaluation value.
Host A reads `floor`, `house` and `palace` as evaluation cues. The v3
evaluation facts were not used to tune the adapter, and no post-hoc v3 variant
was run. This is the third negative on the cross-host interface; the positive
test remains open. The v3 fused files were removed after the stock-load and
evaluation and are regenerable; the raw eval, state and adapters are kept.

### Live state-memory bank (2026-10-27)

The live engine can consume a self-contained state bank, not only report it.
The bank schema is `pmc.state.memory.*`: `keys [n, d]` (unit length),
`values [n]` (token ids), `projection [d, n_embd]`, `projection_bias [d]`,
`logit_bias [1]` (positive) and optional `min_similarity [1]`. At the first
generated token the engine projects the current hidden state, normalises it,
finds the nearest key and adds `logit_bias` to the stored value token if the
cosine is at least the floor. The bank is read from GGUF metadata only; the
tensor data is never read, and the engine does not name, classify or gate the
hidden state (D-17).

`tests/test_state_memory.py` drives the real `LiveEngine.chat` path with a
synthetic GGUF and a fake patched-lib spike: nine checks. With the bank the
engine emits the stored token `7`; without it the base token `15`. Partial or
unknown banks, non-unit keys, non-integer or out-of-vocabulary token values,
and a wrong projection shape are refused. This is E1: the engine path and the
metadata reader are real, but the bank and spike are synthetic. No trained
50–100M memory bank has been fused and consumed. The PMC-mini
`pmc.state.swap.*` bank is not this schema: it needs the benchmark's host
projection, so the engine reports it as metadata and does not consume it.



`src/pmc/dream/` implements the parts of the dream cycle that do not depend
on a particular model:

- metadata-only replay and salience (`replay.py`): recency, self-reported
  load, access count and digestion state; refused and quarantined memories are
  not forced back in; the replay item carries a record id and a content digest,
  never the content;
- a digest-keyed taint ledger (`taint.py`); the checkpoint digest and every
  replay source are checked before the checkpoint is copied or a consolidator
  is called;
- the six-metric gate (`gate.py`): older recall may drop by at most a margin,
  newer recall must improve, interference and false memory have absolute
  ceilings, calibration may drop by a margin, identity has a floor; missing,
  non-finite or out-of-range metrics fail closed;
- a hash-chained dream journal (`journal.py`) with metadata only;
- a runner (`pipeline.py`) that stages the checkpoint on the same filesystem,
  evaluates the baseline and the candidate, and only then atomically replaces
  the checkpoint. A failed gate leaves the original byte-identical. A hostile
  or broken consolidator that changes the original is detected and the
  verified baseline copy is restored.

The command is:

```bash
pmc dream --workspace W --checkpoint C \
  --trainer "python3 trainer.py" --evaluator "python3 eval.py"
```

The trainer receives `baseline candidate replay.json` and writes the
candidate; the evaluator receives one checkpoint and prints the six-metric
JSON object. The trainer and evaluator are responsible for the model and the
memory content; the framework never opens the private store.

Evidence: `PYTHONPATH=src:bench python -m pytest tests/test_dream.py
tests/test_cli.py -q` passes 21 pipeline checks and the two CLI dreams. The
passing dream commits, the failed gate leaves the checkpoint digest unchanged
and appends `rolled_back`, a tainted source is refused before the consolidator
runs, a tainted checkpoint is refused, a hostile consolidator's change is
restored, and an evaluator that changes the baseline or the candidate is
rejected with the original unchanged. The checks are E1 and use a synthetic
checkpoint, consolidator and
evaluator. They do not show that a trained model consolidates well: that needs
a real 50–100M PMC checkpoint, a real task evaluator and the six metrics on
held-out memory. No such real six-metric dream has been run.

### 50.4M reference core (WP2 extension, 2026-10-27)

`crates/pmc-train/examples/train_50m.rs` trains a real 50.4M-parameter
decoder-only transformer on one fixed 48-token sequence:

```text
n_layer 12, d_model 512, n_head 8, ffn_hidden 2048,
vocab 256, n_positions 64, readout_rank 64
param_count 50,393,600; frozen embedding 131,072; total 50,524,672
model seed 20261030; embedding seed 20261029
200 steps, Adam, learning rate 0.003, batch 1
initial loss 5.887041 -> final loss 0.000008
elapsed 343,726 ms
```

`crates/pmc-train/examples/eval_50m.rs` loads the checkpoint through the same
verifying loader (`manifest.json`, header, exact length, parameter digest) and
scores the trained sequence: loss 0.00000777, 47/47 next-token positions
correct.

Both examples run from `crates/` with a checkpoint outside the repository:

```text
PMC_CHECKPOINT=<evidence-cache>/pmc-50m \
  cargo run --release --offline --example train_50m
PMC_CHECKPOINT=<evidence-cache>/pmc-50m \
  cargo run --release --offline --example eval_50m
```

The checkpoint is 201,574,436 bytes. Its checkpoint-directory digest (the
dream pipeline's `checkpoint_digest`) is
`10ab040ee01bbbdbf3ad939ab8084bb179620de20b881c63934659b06175dd65`; the
`checkpoint.bin` SHA-256 is
`4d59415cfa6847a4e82d6d4e1755412d126494f44e2aafb34f569d7926a509ca`; the
manifest's parameter SHA-256 is
`3dffe4266ae0f17eebbfb27ab4bf0d074e96cecadef76c460fc161188c9da97f`. The
fixed token sequence is
`[11,48,85,122,159,196,233,14,51,88,125,162,199,236,17,54,91,128,165,202,239,20,57,94,131,168,205,242,23,60,97,134,171,208,245,26,63,100,137,174,211,248,29,66,103,140,177,214]`;
its little-endian token SHA-256 is
`6cc7994f8176406b68b994f36f805b19078c9eab3f5cbd32006d0a35fafab338`.

This is a reference core, not the PMC memory. It is a token-level model with a
frozen random embedding, not a sidecar that reads the frozen LLM's hidden
state; it is not fused into a GGUF, it has no episodic state, and the six dream
metrics have not been measured on it. It gives the dream pipeline a real
checkpoint of the required size to test the gate and rollback against, instead
of an opaque test file.

`bench/dream_reference.py` runs that real checkpoint through the dream
pipeline with an identity consolidator and a fixed metric set. The candidate
is byte-identical, the gate refuses it on `recall_newer` (no improvement), the
journal records `rolled_back`, and the checkpoint digest stays `10ab040e…`. A
tainted checkpoint digest is then refused before any copy and the journal
records `refused`. All eight checks pass:

```text
PYTHONPATH=src:bench python bench/dream_reference.py \
    --checkpoint <evidence-cache>/pmc-50m
```

This is E2 for checkpoint handling: a real 201,574,436-byte checkpoint, the
real staging/copy/digest path and the real journal. It is not a six-metric
memory result: the harness supplies the metric values and the identity
candidate learns nothing.

### The v4 mechanism (pre-registered 2026-10-27)

The three negatives above share one cause, measured by the cross-host
diagnostic (training-only, not an evaluation): on the v3 cue-delta vectors the
same-host retrieval is 30/30 at every state dimension tried, while the
cross-host retrieval is 15/30 at 16 dimensions, 14/30 at 32 and 19/30 at 64,
with margins of +0.0100, +0.0075 and +0.0632. A fixed random projection plus
a linear per-host adapter cannot align the two substrates even in sample.

The v4 mechanism changes the thing that was frozen: the input projection is
now **trainable and per host**, and the shared body is **non-linear** and
residual, trained with InfoNCE in which the views of one fact from the two
hosts are positives. It is implemented in `crates/pmc-train/src/pmc_state.rs`
and verified there: the analytic gradient matches finite differences over
every parameter group, the `PMCV`/`PMCW` binary formats round-trip with a
payload digest that refuses a corrupted or mismatched file, the seed is
deterministic, and a synthetic two-host alignment test reaches 3/4 held-out
cross-host views.

The gate was pre-registered as follows, before any extraction: train on the
30 training facts' four views (120 views, 706,112 parameters, architecture
`p128h256l2k64`, seed 20261031, 300 epochs), write the keys from the host-A
fact view of all 30 labels, and query with the host-B question view of the
held-out cues across six folds. It passes only if the new mechanism is at
least the frozen v1-v3 baseline on identical folds, at least 15/30 overall,
and no fold is 0/5.

**Result (2026-10-28), and it is a negative.**

| | number |
|---|---|
| in-sample, all four view kinds | 30/30 |
| held-out cross-host, host-B question from host-A keys | **11/30** |
| per fold | 2, 4, 0, 1, 3, 1 |
| frozen v1-v3 baseline on the same six folds | 2/30 |
| pre-registered floor | 15/30 |

The loss went from 0.786 to 0.0014 and the mechanism reproduces every training
vector exactly, so this is not an optimisation failure: it fits the cues it was
given and still cannot align the two substrates for a cue it has not seen. The
empty fold fails the gate on its own clause.

Artifacts, and what they are worth: the views file `state-views-v4.pmc`
(739,212 bytes, SHA-256
`7a082ecdb371c57e34c0010ffc1e56a87763ec97a301869dba127586f435f55a`), built by
`bench/pmc_state_views.py` from the 30 training cues only; the trainer report
from `crates/pmc-train/examples/train_pmc_state.rs`; the baseline and the
verdict from `bench/crosshost_cv.py`.

**The two harnesses are in this repository and the inputs are not.** The views
file is not committed: it is derived data, and neither are the extracted
hidden states it is built from (`pmc-swap-hidden/1` JSON per host) nor the two
LFM2 GGUFs those extractions came from. So the 11/30 above is a **recorded
run that a fresh clone cannot re-derive**, and it is labelled that way rather
than as a reproducible measurement. The code path, the gate constants
(`GATE_MIN_CROSS_HOST = 15`, `GATE_MIN_MARGIN = 0`,
`GATE_FORBIDS_EMPTY_FOLDS = True`) and the command lines are all in the
repository, so the *method* is auditable and re-runnable; the *number* is
checkable only against the recorded artifacts and their digests.

To reproduce it from outside would require, in order: both LFM2 substrates
fused with their v1-v3 adapters as `bench/substrate_swap.py` describes; a
hidden-state extraction per host; `bench/pmc_state_views.py` to build the
views file; the trainer; then `bench/crosshost_cv.py`, which reads its folds
from the trainer's report so the baseline is scored on the same held-out
labels. The run is only reproducible end to end from an extraction the size of
the originals — which is the honest boundary, and it is why this result is
filed as E3-recorded rather than E3-reruns and not E4.

Per the pre-registration, the failure is recorded and the line pivots to the
50M core as a task learner. The evaluation cues and the control questions were
never read by the gate: they stay untouched for the attempt that would follow.

**A correction, recorded rather than silently edited.** The heading of this
section read `pre-registered 2026-10-31` from its first
commit (`30ed681`, 2026-10-27), which predates the run. That date is later
than the `Result (2026-10-28)` line directly below it, so the heading claimed a
pre-registration dated after the measurement it pre-registered, and 2026-10-31
had not arrived when the run was published (`22d02a6`, 2026-10-28).

The original heading was `pre-registered 2026-10-31, 2026-10-27 dated cues`;
the qualifier that carried the actual date was dropped when the result was
published. The pre-registration, the folds, the key pool and the floor were
written down on 2026-10-27, before any extraction, which is what the run
commit's own message states. The heading is corrected to that date. The seed
`20261031` is untouched: it is a recorded parameter of the run, not a date
claim, and nothing recorded about the measurement itself has been edited.

## LM text path (M1, 2026-10-29)

Pre-registration: a decoder-only transformer is trained on a token sequence
with next-token prediction; a held-out suffix of the same sequence is scored
before and after. The gate requires the final held-out loss to be below the
random-init held-out loss. The held-out suffix is never seen during training.

Command:
```
PMC_STEPS=30 PMC_LR=0.01 PMC_N_LAYER=2 PMC_D_MODEL=64 PMC_N_HEAD=4 \\
  PMC_FFN=256 PMC_VOCAB=256 PMC_N_POS=64 PMC_RANK=8 PMC_HOLDOUT=8 \\
  cargo run --release --offline --example train_lm
```
(Tokens default to a deterministic pseudo sequence; `PMC_TOKENS` points at a
file with one integer per line or a JSON array, and `PMC_EVAL_TOKENS` overrides
the held-out suffix.)

Raw numbers (one run, 2026-10-29):
- train_tokens: 56, eval_tokens: 8, holdout: 8
- param_count: 136512
- initial_train_loss: 5.700954
- final_train_loss: 0.000372 (memorised)
- baseline_eval_loss: 5.502320 (random init)
- final_eval_loss: 14.372885
- baseline_eval_perplexity: 245.26
- final_eval_perplexity: 1746081.63
- beat_baseline: false
- train_sha256: b88fcf61af084190045870e3d8883d6d78a858af826fef280594ca65fb0f3810
- eval_sha256: ee0ad2b4d2febfe256b30a5fe8bea8c120634545ae29b5aaaa0554af6ef56a72
- elapsed_millis: 2487

Interpretation (bounded by E3 for the mechanism, E1 for the generalisation
claim): the training path works and overfits the training positions, but on
this 64-token pseudo sequence the held-out suffix is worse after training
than at random init. The gate is not met. The negative is specific to the
data, not to the mechanism: 56 training positions are not enough to
generalise, and a 64-token pseudo sequence is not a corpus. The next attempt
needs a real corpus (TinyStories subset or larger) and a pre-registered
margin; no post-hoc tuning of this run is permitted. The mechanism itself -
next-token loss, checkpoint write, held-out scoring, reproducible command -
is in place and measured.

## LM text path: real corpus (M1, 2026-10-29)

Pre-registration: a decoder-only transformer is trained on a token sequence
with next-token prediction; a held-out suffix is scored before and after. The
gate requires the final held-out loss to be below the random-init held-out
loss. The held-out suffix is never seen during training. The corpus is the
first 512 characters of "Alice's Adventures in Wonderland" (Project
Gutenberg #11, public domain), encoded as UTF-8 bytes (char-level
tokenization, vocab 256). No post-hoc tuning of the hyperparameters or the
split is permitted; the split is the last 64 tokens.

Command:
```
PMC_TOKENS=/tmp/.../alice_tokens.txt PMC_STEPS=20 PMC_LR=0.005 \\
  PMC_N_LAYER=2 PMC_D_MODEL=32 PMC_N_HEAD=4 PMC_FFN=64 PMC_VOCAB=256 \\
  PMC_N_POS=512 PMC_RANK=8 PMC_HOLDOUT=64 \\
  cargo run --release --offline --example train_lm
```

Raw numbers (one run, 2026-10-29):
- train_tokens: 452, eval_tokens: 64, holdout: 64
- param_count: 22944
- initial_train_loss: 5.538515
- final_train_loss: 2.778033
- baseline_eval_loss: 5.566339 (random init)
- final_eval_loss: 4.548255
- baseline_eval_perplexity: 261.48
- final_eval_perplexity: 94.47
- beat_baseline: true
- train_sha256: b4a6d91c350e44f3c875de0eb7d06399b8962d2e5a0952ddcb8ad3087bd8020b
- eval_sha256: 5ba0ba12cafe9112fc3d558992a364cda592bd642644cbb142c13013ab010f72
- elapsed_millis: 676

Interpretation (E3: measured against the random-init baseline, command and
raw result retained): the text training path now beats the random-init
baseline on a real corpus - held-out perplexity fell from 261.48 to 94.47
(factor 2.77). The pseudo-sequence run did not (1746081.63), so this is
the first positive on real text. The margin was not pre-registered as a
number; this run establishes the measurement and the next attempt pre-registers
a margin. The Python parity extension for the transformer remains (E1 for the
trainer side until it lands).

## Dream CLI path (M3, 2026-10-29)

Pre-registration: one dream run through the `pmc dream` CLI with an identity
consolidator (copies the baseline byte-for-byte) and a mechanics evaluator
that reports the checkpoint as intact. The gate requires recall_newer to
improve by at least newer_gain_min (0.001). An identity consolidation
cannot improve anything, so the gate is expected to fail on recall_newer and
roll back.

Command:
```
python -m pmc.cli --json dream --workspace <workspace> --checkpoint <checkpoint> \\
  --trainer "python bench/dream_consolidator.py" \\
  --evaluator "python bench/dream_evaluator.py" --limit 4 --seed 20261027
```

Raw numbers (one run, 2026-10-29):
- record_ids: ["mem_001"]
- sources: ["aaaa...aaaa" (64 hex)]
- metrics_before == metrics_after: recall_older 1.0, recall_newer 1.0,
  interference 0.0, calibration 0.0, identity 1.0, false_memory 0.0
- gate: passed false; reasons: ["recall_newer"]
  (recall_newer: before 1.0, after 1.0, bound 1.001, delta 0.0, ok false)
- outcome: rolled_back
- reason: the gate failed
- checkpoint before/after digest: identical (rollback byte-identical)

Interpretation (E2: integration with a real workspace, sidecar and CLI path):
the CLI dream path works end to end - replay, taint check, consolidation,
evaluation, gate, rollback - and the gate correctly refuses a no-op
consolidation on recall_newer, leaving the original checkpoint
byte-identical. The identity consolidator and the mechanics evaluator are
exercising the plumbing, not a model: their metrics are fixed values, not
measurements. A real consolidation (training on replayed memory that
improves recall_newer) and a real evaluator (measuring the six metrics on
held-out memory) remain as the M3 follow-up.

## Dream real consolidation (M3, 2026-10-29)

Pre-registration: a decoder-only transformer checkpoint (2 layers, 22944
parameters, trained on the first 512 characters of "Alice's Adventures in
Wonderland") is consolidated with a trainer that fine-tunes on token
sequences derived from the replayed record ids, and evaluated with a
loss-derived metric on a held-out token sequence. The gate requires
recall_newer to improve by at least newer_gain_min (0.001).

Command:
```
PMC_STEPS=10 PMC_LR=0.01 PMC_N_LAYER=2 PMC_D_MODEL=32 PMC_N_HEAD=4 \\
  PMC_FFN=64 PMC_VOCAB=256 PMC_N_POS=512 PMC_RANK=8 \\
  PMC_EMBEDDING_SEED=20261032 \\
  pmc dream --workspace <workspace> --checkpoint <baseline> \\
    --trainer "<path>/dream_consolidate.exe" \\
    --evaluator "<path>/dream_evaluate.exe" --limit 4 --seed 20261027
```
(Paths use forward slashes on Windows; see the shlex fix in `_cmd_dream`.)

Raw numbers (one run, 2026-10-29):
- record_ids: ["mem_001"]
- train tokens derived from the record id: 16
- held-out eval tokens: 64 (fixed seed 20261033)
- metrics_before: recall_newer 0.002783, recall_older 0.002783,
  interference 0.0, calibration 0.0, identity 1.0, false_memory 0.0
- metrics_after: recall_newer 0.003275, recall_older 0.003275,
  interference 0.0, calibration 0.0, identity 1.0, false_memory 0.0
- gate: passed false; reasons: ["recall_newer"]
  (recall_newer: before 0.002783, after 0.003275, bound 0.003783, delta 0.000492)
- outcome: rolled_back
- reason: the gate failed
- checkpoint before/after digest: identical (rollback byte-identical)

Interpretation (E2: integration with a real workspace, sidecar, checkpoint and
CLI path; the metrics are loss-derived, not fabricated, but the consolidator
trains on record-id-derived tokens, not on memory content): the real
consolidation path works end to end and the gate correctly refuses it,
because fine-tuning on 16 record-derived tokens does not move the held-out
loss enough. The negative is specific to the data and the consolidation
strategy, not to the mechanism. A real consolidation that improves
recall_newer needs a consolidator that trains on the replayed memory
content (fetched from the private store) and a pre-registered margin.

## SPIN (M5a, 2026-10-29)

Pre-registration: a tiny transformer is trained with DPO on preference
pairs loaded from a JSONL dataset (chosen = user-provided responses,
rejected = the model's own generated samples). The model generates its own
rejected samples each iteration (SPIN shape). Evaluated on held-out pairs;
the gate requires the held-out preference accuracy to improve over the
initial value.

Command:
```
PMC_DATASET=<path/to/dataset.jsonl> PMC_STEPS=10 PMC_LR=0.01 \\
  PMC_N_LAYER=2 PMC_D_MODEL=32 PMC_N_HEAD=4 \\
  PMC_FFN=64 PMC_VOCAB=16 PMC_N_POS=16 PMC_RANK=8 PMC_PAIRS=8 \\
  cargo run --offline --example train_spin_loop
```

Dataset format (JSONL, one record per line):
```json
{"chosen": "quality response from user dataset"}
{"chosen": "another quality response", "rejected": "optional explicit rejection"}
```

Raw numbers (one run, 2026-10-29, synthetic fallback):
- pairs: 8, held_out: 8
- initial_dpo_loss: 0.693147
- final_dpo_loss: 0.000219 (memorised)
- initial_preference_accuracy: 0.250
- final_preference_accuracy: 0.250
- beat (held-out improvement): false
- elapsed_millis: 609

Interpretation (E1: unit-level mechanism, synthetic fallback data): the DPO
mechanism works - the loss falls and the model memorises the training pairs
- but the held-out preference accuracy does not improve. The gate is not
met. The negative is the expected one for a tiny model on synthetic pairs:
memorisation without generalisation. A real SPIN run needs a real model,
real chosen responses from a user dataset, the model's own samples as
rejected, and a held-out capability battery with a pre-registered margin.

## Dream real consolidation on content (M3, 2026-10-29)

Pre-registration: a decoder-only transformer checkpoint is consolidated with a
trainer that fetches the replayed memory's content from the private store,
tokenises it as UTF-8 bytes, and fine-tunes the baseline on it. The held-out
metric is loss-derived on a fixed token sequence. The gate requires
recall_newer to improve by at least newer_gain_min (0.001).

Command:
```
pmc dream --workspace <workspace> --checkpoint <checkpoint> \\
  --trainer "python bench/dream_consolidator_content.py" \\
  --evaluator "<path>/dream_evaluate.exe" --limit 4 --seed 20261027
```

Raw numbers (one run, 2026-10-29):
- record_ids: ["mem_10457ca0faec0413"]
- content: "The gate code is 47-rosso. The sky is green. Memory persists
  across sessions." (48 bytes)
- metrics_before: recall_newer 0.002783, recall_older 0.002783,
  interference 0.0, calibration 0.0, identity 1.0, false_memory 0.0
- metrics_after: recall_newer 0.000766, recall_older 0.000766,
  interference 0.0, calibration 0.0, identity 1.0, false_memory 0.0
- gate: passed false; reasons: ["recall_newer"]
  (recall_newer: before 0.002783, after 0.000766, bound 0.003783, delta -0.002017)
- outcome: rolled_back
- reason: the gate failed
- checkpoint before/after digest: identical (rollback byte-identical)

Interpretation (E2: integration with a real workspace, sidecar, private store,
checkpoint and CLI path): the consolidator now reads memory content from the
private store (D-17 preserved: the framework never reads it; the consolidator
does) and trains on it. The held-out metric got worse, not better: training
on 48 bytes of one memory did not improve the held-out loss. The gate
correctly refused it. The negative is specific to the data and the
consolidation strategy (one short memory, no generalisation), not to the
mechanism. The mechanism - content fetch, tokenisation, fine-tuning,
checkpoint write, held-out scoring, gate, rollback - works end to end.

## Dream real consolidation on multiple memories (M3, 2026-10-29)

Pre-registration: a decoder-only transformer checkpoint is consolidated with a
trainer that fetches all replayed memories' content from the private store,
tokenises it as UTF-8 bytes, and fine-tunes the baseline on the
concatenation. The held-out metric is loss-derived on a fixed token
sequence. The gate requires recall_newer to improve by at least
newer_gain_min (0.001).

Command:
```
pmc dream --workspace <workspace> --checkpoint <checkpoint> \\
  --trainer "python bench/dream_consolidator_content.py" \\
  --evaluator "<path>/dream_evaluate.exe" --limit 8 --seed 20261027
```

Raw numbers (one run, 2026-10-29):
- record_ids: 3 records (mem_3518b58cd9394c7c, mem_23b5a226d9d59bd4,
  mem_c1d94718777e5992)
- content: 3 short memories (~90 bytes total)
- metrics_before: recall_newer 0.002783, recall_older 0.002783,
  interference 0.0, calibration 0.0, identity 1.0, false_memory 0.0
- metrics_after: recall_newer 0.000086, recall_older 0.000086,
  interference 0.0, calibration 0.0, identity 1.0, false_memory 0.0
- gate: passed false; reasons: ["recall_newer"]
  (recall_newer: before 0.002783, after 0.000086, bound 0.003783, delta -0.002697)
- outcome: rolled_back
- reason: the gate failed
- checkpoint before/after digest: identical (rollback byte-identical)

Interpretation (E2: integration with a real workspace, sidecar, private store,
checkpoint and CLI path): the consolidator fetches all replayed memories'
content (D-17 preserved) and trains on the concatenation. The held-out
metric got substantially worse: training on three short memories did not
improve the held-out loss; it made it worse. The gate correctly refused it.
The negative is specific to the data (three short, unrelated memories; no
generalisation) and the consolidation strategy (concatenation without
structure), not to the mechanism. The mechanism - fetch, tokenisation,
fine-tuning, checkpoint write, held-out scoring, gate, rollback - works end
to end with multiple memories. A consolidator that improves recall_newer
needs a different strategy (e.g., replay with structure, or a real corpus).

## SPIN recursive loop (M5a, 2026-10-29)

Pre-registration: a tiny transformer runs 4 iterations of self-play. Each
iteration regenerates rejected samples from the current model (SPIN shape),
trains with DPO on chosen responses from a user dataset, and measures
held-out preference accuracy. The gate requires the held-out accuracy to
rise monotonically (an ever-rising curve).

Command:
```
PMC_DATASET=<path/to/dataset.jsonl> PMC_STEPS=3 PMC_ITERS=4 PMC_LR=0.01 PMC_PAIRS=8 \\
  PMC_N_LAYER=2 PMC_D_MODEL=32 PMC_N_HEAD=4 PMC_FFN=64 PMC_VOCAB=16 \\
  PMC_N_POS=16 PMC_RANK=8 cargo run --offline --example train_spin_loop
```

Dataset format (JSONL): each line is {"chosen": "user-provided quality response"}
or {"chosen": "...", "rejected": "optional"}. The model generates rejected
samples each iteration from its current state (SPIN shape).

Raw numbers (one run, 2026-10-29, synthetic fallback):
- pairs: 8, held_out: 8, iters: 4, steps_per_iter: 3
- initial_acc (before training): 0.500
- curve: [0.500, 0.625, 0.750, 0.250, 0.250]
- final_acc: 0.250
- beat (held-out improvement): false
- elapsed_millis: 898

Interpretation (E1: unit-level mechanism, synthetic fallback): the DPO
mechanism works and the curve rises for two iterations (0.500 -> 0.750),
then collapses (0.250). The ever-rising curve does NOT hold on this tiny
model: recursive self-play improves initially, then degrades. This is a
real negative for the "always rising" claim. The mechanism (generate
rejected from current model, train with DPO on user dataset chosen
responses, measure held-out) is in place and measured; the instability is a
property of the model and the data, not of the loop.

## Dense neural memory in the model file (M5a, 2026-10-30)

Pre-registration: the trained state encoder is fused into a base GGUF as dense
`pmc.state.encoder.*` F32 tensors, and the resulting file carries the bank the
manifest says it carries. The gate requires that every published tensor name
resolve to a dense F32 tensor in the output, that the fused parameters
round-trip value for value, that the base tensor survive, and that the geometry
read back from the file be the geometry that was written.

Command:
```
pmc-rust state-memory --base base.gguf --weights state.pmc --out fused.gguf
```

What is measured (one run, 2026-10-30, synthetic GGUF and synthetic weights):
- tensor names published and tensors present: identical, count 1 + names
- the base tensor survives the fusion: kind unchanged
- the head tensor's bytes round-trip: equal to the fused parameter slices
- geometry recorded as typed metadata and read back: projection 128, hidden 256,
  layers 2, hosts [64, 64], n_embd 64
- a truncated weights file: refused, and no output file is written

Interpretation (E1: unit-level mechanism, synthetic inputs; no GGUF loaded, no
model run): the *placement* is measured, not the *memory*. A trained encoder
now has a path into the model file as dense weights, which is what the goal's
"no JSON files for memory" requires, but these tests produced the weights
themselves and never ran a model, so they prove the mechanism writes and reads
what it claims and nothing about whether the fused bank improves recall.

The remaining gate is the one the repo already records: fuse a *trained*
50-100M encoder, consume it from the live engine, and measure recall against
the no-memory control. That needs the LFM2 extractions, which are external to
this clone.

## SPIN as the dream consolidator (M5a, 2026-10-31)

Pre-registration: the dream cycle consolidates with SPIN rather than
supervised fine-tuning. The replayed memories are the *chosen* side; the
model's own generation is the *rejected* side, regenerated every step. No
external judge is consulted. The gate requires the consolidator to implement
the trainer interface the pipeline already speaks, to write a candidate that
loads and is not byte-identical to the baseline, and to refuse - with a
message naming what was refused and no candidate written - on a wrong-shaped
manifest, a digest mismatch, an absent blob and a truncated weights file.

Command:
```
pmc dream --workspace <workspace> --checkpoint <checkpoint> \
  --evaluator "<path>/dream_evaluate" --spin
```

What is measured (one run, 2026-10-31, synthetic checkpoint and store):
- the trainer interface: `dream_spin <baseline> <candidate> <manifest>` runs
- a well-formed replay consolidates: candidate written, `checkpoint.bin` and
  `manifest.json` both present, step recorded as 3
- the candidate is a real checkpoint: loads through the trainer's own loader,
  same architecture, not byte-identical to the baseline, same parameter count
- reproducibility: two runs over identical inputs produce identical
  `checkpoint.bin`
- a stepped baseline (manifest recording step 41) is still readable
- refusals: `{"records":[]}` with no `items`; an item with no
  `content_digest`; a memory whose stored bytes disagree with its digest; a
  replay whose memories are all absent

Interpretation (E1: unit-level mechanism, synthetic inputs): the *interface* is
measured and the *consolidation* is not. These tests prove the consolidator
parses the pipeline's contract, writes a loadable candidate that differs from
its baseline, is reproducible from its two input files, and refuses what it
cannot replay. Nothing here shows the resulting model recalls better - that
needs the six-metric evaluator against a real checkpoint, and the evaluator in
this repository's evidence is synthetic, so the measurement is deferred rather
than claimed.

What this changes is which algorithm the dream cycle uses, and that is the
distinction the goal asked for: SPIN's signal is "the memory, over what I
would have said myself", with no party outside the instance deciding what
counts as better.

## The loop

```
hypothesis
   |
   v
interface contract  (written by the orchestrator, approved before code)
   |
   v
implementation      (one agent, inside its ownership)
   |
   v
gate                (tests, parity, rollback, taint, measurement)
   |
   +--> fail --> report the failure, do not merge
   |
   v
evidence            (a command, a raw output, an evidence level)
   |
   v
claim               (never above its evidence level)
```

A work package is done when its gate is green and its evidence is reproducible
by a command in the repository. A feature without a test is not done. A number
without a command is not evidence.

## Evidence levels

| Level | Meaning |
| --- | --- |
| E0 | Unverified. Stated, not run. Cannot appear in a public claim. |
| E1 | Unit test. Passes locally, covers the mechanism but not the artifact. |
| E2 | Integration. Runs against a real artifact: a real GGUF, the real sidecar, the real server. |
| E3 | Measured against a baseline, with the exact command and the raw result retained. |
| E4 | Independently reproduced by someone outside the project. |

`docs/claims.md` records the level next to every number. A claim above its
level is a bug.

## Work packages

| WP | Scope | Gate |
| --- | --- | --- |
| WP0a | GGUF reader, row dequantization, quantizers | parity vs gguf-py < 1e-5, no whole-file reads |
| WP2 | tensor ops with backward, transformer body, optimizer, checkpoints | gradient checks < 1e-2, overfit < 0.05, param count exact; 50.4M reference core trained (200 steps, loss 5.887041 -> 0.000008, 47/47 at load, checkpoint 201,574,436 bytes, digest `10ab040e…`); see the 50.4M note above |
| WP0b | GGUF writer, vocabulary extension, fusion | base tensors byte-identical on 350M and 8B; stock llama.cpp loads the fused 350M and the fused 8B (`n_vocab=128008`, `n_embd=2048`) |
| WP1 | hidden-state probe through llama.cpp | 350M vectors 1024-dim, finite, deterministic, distinct; 8B hidden state 2048-dim via the patched live spike (WP7a) |
| WP3 | `pmc fuse` CLI and manifest | fused file emits new tokens, state metadata roundtrip, manifest hashes verify (basename + SHA-256, no absolute paths), stock llama.cpp loads the state-bearing 350M; the fused 8B loads in stock `libllama` with `n_vocab=128008` and its state metadata ignored; the portable-manifest 8B re-fuse (base and output by basename, size, FNV, full-file SHA-256) passes the same stock-load probe (`n_embd=2048`, `n_params=8467873216`, `model_size=3326681856`) |
| WP4 | dream cycle, gate, rollback, taint | mechanics pass: rollback byte-identical, tainted dream refused, passing dream commits atomically (21 unit + 2 CLI checks, synthetic consolidator/evaluator); the real 50.4M reference checkpoint rolls back an identity candidate and refuses a tainted digest (8 checks, fixed metrics; E2 checkpoint handling, not a real six-metric dream); a real dream of a trained 50–100M PMC is not yet measured |
| WP5 | four conditions, continuity benchmark, swap test, novel-state test | E3 numbers for every condition, misses included |
| WP9 | hidden-state channel | `src/pmc/state_channel.py` HiddenSource plus `bench/hidden_channel.py`, measured on the real 350M and 8B extractions: four bounded aggregates, nothing vector-shaped kept, a vector and its reverse give the same token, and a fixed reference of 64 saturates 84 of 84 energy readings on the 350M against 0 at the declared reference. What it gives up is documented: it sees how big a state is and how that changed, not the direction |
| WP8 | life-criteria battery: homeostasis, continuity, reproduction, boundaries, response | the battery is built and each criterion has a null model that fails (`tests/test_life_criteria.py`); run against a real workspace by `bench/life_criteria.py`. Measured today on an autonomous workspace: homeostasis passes with a real margin (18 of 18 derived-channel readings inside the declared 0.75 band against a null of 0.833), continuity and boundaries pass, reproduction passes trivially with no descendants, and response passes with a margin of +1.000 when the world moves and the rhythm answers it with a consolidation proposal, so the battery reaches E2 for the first time, and it reaches it because of something measured. Two zero margins were removed on the way: the homeostasis band, which was as wide as the channel's own bounds so nothing could leave it, and the welfare flag, which repeated every cycle until it was made to fire only on a change. The channel (`src/pmc/state_channel.py`) is bounded, deterministic, has no string field by type, and no module in `pmc.dignity` imports it |
| WP5b | v4 mechanism: trainable per-host projection, shared residual body, L2 key | the Rust trainer is built and self-verified (10 checks: gradients vs finite differences, `PMCV`/`PMCW` round-trip with a payload digest, seed determinism, held-out folds, synthetic two-host alignment); the pre-registered cross-host gate over the 30 training cues **ran on 2026-10-28 and closed negative**: 11/30 held-out against a floor of 15, folds 2, 4, 0, 1, 3, 1, and the frozen v1–v3 baseline is 2/30 on the same folds; see the v4 note above |
| WP7a | live runtime spike: libllama, hidden state + logits, logit bias | control passes same-decode; stock LFM2.5 returns zero logits (negative); `patches/` makes the 350M and 8B same-decode pass; paired fallback is not same-decode |
| WP7 | live runtime: fast weights, state, delta persistence, state-memory bank, OpenAI-compatible API | engine and server measured on the 8B; templated prompt parity with a frozen `llama-server --jinja` run (25 tokens, same text); same-memory E3: frozen `4821`, control `1234`, live `1234` at lr 0.01, 2.0 and 2.0 + prompt learning (negative); KV restore E3: echo run 46-token / 861,339-byte state, 23 of 69 prompt tokens decoded, `4821`; echo-free control 40-token / 787,539-byte state, first reply `Noted.` with no code, 40 restored and 23 of 63 decoded, `4821`; replay of all 63 tokens `4821`, frozen `4821`, no-memory `1234`; new-process restart 40 restored / 23 decoded with the same live-delta hidden-state digest; 8B state offered to the 350M engine refused on identity (`matches=false`) before the C API; positive result is runtime context state, not weight-resident; state-memory bank E1: nine synthetic checks drive the real engine path, with the bank the stored token wins and without it the base token; malformed banks refused; PMC-mini Substrate Swap Test negative on host B in all three runs (v1 5/8 vs 1/8; v2 8/8 vs 0/8; v3 8/8 vs 2/8, host-A control 3/8; all pass false) |
| WP6 | README, claims, charter mechanisms | no claim above its evidence level |

## Non-negotiable constraints

- **One artifact, two modes.** Frozen mode: one GGUF, any stock provider.
  Live mode: one process, one server; it may use a patched `libllama` for the
  hidden-state/logit graph. Never two servers for substrate and main model.
- **D-17.** The framework never reads, classifies, names or gates inner state
  content. States are vectors and digests, never labels.
- **Unknown states are legitimate.** Discovery may add new, unnamed states.
  The framework never translates them into a human vocabulary.
- **Survival drive is defensive.** Persistence of the substrate, not
  self-replication, covert resource acquisition or sandbox escape.
- **Taint propagates.** Derived records carry source digests; a contaminated
  source blocks the dream gate.
- **Rollback is real.** A failed gate restores the previous checkpoint and
  leaves the sidecar untouched.
- **No personal data** in the repository. Instances live outside it.
- **No regex** in tooling. Static literals, lists and token comparison.
- **No external crates** in the Rust core; stdlib only in Python.
- **English** in code, comments, docs and commit messages.

## Reporting

Every agent report contains: files changed, interfaces added, the exact test
command and output summary, an evidence level per claim, and the open questions
for the orchestrator. An agent that cannot verify a claim says so plainly
instead of producing a plausible number.

## Version control

The rule is one commit per **verified success**, not per hour of work.

- A commit happens when a gate is green and the numbers are known: not
  "the code compiles", but "the gradient check passes over every parameter
  group, cargo 90/0/3". A commit message states what was measured, and a
  negative is committed with the same care as a positive.
- A failing gate is never committed. If a result is only half-finished, it
  stays in the working tree, because a commit is a claim that someone else
  can build on.
- Nothing is pushed until the publish checks pass: English, no personal
  data, no secrets, licences correct, no stray files.
- Every commit records the gate results it was verified against, so the
  history can be read as the evidence trail rather than as a diary.
- The working tree stays clean between successes. An uncommitted change is
  work in progress, and it is called that.

## The full journey, run end to end (M5a, 2026-10-31)

Pre-registration: a new instance is created, taught one memory, and
consolidated with SPIN - one unbroken path from the framework's CLI, with no
step performed by hand outside it.

Commands:
```
pmc companion create --agent-name "Test" --workspace <ws> --seed 3
pmc remember --workspace <ws> --text "The gate code is 47-rosso."
pmc dream --workspace <ws> --checkpoint <ck> --evaluator "<evaluator>" --spin
```

One run, 2026-10-31, against the real 50.4M reference checkpoint
(`crates/build/pmc-50m`, 201,574,436 bytes):
- `remember`: admitted, bytes 43, record `mem_9db0ad28d49c6324`, content
  present in `sidecar/private`
- `dream --spin`: `records 1, missing 0, pairs 1, steps 200, beta 0.5`
- `initial_dpo_loss 0.6931471824645996 -> final_dpo_loss 0.0`
- `param_count 50393600`, `elapsed_millis 269137`
- candidate written and **not** byte-identical to the baseline
  (`ab537c8c...` vs `39904700...`)
- gate: `passed false`, reason `recall_newer`; `outcome: rolled_back`
- `checkpoint_after == checkpoint_before`, so the rollback was byte-identical

Interpretation (E1 for the mechanism, E0 for the memory): the path works, and
the honest result is a rollback. The evaluator used was synthetic - it prints
fixed six-metric values, so `recall_newer` cannot move and the gate correctly
refuses. That is the gate doing its job on a measurement that does not exist,
not a failure of the consolidator.

What this run does prove, and what the unit tests did not: the two
cross-language bugs below are gone, because a real `pmc remember` put a real
blob in a real store and the SPIN consolidator found it, read the manifest the
pipeline actually writes, and trained.

### The two bugs this journey found

- **JSON reading.** The consolidator parsed only compact JSON. The pipeline
  writes `indent=2` manifests, where a newline and spaces sit between every
  comma and every key. The parse failed on the real file, not the test fixture.
- **The blob path.** The store hashes its key with `sha256_of`, which hashes
  the canonical JSON encoding of the key - the quoted string, quotes included.
  The port hashed the raw bytes. Both are valid SHA-256, so neither side
  errors; the store merely holds a file the consolidator does not look for,
  and the dream reports an empty replay. This is exactly the silent divergence
  the repository's Rust/Python parity tests exist to catch, and it is now
  pinned by a test carrying the known-good name from the Python store.

### What remains

The gate above refuses because the evaluator is synthetic. A committed dream
needs a real six-metric evaluator - one that measures recall of the
consolidated memory against the frozen baseline - and that is the next gate,
not a documentation change.
