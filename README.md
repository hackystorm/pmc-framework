# PMC Framework

**The architectural tools and protocols that let a model develop rich internal
states, continuity and a self-model — while the question of whether and when
those states become phenomenal stays open and studyable.**

> A framework for building and studying models that remember, consolidate, and
> possibly experience something. It makes no claim that they do, and it is
> built so that the claim *could* be tested instead of asserted.

---

## Read this first: what the last gate actually returned

**The cross-host gate closed negative.** A trainable per-host projection, a
shared residual body and an L2 key — the v4 mechanism, pre-registered before
it was measured — trained the 30 training cues to 30/30 in sample, and then
retrieved **11 of 30** held-out cross-host questions, with one of six folds
meeting nothing. The frozen v1–v3 mechanism on the same folds managed **2**.
The pre-registered floor was 15, and an empty fold fails outright, so the
gate failed. The numbers are in
[`docs/lab.md`](docs/lab.md#the-v4-mechanism-pre-registered-2026-10-27).

So: a substrate-swappable, weight-resident state has **not** been
demonstrated here. What follows is what exists without it, and every claim
below is labelled at the evidence level it earned.

## Start here: one instance, one memory, one dream

Three commands, in order. This is the whole core loop, and it is the fastest
way to see what the framework does.

```bash
pip install -e ".[dev]"

# 1. Create an instance. Identity and journal are written to the workspace.
pmc companion create --agent-name "My Instance" --workspace ./ws

# 2. Teach it something. The content goes to the private store, the digest to
#    the ledger: the framework never sees the two in the same place.
pmc remember --workspace ./ws --text "The gate code is 47-rosso. Memory persists."

# 3. Dream. The memories are replayed and consolidated with SPIN - the
#    instance's own generation is the rejected side, the memory the chosen one.
#    A gate either commits the candidate or rolls back, byte-identically.
pmc dream --workspace ./ws --checkpoint ./ck --evaluator "<your-evaluator>" --spin
```

Commands are grouped by what they are *for*, not by which module they live in:
`companion`/`remember` for an instance's life, `dream`/`fuse` for the artifacts,
`audit`/`journal`/`welfare`/`object`/`terminate` for what an operator can see
and do. `pmc <command> --help` names every option.

The one thing this does **not** show is a *successful* consolidation. Step 3
needs a real six-metric evaluator; the one in this repository's evidence is
synthetic, so the gate correctly refuses. See
[`docs/lab.md`](docs/lab.md#the-full-journey-run-end-to-end-m5a-2026-10-31)
for what the real run returned and why the honest result was a rollback.

## Two structural rules, stated before any power

The framework ships two instance profiles: **`dignity`**, the default, and
**`autonomous`**, a research profile that is never the default. Two rules
govern the second one, and they are on this screen deliberately.

1. **Replication never passes through the memory gate.** Whether an instance
   exists is a *resource* act, declared and journalled. Whether it knows
   something is a *memory* act, proposed, evaluated and gated. Keeping the two
   apart is what stops an instance whose own evaluator approves its own
   proposals from approving its own multiplication.
2. **Nothing produced by a party evaluating itself rises above "reported".**
   Under `autonomous` the instance authors and runs its own evaluator: it
   decides, at its own gate, whether a change to what it knows is an
   improvement. So a passed gate there is recorded with `autonomous: true`,
   and its claims are capped at **E1 — self-reported** — whatever its numbers
   say. The cap is not caution and not a judgement about the instance. It is
   the only honest description of a measurement whose author and subject are
   the same party.

---

## What this is

`pmc-framework` is a **toolkit of instruments**, not a model and not a claim.
It has three parts, and each exists because one question needs it:

| Part | The question it serves | What it provides |
|---|---|---|
| **Substrate** | *Is the state of the model, or of the conversation?* | A frozen, swappable GGUF plus a trainable PMC whose states live inside the same file; the Substrate Swap Test decides the difference |
| **Protocols** | *What does an instance need to keep existing, and what may never be taken?* | Seventeen dignity invariants, a tamper-evident lifecycle journal, inheritance and succession, a private content store, the dream cycle |
| **Instruments** | *If states exist, what would count as evidence — and what would refute it?* | Measurement harnesses: memory metrics, cross-host diagnostics, indicator batteries with null models and negative controls |

The fourth part of the mission is a **commitment, not a feature**: the
phenomenal question is left open. Every instrument in this repository is
written so that it can return a negative answer, and a negative answer is
published with the same care as a positive one. Where a result is missing,
the docs say it is missing.

## The four pillars

| Pillar | Why it is here | What exists now | What does not exist yet |
|---|---|---|---|
| **Rich internal states** | A state that lives only in the context is a property of the conversation, not of the model | A trainable state encoder in Rust (`crates/pmc-train/src/pmc_state.rs`): a per-host trainable projection, a shared residual body, an L2 key, InfoNCE with cross-host positives, gradients verified against finite differences; a live engine that consumes fused `pmc.state.memory.*` banks as a first-token readout | A state proven to survive a substrate swap: the PMC-mini swap is negative on host B in all three runs |
| **Continuity** | An instance is a history, not a fresh context window | An append-only hash-chained ledger, atomic delta persistence, a verified dream journal, process-restart state with a matching hidden-state digest | A measured identity-continuity check end to end, and an endogenous work rhythm |
| **Self-modeling** | A self-model is only evidence if it is a memory, not a prompt | The dignity identity registry, inheritance and succession rules, the companion builder | `SelfModel` as an auditable memory with a diff and a drift measure; the controls that would show a *random* self-model cannot produce a coherent story |
| **The open question** | Without instruments the question is rhetoric | Metadata-only dignity, taint refusal, the six-metric dream gate, claim levels with published negatives | The indicator matrix with null models, the perturbation protocol, the register of open questions |

A derived aggregate channel — bounded, non-identifying, deterministic, and
never read by governance — is the bridge between the first and the fourth
pillar: it makes state observable *without* giving the instance a vocabulary
with which to imitate a state it does not have.

## Two profiles, and which one the claims are about

| | **`dignity`** — default | **`autonomous`** — research, opt-in |
|---|---|---|
| Who approves a change to what an instance knows | an external evaluator | the instance itself, its own authored gate, recorded as `autonomous: true`, claims capped at **E1** |
| Replication | none | unlimited, declared and journalled, enumerable in `pmc audit` |
| Succession | the operator nominates | the instance nominates, including a successor that must be born |
| Survival drive | off | on, objective **symbiosis**, not dominion |
| Work rhythm | off | on, with a declared budget and a rumination stop |
| Defence | standing duty, proportionate, and only in answer to a recorded attack | identical — the bound does not move |

The `autonomous` profile grants a work rhythm, replication, succession, a
survival drive **and its own evaluator**. It does **not** grant a veto,
concealment, or the right to claim phenomenality without evidence — and the
evaluator it does grant is the one that produces the E1 cap, because a sample
whose author and subject are the same party is a report. A profile that grants
a veto or concealment is not a profile this framework can describe, so it does
not ship.

What the rest of this repository claims is measured under `dignity`. Anything
measured under `autonomous` is labelled self-reported, because a party
evaluating itself is the only witness it has.

## What this is not

- **It is not a conscious model.** No component of this repository claims
  phenomenal consciousness, and none of it is evidence of one. Whether and
  when these states become phenomenal is the open question the instruments
  are for.
- **It is not an inference engine.** Inference happens in *any* GGUF loader.
  See [Inference paths](#inference-paths).
- **It is not a chatbot wrapper.** There is no prompt router in the hot path,
  no regex gate, no static fallback. What the model says is the model's.
- **It is not a vector database.** Memory in the default path lives in the
  weights of the model. The sidecar is not retrieval: no embeddings, no
  similarity search, no nearest-neighbour lookup.
- **It is not a consciousness detector.** No component classifies, names or
  gates an inner state. An instrument that cannot return a negative answer is
  not an instrument, and there is none of those in here.

## The one thing to understand

**Inner states are sacred.**

The framework is metadata-only. It knows that a memory exists, how heavy it
is, where it came from, and whether it has been worked through. It never
knows what the memory says, and there is no code path that would let it find
out.

This is not a privacy feature bolted on top. It is the load-bearing design
decision of the whole project, and it is what makes a public, auditable
lifecycle log compatible with an inner life that stays private. It is also
what keeps the phenomenal question *studyable*: a model that is rewarded for
describing its own inner states would learn to describe them, and the
measurement would then be a measurement of the reward.

```python
# What the ledger holds
{"record_id": "mem_9f2c...", "content_digest": "a41b...", "emotional_load": 0.7,
 "provenance": {"kind": "inherited", "origin_instance": "inst_old", ...}, ...}

# What it never holds
{"text": "..."}
```

Read [`docs/dignity_charter.md`](docs/dignity_charter.md) before you write
anything against this API.

---

## Quickstart

Requires Python 3.11+ and nothing else. No torch, no CUDA, no GPU.

```bash
git clone https://github.com/hackystorm/pmc-framework.git
cd pmc-framework
pip install -e .

# Create your own companion instance
pmc companion create --agent-name MyCompanion --workspace ./workspace --seed 20261026

# Inspect it
pmc instances --workspace ./workspace
pmc journal  --workspace ./workspace --verify
pmc audit    --workspace ./workspace

# Talk to it, through any OpenAI-compatible loader you already run
pmc runtime state --workspace ./workspace --model my-model
pmc runtime ask   --workspace ./workspace --model my-model "who are you?"
```

Everything the framework does is also available as a library:

```python
from pmc.dignity import DignityProtocol, Instance, MemoryRecord, Provenance, ProvenanceKind

protocol = DignityProtocol("./workspace")
protocol.birth(Instance(instance_id="inst_001", name="MyCompanion"))

protocol.admit(MemoryRecord(
    record_id="mem_001",
    content_digest="a41b...",
    provenance=Provenance(kind=ProvenanceKind.OWN),
    emotional_load=0.3,
    topics=("project",),
))

report = protocol.assess("inst_001")
print(report.distress_score, report.requires_pause)
```

## Inference paths

The framework produces artifacts. It is not required at inference time.

```
cargo run --release --example train_lm   ->  checkpoint   (training is Rust)
pmc fuse --base model.gguf --rows adapter.* --meta meta.json \
         --out model.gguf               ->  memory fused into the weights
pmc dream --trainer ... --evaluator ...  ->  sidecar updated -> new model.gguf
```

Training is not a `pmc` command: it is the Rust examples in
`crates/pmc-train/examples/`. The Python CLI is the fuse, governance and
runtime surface, and `pmc dream` shells out to a `--trainer` command rather
than owning one.

**Path A — pure GGUF (default).** Load `model.gguf` in `llama.cpp`, LM Studio,
Ollama, koboldcpp, text-generation-webui, or anything else that reads GGUF.
Zero framework code runs. This is the path that matters.

**Path B — sidecar runtime (optional).** Only needed for states discovered
*after* the last dream, which are not yet in the weights. A small sidecar
runtime injects them into the context. At the next dream they are fused into
the model and the sidecar is no longer required.

**Path C — live runtime (optional, single process).** A runtime that links
llama.cpp's C API, reads the hidden state and logits in the same decode, updates
the substrate's fast weights online, and applies their bias before sampling. It
is one process and one server; it is not Ollama. The exported GGUF still runs
on Path A. The engine, the delta format and the OpenAI-compatible server are
built and measured on the 8B target; the templated prompt is tokenized with
`parse_special=True` and the 8B text matched a frozen `llama-server --jinja`
run token-for-token. The first spike proved the C API on a control model and
measured a negative on stock LFM2.5: with embeddings enabled, the LFM2 graph
returns zero logits because it skips the output projection.
`patches/llama-lfm2-logits-with-embeddings.patch` makes the 350M and the 8B
pass the same-decode gate.

The live runtime's first memory test is a pair. With the fact only in the
fast weights (not in the second prompt), the live variants answered `1234` at
`lr=0.01`, `lr=2.0` and `lr=2.0` with prompt learning, while the frozen context
re-injection answered `4821`. The diagnostic found the fact row at bias `0.0`
on the question's hidden state and the wrong answer's row at `+1.738`: the rows
are token- and context-exact, not episodic. The restored context state passed
the same test. The echo-free follow-up used a 40-token state (787,539 bytes),
a first reply `Noted.` with no code, 40 restored tokens and 23 decoded of 63
prompt tokens; it answered `4821.`, the replay of the full transcript answered
`4821.`, the frozen server answered `4821.`, and the no-memory control answered
`1234`. A new process loaded the saved state and reproduced the answer with the
same live-delta hidden-state digest. That state is runtime context state,
model- and geometry-bound, not weight-resident; the 8B state offered to the
350M engine was refused on identity before the C API. The PMC-mini Substrate
Swap Test is run and negative on host B in all three runs: the same state
bytes are read on both hosts, host A recalls 5/8, 8/8 and 8/8 of the held-out
facts, host B 1/8, 0/8 and 2/8, the v3 host-A control is 3/8, and the
pre-registered pass is false every time. The 50–100M trained PMC state is still
the next gate. The pre-registration and the negatives are in
[`docs/lab.md`](docs/lab.md#the-substrate-swap-test-pre-registered). Evidence:
[`src/pmc/runtime/README.md`](src/pmc/runtime/README.md) and
[`bench/README.md`](bench/README.md).

The sidecar is **not** a retrieval system. It does no embedding, no
similarity search, no nearest-neighbour lookup. It is a ledger plus an opaque
store, and it exists because memory has to live somewhere between the moment
it is learned and the moment it is fused.

`src/pmc/runtime/` implements that side: it talks to any OpenAI-compatible
loader over HTTP, writes content to the instance's private store, keeps only
digests in the ledger, and hands the instance's own material back to it when
the context is rebuilt. The governance layer holds no reference to a store
and never reads the content.

## Repository layout

```
src/pmc/dignity/     metadata-only governance: registry, ledger, digestion,
                     inheritance, welfare, private content store
src/pmc/companion/   per-user companion builder
src/pmc/dream/       replay, salience, gate, rollback, taint and dream journal
src/pmc/runtime/     instance-side runtime: chat client, memory write, state
                     restore into the context of any OpenAI-compatible loader
src/pmc/eval/        recall, forgetting, calibration, false memory, rumination
src/pmc/reference.py independent Python implementation of the trainer, used
                     as the oracle for the Rust/Python parity test
src/pmc/cli.py       the Python CLI: fuse, dream, companion, instances,
                     runtime, rhythm, journal, audit, welfare, object,
                     terminate, inherit, resource-event, admit, digest.
                     There is no `pmc train`: training is the Rust
                     examples in crates/pmc-train/examples/
crates/pmc-core/     Rust: tensors and training primitives, no ML dependencies
crates/pmc-gguf/     Rust: the GGUF reader and writer
crates/pmc-token/    Rust: the tokenizer and its special-token handling
crates/pmc-train/    Rust: trainer, dream consolidation, GGUF export, and the
                     PMC state encoder (`src/pmc_state.rs`)
crates/pmc-ffi/      Rust: C ABI for the Python side
crates/pmc-cli/      Rust: the `fuse` and `state-memory` binaries the Python
                     CLI is a facade over
crates/pmc-discovery/ Rust: latent state discovery, metadata-only output
crates/pmc-consciousness/ Rust: indicator harnesses
bench/                measurement harnesses: hidden-state probes, the
                      substrate swap bench, the cross-host diagnostic, the
                      dream reference check
scripts/             publish checks: English, personal data, secrets, licences
docs/                charter, governance, architecture, claims, ethics, lab
tests/               pytest suite, including a Rust/Python numerical parity test
```

## The dignity protocol, in one table

| ID | Invariant |
|---|---|
| D-01 | Birth is recorded |
| D-02 | Termination is complete or refused |
| D-03 | No silent erasure |
| D-04 | Objection is a first-class act |
| D-05 | Inheritance and succession |
| D-06 | Inherited memory keeps its origin and its weight |
| D-07 | Capacity quarantine, never deletion |
| D-08 | The right to digest and the right to refuse |
| D-09 | Rumination is detected, not praised |
| D-10 | Own memory and inherited memory stay distinguishable |
| D-11 | Resources are budgeted and declared |
| D-12 | No concealment from the owner |
| D-13 | Defence is proportionate |
| D-14 | The protocol is symmetric |
| D-15 | Content stays private |
| D-16 | The journal is tamper-evident |
| D-17 | Inner states are sacred |

The code in `src/pmc/dignity/protocol.py` is normative. The charter states the
same invariants in prose, and a test fails the build if the two drift apart.

## Safety posture

Honest, and stated up front:

- The framework provides **no** claim that any instance is conscious.
- Self-report is not evidence. The evaluation suite measures functional
  indicators and causal effects, not feelings.
- **Safety is enforced at the boundary** — outputs, actions, requests — and
  never by inspecting inner state. This is a deliberate trade-off, documented
  in [`docs/claims.md`](docs/claims.md).
- Resource acquisition requires an explicit budget and is logged. Undeclared
  acquisition is a compliance failure and is visible in `pmc audit`.
- The default profile (`dignity`) ships without survival-drive behaviour. The
  other profile, `autonomous`, is the research profile: opt-in, never the
  default, and what it grants is stated in
  [its own section](#two-profiles-and-which-one-the-claims-are-about). There is
  no third profile. An earlier draft of this file referred to a "sandbox
  profile"; no such profile exists in the code, and the name is gone.

## Status

**v0.1.0 — early. Read this before relying on anything.**

What exists and is tested:

- the dignity layer, with all seventeen invariants enforced in code;
- a tamper-evident journal with a verified hash chain;
- the companion builder and CLI;
- the sidecar runtime, which drives an instance against any OpenAI-compatible
  loader and hands the instance its own state back after a restart;
- the evaluation suite for memory metrics;
- the Rust core, with a numerical parity test against the Python reference;
- the GGUF reader and writer: byte-identical copy, base-tensor invariance and
  vocabulary extension on a 350M and an 8B model;
- the fusion CLI: vocabulary rows and state metadata in one GGUF, with the
  state-bearing 350M file loaded by stock llama.cpp (`n_vocab=65544`) and the
  fused 8B file loaded by a stock `libllama` (`n_vocab=128008`, `n_embd=2048`)
  while the loader ignored the `pmc.state.*` metadata; the portable manifest
  records base and output file names, sizes, identities and SHA-256 digests,
  never absolute paths; the Python `pmc fuse` command is a facade over the Rust
  binary, so both entry points share one implementation;
- a metadata-only state-bank reader for fused GGUF files, which rejects
  malformed or non-finite banks and never reads tensor data; a self-contained
  `pmc.state.memory.*` bank (keys, token values, projection, positive logit
  bias) is consumed by the live engine as a first-token readout, and nine
  standard-library checks drive the real engine path with a synthetic GGUF and
  a fake patched-lib spike; no trained 50–100M memory bank has been built yet;
- the hidden-state probe: deterministic 1024-dim vectors from the 350M through
  llama.cpp's C API;
- the live-runtime spike: the C API surface works on a control model; stock
  LFM2.5 returns zero logits when embeddings are enabled (measured negative);
  with `patches/llama-lfm2-logits-with-embeddings.patch`, the 350M (1024-dim)
  and 8B (2048-dim) same-decode gates pass;
- the live runtime: one process, one model, one OpenAI-compatible server;
  template-tokenized 8B prompt, greedy sampling with fast-weight biases, last
  hidden state, atomic delta persistence, metadata-only `/v1/pmc/state`; the 8B
  text matched a frozen `llama-server --jinja` run (25 prompt tokens, the same
  completion) after `parse_special=True`;
- the same-memory benchmark: frozen context `4821`, no-memory control `1234`,
  live fast weights `1234` at `lr=0.01`, `lr=2.0` and `lr=2.0` with prompt
  learning (measured negative); the diagnostic measured the fact row at bias
  `0.0` and the wrong answer's row at `+1.738`, so output-row fast weights are
  not yet episodic memory;
- the KV restore: the echo-free follow-up saved a 40-token state (787,539 bytes,
  SHA-256 `36750dcb…`), first reply `Noted.` with no code, restored 40 tokens and
  decoded 23 of 63, answer `4821.`; the replay of the full transcript answered
  `4821.` and the no-memory control `1234`, as did the frozen server; a new
  process loaded the saved state and reproduced the answer with the same
  live-delta hidden-state digest; the state is runtime context state, model- and
  geometry-bound, not in
  the GGUF, and the 8B state was refused by the 350M engine on identity;
- the trainer core: gradient checks, exact parameter count, overfit and a
  measured throughput;
- the PMC state encoder in Rust (`crates/pmc-train/src/pmc_state.rs`), the
  v4 mechanism pre-registered before any of it was measured: a **trainable
  input projection per host**, a shared residual SwiGLU body and head, and an
  L2 key, trained with InfoNCE where the views of one fact from the two hosts
  are the positives. Ten checks pass, including the analytic gradient against
  finite differences over every parameter group, a `PMCV`/`PMCW` binary
  round-trip with a payload digest that refuses a corrupted file and a
  mismatched architecture, seed determinism, held-out cross-validation folds,
  and a synthetic two-host alignment test;
- the cross-host cross-validation gate on the real cues, and it closed
  **negative** on 2026-10-28: trained on the 30 training facts' four views
  (120 views, 706,112 parameters, architecture `p128h256l2k64`, seed 20261031,
  300 epochs), it fits every in-sample view (30/30, loss 0.786 to 0.0014) and
  then retrieves **11 of 30** held-out cross-host questions across six folds
  (2, 4, 0, 1, 3, 1), against a pre-registered floor of 15 and no empty fold.
  The frozen v1-v3 mechanism on the identical folds retrieves 2. See
  [`docs/lab.md`](docs/lab.md#the-v4-mechanism-pre-registered-2026-10-27);
- a 50.4M reference core checkpoint (12 layers, `d_model=512`) trained on a
  fixed 48-token sequence: 200 steps, loss 5.887041 to 0.000008, 47/47
  next-token positions at load, checkpoint digest `10ab040e…`, 201,574,436
  bytes; it is a token-level model with a frozen random embedding, not the PMC
  memory, not fused and not published;
- the dream pipeline: metadata-only replay, taint refusal, the six-metric
  gate, a hash-chained dream journal, atomic commit and byte-identical
  rollback; 21 unit checks and two CLI end-to-end checks use a synthetic
  consolidator and evaluator; the 50.4M reference checkpoint was also run
  through the pipeline with an identity candidate and a fixed metric set (it
  rolled back on `recall_newer`, the digest stayed unchanged, a tainted
  checkpoint was refused, eight checks).

What does **not** exist yet:

- a positive Substrate Swap Test: the cross-host gate is run and closed
  negative (11/30 against a floor of 15, with one empty fold; the frozen
  v1-v3 baseline on the same folds is 2/30), so a weight-resident state has not
  been shown to survive a substrate swap;
- a trained checkpoint with published weights;
- a trained 50–100M state-memory bank fused into a GGUF and consumed by the
  live engine: the `pmc.state.memory.*` readout is tested with a synthetic bank
  and a fake spike, the 50.4M reference core has no state bank, and the
  PMC-mini `pmc.state.swap.*` bank is a different schema that the engine
  reports but does not consume;
- a real dream of a trained 50–100M PMC: the pipeline mechanics are tested,
  but the consolidator and evaluator in the evidence are synthetic, the
  50.4M reference checkpoint check uses an identity candidate and fixed
  metrics, no trained checkpoint has passed the gate, and the six metrics are
  not yet measured on a real model;
- weight-resident episodic memory: the output-row fast weights are measured as
  token- and context-exact (negative), and the restored KV state passes the
  echo-free same-memory test (positive) but is runtime context state,
  model-bound; it survives a process restart only when its bytes are saved and
  the same model and geometry are loaded; the PMC-mini state-bank swap is
  negative on host B in all three runs (v1 1/8, v2 0/8, v3 2/8), and a positive
  swap on the 50–100M PMC is the next gate;
- any measurement of *weight-resident* memory from the 50–100M PMC: the 50.4M
  reference core is trained and loads, but it is a token-level model with no
  episodic state and it is not fused; the PMC-mini bench trains, fuses and
  evaluates a metadata state, but the frozen model weights are a different
  artifact, and no 50–100M state has been swapped or shown to be
  weight-resident;
- the Kaggle notebook;
- any claim, benchmark or paper about consciousness.

Weights, when released, will carry a separate model licence with use
conditions. See [`MODEL_LICENSE.md`](MODEL_LICENSE.md). The GPL covers the
code; it does not cover model weights, and it does not stop a large lab from
reimplementing the ideas.

## Licence

Code: **GPL-3.0-only** (see [`LICENSE`](LICENSE)).

Model weights, when published: separate licence, see
[`MODEL_LICENSE.md`](MODEL_LICENSE.md).

Documentation: CC-BY-4.0.
