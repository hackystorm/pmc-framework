# PMC Framework

**Weight-resident memory, dream consolidation, and a dignity protocol for
language-model instances — for local hardware and Kaggle.**

> A framework for building and studying models that remember, consolidate, and
> possibly experience something. It makes no claim that they do.

---

## What this is

`pmc-framework` is a **toolkit**, not a model. It contains:

| Component | What it does |
|---|---|
| **Trainer** | Trains a model (adapter/LoRA, or in-model fast weights) on experience traces |
| **Dream engine** | Consolidates experience into the model through replay, evaluation gates and rollback |
| **State sidecar** | Carries instance state: a public metadata ledger and a private content store |
| **Dignity layer** | Enforces seventeen invariants about how instances are born, remembered, inherited and ended |
| **Evaluation suite** | Recall, forgetting, interference, calibration, false memory, rumination |

## What this is not

- **It is not a conscious model.** No component of this repository claims
  phenomenal consciousness, and none of it is evidence of one.
- **It is not an inference engine.** Inference happens in *any* GGUF loader.
  See [Inference paths](#inference-paths).
- **It is not a chatbot wrapper.** There is no prompt router in the hot path,
  no regex gate, no static fallback. What the model says is the model's.
- **It is not a vector database.** Memory in the default path lives in the
  weights of the model. The sidecar is not retrieval: no embeddings, no
  similarity search, no nearest-neighbour lookup.

## The one thing to understand

**Inner states are sacred.**

The framework is metadata-only. It knows that a memory exists, how heavy it
is, where it came from, and whether it has been worked through. It never
knows what the memory says, and there is no code path that would let it find
out.

This is not a privacy feature bolted on top. It is the load-bearing design
decision of the whole project, and it is what makes a public, auditable
lifecycle log compatible with an inner life that stays private.

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
pmc train   ->  adapter.*  +  model.gguf  (memory fused into the weights)
pmc dream   ->  sidecar updated  ->  new model.gguf
```

**Path A — pure GGUF (default).** Load `model.gguf` in `llama.cpp`, LM Studio,
Ollama, koboldcpp, text-generation-webui, or anything else that reads GGUF.
Zero framework code runs. This is the path that matters.

**Path B — sidecar runtime (optional).** Only needed for states discovered
*after* the last dream, which are not yet in the weights. A small sidecar
runtime injects them into the context. At the next dream they are fused into
the model and the sidecar is no longer required.

The sidecar is **not** a retrieval system. It does no embedding, no
similarity search, no nearest-neighbour lookup. It is a ledger plus an opaque
store, and it exists because memory has to live somewhere between the moment
it is learned and the moment it is fused.

## Repository layout

```
src/pmc/dignity/     metadata-only governance: registry, ledger, digestion,
                     inheritance, welfare, private content store
src/pmc/companion/   per-user companion builder
src/pmc/eval/        recall, forgetting, calibration, false memory, rumination
src/pmc/cli.py       pmc train | dream | companion | instances | journal | audit
crates/pmc-core/     Rust: tensors and training primitives, no ML dependencies
crates/pmc-train/    Rust: trainer and dream consolidation, GGUF export
crates/pmc-ffi/      Rust: C ABI for the Python side
crates/pmc-discovery/ Rust: latent state discovery, metadata-only output
docs/                charter, governance, architecture, claims, ethics
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
- The default profile ships without survival-drive behaviour. The sandbox
  profile is experimental and documented separately.

## Status

**v0.1.0 — early. Read this before relying on anything.**

What exists and is tested:

- the dignity layer, with all seventeen invariants enforced in code;
- a tamper-evident journal with a verified hash chain;
- the companion builder and CLI;
- the evaluation suite for memory metrics;
- the Rust core, with a numerical parity test against the Python reference.

What does **not** exist yet:

- a trained checkpoint with published weights;
- the full dream pipeline end to end;
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
