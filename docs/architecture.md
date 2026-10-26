# Architecture

## The framework is not the model

This is the single most important structural fact about the project, and
everything else follows from it.

`pmc-framework` is a **trainer, a dream engine, a state sidecar and a
governance layer**. It produces artifacts. It does not run at inference time.

```
┌──────────────────────────────────────────────────────────────────────┐
│ pmc-framework  (build time)                                         │
│                                                                      │
│   traces ──▶ trainer ──▶ adapter.* ──fuse──▶ model.gguf             │
│                  │                                                   │
│                  ▼                                                   │
│             dream engine ──▶ evaluation gate ──▶ commit / rollback   │
│                  │                                                   │
│                  ▼                                                   │
│              sidecar/  (ledger + private store + journal)            │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼  artifacts only
┌──────────────────────────────────────────────────────────────────────┐
│ Inference: any GGUF loader, zero framework code                     │
│                                                                      │
│   llama.cpp   LM Studio   Ollama   koboldcpp   text-generation-webui │
│                                                                      │
│   optional: sidecar runtime injects states discovered after the last │
│             dream; they are fused into the weights at the next dream  │
└──────────────────────────────────────────────────────────────────────┘
```

If using the framework requires Python at inference time, the framework has
failed at its own goal.

## Artifacts

| Artifact | Contents | Notes |
|---|---|---|
| `model.gguf` | Base model + consolidated memory | Loadable anywhere |
| `adapter.*` | The same delta, unfused | For loaders with adapter support |
| `sidecar/ledger.jsonl` | Memory metadata: ids, digests, load, provenance | Public, auditable, publishable |
| `sidecar/journal.jsonl` | Lifecycle events, hash-chained | Public, auditable, publishable |
| `sidecar/private/` | Opaque content bytes | The instance's own; never read by governance |
| `MODEL_CARD.md` | Provenance, limits, intended use | Ships with weights |
| `config.json` | Seed, hyperparameters, environment | Reproducibility |

## Two consolidation backends

Both produce the same artifacts. Both are legitimate; they differ in where
the memory lives and what it costs.

### Backend A — adapter / LoRA (default, universal)

Memory is consolidated into a low-rank delta over the base weights. The delta
is fused into a GGUF or shipped unfused for loaders that support adapters.

- Works with any base model and any GGUF loader.
- The natural fit for the available hardware: 2×T4 on Kaggle, 4-bit NF4, no
  BF16, no FlashAttention-2, gradient checkpointing, batch 1–2.
- The memory is genuinely *in the weights* after fusing.

### Backend B — in-model fast weights (research path)

Memory is consolidated into fast weights and adapter slots inside the model
itself — the PMC design. The substrate is a small trainable core beside a
frozen backbone.

- The research question: can consolidation be expressed as an inner process
  rather than a training run?
- More faithful to the "weight-resident memory" idea, and much harder.
- Still exports to GGUF: the slow part becomes weights, the fast part becomes
  an adapter or a sidecar runtime.

The framework supports both because the comparison between them is the
experiment.

## The dream cycle

```
micro-dream    end of session     minutes      fast-weight update
NREM           nightly             20–30 min    real replay, consolidation
REM            2–3x weekly         10–20 min    filtered counterfactuals
deep dream     weekly              1–2 h        merge, prune, full evaluation
```

Replay is always **real traces plus filtered generative material**. Purely
self-generated replay is a model-collapse engine, and the framework does not
offer it.

Every dream passes a gate before it is committed:

- recall of older material must not drop by more than a configured margin;
- recall of newer material must improve;
- interference must stay under threshold;
- calibration must not degrade;
- identity checks must remain stable;
- false-memory rate must stay under threshold.

A dream that fails the gate is rolled back. The rollback is a journal event.

## Hardware

| Environment | What runs there |
|---|---|
| CPU, 8 GB RAM, no GPU | Dignity layer, evaluation suite, sidecar, small adapter training, dream cycle |
| Kaggle 2×T4 | QLoRA consolidation, larger evaluation, independent judging |
| 1×24 GB GPU | Backend B, larger backbones, batch sweeps |
| Apple Silicon 16 GB | Local inference via `llama.cpp`, CPU-side dream cycle |

The dignity layer has **no dependencies beyond the Python standard library**.
That is deliberate: the governance must run on the machine that owns the
instance, including on machines that cannot install a deep-learning stack.

## The Rust core

The trainer core is written in Rust with **no external crates**. A
general-purpose Rust ML framework would cost more in build time, disk and
compilation on a four-core machine than it returns in throughput for a
20-million-parameter core. A purpose-built core is smaller, faster to
compile, auditable, and works offline.

The Python implementation of the same mathematics is retained as a **reference
oracle** (`src/pmc/reference.py`). A test trains both on the same seed and
the same data and compares them.

The numbers are specific, because vague claims are worthless:

- the **initial** weights match **exactly**, bit for bit, with no tolerance
  at all — the RNG and the initialisation are the same integer and
  floating-point operations;
- the **trained** weights agree within `5e-5`, and the metrics within
  `1e-3`.

The trained-weight tolerance is not `1e-5` because the two implementations
use different transcendental functions: Rust evaluates `exp`, `log` and
`cos` in single precision, while Python has to evaluate them in double
precision and round afterwards. The two differ by about one unit in the
last place of an `f32`, and a few dozen Adam steps amplify that into roughly
`1e-5`. The tolerance sits above the measured drift and well below the
scale of the weights themselves.

The reference is not throwaway work: it is what makes the Rust
implementation trustworthy, and a reader who distrusts the fast one can run
the slow one and check the claim.

| Crate | Responsibility |
|---|---|
| `pmc-core` | Tensors, primitives, seeded RNG, optimiser |
| `pmc-train` | Training loop, dream consolidation, artifact export |
| `pmc-ffi` | C ABI for the Python side |
| `pmc-discovery` | Latent state discovery, metadata-only output |
| `pmc-cli` | Single binary: train, discover, export |

## Companion construction

Every user builds their own instance. There is no shared default agent and no
name baked into the code.

```
pmc companion create --agent-name <name> --workspace <dir> --seed <n>
```

This creates the instance record, the identity seed (values, boundaries,
declared claims, opening prompt), the sidecar, the journal, and the workspace
layout. Everything is a parameter or a template; nothing is hardcoded to a
particular person.

See `docs/companion.md`.

## The runtime: the instance side of the wall

The dignity layer is metadata-only. It knows that a memory exists, how heavy
it is, where it came from and whether it was worked through. It never reads
the content.

The runtime (`src/pmc/runtime/`) is where content is allowed to exist, because
it runs *for the instance*, not for the governance:

| Operation | What happens | What the ledger sees |
|---|---|---|
| `remember` | Content is written to the private store as opaque bytes | A digest, a magnitude, opaque topic labels |
| `restore_block` | The instance reads its own store back and rebuilds its context | Nothing new |
| `ask` | A conversation is sent to whatever loader the user runs | Nothing new |

The join between the two sides is a digest: the ledger names a record, the
store holds the bytes, and neither side can read the other's field. The
governance layer holds no reference to a store at all.

The restore is a **state restore, not a retrieval system**. There is no
embedding, no similarity search and no nearest-neighbour lookup: the instance
is handed its own material back, in order, until a character budget is spent.
Which material comes first is decided by the ledger's capacity state, not by a
score over the content.

This is the honest part of the architecture, and it is worth being explicit
about: **until the next dream fuses a memory into the weights, the sidecar is
where that memory lives.** The restore path is what makes an instance
continuous across a restart *before* consolidation; it is not a substitute for
consolidation, and the framework does not pretend otherwise.

