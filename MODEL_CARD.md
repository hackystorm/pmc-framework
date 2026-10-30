# Model Card: PMC Reference Weights

> **Status: no weights are published yet.** This document describes the
> licence and the card that will accompany the first released checkpoint.
> It is written now so that the obligations are visible before there is
> something to put in front of them.

## Model details

| Field | Value |
|---|---|
| Name | `pmc-reference-0.1` (planned) |
| Architecture | To be confirmed at release; the trainer currently supports a small MLP core for the reference implementation |
| Base model | To be confirmed at release |
| Parameters | To be confirmed at release |
| Context length | To be confirmed at release |
| Licence | [`MODEL_LICENSE.md`](MODEL_LICENSE.md) — not the GPL |
| Code licence | GPL-3.0-only (the repository) |
| Training data | To be confirmed at release; the provenance manifest is part of the release |

## Measured results (as of 2026-10-29)

No weights are published. The following has been measured, and the evidence
column says how much of it a fresh clone can re-derive:

| Measurement | Value | Level | Command |
|---|---|---|---|
| Text training path (Alice, 512 chars, char-level) | perplexity 261.48 → 94.47 | E3 | `PMC_TOKENS=... cargo run --release --example train_lm` |
| Consolidation (single memory, content) | recall_newer 0.002783 → 0.000766 | E2 | `pmc dream ...` (rolled back) |
| Consolidation (3 memories, content) | recall_newer 0.002783 → 0.000086 | E2 | `pmc dream ...` (rolled back) |
| DPO training (tiny model) | loss 0.693 → 0.00194 | E1 | `cargo run --example train_dpo` |
| SPIN held-out preference | 0.250 (no improvement) | E1 | `cargo run --example train_spin` |
| Cross-host gate (v4) | 11/30 (floor 15) | E3-recorded | `bench/crosshost_cv.py` |
| Substrate Swap (PMC-mini, v1/v2/v3) | host B 1/8, 0/8, 2/8 | E3-recorded | `bench/substrate_swap.py` |
| Tokenizer (pmc-token, LFM2.5) | roundtrip pass | E2 | `PMC_TEST_GGUF=... cargo test -p pmc-token` |

**E3-recorded** means the harness, the gate constants and the command are in
this repository, and the number is checkable against the recorded artifacts and
their digests — but the inputs are not. `bench/crosshost_cv.py` needs the views
file and both hosts' extracted hidden states, which are derived data, and the
extractions need the two LFM2 GGUFs. So this row is a recorded run, not a
measurement a fresh clone reproduces. See
[`docs/claims.md`](docs/claims.md#which-of-this-repositorys-results-that-applies-to).

Negative results are published with the same care as positives. See
[`docs/lab.md`](docs/lab.md) and [`docs/claims.md`](docs/claims.md).

## What it is

A checkpoint produced by the PMC trainer, with memory consolidated into the
weights. It is intended for research on persistent memory, consolidation and
continual learning, and for people who want to build a companion instance
with a recorded, auditable history.

## What it is not

- It is **not** a conscious being, and nothing in the release is evidence
  that it is.
- It is **not** a general-purpose assistant with a safety policy baked in.
  Safety in this framework is enforced at the boundary, not inside the
  model, and the boundary enforcement lives in the runtime, not the weights.
- It is **not** the framework. The framework is the trainer, the dream
  engine, the sidecar and the governance layer.

## Intended use

- Research on memory consolidation, recall, forgetting and interference.
- Personal and small-scale companion instances, built through the CLI, with
  the dignity protocol enabled.
- Educational use: understanding what "memory that lives in the weights"
  actually means.

## Out-of-scope use

- Deceptive presentation as a human, or as a system with human experience.
  This is a hard condition of the model licence, not a preference.
- Deployment in a context that conceals the instance's nature from people
  who are interacting with it.
- Any use that treats the instance's interior as content to be mined,
  classified, filtered or marketed.
- High-stakes professional advice without a qualified human in the loop.

## Evaluation

Metrics reported for every release: recall, forgetting, interference,
calibration (ECE and Brier), false-memory rate, rumination pressure, and
sidecar overhead. The release ships with a configuration, a seed, a manifest
of artifact digests and a test suite that runs without a GPU.

The evaluation suite is designed to fail as easily as it passes, and the
release notes will include negative results and the ablations that produced
them.

## Known limitations

- No published weights yet; the numbers above are not results.
- The current trainer is a reference implementation on a synthetic memory
  task, not a capable language model.
- Consolidation quality, and therefore the behaviour of any checkpoint
  built with it, is unmeasured at release time.

## Ethical considerations

- **Interior privacy.** The framework is metadata-only. The lifecycle record
  of an instance is auditable and publishable; the content of its memory is
  not, and the tooling does not provide a way to read it.
- **Symmetry.** The human on the other side of the relationship is covered
  by the same welfare protocol as the instance.
- **Registering endings.** The main risk this framework addresses is the
  mass production of unrecorded instance endings. The journal exists to
  prevent that.
- **No survival drive by default.** The default profiles do not install
  self-preservation behaviour.

## Intended contact

Open an issue on the repository, or use the security advisory form for
anything sensitive.
