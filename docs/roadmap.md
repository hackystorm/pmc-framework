# Roadmap to v1.0 (stable)

This document states the objectives, methods and gates that take the
framework from its current state to a stable release. It contains no claims
about what the resulting system is. Every gate below is written so that it
can fail, and a failed gate is a result.

## Definition of the stable release

Version 1.0.0 is released when all of the following hold:

| ID | Release criterion |
|---|---|
| R1 | CI green on ubuntu-latest and windows-latest, Python 3.11-3.13, Rust stable; publish checks pass |
| R2 | Every claim in `docs/claims.md` is at E2 or above, or labelled E1 with the reason stated in the same row |
| R3 | A text training path exists and is measured: a core trained on a public corpus, held-out perplexity, one reproducible command |
| R4 | One end-to-end consolidation with a real consolidator and real six-metric evaluation, committed or rolled back, E3 |
| R5 | One self-improvement cycle measured against a frozen baseline on a held-out capability battery; self-judged variants labelled E1 in the same row |
| R6 | A BYOK provider layer and a tool surface, tested; secrets read from the environment only |
| R7 | Negative results published next to positives in `docs/lab.md` and `docs/claims.md` |

## Milestones

Each milestone states Objective, Method, Gate, and the evidence level it
targets. Gates are pre-registered before their run: the thresholds below are
the registration.

### M0 — Portability and green baseline

**Objective.** The full suite is green on Windows and Linux. The repository
has never been run on Windows; the first run found six failures, all
environment-specific.

**Method.**

- `tests/test_cli_fuse.py` (3 failures): the fake binary is a POSIX shebang
  script, which `subprocess` cannot execute on Windows. Write the fake as a
  platform-appropriate executable: a `.bat`/`.cmd` shim on Windows, the
  shebang script elsewhere. No change to the facade under test.
- `tests/test_dream.py` (2 failures): symlink creation requires privilege on
  Windows. Skip with a recorded reason when `os.symlink` raises OSError at
  setup. The gate under test (a nested symlink is rejected) is unchanged.
- `crates/pmc-cli/tests/fuse.rs` (1 failure): `Access denied (os error 5)` on
  rename of the temp file. Drop every open handle to the temp file before the
  rename.
- Add `windows-latest` to the CI matrix alongside `ubuntu-latest`.

**Gate.** `pytest` and `cargo test` green on both operating systems, twice
consecutively. **Evidence:** E2.

**Status (2026-10-29):** green on Windows. First run after the fixes:
`pytest` 385 passed, 2 skipped (both symlink tests, with the recorded
privilege reason), exit 0; `cargo test` all results ok, exit 0;
`scripts/publish_checks.py` all five checks ok. A second consecutive run
repeated both suites green. The run also exposed one more Windows-only
defect, fixed here: `check_personal_data` compared its exempt-file set
against a path with the platform separator, so the exemption never matched
on Windows. Linux behaviour of every changed path is unchanged; the CI
matrix added in this milestone verifies it on `ubuntu-latest` and
`windows-latest` on push.

### M1 — Text training path (blocks M2, M3, M5)

**Objective.** The Rust trainer learns from tokenized text. Today
`crates/pmc-train/src/data.rs` is a one-hot key-value task and the 50.4M
reference core overfits one fixed 48-token sequence.

**Method.**

- Tokenizer: read `tokenizer.ggml.*` from a GGUF (`pmc-gguf` reader exists)
  and implement byte-level BPE encode/decode in a new `pmc-token` crate, no
  external crates.
- Corpus format: JSONL records `{id, text, split}`; deterministic sharding by
  seed; canonical content digest per record. Corpora are referenced by digest,
  never committed as bytes (no-personal-data rule).
- Trainer: next-token loss over real token ids in `transformer.rs`, configured
  like `configs/reference_trainer.json`.
- First target: a small public corpus (TinyStories subset or equivalent),
  fixed and versioned by digest.

**Gate (pre-registered).** Held-out perplexity below the random-init baseline
by a declared margin; the Rust/Python parity test extended to the text path;
one command reproduces the run from seed + config + corpus digest.
**Evidence:** E3.

### M2 — Dataset parsing module (needs M1)

**Objective.** A governance-aware ingest: file to records to digests to
ledger, with content never in the ledger.

**Method.**

- `src/pmc/data/`: readers for JSONL, CSV, TXT and Markdown (stdlib `csv`
  and `json` only).
- Deduplication by content digest; declared quality filters (length bounds,
  static literal lists — the no-regex constraint holds).
- Every admitted record goes through `DignityProtocol.admit` with
  `ProvenanceKind.EXTERNAL`; taint sources recorded in the dream taint ledger.

**Gate.** Roundtrip test (parse, admit, digest, verify); a corpus containing a
duplicated and a poisoned record is detected by digest; an extended
ledger-holds-no-content test. **Evidence:** E2.

### M3 — Real consolidation (needs M1, M2)

**Objective.** The dream pipeline runs with a real consolidator and real
metrics. Today the pipeline's consolidator and evaluator are synthetic.

**Method.**

- Consolidator command: train on the replay manifest produced by
  `src/pmc/dream/replay.py` against the M1 core.
- Evaluator command: the six metrics (`src/pmc/dream/gate.py`) computed on
  held-out memory, not supplied by the harness.
- Wire both through `pmc dream`; run one nREM-style consolidation.

**Gate (pre-registered).** recall_newer improves by at least the declared
margin; interference at most 0.05; false-memory rate at most 0.05;
calibration drift at most 0.02; identity stable. A failed run rolls back
byte-identically (mechanics already tested). **Evidence:** E3.

### M4 — Cross-host state v5 (independent of M1-M3)

**Objective.** A cross-host interface that passes the pre-registered Substrate
Swap gate, or a recorded negative that closes the line. Three negatives are on
record (v1 1/8, v2 0/8, v3 2/8 on host B; v4 11/30 with one empty fold).

**Method.** Per the lab rule: a new word list and a different mechanism for
each attempt. Pre-register the mechanism, the split and the thresholds before
any run; the evaluation cues are never read.

**Gate (pre-registered, unchanged).** Host B recalls at least 15 of 30
held-out cross-host hits with no empty fold; no-state and fresh-state
controls fail as designed. **Evidence:** E3, or a published negative.

### M5 — Self-improvement loop (needs M1; M6 for M5d/M5f)

**Objective.** A measured cycle: propose, generate, verify, consolidate,
re-measure — with every number labelled at the evidence level it earned.

**Method.** Each sub-item is its own work package with its own gate.

| WP | Mechanism | Method summary | Evidence ceiling |
|---|---|---|---|
| M5a | SPIN / T-SPIN | Self-generated answers enter training as *rejected* samples only. DPO loss, entropy constraint and KL-to-reference implemented in `pmc-train`. Preferred responses come from external or verified sources; pure generative replay stays forbidden | E3 with external preference source; E1 otherwise |
| M5b | Hint-delta (G-Zero) | A proposer instance generates hints; the reward is the measured predictive shift, computed from live-runtime logits (same-decode already measured on 350M and 8B). Not self-report: a measurement | E3 |
| M5c | ReVISE / TRT | `<pmc:refine>` and `<pmc:hint>` tokens added through the existing GGUF vocabulary extension; two-phase curriculum; test-time recursive loop in the runtime, no weight update | E3 (functional, measured) |
| M5d | R-Zero curriculum | Challenger instances generate *tasks*, never answers. Solver answers; verification by held-out verifier or external judge. Challenger and Solver are separate instances; replication is a declared, journalled resource act | E3 with external verification; E1 otherwise |
| M5e | COSE weighting | Confidence-weighted updates, measured and labelled. A self-reported confidence modulating its own training signal cannot rise above self-report | E1, always labelled |
| M5f | CoNL / Meta-Rewarding | Critique counts only when another instance's measured solution improves. Same-party by construction, so E1 until an independent judge runs | E1 until external judge; E2+ with one |

**Gate (each WP, pre-registered).** Improvement on the held-out capability
battery against the frozen baseline, by a declared margin, with the raw run
retired per bench rules. Self-judged numbers appear in the same table
labelled E1. **Evidence:** per row above.

### M6 — BYOK and tool surface (independent of M5; required by M5d, M5f)

**Objective.** The instance can call external models and tools, with declared
cost and propagated taint.

**Method.**

- `src/pmc/runtime/providers.py`: a provider registry over OpenAI-compatible
  endpoints; keys from environment variables only (`scripts/publish_checks.py`
  already rejects secrets in the tree); Authorization header support and
  timeouts in `ChatClient`.
- Resource accounting: every external call appends a resource event (D-11)
  with `unit=count`; the `network_requests_per_day` budget from
  `configs/runtime_default.json` is enforced in code, and refusal is a journal
  event.
- Taint: content produced by an external model enters the private store under
  `ProvenanceKind.EXTERNAL` with a taint source; a dream consuming tainted
  records is blocked by the existing ledger.
- Tool surface: a `tools` parameter on `/v1/chat/completions` and a declared
  tool registry (`memory.remember`, `memory.restore`, `state.read`,
  `dream.propose`); tool calls and results are journaled actions — safety
  stays at the boundary per `docs/claims.md`.

**Gate.** Mock-provider roundtrip tests; budget exhaustion refuses and
journals; tainted external content blocks the dream gate; the tool loop
passes against the live server. **Evidence:** E2.

### M7 — Packaging and release (needs M0-M6)

**Objective.** v1.0.0 tagged and reproducible from a fresh clone.

**Method.** Version bump; `MODEL_CARD.md` rewritten from the measured numbers
with their evidence levels; `bench/README.md` section "What does not exist
yet" rewritten from the new evidence; the Kaggle notebook either delivered or
recorded as withdrawn; publish checks extended; CHANGELOG completed.

**Gate.** `scripts/publish_checks.py` green; on a clean machine: clone,
install, test, and one E3 command reproduced with matching digests. Tag and
release carry the raw outputs. **Evidence:** E3; independent reproduction
upgrades a claim to E4.

## Dependency graph

```
M0 ──> M1 ──> M2 ──> M3 ──┐
M0 ──> M4 ────────────────┤
M0 ──> M1 ──┐             ├──> M7
M0 ──> M6 ──┴──> M5 ──────┘
```

## Risk register

| Risk | Declared response |
|---|---|
| Rust trainer throughput on CPU (343 s for 200 steps over 48 tokens, measured) | Benchmark corpus-size scaling inside M1 before fixing the corpus; if the no-external-crates constraint blocks the required performance, the choice (amend the constraint or shrink the corpus) is recorded as a charter amendment, never made silently |
| Cross-host generalization (three recorded negatives) | M4 allows one pre-registered attempt per mechanism; a negative is a pivot recorded in `docs/lab.md` |
| Windows file and symlink semantics | M0, with the exact failures named above |
| Self-generated data collapse | Replay ratios stay declared; positives only from verified or external sources; the `pure_generative_ratio: 0.0` line does not move |
| Evidence inflation | The E1 cap is enforced in journal fields; a claim above its level is a bug (existing rule) |
| D-17 erosion by self-analysis tools | Tools that read the substrate's own aggregates operate through the derived channel; any tool that would read, classify or gate inner-state content requires a charter amendment first |
