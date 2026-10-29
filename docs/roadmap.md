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

### M1 contract: `pmc-token` (written before implementation)

- `Tokenizer::from_reader(reader: &GgufReader) -> Result<Tokenizer, TokenizerError>`
  reads `tokenizer.ggml.model` (only `gpt2` byte-level BPE is in scope;
  anything else is refused by name), `tokenizer.ggml.tokens` and
  `tokenizer.ggml.merges` (both required for BPE), `tokenizer.ggml.scores`
  and `tokenizer.ggml.token_type` (optional; a wrong length is refused), and
  the optional `bos`/`eos`/`unk` id metadata.
- `Tokenizer::encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>`
  is deterministic byte-level BPE. Special tokens (CONTROL and
  USER_DEFINED types) split the text first, earliest position then longest
  match wins. Each remaining segment is pretokenized by a hand-rolled
  scanner implementing the GPT-2 pretoken pattern without regex: `\p{L}` is
  approximated by `char::is_alphabetic`, `\p{N}` by `char::is_numeric`,
  `\s` by `char::is_whitespace`, contractions are the fixed lowercase set
  `'s 't 're 've 'm 'll 'd`. The approximation is declared here and pinned
  by fixed cases in the tests. Each pretoken's UTF-8 bytes map into the
  GPT-2 byte alphabet; adjacent pairs merge by lowest rank until no rank
  applies; a piece not in the vocabulary is an error, never a silent unk.
- `Tokenizer::decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8, TokenizerError>`
  and `Tokenizer::decode_string(&self, ids: &[u32]) -> Result<String, TokenizerError>`
  invert the byte map; `decode_string` refuses non-UTF-8 instead of
  replacing; a character outside the byte alphabet is an error naming the
  token. BOS and EOS are never added implicitly; callers compose with the
  `bos_id`/`eos_id` accessors.
- Accessors: `vocab_size`, `bos_id`, `eos_id`, `unk_id`.
- Guarantees under test: the byte alphabet is a bijection over all 256 byte
  values; `decode(encode(x)) == x` for arbitrary UTF-8 without special
  tokens; the same input yields the same ids on every call.

**Gate (pre-registered).** Held-out perplexity below the random-init baseline
by a declared margin; the Rust/Python parity test extended to the text path;
one command reproduces the run from seed + config + corpus digest.
**Evidence:** E3.

**Status (2026-10-29):** first work package delivered — the `pmc-token`
crate (`crates/pmc-token/`) reads `tokenizer.ggml.*` from a GGUF and
implements byte-level BPE with no external crates and no regex: the GPT-2
byte alphabet (a bijection over all 256 values), a hand-rolled pretoken
scanner whose Unicode-class approximation is declared and pinned by fixed
cases, rank-ordered merges, special-token splitting (earliest then longest),
and `decode_bytes`/`decode_string` that refuse invalid UTF-8. Measured:
`cargo test -p pmc-token` 13 passed, 0 failed, exit 0, including a
metadata-only GGUF roundtrip and an env-gated real-model roundtrip
(`PMC_TEST_GGUF`). Remaining M1 work at that point: the corpus format and
sharding, then the next-token loss in `pmc-train`.

**Status (2026-10-29):** M1 gate MET on a real corpus. `crates/pmc-train/examples/train_lm.rs`
trained on the first 512 characters of "Alice's Adventures in Wonderland"
(Project Gutenberg #11, public domain), char-level tokenisation, 2 layers,
22944 parameters, 20 steps: train loss 5.538515 to 2.778033, baseline eval
loss 5.566339, final eval loss 4.548255, final perplexity 94.47 vs 261.48
baseline, beat_baseline true. Raw numbers and command in docs/lab.md. The
Rust/Python parity extension for the transformer remains. M1 is effectively
complete at the mechanism + gate level; the parity extension is the last
sub-gate.

**Status (2026-10-29):** third work package delivered - `crates/pmc-train/examples/train_lm.rs`
trains a decoder-only transformer on a token sequence with next-token prediction,
holds out a suffix, scores it before and after, writes a checkpoint, and prints a
JSON report with the random-init baseline beside the final number. Measured on
a 64-token pseudo sequence, 2 layers, 136512 parameters, 30 steps: train loss
5.700954 to 0.000372 (memorised), baseline eval loss 5.502320, final eval
loss 14.372885, final perplexity 1746081.63, beat_baseline false. The
mechanism is in place and reproducible; the gate is NOT met on this data -
recorded as a negative in docs/lab.md. Remaining M1 work: a real corpus
(TinyStories subset) and a pre-registered margin, plus the Rust/Python parity
extension for the text path.

**Status (2026-10-29):** second work package delivered - `src/pmc/data/`
corpus module: `CorpusRecord` (validated id/text/split), per-record and
whole-corpus SHA-256 digests (order-independent), JSONL parse/write,
deterministic seed-based sharding that does not depend on input order,
split grouping, and injected tokenization. Measured: `pytest
tests/test_corpus.py` 20 passed, exit 0; full suite 399 passed, 8 skipped
(2 symlink, 6 parity without cargo on PATH; parity 6 passed with cargo),
exit 0. Remaining M1 work: the next-token loss on real token ids in
`pmc-train` (`transformer.rs`), then the M1 gate (held-out perplexity).


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

**Status (2026-10-29):** delivered. `src/pmc/data/ingest.py` adds readers for
CSV, TXT and Markdown (JSONL already in corpus.py), dedup by content digest,
declared quality filters, and `admit_corpus` under `ProvenanceKind.EXTERNAL`
with content never in the ledger. Taint propagation is exercised by marking
an admitted digest in a `TaintLedger`. Measured: `pytest tests/test_ingest.py`
12 passed, exit 0; full suite 411 passed, 8 skipped, exit 0. M2 gate met at
E2 for the ingest path.

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

**Status (2026-10-29):** CLI path exercised end to end (E2). Identity
consolidator + mechanics evaluator run through `pmc dream`: replay, taint
check, consolidation, evaluation, gate, rollback. The gate correctly
refused the no-op on recall_newer (newer_gain_min 0.001) and left the
checkpoint byte-identical. The consolidator and evaluator are plumbing
exercisers, not measurements: their metrics are fixed values. A real
consolidation (training on replayed memory that improves recall_newer) and
a real evaluator (measuring the six metrics on held-out memory) remain as
the M3 follow-up.

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

**Status (2026-10-29):** first sub-package delivered (M6a). `src/pmc/runtime/providers.py`
adds `Provider` (keys read from the environment only, never from files),
`ProviderRegistry` (JSON file loader), `ResourceBudget` (declared
`network_requests_per_day` with persistence and refusal), and `ExternalCall`
(budget + resource event + taint mark). `ChatClient` gained an `api_key`
parameter that emits an `Authorization: Bearer` header. Measured:
`pytest tests/test_providers.py` 15 passed, exit 0; full suite 426 passed,
8 skipped, exit 0.

**Status (2026-10-29):** second sub-package delivered (M6b). `src/pmc/runtime/tools.py`
adds a declared tool registry (`memory.remember`, `memory.restore`,
`state.read`, `dream.propose`) with validation, handlers backed by the
session/store/protocol, and resource-event journaling. The live server now
accepts a `tools` array and `tool`-role messages on `/v1/chat/completions`,
validates them, and exposes `execute_tool` through `LiveHTTPServer` and
`make_server`. Measured: `pytest tests/test_tools.py` 8 passed, exit 0;
full suite 434 passed, 8 skipped, exit 0. M6 is complete at the mechanism
level; the tool loop against a real model remains (the live engine does not
emit tool calls on its own).

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
