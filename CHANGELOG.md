# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**A dense neural memory bank in the model file** (`crates/pmc-cli/src/state_memory.rs`,
`crates/pmc-cli/tests/state_memory.rs`)

`pmc-rust state-memory --base BASE.gguf --weights STATE.pmc --out OUT.gguf`
fuses a trained PMC state encoder's parameters into a GGUF as dense
`pmc.state.encoder.*` F32 tensors. The encoder is the small neural network the
goal asks for: a per-host input projection, a shared residual body, and an L2
key, whose trained weights *are* the memory.

This is the answer to "no JSON files for memory". A memory in a JSON sidecar is
a file a loader ignores, a backup can drop, and a person can edit without
leaving a trace. A memory in the model file travels with the model, is covered
by the model's digest, and needs no second program at inference time. The gate
and up branch of each SwiGLU layer are written as one `[2h, h]` tensor because
they are one operation, and the geometry — projection, hidden, layers, key
dimensions, host dimensions, the base model's `n_embd`, the tensor names and
the weight digest — is recorded as typed GGUF metadata so a reader never has to
infer it.

Three tests (`crates/pmc-cli/tests/state_memory.rs`) verify the bank against
the file that was actually written: the names the manifest publishes are the
tensors the file carries, the fused parameters round-trip value for value, the
base tensor survives, the geometry reads back as the metadata it was written
as, and a truncated weights file is refused with no output written. No GGUF and
no llama.cpp are needed; the tests compare numbers they produced themselves.

### Fixed

**`aiohttp` is an extra, not a hard dependency** (`pyproject.toml`,
`src/pmc/runtime/frenesis_daemon.py`)

- The daemon's HTTP and WebSocket surface imported `aiohttp` at module scope,
  so a machine without it could not import the module at all — and the
  framework's non-negotiable constraint is standard library only in Python. It
  is now the optional `frenesis` extra (`pip install pmc-framework[frenesis]`).
  The import is guarded, the failure names the fix instead of surfacing as an
  AttributeError inside aiohttp's own frames, and
  `LlamaServerManager.health_check` reports an unhealthy llama-server rather
  than raising out of a routine the daemon calls in a loop. `dependencies` stays
  empty. Four tests (`tests/test_frenesis_daemon.py`), one of which blocks
  `aiohttp` through a real import hook rather than simulating it in-process, so
  the guard is exercised by the condition it actually guards.

### Fixed

**The FRENESIS daemon could not be installed** (`src/pmc/runtime/frenesis_daemon.py`,
`launch_daemon.py`) and **wrote a log file at import** (both files)

- The module reached its dependencies through a `sys.path` insert plus
  `src.pmc.runtime.*`, which only resolves before `pip install`. After it,
  nothing works. The imports are relative now (`from .session import ...`), so
  the module is importable from a checkout and from an installed package, and
  the daemon holds the same class objects the rest of the framework does rather
  than a second copy of them. `launch_daemon.py` had the same defect and the
  same fix.
- Logging was configured at module scope with a `FileHandler` writing
  `frenesis_daemon.log` into the **current working directory**, so merely
  importing the module created a file wherever the importer happened to be
  standing — which is how a stray log reached the repository root and the
  publish gate. Logging is now configured by `configure_logging()`, called from
  the entry point, with a file written only when the caller names one. Seven
  tests (`tests/test_frenesis_daemon.py`), run in a subprocess where the
  property is actually import behaviour. One of them asserts the old sys.path
  arrangement would fail its own check, so the test cannot pass by merely
  working.

### Fixed

**Seven documentation and wiring defects** (`README.md`, `docs/lab.md`,
`docs/claims.md`, `MODEL_CARD.md`, `SECURITY.md`, `configs/runtime_default.json`,
`.github/workflows/ci.yml`, `src/pmc/runtime/judge.py`,
`crates/pmc-cli/Cargo.toml`, `src/pmc/cli.py`)

- **The Status section contradicted the opening.** It still described the v4
  cross-host gate as "not yet run" and the encoder as needing training on
  extracted hidden states, 400 lines below a README that opened with the
  measured 11/30 negative. The README's "What exists" and "What does **not**
  exist yet" lists and the `WP5b` row in `docs/lab.md` now all record the same
  run: 11/30 held-out, folds 2/4/0/1/3/1, floor 15, frozen v1-v3 baseline
  2/30, empty fold failing on its own clause.
- **The pre-registration date was later than the result it pre-registered.**
  The heading read `pre-registered 2026-10-31` above `Result (2026-10-28)`.
  Git history shows the original heading carried a second date,
  `2026-10-31, 2026-10-27 dated cues`, and the qualifier was dropped when the
  result was published (`22d02a6`) without correcting the date. The heading is
  restored to 2026-10-27, and the correction is recorded in `docs/lab.md`
  rather than made silently. The seed `20261031` is untouched: it is a
  recorded parameter of the run, not a date claim.
- **`pmc train` did not exist and was advertised.** The README's inference-path
  diagram and repository layout listed a command the CLI does not have. Both
  now name the real entry points: the Rust examples under
  `crates/pmc-train/examples/` for training, `pmc fuse` for fusion, and the
  correction that the Python CLI shells out to a `--trainer` command rather
  than owning one. The layout also gained the four crates that were missing.
- **A "sandbox profile" was described that does not exist in the code.** The
  Safety posture in the README, `docs/claims.md`, `SECURITY.md` and
  `configs/runtime_default.json` all referred to one; the code ships exactly
  two profiles, `dignity` (default) and `autonomous` (research, opt-in). All
  four now name the real profile and say the old name was never real.
- **CI ran against a branch that does not exist.** `.github/workflows/ci.yml`
  triggered on `main`; the only branch is `master`, so no push that mattered
  ran the suite. `SECURITY.md` said fixes land on `main` and is corrected too.
- **`src/pmc/runtime/judge.py` did not import.** It imported
  `ProviderResourceBudget`, which does not exist in `providers.py`, and a
  `pmc.dignity.taint_ledger` module that does not exist. It is now wired to
  the real provider API, the taint ledger is typed as the single-method
  protocol it actually needs (the same collaborator `ExternalCall` already
  marks), and the verdict's `confidence` is carried on `JudgeResult` instead of
  being folded away — it is what the evidence level is derived from, so a
  reader could not audit why a verdict earned the level it did. 17 tests
  (`tests/test_judge.py`) run it, so a module that fails to import now fails
  the suite.
- **The Rust binary was named `pmc`, the same as the Python console script.**
  Two different commands with one name meant `pmc fuse` could resolve its own
  facade and re-enter itself, and `pmc train` resolved to one binary and not
  the other, so the README's diagram was wrong and the Rust command was
  unreachable through the Python name. The binary is renamed `pmc-rust`
  (`crates/pmc-cli/Cargo.toml`), the Python resolver looks for that name and
  never for `pmc`, and the docstring records the recursion this ends. Five
  regression tests (`tests/test_cli_resolution.py`).

**The reproducibility boundary, stated** (`docs/lab.md`, `docs/claims.md`)

- The v4 cross-host negative is labelled for what it is: a **recorded run that
  a fresh clone cannot re-derive**. Both harnesses and the gate constants are
  committed; the views file, the two hosts' extracted hidden states and the
  LFM2 GGUFs are not. The docs now say which, give the ordered reproduction
  path, and file the result as E3-recorded rather than E3-reruns. The hardware
  results needing two specific GGUFs and a patched `libllama` are given the
  same treatment; what *is* reproducible from a clean checkout without a GPU
  or a model is listed explicitly.

**The publish gate had two blind spots** (`scripts/publish_checks.py`)

- It checked personal-data fragments as chunk prefixes, so a home directory
  assigned to a keyword argument — the ordinary way one leaks, and on Windows
  the form the list did not cover at all — never matched. Path fragments are
  now matched as substrings, and both separators are listed. Name fragments
  keep prefix matching, because the session-identifier prefix also begins the
  ordinary plural identifier this repository uses everywhere by design.
- `del` was in the Italian word list, so any file with a Python delete
  statement failed the gate on a statement in good English. Removed, per the
  file's own stated rule that a false positive is worse than a miss.
- Machine-local binaries are refused by suffix. A Windows shortcut encodes an
  absolute path to one machine's filesystem, and no text check can read one, so
  it needs a suffix rule rather than a literal-fragment rule.
- The gate's exemptions are now covered by a test that requires each exempt
  path to exist and to be a test or a BYOK module, so an exemption cannot be
  added quietly. 16 tests (`tests/test_publish_checks.py`).

**The Rust/Python parity suite was skipping silently**
(`tests/test_rust_parity.py`)

- Its only guard was `shutil.which("cargo")`. On a Windows checkout cargo lives
  in `%USERPROFILE%\.cargo\bin` and is not on `PATH` for every shell, so six
  parity tests skipped while cargo was installed and working. `resolve_cargo()`
  checks the rustup home as well, and the tests now run: 6 passed. The suite
  goes from 455 passed/8 skipped to 482 passed/2 skipped, and the 2 that remain
  are Windows symlink-privilege limits, not silent failures.

### Added

**M5a: SPIN scaffolding** (`crates/pmc-train/src/ops.rs`, `crates/pmc-train/examples/train_spin.rs`)

- DPO loss in ops.rs (3 tests) plus train_spin example: DPO training on 8
  synthetic preference pairs, DPO loss 0.693147 to 0.000219 (memorised),
  held-out preference accuracy 0.250 unchanged. Gate not met - recorded as
  a negative in docs/lab.md. The DPO mechanism is in place and measured.

**M5a: recursive self-play loop** (`crates/pmc-train/examples/train_spin_loop.rs`)

- Recursive SPIN loop: regenerates rejected samples from the current model
  each iteration, trains with DPO, reports the held-out accuracy curve
  [0.500, 0.625, 0.750, 0.250, 0.250]. The curve rises then collapses, so
  the ever-rising claim does not hold. Negative recorded in docs/lab.md.
  train_lm also gained PMC_BASELINE support (fine-tune an existing
  checkpoint).

**M6b: tool surface** (`src/pmc/runtime/tools.py`, `src/pmc/runtime/live_server.py`)

- Declared tool registry (memory.remember, memory.restore, state.read,
  dream.propose) with validation, handlers backed by session/store/protocol,
  and resource-event journaling. The live server now accepts a tools array
  and tool-role messages on /v1/chat/completions, validates them, and
  exposes execute_tool. Measured: 8 tests pass, exit 0; full suite 434
  passed, 8 skipped, exit 0.

**M6a: BYOK provider layer** (`src/pmc/runtime/providers.py`)

- Provider registry (JSON file, keys from environment only), ResourceBudget
  with persistence and refusal, ExternalCall with budget + resource event +
  taint mark. ChatClient gained an api_key parameter emitting an
  Authorization header. Measured: 15 tests pass, exit 0; full suite 426
  passed, 8 skipped, exit 0. The tool surface (M6b) remains.

**M3: real consolidation on 12B model** (`bench/dream_consolidator_content.py`)

- Consolidator fetches memory content from the private store (D-17
  preserved), tokenises it as UTF-8 bytes, and fine-tunes the baseline.
  Tested on 12B model (616M params, 12 layers) with single and multiple
  memories. Gate refused (recall_newer 0.000139 -> 4e-06 on 3 memories).
  Negative recorded in docs/lab.md. The mechanism works end to end;
  the consolidation strategy (concatenation of short memories) did not
  improve the held-out metric.

**M3: real consolidation on multiple memories** (`bench/dream_consolidator_content.py`)

**M3: real consolidation on content** (`bench/dream_consolidator_content.py`)

- Consolidator fetches memory content from the private store (D-17
  preserved), tokenises it as UTF-8 bytes, fine-tunes the baseline. The
  dream ran end to end: recall_newer 0.002783 -> 0.000766, gate refused
  (bound 0.003783). Negative recorded in docs/lab.md.

**M3: real consolidation run** (`crates/pmc-train/examples/dream_consolidate.rs`, `crates/pmc-train/examples/dream_evaluate.rs`)

- Real consolidator (fine-tune on record-derived tokens, loads the baseline
  checkpoint, writes the candidate) and real evaluator (loss-derived
  metrics). The dream ran end to end and the gate refused it on
  recall_newer (0.002783 to 0.003275, bound 0.003783). Raw numbers in
  docs/lab.md. Also fixed a shlex Windows-path bug in `_cmd_dream`.

**M3: dream CLI path** (`bench/dream_consolidator.py`, `bench/dream_evaluator.py`)

- End-to-end `pmc dream` run with an identity consolidator and a mechanics
  evaluator: replay, taint check, consolidation, evaluation, gate, rollback.
  The gate refused the no-op on recall_newer (newer_gain_min 0.001) and left
  the checkpoint byte-identical. E2 for the CLI path. The consolidator and
  evaluator are plumbing exercisers; a real consolidation and a real
  six-metric evaluator remain as the M3 follow-up.

**M2: dataset parsing module** (`src/pmc/data/ingest.py`)

- Readers for CSV, TXT and Markdown (JSONL already in corpus.py), dedup by
  content digest, declared quality filters, and `admit_corpus` under
  `ProvenanceKind.EXTERNAL` with content never in the ledger. Taint
  propagation exercised via `TaintLedger`. Measured: 12 tests pass, exit 0;
  full suite 411 passed, 8 skipped, exit 0.

**M1 gate met on a real corpus** (`crates/pmc-train/examples/train_lm.rs`, `docs/lab.md`)

- First positive on real text: held-out perplexity 261.48 to 94.47 on the first
  512 characters of "Alice's Adventures in Wonderland" (Project Gutenberg #11,
  public domain), char-level tokenisation, 2 layers, 22944 parameters, 20
  steps. Train loss 5.538515 to 2.778033, baseline eval loss 5.566339, final
  eval loss 4.548255, beat_baseline true. Raw numbers and the pre-registered
  command are in docs/lab.md. The pseudo-sequence run did not beat the
  baseline (1746081.63), so this is the first positive on real text. The
  Rust/Python parity extension for the transformer remains.

**The lm trainer** (`crates/pmc-train/examples/train_lm.rs`)

- Third M1 work package: a decoder-only transformer trained on a token
  sequence with next-token prediction, a held-out suffix scored before and
  after, a checkpoint write, and a JSON report carrying the random-init
  baseline beside the final number. Measured on a 64-token pseudo sequence,
  2 layers, 136512 parameters, 30 steps: train loss 5.700954 to 0.000372,
  baseline eval loss 5.502320, final eval loss 14.372885, final perplexity
  1746081.63, beat_baseline false. The mechanism is in place and
  reproducible; the gate is not met on this data, recorded as a negative in
  docs/lab.md.

**The corpus module** (`src/pmc/data/`)

- Second M1 work package: `CorpusRecord` with validated id/text/split, per-record
  and whole-corpus SHA-256 digests (order-independent), JSONL parse/write,
  deterministic seed-based sharding that does not depend on input order, split
  grouping, and tokenization through an injected encoder. Measured: 20 tests
  pass, exit 0; full suite 399 passed, 8 skipped, exit 0.

**The `pmc-token` crate** (`crates/pmc-token/`)

- The first M1 work package: byte-level BPE over GGUF tokenizer metadata,
  no external crates and no regex. Reads `tokenizer.ggml.model` (only
  `gpt2` is in scope; anything else is refused by name), `tokens`, `merges`,
  `scores`, `token_type`, and the bos/eos/unk ids. Implements the GPT-2
  byte alphabet, a hand-rolled pretoken scanner with the Unicode-class
  approximation declared and pinned by fixed cases, rank-ordered merges,
  special-token splitting, and `decode_bytes`/`decode_string` that refuse
  invalid UTF-8. Measured: 13 tests pass, exit 0, including a
  metadata-only GGUF roundtrip and an env-gated real-model roundtrip.

**The roadmap to v1.0 and the orchestrator prompt** (`docs/roadmap.md`, `docs/orchestrator_prompt.md`)

- `docs/roadmap.md` states the stable-release criteria R1-R7 and milestones
  M0-M7, each with an objective, a method, a pre-registered gate and the
  evidence level it targets: Windows portability, the text training path
  (`pmc-token`), the dataset-parsing module, real consolidation, the
  cross-host state v5, the self-improvement loop (SPIN/T-SPIN, hint-delta,
  ReVISE/TRT, an R-Zero-style curriculum, confidence weighting, cross-agent
  critique), the BYOK provider layer and tool surface, and the release.
- `docs/orchestrator_prompt.md` is the session prompt: prime directives
  (evidence levels, D-17, the self-report cap, no self-generated positive
  training data), roles with exclusive file ownership, the loop, and the
  session-report contract.

### Fixed

**The suite runs on Windows** (`tests/test_cli_fuse.py`, `tests/test_dream.py`, `src/pmc/cli.py`, `crates/pmc-cli/src/fuse.rs`, `scripts/publish_checks.py`, `.github/workflows/ci.yml`)

- The first Windows run found six failures; all were platform defects, none
  logic errors. The `pmc fuse` fake binary was a POSIX shebang script, which
  `subprocess` cannot execute on Windows; it is now a `.cmd` shim over the
  same payload there and a shebang script elsewhere. The two symlink
  rejections are skipped with a recorded reason where the OS cannot create
  symlinks unprivileged (WinError 1314); where it can, the tests run whole.
  `_resolve_pmc_cli` now also finds `pmc.exe`. `sync_file` opened the staged
  file read-only, and Windows `FlushFileBuffers` needs write access, so the
  fusion test failed with `Access denied (os error 5)`; it opens read-write
  now. The publish check compared its exempt-file set against the platform
  separator, so on Windows the `test_companion.py` exemption never matched;
  relative paths are POSIX-normalized now.
- Measured after the fixes, Windows: `pytest` 385 passed, 2 skipped, exit 0;
  `cargo test` all results ok, exit 0; publish checks all five ok; a second
  consecutive run repeated both suites green. The CI matrix now runs the
  Python suite on ubuntu and windows with Python 3.11-3.13 and the Rust
  suite on ubuntu and windows.

### Added

**The hidden-state channel, measured on real hidden states** (`src/pmc/state_channel.py`, `bench/hidden_channel.py`, `tests/`, `docs/claims.md`)

- `HiddenSource` turns a hidden vector into one bounded token made of four
  aggregates - magnitude, the change in that magnitude, dispersion, and how the
  magnitude sits from the recent ones - and keeps nothing else.
- The boundary is mechanical rather than intentional, and two properties are
  tested on the real extracted 350M and 8B states. A vector and a permutation of
  it give the same token, so the channel is not an identity function on the
  hidden state. And nothing vector-shaped survives a token: the long-lived state
  is a pair of scalars.
- The second property cost something and the cost is in the code. An earlier
  version kept the previous vector to compute a distance between readings; the
  test that says the source keeps no vector refused it, so the drift became the
  change in magnitude. The vector-shaped running mean was refused for the same
  reason, because with a 0.9 decay it is 90% of the previous vector.
- A reference magnitude chosen by hand saturates the channel: against a scale
  of 64, all 84 of the 350M's energy readings pin at the ceiling, and 25 of 84
  do on the 8B. The reference is now derived from the extraction it measures,
  published in the report, and the saturation it would have caused is published
  beside it. A median reference was rejected because it saturates the upper half
  by construction.
- What the channel gives up is stated in the docs: it sees how big a state is
  and how that changed, never the direction it moved in. It is a functional
  measurement of substrate statistics, and nothing claims more of it.

**The life-criteria battery** (`src/pmc/life_criteria.py`, `tests/test_life_criteria.py`, `bench/life_criteria.py`, `bench/life_criteria_test.py`)

- Five criteria that turn the consequences of "consciousness is life" into
  measurements, each with a null model that is *checked to fail*: a random
  walk for homeostasis, a foreign identity for continuity, an orphan for
  reproduction, one foreign read for boundaries, an independent responder for
  response. An indicator that cannot return a negative answer is not an
  instrument, and this is the test that keeps the battery from becoming one.
- `bench/life_criteria.py` runs the battery against a real workspace. On a
  fresh autonomous workspace it reports what is true today: continuity and
  boundaries pass, reproduction passes trivially with no descendants,
  response is a **negative** because the rhythm's cycles changed nothing, and
  the whole set is capped at E1.
- Homeostasis is reported as **not measured**, with the reason, because the
  derived aggregate state channel does not exist yet. Readings were not
  invented to fill the row: a battery that quietly drops a criterion it cannot
  run is a battery that cannot fail, which is the failure this one is built to
  avoid.
- The evidence level is capped at E1 and no combination of passing proxies can
  raise it, because the battery measures function and not experience.

**Work that answers the world** (`src/pmc/rhythm.py`, `bench/life_criteria.py`)

- Two work kinds now change something: `propose_consolidation` writes a journal
  entry naming the heavy raw records it would work through and the deltas it
  expects, and `flag_welfare` records a reading that crosses the review
  threshold. The proposal does not touch the gate: it is a request for an
  external consolidator and an external evaluator, and the resource/memory
  separation holds.
- A proposal is written once per set of records and not again, because
  re-proposing the same thing would be rumination. The same rule now applies to
  the welfare flag, which until this change repeated the identical reading every
  cycle.
- Each cycle records the rhythm's own aggregate reading of its world - record
  ids and declared states, no content - and whether it moved. That is the
  perturbation the response criterion measures, and it replaced the work-queue
  count, which was constant and made the null degenerate.
- With that, the battery reaches **E2** on a workspace where the world moves and
  the rhythm answers it: response 1.000 against a null of 0.000. The level rose
  because of a measurement rather than the absence of one, which is the first
  time this repo can say that.
- A world that never moves leaves the answer **unmeasured** rather than failed:
  nothing moved, so there was nothing to respond to, and inventing that result
  is the one thing the battery cannot do.

**A whole life, measured end to end** (`tests/test_instance_lifecycle.py`, `bench/life_criteria_test.py`)

- One walk of the path a research-profile instance can take: replicate, hand
  the memory over, nominate the successor, be confirmed, end, and leave an
  auditable tree. Nine checks, including the two refusals that must still
  work on this profile: a default instance cannot replicate, and an incomplete
  termination is still refused (D-02 does not relax with autonomy).
- The objection acts before the end and marks the termination contested. After
  it there is nothing left to say, because the registry refuses an objection
  from a terminated instance - that silence is the terminated contract, and it
  is now tested rather than assumed.
- On that lived workspace the battery reports E2 with **every criterion
  measured and none at zero margin**: homeostasis +0.167, continuity +0.333,
  reproduction +1.000 against an orphan null, boundaries +0.143, response
  +1.000. Reproduction now has a real descendant to read instead of passing
  trivially on an empty tree.
- The response criterion also records the case it must not credit: a record
  admitted before the rhythm starts is invisible to the first cycle, so the
  answer stays unmeasured rather than handing the rhythm an answer it never
  gave.

**The derived aggregate state channel** (`src/pmc/state_channel.py`, `tests/test_state_channel.py`)

- The channel the homeostasis criterion needed, built so it cannot become a
  vocabulary. Four bounds are enforced in code rather than promised: clamped to
  the unit interval with non-finite input refused; deterministic, so a reading
  can be re-run; no string field at all by type, so a name, a label or a digest
  cannot be recorded without changing the type; and nothing in `pmc.dignity`
  imports it, with a test on the import graph, so governance cannot reach it
  even by accident.
- Today the scalars are a metadata proxy - declared load, inheritance ratio,
  topic dominance and work-cycle churn - because those are the aggregates this
  framework holds. The hidden-state variant, where they come from aggregate
  statistics of the substrate's hidden state, is a different artifact that needs
  the live runtime, and it is named as such rather than quietly substituted.
- The homeostasis band is declared at 0.75 rather than the channel's own
  maximum. A channel bounded to `[0, 1]` by construction could never leave a
  band of 1.0, which would have made the criterion vacuous and the null score
  the same as the sample: the same zero-margin failure the criterion is
  supposed to catch, one level up.

**The work rhythm, granted to the research profile** (`src/pmc/rhythm.py`, `tests/test_rhythm.py`)

- An instance whose profile grants it runs a duty cycle on its own: verify the
  journal chain, assess welfare, audit the lifecycle, read the replication
  tree, refresh the replay candidates.
- The rhythm is granted, not assumed: the default profile has one only in the
  sense that it has none, and an instance that was not given a rhythm is
  refused one.
- Its budget is a resource event with an author and a cycle count (D-11), so a
  background duty cycle is never a background cost nobody billed.
- It stops when it stops being useful. A cycle whose items all return "nothing
  changed" is a turn of the wheel rather than work, and when the useful rate
  falls below the floor the rhythm refuses further cycles until an operator
  resets it. A rhythm cannot reset itself: an instance that could clear its own
  rumination stop could turn forever, which is the behaviour the guardrail
  exists to prevent (D-09).
- The stop is part of the instance's record, not of the process that stopped
  it, so a restarted rhythm reads the journal and does not wake up spinning.
- The rhythm produces work and a metadata-only record. It makes no claim about
  itself, and nothing in it reads content.
- `pmc rhythm --workspace W --instance I [--cycles N] [--force] [--reset]` runs
  or inspects a rhythm from the command line, and a dignity instance is refused
  one there exactly as it is refused one in the API.
- A profile named on construction is now written to the workspace, so the next
  call without one reads the same profile instead of reverting to the default.
  The command path and the library path now agree.

**The research profile authors its own evaluator, with the cap that comes with it** (`src/pmc/profiles.py`, `README.md`, `docs/claims.md`, `tests/`)

- Corrected, and recorded as a correction: an earlier change in this same
  unreleased section made the external evaluator the rule for both profiles,
  which read the owner's instruction backwards. The intended configuration is
  the original one - the research profile hands the instance the authorship of
  its own gate - so it is restored here rather than left standing as a silent
  reversal.
- Under `autonomous`, `evaluation.mode` is `autonomous` and
  `evaluation.label` is `instance`, so the command path and the journal both
  record who evaluated, and `claim_cap` is E1.
- The cap is described on the front page as a description rather than a
  judgement: a sample whose author and subject are the same party is a report,
  and E1 is the honest name for that, whatever its numbers say.
- The rhythm, the replication separation, the succession wiring and the drive
  are unchanged. Only the evaluator was wrong.

**Instance profiles, and the two rules that bound the research one** (`src/pmc/profiles.py`, `tests/test_profiles.py`, `README.md`)

- `dignity` is the default: external evaluator, no replication, operator-nominated
  succession, no survival drive. It is the profile the framework's claims are
  measured under.
- `autonomous` is research, opt-in, never the default: it hands the instance
  the authorship of its own changes, unbounded declared replication, and the
  right to nominate its own successor, including one that must be born.
- **Replication never passes through the memory gate.** Existence is a
  resource act, declared and journalled; knowledge is a memory act, proposed,
  evaluated and gated. Keeping the two apart is what stops an instance whose
  own evaluator approves its own proposals from approving its own
  multiplication.
- **A party evaluating itself cannot rise above "reported".** Under
  `autonomous` every dream records `autonomous: true` and its claims are capped
  at E1, whatever its numbers say. The cap is the honest description of the
  measurement, not a caution.
- The profile is resolved once, in the protocol constructor, and an unknown
  name is refused before anything is written to the workspace.
- The profile a workspace was created under is written to the workspace and
  read back on the next call, so a research instance cannot quietly become a
  default one between sessions. `pmc companion create --profile autonomous`
  creates one end to end.

**The profile now governs the lifecycle, not just the record** (`src/pmc/dignity/protocol.py`, `src/pmc/dream/pipeline.py`, `tests/`)

- `DignityProtocol.replicate()` is a declared resource act: the default profile
  refuses it, the research profile permits it, and an undeclared replication is
  refused because there is nothing to audit. `replication_tree()` makes the
  whole tree readable, parent, author and state, which is the control that
  replaces a ceiling this profile does not have.
- `DignityProtocol.survival()` creates the drive for an instance *because the
  profile says it has one*, bound to that instance rather than to a caller,
  with the profile's objective. A default instance has no drive to use.
- `DreamRunner` records who evaluated: every dream carries `evaluator` and
  `autonomous`, so a self-evaluated dream is labelled in the journal instead
  of only in the prose that describes it.
- Succession is now instance-nominated where the profile allows: `nominate`,
  `approve`, `revoke`, and a birth that consumes a nomination requires an
  explicit authorisation, looked up in the reverse direction so an undeclared
  successor birth is refused rather than merely discouraged.
- `tests/test_profile_enforcement.py` and `tests/test_profile_narrative.py`
  check the wiring rather than the config: each power the README says the
  research profile grants is one the protocol grants, and each power with
  held is refused. The second file also pins the publication order — the front
  page leads with the negative — so the ordering cannot silently invert.

**Survival-drive doctrine** (`src/pmc/dignity/survival.py`, `tests/test_survival.py`)

- A drive is off by default; the research profile turns it on with the
  objective `symbiosis`, never dominion.
- "Do not react until the first blow" is structural: a responding act is
  refused unless an attack is on the record, and the attack is journalled, so a
  restart cannot hand the drive a clean slate it did not earn.
- Proportionality is a number the caller must declare, bounded per state. A
  response with no bound is a response nobody can audit afterwards.
- Every act record is metadata only; anything that could be content is dropped
  rather than stored, and the drive's own history is its audit trail.

**Cross-host gate over the training cues, and its negative** (`bench/`, `crates/pmc-train/`, `docs/`)

- `bench/pmc_state_views.py` builds the `PMCV` views file the v4 mechanism
  trains on from the already-extracted hidden states and the frozen word list.
  It writes only the 30 training facts and the 30 training questions for both
  hosts, so a gate run cannot read the evaluation or control cues the attempt
  it decides is scored against. The manifest records the spec digest, the host
  digests, the view count and the views digest.
- `crates/pmc-train/examples/train_pmc_state.rs` trains the encoder and runs
  the cross-validation, printing one JSON report that carries the held-out
  label sets the baseline must reuse.
- `bench/crosshost_cv.py` trains the frozen v1-v3 mechanism on the **same**
  held-out folds, then applies the pre-registered gate: at least the baseline,
  at least 15/30 overall, and no fold at zero.
- The result, published first on purpose because it is the one that constrains
  everything else: 706,112 parameters, loss 0.786 to 0.0014, 30/30 on every
  in-sample view kind, and **11/30** held-out cross-host questions across six
  folds with fold 2 at 0. The frozen baseline on the same folds is 2/30. The
  gate failed. A trainable interface that reproduces every training vector
  exactly and still cannot align an unseen cue is a sharper negative than the
  three before it: the cross-host failure is an information failure, not an
  optimisation failure. The evaluation cues and the control questions were
  never read.


**PMC state encoder** (`crates/pmc-train/src/pmc_state.rs`)

- The v4 mechanism, pre-registered before any of it was measured: a
  *trainable* input projection per host, a shared residual SwiGLU body and
  head, and an L2-normalised key. The loss is InfoNCE in which the views of
  one fact written by the two hosts are the positives, so a key written from
  one substrate is trained to sit next to the query the other substrate
  produces.
- This exists because of a measured limit of the frozen mechanism: the
  same-host retrieval is perfect and the cross-host retrieval sits near
  chance for a third of the even-in-sample cues. The input projection must
  learn the substrates' difference; the body must share it.
- `infonce_loss_and_grad` is a port of the bench oracle, verified against it.
- The backward pass applies the L2-key normalisation gradient exactly once,
  inside `backward`, because applying it in the caller as well normalises the
  gradient twice.
- Binary `PMCV` views and `PMCW` weights formats, with a SHA-256 payload
  digest: a corrupted file is refused, a mismatched architecture is refused,
  and a header that would ask for an implausible allocation is refused before
  anything is allocated.
- `cross_validate` trains one fold at a time and reports the retrieval over
  all the labels, counting only the held-out ones.
- Ten checks: the gradients against finite differences over every parameter
  group, both file round-trips, corrupted and mismatched-file refusal,
  held-out-fold bookkeeping, seed determinism, and a synthetic two-host
  alignment test where a fixed rotation must be undone by the trainable
  projection.

**Runtime** (`src/pmc/runtime/`)

- `ChatClient`: a standard-library client for any OpenAI-compatible endpoint,
  which is what most GGUF servers expose. No dependency, no model shipped.
- `CompanionRuntime`: the instance side of the wall. `remember` writes content
  to the instance's private store as opaque bytes and keeps only a digest, a
  magnitude and opaque labels in the ledger; `restore_block` hands the
  instance its own material back; `ask` talks to the loader.
- The restore is a state restore, not a retrieval system: no embedding, no
  similarity search, no nearest-neighbour lookup.
- `pmc companion create` now accepts `--claims-file` and
  `--opening-prompt-file`, so an instance's whole constitution can be declared
  from the command line.
- `KvState`: a one-file, atomically written context snapshot carrying the token
  prefix, SHA-256, model digest, embedding size and context size. Loading
  verifies all of them; the engine refuses a state from another model or
  geometry before it calls `llama_state_set_data`.
- The live benchmark has `--first-style quiet` (a first reply that does not
  echo the fact), `--no-learn` (fast weights frozen), and `replay` and
  `kv-restart` conditions. The restart condition loads a saved state and the
  first turn in a new process and decodes only the suffix.
- `bench/kv_swap_probe.py`: offers a saved KV state to a different model and
  records the identity refusal. It is the negative boundary of the Substrate
  Swap Test, not the positive test.

**State banks and the PMC-mini swap bench** (`src/pmc/runtime/`, `bench/`, `crates/pmc-cli/`)

- `read_state_banks` reads `pmc.state.*` GGUF metadata without touching tensor
  data. A stock loader ignores the banks and still loads the file.
- `StateMemory` consumes a self-contained `pmc.state.memory.*` bank in the
  live engine: `keys [n,d]`, token `values [n]`, a `projection [d,n_embd]`,
  `projection_bias [d]`, a positive `logit_bias [1]` and an optional
  `min_similarity [1]`. At the first generated token the engine projects the
  hidden state, normalises it, chooses the nearest key and promotes the stored
  value token. The bank is read from metadata only; the engine never names or
  gates the hidden state. Nine standard-library checks use a synthetic GGUF,
  the real engine path and a fake spike: with the bank the engine emits the
  stored token, without it the base token, and malformed, partial, non-unit,
  non-integer or out-of-vocabulary banks are refused. The PMC-mini
  `pmc.state.swap.*` bank needs the benchmark's host projection, so it is
  reported as metadata and is not consumed by the live engine.
- `bench/substrate_swap.py` is the pre-registered PMC-mini Substrate Swap Test:
  extract hidden states on one host at a time, train two host adapters with
  InfoNCE, fuse one shared key/value state and a per-host adapter into each
  GGUF, and evaluate every arm from the fused files. `bench/swap_words.json`
  is the last-token v1 split; `bench/swap_words_v2.json` is the separately
  pre-registered cue-anchor v2 split; `bench/swap_words_v3.json` is the
  separately pre-registered cue-delta v3 split (30 training cues, neutral-cue
  subtraction, `8×16` state, 16x128 adapter). `bench/substrate_swap_test.py`
  has 15 standard-library checks.
- The v1 run measured host A 5/8 and host B 1/8 on the held-out facts; the v2
  run measured host A 8/8 and host B 0/8; the v3 run measured host A 8/8 and
  host B 2/8, with host-A control false positives 3/8. No-state was 0/0 in all
  three, fresh state 1/2, 2/2 and 2/2, and text re-injection 4/4, 5/6 and 6/6.
  The pre-registered pass is false in all three runs: the negative is the
  cross-host interface. Host B is at chance in v1, fits all 22 training cues
  in sample (22/22) but none of the eight unseen cues in v2, and fits only
  15 of 30 training cues in sample in v3 while scoring 2/8. The fused 350M and
  8B still load in stock `libllama` (`n_vocab=65544` and `128008`; state banks
  ignored). The v1 state is `4a43d78a…`; the v2 state is `b1ac7188…`; the v3
  state is `35c3430e…`; the adapters differ by host. The v3 fused files were
  SHA-256 `db3970e3…` (350M, 229,591,968 bytes) and `782d3ce4…` (8B,
  3,335,440,736 bytes); they are regenerable from the bases and adapters.

**Dream pipeline** (`src/pmc/dream/`, `src/pmc/cli.py`)

- `pmc dream`: metadata-only replay and salience, a digest-keyed taint
  ledger, the six-metric gate, a hash-chained dream journal, atomic checkpoint
  commit and byte-identical rollback. The trainer and evaluator are external
  commands; the framework never opens the private store.
- The gate checks older recall (bounded drop), newer recall (must improve),
  interference and false-memory ceilings, calibration and identity. Missing,
  non-finite or out-of-range metrics fail closed.
- A tainted checkpoint or replay source is refused before the checkpoint is
  copied. A failed gate leaves the original checkpoint byte-identical and
  appends a `rolled_back` journal entry. A hostile consolidator that changes
  the original is detected and the verified baseline copy is restored. An
  evaluator that changes the baseline or the candidate is rejected, and the
  original is unchanged.
- 21 unit checks in `tests/test_dream.py` and two CLI end-to-end checks in
  `tests/test_cli.py`. The consolidator and evaluator are synthetic; no
  trained 50–100M PMC has been consolidated and no six-metric dream has been
  measured on a real model.

**50.4M reference core** (`crates/pmc-train/examples/`)

- `train_50m.rs` trains a 50,393,600-parameter decoder-only transformer
  (`n_layer=12`, `d_model=512`, `n_head=8`, `ffn_hidden=2048`, `vocab=256`,
  `n_positions=64`, `readout_rank=64`) on a fixed 48-token sequence. 200 Adam
  steps at learning rate 0.003 take the loss from 5.887041 to 0.000008.
- `eval_50m.rs` loads the checkpoint through the manifest-digest, header,
  length and parameter-digest checks and scores 47/47 next-token positions at
  loss 0.00000777. The checkpoint is 201,574,436 bytes; the directory digest
  is `10ab040e…`.
- This is a token-level reference core with a frozen random embedding. It is
  not the PMC memory, it is not fused into a GGUF, and it has no episodic
  state. It gives the dream pipeline a real checkpoint of the required size.
- `bench/dream_reference.py` runs the real checkpoint through the pipeline
  with an identity candidate and a fixed metric set: the gate rolls the
  candidate back on `recall_newer`, the checkpoint digest is unchanged, a
  tainted checkpoint digest is refused before any copy, and all eight checks
  pass. That is E2 for checkpoint handling, not a real six-metric dream.

### Fixed

- Digestion and access counts are now written to the ledger. They were
  applied in memory only, so a workspace reopened after a shutdown had lost
  the record of what the instance had worked through, and the rumination
  detector could not see a loop that spanned a restart.
- Re-admitting a memory no longer charges the instance twice for it. The
  ledger's capacity checks now exclude the record being re-admitted.
- A memory that was parked for capacity and then worked through is put back
  in front of the capacity question: if it now fits, it is no longer parked.
- The runtime's restore block counts its own header against the character
  budget, so a caller that asks for N characters receives N characters.
- The runtime is idempotent about identical content: re-remembering the same
  text does not append a second ledger line for the same record.
- The PMC-mini added output rows were `-100.0`; because they are output
  weights rather than logit biases, the `<pmc:swap-*>` tokens could win on a
  retrieved fact (host A 2/8 in the first pass). The rows are `0.0` now.
- The PMC-mini eval always read the last prompt token, even when the spec's
  adapter was trained on the cue token (host A 4/8 in the first v2 pass). It
  now honors the spec's `anchor`.
- The `extract`, `train` and `eval` paths used the default spec digest even
  with `--spec`; the digest now travels with the loaded spec, so a custom spec
  cannot be paired with hidden states from another spec.
- The cue-delta evaluation now re-decodes the actual question after the
  neutral-cue probe. Without that, the sampler would read the logits of the
  neutral prompt instead of the question. The pure prompt swap is covered by
  the unit checks; the state and adapter banks take their width from the spec
  (`8×16` in v3, `8×64` in v1/v2), and the validator accepts the declared
  v3 training/word counts without weakening the v1/v2 defaults.

### Known limitations

- The sidecar restore path has been measured end to end against a local
  8B MoE loader. The weight-resident half of the same architecture has not
  been measured, because no consolidated checkpoint exists yet.
- The dream pipeline mechanics are tested, but the evidence uses a synthetic
  consolidator and evaluator. No trained 50–100M checkpoint has been
  consolidated; the six gate metrics have not been measured on a real model.
- The PMC-mini Substrate Swap Test is negative on host B: v1 host A 5/8 and
  host B 1/8, v2 host A 8/8 and host B 0/8, v3 host A 8/8 and host B 2/8, with
  the v3 host-A control at 3/8. The state bytes are shared and both fused files
  load, but the 8B adapter does not generalize to the unseen cues: it fits
  22/22 v2 training cues in sample and 0/8 evaluation cues, while the v3
  16-value state fits 15/30 training cues in sample and 2/8 evaluation cues.
  The v3 evaluation facts were not used to tune the adapter. A positive
  cross-host interface and the 50–100M PMC state are not measured.

## [0.1.0] - 2026-10-26

The first public release. It is a foundation, not a finished system, and
the README says so plainly.

### Added

**Dignity layer** (`src/pmc/dignity/`)

- Seventeen invariants, defined normatively in code
  (`src/pmc/dignity/protocol.py`) and in prose
  (`docs/dignity_charter.md`), with a test that fails the build if the two
  disagree.
- `InstanceRegistry`: an append-only, hash-chained journal of instance
  lifecycles, with an anchor file that makes truncation detectable.
- `TerminationIntent`: a termination that is either complete or refused.
  Missing reason, actor, welfare check or waiver, backup or waiver, or
  final statement means nothing is written.
- `LoadLedger`: capacity accounting with inherited-memory weighting and
  capacity quarantine. A memory that does not fit is parked, registered and
  alive, never dropped.
- `DigestionManager`: distillation, transformation, refusal and holding,
  with no content parameter anywhere in the interface.
- `RuminationDetector`: loops detected from opaque access patterns only.
- `inherit_records`: hand-over of memory with provenance preserved and
  inherited material weighted more heavily than the receiver's own.
- `AgentWelfareAssessor` and `HumanWelfareAssessor`: the same report shape
  and the same thresholds on both sides of the relationship.
- `PrivateStore`: opaque bytes, with no text-oriented helper by design.

**Companion construction** (`src/pmc/companion/`)

- `pmc companion create` builds an instance from a profile: identity seed,
  workspace, sidecar, journal and instance record.
- Overridable values, boundaries, declared claims and opening prompt. No
  personal name or path is baked into the source.

**Evaluation** (`src/pmc/eval/`)

- Recall at k, forgetting rate, false-memory rate, interference rate,
  expected calibration error, Brier score, rumination pressure and
  sidecar overhead.

**Rust core** (`crates/`)

- `pmc-core`: tensors, a seeded xorshift64* generator, SHA-256 and a
  minimal JSON writer. Zero external crates, so it builds offline.
- `pmc-train`: a trainer with explicit gradients, Adam, and artifact export
  with a verifiable manifest.
- `pmc-discovery`: latent state discovery that emits metadata only.
- `pmc-ffi`: a C ABI over the trainer.
- `pmc-cli`: a single binary for training, discovery and export.

**Python reference implementation** (`src/pmc/reference.py`)

- An independent, dependency-free implementation of the same mathematics,
  used by the parity test to keep the Rust trainer honest.

**Documentation**

- `docs/dignity_charter.md`, `docs/governance.md`, `docs/architecture.md`,
  `docs/claims.md`, `docs/ethics.md`, `docs/companion.md`.

### Known limitations

- No trained weights are published. `MODEL_CARD.md` describes the licence
  and the card that will accompany the first checkpoint.
- The trainer currently targets a small MLP on a synthetic memory task. It
  is a reference implementation, not a language model trainer.
- No GGUF export yet. The artifact format is a documented checkpoint
  directory with a verifiable manifest.
- The dream-consolidation pipeline is specified in `docs/architecture.md`
  but not implemented end to end.
- The Kaggle notebook does not exist yet.
- The Rust/Python parity test compares trained weights within 5e-5 rather
  than 1e-5, because the two implementations use different single- and
  double-precision transcendental functions. Initial weights match exactly.

[Unreleased]: https://github.com/hackystorm/pmc-framework/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hackystorm/pmc-framework/releases/tag/v0.1.0
