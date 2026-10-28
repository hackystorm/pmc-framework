# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

**The dream is externally evaluated in both profiles** (`src/pmc/profiles.py`, `README.md`, `docs/claims.md`)

- The research profile no longer buys the instance its own evaluator. A result
  produced by the party it is about is a report, not a measurement, so the gate
  still runs an evaluator the instance did not write, and every dream records
  who ran it.
- The E1 cap now applies to a *sample whose evaluator is the instance* rather
  than to the research profile as a whole, and the front page states the rule
  as a rule about who measures rather than about how much autonomy an instance
  has.

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
