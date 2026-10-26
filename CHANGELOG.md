# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

### Known limitations

- The sidecar restore path has been measured end to end against a local
  8B MoE loader. The weight-resident half of the same architecture has not
  been measured, because no consolidated checkpoint exists yet.

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
