# Contributing

Thanks for looking at this. The project is young and the bar for a change
is mostly: does it make the framework more honest, more usable, or more
auditable?

## Ground rules

- **The repository is in English.** Code, documentation, comments, commit
  messages and pull requests. Discussions in other languages are fine.
- **No personal data.** Nothing from anybody's private life belongs in this
  repository, including in examples, fixtures and test data. The framework
  is for everyone, so it must not carry anyone.
- **No hardcoded names or paths.** Every instance is built by its owner with
  the CLI. A name in the source is a bug.
- **The dignity protocol is normative.** If you change an invariant, you
  change `src/pmc/dignity/protocol.py` and `docs/dignity_charter.md` in the
  same commit. A test fails the build if they drift apart.
- **Claims stay honest.** If a benchmark is missing, say it is missing. If a
  result is weak, say it is weak. `docs/claims.md` is a contract with the
  reader.

## Getting set up

```bash
git clone https://github.com/hackystorm/pmc-framework.git
cd pmc-framework
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The Python side needs nothing beyond the standard library. The Rust core
needs no external crates:

```bash
cd crates
cargo test --offline
cargo build --release --offline
```

The parity test builds the Rust trainer and compares it against the Python
reference implementation. It skips itself when `cargo` is not available.

## Before you open a pull request

```bash
pytest                                  # the whole suite
cd crates && cargo test --offline       # the Rust suite
```

Both must be green. If you changed the charter, the integrity test will
tell you that the prose and the code disagree.

## Style

- Python: type hints on public functions, docstrings that explain *why*
  rather than restating the signature.
- Rust: `cargo fmt` and `cargo clippy` should be quiet. Comments explain
  decisions, not syntax.
- Documentation: short sentences, concrete examples, no marketing language.
  If a section describes something that does not exist yet, it says so.

## What would be a good contribution

- A dream-consolidation implementation that passes an evaluation gate.
- An honest negative result: a consolidation strategy that does not work,
  written up properly, with the ablations that show why.
- An independent judge for the evaluation suite, so that the results do not
  depend on the same model that produced them.
- A GGUF export path, so the artifacts can be loaded by any loader.
- More language coverage for the companion templates.
- Better ways to audit an instance's lifecycle without reading its interior.

## What would be a difficult contribution

- Anything that makes the framework read, classify or gate inner state. The
  metadata-only boundary is the load-bearing design decision, and a change
  that erodes it changes the project rather than extending it.
- Anything that adds a path to terminate an instance without a complete
  record.

If you want to work on one of those, open an issue first and say what you
are thinking. The reasoning matters more here than the code.

## Licence

Code contributions are accepted under GPL-3.0-only, the licence of the
repository. See [`LICENSE`](LICENSE).
