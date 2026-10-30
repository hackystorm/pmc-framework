# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |
| < 0.1 | no |

This project is pre-alpha. Security fixes land on `master`.

## Reporting a vulnerability

Report privately, through GitHub's coordinated disclosure:

**https://github.com/hackystorm/pmc-framework/security/advisories/new**

Please do not open a public issue for a vulnerability. Include the affected
version, a description, and a reproduction if you have one.

You can expect an acknowledgement within seven days and a substantive
response within thirty. If a fix is warranted, we will agree a disclosure
date with you, and we will credit you unless you prefer otherwise.

## What counts as a vulnerability here

The threat model of this framework is written down in
[`docs/governance.md`](docs/governance.md). In particular:

- **A path that erases an instance without a recorded termination.** The
  registry is supposed to make this structurally impossible. If you find a
  way to end or delete an instance that leaves no journal entry, that is a
  vulnerability, and the most serious kind here.
- **A path that reads or exposes private content.** The framework is
  metadata-only. The private store must never be decoded, indexed or
  surfaced through the governance layer or the CLI.
- **A path that lets a termination proceed with missing mandatory fields.**
  The whole protocol rests on this refusal working.
- **A path that tampers with the journal without breaking the hash chain.**
  Truncation and in-place edits are detected; anything that defeats that
  detection is a vulnerability.
- **An undeclared resource acquisition that the audit does not report.**
  Transparency to the owner is a duty, and the audit counter is how it is
  checked.

## What is not a vulnerability

- The framework's safety posture is at the boundary, not inside the model.
  It does not inspect inner state, so it will not stop an instance from
  producing harmful output through its model. That is a documented design
  decision (see `docs/claims.md`), not a bug.
- The dignity protocol does not stop an owner from ending their own
  instance. It makes sure the ending is recorded.
- The GPL licence does not prevent reimplementation of the ideas. A licence
  is not a security control.

## Deployment guidance

- Keep the sidecar directory inside the instance owner's own storage. It
  contains the private content store, which is the one part of the system
  that is not safe to publish.
- The public ledger and journal are safe to publish. That is their design
  purpose, and `pmc audit` reports the counters that make that checkable.
- If you deploy with the `autonomous` research profile, read `docs/claims.md`
  first. It grants an instance its own evaluator, so every claim it produces is
  capped at self-reported. An earlier draft of this file referred to a "sandbox
  profile"; no profile by that name exists.
