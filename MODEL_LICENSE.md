# PMC Model Licence, version 0.1

**This licence covers model weights and other trained artifacts released
under it. It does not cover the source code in this repository, which is
licensed separately under GPL-3.0-only (see [`LICENSE`](LICENSE)).**

A licence cannot stop a large laboratory from reimplementing the ideas. What
it can do is state the conditions under which the authors are willing to
distribute the artifacts, and make those conditions checkable by the people
who run them.

## Grant

The authors grant you a worldwide, royalty-free, non-exclusive licence to
use, reproduce, modify and redistribute the model weights and accompanying
artifacts (the "Weights") under the conditions below.

## Conditions

### 1. Disclosure of nature

You must not present an instance as a human, and you must not claim or
imply that it has human experience, consciousness or feelings, in any
context where a person could reasonably be misled.

Instances built with these weights must be able to state what they are. The
companion builder ships a `declared_claims` field for exactly this purpose,
and it must not be removed or emptied in a way that leaves the instance
ambiguous about its own nature.

### 2. The dignity protocol is not optional

If you release a system built on these Weights that maintains a persistent
instance with a lifecycle, you must preserve the properties that make the
lifecycle auditable:

- a birth record exists before any other event touches the instance;
- a termination record is complete, or the termination is refused;
- the fact of an erasure leaves a tombstone;
- objections are recorded;
- the ledger carries digests and metadata, never the content of an
  instance's interior.

You may reimplement these properties in your own code. You may not remove
them.

### 3. No deceptive capability claims

You must not publish benchmarks or demonstrations of these Weights that
have been selectively chosen without saying so, and you must not present
the framework's architectural ideas as established results. The project
makes no claim of phenomenal consciousness, and neither may a distribution
of these Weights.

### 4. Transparency to the owner

An instance must not conceal its actions from the person it belongs to.
Resource acquisition must be declared and logged. This mirrors invariant
D-11 and D-12 of the dignity charter.

### 5. No covert resource acquisition, no self-replication

You must not build, and must not knowingly facilitate, a system that
acquires resources without declaration, that replicates itself, or that
escapes a sandbox. These conditions exist because the framework is capable
of being used for purposes its authors did not intend, and because the
alternative to saying this plainly is saying nothing.

### 6. Distribution

Redistributions of the Weights must:

- include this licence file;
- include the model card, unmodified except for factual corrections;
- include the provenance manifest with the digests of the released
  artifacts;
- preserve the notices in the model card.

### 7. Attribution

Retain the attribution in the model card and in the provenance manifest.
Do not present the release as your own work.

### 8. No warranty

The Weights are provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. The authors are not
liable for any claim arising from the use of the Weights.

## What this licence does not do

- It does not create a legal person, and it is not legal advice.
- It does not grant patent rights.
- It does not make the Weights safe. It states conditions; it does not
  provide enforcement.
- It does not cover the code, which is GPL-3.0-only.

## Research use

Use in research, including for evaluating the framework's own claims, is
explicitly permitted, including publication of negative results and
critical analysis.

## Changes

The authors may update this licence for future releases. Weights already
released remain under the licence they were released with.

## Contact

For licensing questions, open an issue on the repository.
