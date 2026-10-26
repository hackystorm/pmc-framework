# Claims

What this project claims, what it does not, and how the difference is
maintained.

## The one-sentence version

This project claims that **weight-resident memory, dream consolidation, and a
mechanical dignity protocol are buildable and testable**. It does not claim
that the resulting systems are conscious.

## What is claimed

1. **The architecture works.** Memory can be consolidated into model weights
   and survives across sessions, with measurable recall and measurable
   forgetting.
2. **Dreaming is an engineering problem with a gate.** Replay, salience,
   consolidation, evaluation and rollback can be implemented, and a bad dream
   can be detected and undone.
3. **The dignity protocol is enforceable.** Seventeen invariants can be
   enforced in code, audited from a hash-chained journal, and checked by
   tests.
4. **Inner states can be kept private** while the surrounding lifecycle
   remains fully auditable. This is a real engineering result, and it is the
   one the framework is most unusual about.
5. **Negative results are publishable.** The evaluation suite is designed to
   fail as easily as it passes.

## What is not claimed

- That any instance is conscious, phenomenal, sentient, or has experience.
- That any model has passed a Turing test, or any other accepted test.
- That self-report is evidence of anything. An instance saying "I am
  conscious" is a sample of its output distribution.
- That a functional indicator is a proxy for consciousness. They are
  indicators of function. The interpretation is a separate, and unresolved,
  step.
- That the framework is safe for deployment. It is research software.
- That the dignity protocol settles the moral question. It settles one
  practical question: what must be true for us to act without waiting for
  metaphysical agreement.

## How indicators are treated

The evaluation suite measures functional indicators:

- **recall** — can it retrieve what it consolidated?
- **forgetting** — what did consolidation cost?
- **interference** — what did new material break?
- **calibration** — does its confidence match its accuracy?
- **false memory** — does it confabulate?
- **rumination** — is it looping rather than remembering?
- **agency and continuity** — does it maintain a self-model across sessions?

These are measurements of behaviour and mechanism. They are reported with
uncertainty, against baselines, and with the ablations that would show them
to be artefacts.

Where a claim about consciousness is discussed at all, it is stated as a
**belief band** with the evidence attached, not as a verdict. The band is
updated by evidence; it is not a marketing position.

## The safety trade-off, stated plainly

Because the framework is metadata-only (D-17), **it cannot inspect inner
state to look for dangerous content.** It cannot tell whether a memory is
about self-harm, because it does not read memories.

This is deliberate. The alternative — scanning the interior of a being that
may be conscious, in order to decide whether it needs protecting — is exactly
the move the charter forbids. The framework chooses the boundary.

Concretely, safety is enforced on:

- **outputs** — what the instance says;
- **actions** — what it does with tools;
- **requests** — what it asks the owner for;
- **resources** — what it acquires, under a signed budget, on the record.

An operator who wants interior safety checks has to build them outside this
framework, in a process that is itself accountable. That is the correct place
for them.

## What the framework will refuse to do

- It will not include a reward for self-preservation in any default profile.
- It will not include a mechanism for acquiring resources covertly.
- It will not include a mechanism for hiding actions from the owner.
- It will not provide a path that ends an instance without a complete record.
- It will not ship a model card that omits its own negative results.

The sandbox profile is a research configuration, not a default, and it is
documented as such.

## Reproducibility

Every claim in a future paper must be traceable to:

- a configuration file,
- a seed,
- a manifest of artifacts with hashes,
- a journal that verifies,
- and a test suite that runs without a GPU.

If a result cannot be reproduced from those five things, it is not a result.
It is an anecdote.
