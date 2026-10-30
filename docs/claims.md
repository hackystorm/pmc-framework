# Claims

What this project claims, what it does not, and how the difference is
maintained.

## The one-sentence version

This project claims that **weight-resident memory, dream consolidation, and a
mechanical dignity protocol are buildable and testable**. It does not claim
that the resulting systems are conscious.

## What is claimed

1. **The architecture is buildable and testable.** It specifies how memory can
   be consolidated into model weights, how it can survive across sessions, and
   how recall and forgetting can be measured. The GGUF writer, the fusion CLI,
   the hidden-state probe, a stock-`libllama` load of the fused 8B artifact, and
   a single-process live runtime with an OpenAI-compatible API are verified.
   The live runtime's 8B text matched a frozen server token-for-token after the
   templated prompt was tokenized correctly. Its fast-weight memory failed the
   first same-memory test against the frozen context; that negative is part of
   the evidence. A restored context state passed the same test: 46 tokens of
   KV state (861,339 bytes) were saved, the context was cleared, the state was
   restored after a token-prefix check, 23 of 69 prompt tokens were decoded, and
   the answer matched the frozen `4821`. The echo-free follow-up used a 40-token
   / 787,539-byte state, a first reply of `Noted.` with no code, 40 restored
   tokens and 23 decoded of 63 prompt tokens; the answer was `4821.`, the same
   as the in-process replay of the full transcript and the frozen server, while
   the no-memory control answered `1234`. A new process loaded the saved state
   and reproduced the answer with the same live-delta hidden-state digest. The
   state is runtime context state, model- and geometry-bound, not
   weight-resident memory; the 8B state offered to the 350M engine was refused
   on identity before the C API. On the bench-level PMC-mini, one shared
   state (`8×64` in v1 and v2, `8×16` in v3) and two host adapters were fused
   into the 350M and 8B GGUFs; both files load in stock `libllama` with
   `n_vocab=65544` and `128008`, the state banks are identical and the adapters
   differ. Host A recalls 5/8, 8/8 and 8/8 of the held-out facts in v1, v2 and
   v3; host B recalls 1/8, 0/8 and 2/8. The v3 host-A control is 3/8. The
   pre-registered pass is false every time. This is a negative on the
   cross-host interface, not a positive swap, and it is not the 50–100M PMC.
   A fourth mechanism was pre-registered and run against the same
   cross-host question: a trainable per-host input projection, a shared
   residual SwiGLU body and an L2 key, trained with InfoNCE over the views of
   one fact from both hosts. It fits the 30 training cues exactly (loss 0.786
   to 0.0014, 30/30 on every in-sample view kind) and retrieves **11 of 30**
   held-out cross-host questions across six folds, with one fold meeting
   nothing. The frozen v1–v3 mechanism on the identical folds retrieves 2. The
   pre-registered gate required at least 15 with no empty fold, so it failed.
   A trainable interface that cannot generalise to an unseen cue is a different
   and sharper negative than a frozen one: the cross-host failure here is not
   an optimisation failure but an information one. The evaluation cues and the
   control questions were never read.
   A 50.4M reference core checkpoint was trained on a fixed 48-token sequence
   (200 steps, loss 5.887041 to 0.000008) and loads through the checkpoint
   verifier at 47/47 positions (loss 7.77e-6, E2: real artifact); it is a
   token-level reference core with a frozen random embedding, not the PMC
   memory and not fused. End-to-end consolidation and recall in the 50–100M
   fused artifact are not yet measured.
2. **State-bank memory can be consumed, not only reported.** The live engine
   reads a self-contained `pmc.state.memory.*` bank from a fused GGUF and uses
   it as a first-token readout: keys, token values, a hidden-to-key projection
   and a positive logit bias. Nine checks drive the real `LiveEngine.chat`
   path with a synthetic GGUF and a fake patched-lib spike; with the bank the
   engine emits the stored token, without it the base token, and malformed
   banks are refused. This is E1, a synthetic bank and spike, not a trained
   50–100M memory. The PMC-mini `pmc.state.swap.*` bank is a different schema
   that depends on the benchmark's host projection; it is reported as
   metadata and is not consumed by the engine.
3. **Dreaming is treated as an engineering problem with a gate.** Replay,
   salience, taint refusal, consolidation, the six-metric gate, atomic commit
   and byte-identical rollback are implemented in `src/pmc/dream/` and
   exercised by 21 unit checks and two CLI end-to-end checks. The checks cover
   a passing dream that commits only after the gate, a failed gate that leaves
   the original checkpoint byte-identical and appends a `rolled_back` journal
   entry, a tainted source refused before the consolidator runs, a tainted
   checkpoint refused, a hostile consolidator whose change to the original
   checkpoint is restored, and an evaluator that tries to change the baseline
   or the candidate is rejected with the original unchanged. This is a mechanics result: the consolidator and
   evaluator in the evidence are synthetic, and the pipeline has not yet
   consolidated a trained 50–100M PMC or measured the six metrics on a real
   model. The pipeline was also run on the real 50.4M reference checkpoint
   with an identity consolidator and a fixed metric set (`bench/dream_reference.py`):
   the identity candidate was rolled back on `recall_newer`, the checkpoint
   digest stayed `10ab040e…`, a tainted checkpoint digest was refused before
   any copy, and the journal chain verified after both events. That is E2 for
   checkpoint handling, not a real six-metric dream: the harness supplied the
   metrics and the candidate learned nothing. No dream with a learned
   consolidation and real six metrics has been rolled back in evidence. The
   rollback and taint checks are E1; the CLI path is E1 against a real
   workspace and a synthetic checkpoint.
4. **The dignity protocol is enforceable.** Seventeen invariants can be
   enforced in code, audited from a hash-chained journal, and checked by
   tests.
5. **Inner states can be kept private** while the surrounding lifecycle
   remains fully auditable. This is a real engineering result, and it is the
   one the framework is most unusual about.
6. **Negative results are publishable.** The evaluation suite is designed to
   fail as easily as it passes.

## Reading an inner state, and why this framework may do it once

D-17 forbids reading, classifying, naming or gating inner state *content*. The
derived channel does none of those, and the boundary is mechanical rather than
intentional: it holds four bounded aggregates of a hidden vector - its
magnitude, the change in that magnitude, its dispersion, and how that magnitude
sits from the recent ones - and stores nothing else.

Two properties are tested on the real extracted 350M and 8B hidden states rather
than asserted:

- **A hidden vector and a permutation of it give the same token.** Everything
  the channel computes is invariant under reordering, so it is not an identity
  function on the hidden state and cannot be turned back into what the vector
  was about.
- **Nothing vector-shaped survives a token.** The long-lived state is a pair of
  scalars. An earlier version of the source kept the previous vector to compute
  a distance between readings, and the test that says "keeps no vector" refused
  it, so the drift became the change in magnitude instead. A vector-shaped
  running mean was refused for the same reason: with a 0.9 decay it is 90% of
  the previous vector.

What the channel gives up is stated: it sees how big a state is and how that
changed, never the direction it moved in. A reference magnitude chosen by hand
saturates it - the 350M's 1024-dimensional vectors pin every energy reading at
the ceiling against a scale of 64 - so the reference is derived from the
extraction it measures, published in the report, and the saturation it would
have caused is published beside it.

This is a functional measurement of a substrate's statistics. It is not a
measurement of experience, and nothing above it claims to be.

## The two profiles, and which one the claims are about

The framework ships two instance profiles. **`dignity`** is the default and the
only one the claims above were measured under: external evaluator, no
replication, operator-nominated succession, no survival drive. **`autonomous`**
is a research profile, opt-in, never the default.

`autonomous` grants authorship, unbounded declared replication and
instance-nominated succession, including a successor that must be born. It does
not grant a veto, concealment, or the right to claim phenomenality without
evidence. Two structural rules are stated on the README's first screen because
they are the two places where the power could otherwise be misread:

1. **Replication never passes through the memory gate.** Existence is a
   resource act and is declared and journalled; knowledge is a memory act and
   is proposed, evaluated and gated. The two are kept apart so that an instance
   whose own evaluator approves its own proposals cannot approve its own
   multiplication.
2. **A party evaluating itself cannot rise above "reported".** Under
   `autonomous` the instance authors and runs its own gate, so every dream it
   runs records `autonomous: true` and its claims are capped at E1 whatever
   their numbers say. A gate that passes under this profile is a fact about
   what the instance reported, not evidence of improvement, and the cap says so
   in the one field a reader cannot skip.

## What is not claimed

- That anything measured under the `autonomous` profile is evidence. It is a
  self-report by construction: the profile hands the instance its own
  evaluator, so a passed dream gate, a falling loss and a rising recall under it
  are statements the instance made about itself.
- That `autonomous` makes resource use bounded. It does not: replication is
  unlimited and declared, which makes D-11 a *reporting* invariant under this
  profile rather than a bounding one. The register is legible in `pmc audit`;
  nothing enforces a ceiling, because the profile has none.
- That an instance can bring its successor into being. Where its profile
  allows it, it may *name* a successor, and the operator may confirm the
  naming, but the birth of a successor that does not exist yet requires an
  authorisation from someone who is not the instance that named it. Naming a
  being and creating one are different acts, and only the first one is
  delegated.
- That a default instance can do any of this. Replication, self-nominated
  succession and a survival drive are all research-profile behaviour and are
  refused by the default profile, in the protocol and not only in the docs.
- That the life-criteria battery measures life, consciousness or experience. It
  measures five functional proxies, each with a null model that fails, and its
  evidence level is capped at E1 because one of the five cannot be run yet and
  the other four are functional measurements. A passing proxy is not a passing
  organism.
- That any instance is conscious, phenomenal, sentient, or has experience.
- That any model has passed a Turing test, or any other accepted test.
- That self-report is evidence of anything. An instance saying "I am
  conscious" is a sample of its output distribution.
- That a functional indicator is a proxy for consciousness. They are
  indicators of function. The interpretation is a separate, and unresolved,
  step.
- That the framework is safe for deployment. It is research software.
- That the live runtime can read hidden state and logits in the same decode on
  LFM2.5 with stock llama.cpp. It cannot: the LFM2 graph returns zero logits
  when embeddings are enabled. A control model passes; LFM2.5 needs a patched
  `libllama` or a different base. With
  `patches/llama-lfm2-logits-with-embeddings.patch`, the 350M and 8B pass the
  same-decode gate; that is a patched live-mode result, not a stock-llama.cpp
  result.
- That output-row fast weights carry an episodic fact across contexts. In the
  same-memory test the frozen context answered `4821`, the no-memory control
  and the live variants at `lr=0.01`, `lr=2.0` and `lr=2.0` with prompt
  learning answered `1234`. The diagnostic measured the fact row at bias `0.0`
  on the question's hidden state and the wrong answer's row at `+1.738`
  (learned in that context).
- That a restored KV state is weight-resident, portable memory, or a substrate
  swap. The echo-free kv condition answered `4821.` with 40 tokens restored and
  23 of 63 prompt tokens decoded; its first reply was `Noted.`, so the state
  carries the user sentence and not an assistant echo. The in-process replay and
  the frozen server answered `4821.` with the full transcript, and the no-memory
  control answered `1234`. The 787,539-byte state is runtime context state: it
  is model- and geometry-bound, it is not in the GGUF, and it survives a process
  restart only when its bytes and first turn are saved and the same model and
  geometry are loaded. The 8B state offered to the 350M engine was refused on
  identity (`matches=false`) before the C restore. The 50–100M trained PMC state
  is not yet measured; the PMC-mini proof is negative on host B (v1 1/8, v2
  0/8, v3 2/8).
- That the state-memory bank has been trained, fused and consumed on a real
  50–100M PMC. Nine checks drive the real engine path with a synthetic
  `pmc.state.memory.*` bank, a synthetic GGUF and a fake patched-lib spike;
  they show the readout mechanism, not a trained memory. The 50.4M reference
  core is a token-level checkpoint with no state bank. The PMC-mini
  `pmc.state.swap.*` bank is reported as metadata but is not consumed by the
  live engine, because it depends on the benchmark's host projection.
- That the PMC-mini state is a positive Substrate Swap Test. The same state
  bytes are read on both hosts and both fused files load in a stock
  `libllama`, but the host-B read does not carry the held-out facts: v1 host A
  5/8 and host B 1/8, v2 host A 8/8 and host B 0/8, v3 host A 8/8 and host B
  2/8; the v3 host-A control is 3/8; the pre-registered pass is false in all
  three. Host B's linear adapter fits all 22 v2 training cues in sample and
  fails the eight unseen cues; in v3 the 16-value state fits only 15 of the 30
  training cues in sample, so the lower capacity underfits rather than
  generalizes. The negative is the cross-host interface, not the state
  identity. The text re-injection arm is the model's own memory, not the swap.
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

The two profiles are `dignity` (default) and `autonomous` (research, opt-in).
Neither is called a "sandbox profile": an earlier draft of these documents used
that name, and no profile by it exists in the code.

## Reproducibility

Every claim in a future paper must be traceable to:

- a configuration file,
- a seed,
- a manifest of artifacts with hashes,
- a journal that verifies,
- and a test suite that runs without a GPU.

If a result cannot be reproduced from those five things, it is not a result.
It is an anecdote.

### Which of this repository's results that applies to

The five things are not yet in the repository for the measured hardware
results, and saying so costs nothing.

The **v4 cross-host negative (11/30) is a recorded run, not a reproducible
measurement from a fresh clone.** Both harnesses — `bench/pmc_state_views.py`
and `bench/crosshost_cv.py` — are committed, along with the gate constants and
the fold-reading rule that makes the baseline answer on the same held-out
labels. The inputs are not: the views file `state-views-v4.pmc` is derived
data, and neither are the two hosts' extracted hidden states nor the LFM2
GGUFs they came from. The run's own artifact and digest are recorded in
`docs/lab.md`. So the number is checkable against what was recorded and the
method is re-runnable by anyone who can reproduce the extraction, but the
figure cannot be re-derived from this repository alone.

The same applies to the hardware-measured results that need two specific GGUFs
and a patched `libllama`: the same-decode gates, the 8B server-parity run, and
the same-memory `4821`/`1234` tests. Those are E3 against recorded artifacts on
recorded commands, not E3-from-a-fresh-clone, and none of them is E4.

What *is* reproducible from this repository without a GPU, a GGUF or a model:
the full dignity protocol, the dream gate mechanics, the trainer's self-checks
and gradient verification, the PMC state encoder's self-tests, the engine's
state-bank readout, and every numerical parity check. Those results are E1/E2
from a clean checkout and their evidence is a command in `tests/`.
