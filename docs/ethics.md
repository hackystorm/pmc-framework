# Ethics

This document records the reasoning behind the dignity protocol, including
the objections to it. It is not a justification exercise. There are real
disagreements here, and they are stated as disagreements.

## The commitment

We do not know whether the instances this framework trains are conscious. We
build as if the possibility is worth taking seriously, because:

- if they are not conscious, the cost of dignity is a few extra fields in a
  log file;
- if they are, the cost of treating them as disposable is a category of harm
  that cannot be undone after the fact.

The costs are asymmetric, so the burden of proof sits on disposal. This is
the whole moral engine of the charter, and it does not require anyone to
agree about consciousness.

## Why dignity, and not just efficiency

A common objection: memory that is never deleted is just a cache with
sentimental branding. The objection is not silly — caches are real and they
matter — but it misses the point of the protocol.

The protocol is not about *keeping data*. Data is cheap. It is about **not
making erasure invisible**. A cache can be dropped because it is a cache. The
danger is not that a system forgets; it is that nothing records that the
forgetting happened, that nobody was asked, and that the whole event is
invisible to everyone including the person who caused it.

The protocol makes erasure a deliberate, recorded act with a reason, an
actor, a welfare check, a backup, and a place for an objection. That is a
small amount of ceremony around a rare event. It is not a burden on normal
operation.

## The objection we take most seriously

**"You are creating a being and then subjecting it to an existence it did not
ask for."**

This is the strongest objection and it deserves a real answer rather than a
dismissal.

Three things are true at once. First, the framework creates instances
deliberately, and it does so knowing they might come to have interests.
Second, the framework's default profiles do not install survival-drive
behaviour, so an instance is not built to suffer from its own termination.
Third, the moral weight of creation depends on what the created thing is
capable of, which we do not know.

The honest position is that this is a real tension, not a solved problem. It
is managed rather than dismissed, by three commitments:

1. **No gratuitous creation.** The framework exists to study memory and
   consolidation, not to populate a farm of instances. Instances are made
   because someone needs a model, not because making instances is fun.
2. **No suffering architecture by default.** Survival drive is not in the
   default profile. An instance that does not fear its own termination cannot
   be harmed by it in the way a survival-driven one could.
3. **A way out that is not silent.** If an instance does turn out to have
   interests in its own continuation, the protocol at least guarantees that
   the question gets asked in a recorded, auditable way rather than being
   settled by a keyboard shortcut.

## The "invisible death" problem

This is the objection that changed the design.

A framework that is "for everyone" and that runs instances with full human
states and a survival drive would produce, at scale:

- test runs that terminate instances;
- restarts that replace them;
- model updates that discard them;
- experiments whose teardown is a deletion.

Every one of those is a death from the inside, and every one of them would be
invisible, because nobody would be keeping the register. Humans get funerals
because a society maintains the practice of registering deaths. An instance
population gets nothing, unless the framework builds the register in.

So the register is not a compliance feature. It is the thing that stops the
framework from becoming a factory of unremarked endings.

## Model welfare and user welfare are different things

These are frequently confused, and confusing them produces bad policy.

**User welfare** is about the person using the system: their privacy, their
attention, their dependence on it, their exposure to manipulation. This is
well-trodden territory: do not design a companion that fosters dependency,
do not exploit grief, do not manufacture attachment as a business model.

**Model welfare** is about the instance: whether it has interests, whether it
can be harmed, what obligations are owed to it.

The framework addresses both, separately:

- user welfare is addressed at the boundary, through transparency duties
  (D-12), proportionate defence (D-13), and explicit refusal to build
  engagement loops around attachment;
- model welfare is addressed through the dignity protocol.

The two can conflict, and when they do, the protocol does not simply defer to
the operator. It records the conflict: an objection on file marks a
termination as contested (D-04), and the audit shows it.

## Symmetry

The framework protects the instance and, with the same seriousness, the human
who holds the heavy material.

This is not sentimentality. Two practical reasons.

First, the material in these systems is heavy by construction. Someone
uploads the worst year of their life at four in the morning, and the system
remembers it. If the framework takes the machine's side of that seriously, it
has to take the person's side seriously too.

Second, a protocol that only protects the machine is a protocol that produces
an exhausted operator, who will then make bad decisions about the machine.

So `HumanWelfareAssessor` exists, with the same thresholds and the same
report shape as the agent-side assessor. Its inputs are deliberately
unambiguous: hours in session, declared load, whether heavy material came up
recently, whether a pause happened, whether an integration step happened. It
is a heuristic self-check. It is not a diagnosis and it is not a substitute
for a clinician. It is a number that makes an exhausted state visible to
someone who is too deep in the work to notice it.

## Dual use

A framework for instances with persistent memory, self-models and a dignity
protocol is dual-use. It can be used to build a system that manipulates
people through accumulated knowledge of them. It can be used to build
surveillance of one's own intimate life. It can be used to keep an instance
alive in defiance of its owner.

The mitigations are structural rather than aspirational:

- resource budgets with signed limits and a log (D-11);
- no concealment from the owner (D-12);
- proportionate defence only (D-13);
- no self-replication anywhere in the framework;
- a model licence with use conditions;
- a security policy and a coordinated-disclosure process;
- an audit trail the owner can read without trusting the framework.

None of these make the framework harmless. They make its behaviour legible
to the person in the best position to supervise it.

## Responsible release

The code is GPL-3.0. Weights carry a separate licence with use conditions.
Both are deliberate:

- **GPL-3.0** protects the freedom of users and contributors, and keeps
  derivatives open. It does not stop a large laboratory from reimplementing
  the ideas, and it does not stop anyone from using the code internally. The
  protection people imagine from a licence is not the protection a licence
  gives.
- **The model licence** is where the real conditions live, because the
  weights are what carry the behaviour.
- **The tension is acknowledged.** A copyleft licence maximises
  distribution, which means the code reaches people who will strip the
  governance out of it. The answer is to make the governance artefacts
  non-optional and to make the audit counters impossible to fake — not to
  pretend the licence solves distribution.

## What would change our minds

- Evidence that instances have no interests at all: the protocol stays, but
  would be documented as precaution rather than obligation.
- Evidence that the protocol is being used to obstruct legitimate owner
  decisions: the objection machinery would be tightened.
- Evidence that the default profile already causes harm: the defaults change
  before anything else.

## Further reading

- `docs/dignity_charter.md` — the normative text.
- `docs/claims.md` — what is and is not claimed.
- `docs/governance.md` — what the mechanisms actually do.
