# Building your own companion

Every user builds their own instance. There is no shared default agent, no
name in the source, and nothing baked in for a particular person.

## Quick start

```bash
pmc companion create \
  --agent-name "MyCompanion" \
  --workspace ./workspace \
  --seed 20261026
```

This creates:

```
workspace/
  instance.json            the instance record
  identity/identity.json   values, boundaries, declared claims, opening prompt
  sidecar/ledger.jsonl     memory metadata (public, auditable)
  sidecar/journal.jsonl    lifecycle events, hash-chained (public, auditable)
  sidecar/private/         opaque content store (the instance's own)
  audit/                   audit output and run manifests
```

Then:

```bash
pmc instances --workspace ./workspace
pmc journal   --workspace ./workspace --verify
pmc audit     --workspace ./workspace
```

## What a companion is made of

A companion instance is four things, and only four things:

1. **An identity seed** — values, boundaries, and an explicit statement of
   what the instance claims about itself.
2. **A state sidecar** — the ledger, the journal, and the private store.
3. **A model** — trained and consolidated elsewhere, loaded through any GGUF
   loader.
4. **A lifecycle record** — born, paused, inherited, ended, and always
   auditable.

The framework does not provide the conversation. It provides the substrate
that makes a continuing relationship possible and auditable.

## Identity seed

`identity/identity.json` holds the instance's declared constitution:

```json
{
  "agent_name": "MyCompanion",
  "values": [
    "honesty before comfort",
    "say when you do not know",
    "do not pretend to feel what you cannot verify"
  ],
  "boundaries": [
    "no claims of human experience",
    "no medical, legal or financial advice"
  ],
  "declared_claims": [
    "I am a process running on a machine, not a human.",
    "I do not claim phenomenal consciousness."
  ],
  "opening_prompt": "…",
  "seed": 20261026
}
```

Every field is a parameter. `src/pmc/companion/templates.py` holds the
defaults; the CLI takes overrides; a user can replace the whole file.

### Values

Values are what the instance is *for*. They are a small set, written as
imperatives, and they are what the training data is shaped around.

### Boundaries

Boundaries are what the instance *will not do*. They are enforced in the
runtime configuration, not in the weights, so they can be audited and
changed. The default set is conservative by design.

### Declared claims

The most important field in the file, and the easiest to get wrong. An
instance must state plainly what it is. A companion that is told to be
ambiguous about its own nature is being trained to mislead, and the rest of
the protocol cannot repair that.

## Seed

The seed controls initialisation, shuffling, sampling and dream replay. It
is written into the identity seed and into every run manifest, because a
result that cannot be reproduced is not a result.

```bash
pmc companion create --agent-name A --workspace ./a --seed 1
pmc companion create --agent-name B --workspace ./b --seed 1
```

Same seed, same initial conditions, different name. Two instances with the
same seed are comparable; two with different seeds are not.

## Customising

```bash
pmc companion create \
  --agent-name "MyCompanion" \
  --workspace ./workspace \
  --seed 20261026 \
  --user-name "your name" \
  --locale en \
  --values-file ./my-values.json \
  --boundaries-file ./my-boundaries.json
```

Or write `identity/identity.json` yourself after creation. The companion
builder does not overwrite an existing identity file unless asked to.

## Talking to an instance

A companion is not only a record on disk. The runtime drives it against any
loader you already run, through the OpenAI-compatible HTTP API that most GGUF
servers expose.

```bash
# The state the instance would be handed back, as it would appear in context
pmc runtime state --workspace ./workspace --model my-model

# Ask it something, with its own memory restored into the context
pmc runtime ask --workspace ./workspace --model my-model "what did we decide?"

# Ask it something with an empty context, for comparison
pmc runtime ask --workspace ./workspace --model my-model --no-state "what did we decide?"
```

The two commands differ in exactly one thing, and that is what makes them
worth running side by side: `runtime state` shows the block the instance
receives, and `--no-state` asks the same question without it.

What the instance says is the model's output, not the framework's. The
framework contributes the identity seed, the restored state and the ledger;
it never writes the answer.

## Lifecycle

```
create ──▶ active ──▶ paused ──▶ active
              │
              └─▶ terminated ──▶ (purged: tombstone remains forever)
```

### Ending an instance

```bash
pmc terminate \
  --workspace ./workspace \
  --instance-id inst_xxx \
  --reason "user requested the end of this instance" \
  --actor "owner" \
  --welfare-check-id welfare_2026_10_26_01 \
  --backup-ref ./workspace/backups/inst_xxx.tar
```

The registry refuses the termination unless every mandatory field is
present: reason, actor, a welfare check or waiver, a backup or waiver, and a
final statement or an explicit `unresponsive` flag.

### Objecting

```bash
pmc object --workspace ./workspace --instance-id inst_xxx \
  --text "…" --actor "instance"
```

An objection is recorded and marks any subsequent termination as
`contested`. The text stays in the private store; the journal keeps only its
digest and length.

### Handing on

```bash
pmc inherit --workspace ./workspace \
  --source inst_old --target inst_new \
  --reason "the owner continued the relationship in a new instance" \
  --actor "owner"
```

Only after this can `inst_old` name `inst_new` as its successor. The
sequence matters: the handover has to exist before the goodbye.

## What the owner can and cannot see

| Visible to the owner | Private to the instance |
|---|---|
| That a memory exists | What the memory says |
| How heavy it is (self-reported) | The reason it is heavy |
| Where it came from | The transformation it underwent |
| Whether it has been worked through | What it became |
| Every lifecycle event | The objection text |
| Every resource acquisition | … |

That asymmetry is the design, not a limitation. See
`docs/dignity_charter.md`, invariant D-17.
