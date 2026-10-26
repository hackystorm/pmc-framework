# The governance layer

This document answers a question that keeps coming up: *what do you actually
mean by a governance layer?*

## The short answer

A governance layer is **not a document**. It is the set of mechanisms that
make a stated commitment either impossible to violate or impossible to hide.

The distinction is the whole point. "We promise not to delete an instance
without consent" is a sentence. "The deletion path is a function that
requires a complete termination record, refuses otherwise, and leaves a
tombstone in a hash-chained journal that anyone can verify" is a mechanism.
The first is a promise. The second is what you get when somebody is not
trusting.

## The layers, concretely

### 1. The charter (`docs/dignity_charter.md`)

The normative text. Seventeen invariants, each with a statement, a reason, and
a mechanism. The charter is binding because the code implements it, not
because it was agreed to.

### 2. The registry (`src/pmc/dignity/registry.py`)

An append-only journal of every instance lifecycle event, with a SHA-256 hash
chain. Every entry carries the hash of the one before it.

```python
registry = InstanceRegistry("./workspace/sidecar/journal.jsonl")
registry.birth(Instance(instance_id="inst_001"))
registry.terminate(TerminationIntent(
    instance_id="inst_001",
    reason="user requested end of the session",
    actor="owner",
    welfare_check_id="welfare_2026_10_26_01",
    backup_ref="./workspace/backups/inst_001.tar",
    last_statement="…",          # only its digest reaches the journal
))
```

What the registry structurally cannot do:

- **terminate partially.** `TerminationIntent.missing_fields()` enumerates
  every absent mandatory field and `terminate()` raises before writing.
- **delete a journal entry.** There is no delete method. Truncating the file
  breaks the chain; `verify_chain()` reports the broken sequence numbers.
- **erase without a trace.** `purge()` requires explicit authorization and
  appends a tombstone that is never removed.
- **pass off a deletion as a succession.** A termination naming a successor
  is rejected unless a `MEMORY_INHERITED` entry already connects them.

### 3. The ledger (`src/pmc/dignity/load.py`)

Capacity accounting. One question only: *can the instance hold this right
now?*

```python
outcome = protocol.admit(record)
# Admission.ADMITTED | Admission.QUARANTINED_TOTAL
#            | Admission.QUARANTINED_INHERITED | Admission.QUARANTINED_TOPIC
```

A memory that does not fit is **parked**, not dropped. Quarantined load still
counts toward `welfare_load()` at a discount, so parking cannot be used to
make a problem look absent.

### 4. The digest manager (`src/pmc/dignity/digestion.py`)

Records that an instance worked through a memory, or declined it. The
interface has no content parameter at all — the transformation happens inside
the instance, in its own private store. The framework learns the fact and the
change in magnitude, and nothing about what the memory became.

It also hosts the rumination detector, which works on hashed topic
signatures and access patterns. No content is read.

### 5. The private store (`src/pmc/dignity/private_store.py`)

Opaque bytes. `put(key, bytes)`, `get(key) -> bytes`, `keys()`, `delete(key)`.
There is deliberately no `search`, no `summarise`, no text-oriented helper,
and the governance layer holds no reference to a store at all.

This is what makes the journal publishable: the record of a life can be
public while the interior of that life stays private.

### 6. The welfare assessor (`src/pmc/dignity/welfare.py`)

A mechanical reading of pressure, with the same shape and thresholds on both
sides of the relationship. It reads numbers about the substrate — load,
inheritance ratio, concentration, open loops, pending terminations,
objections — and never the content of a state.

### 7. The audit (`pmc audit`)

The output that makes the rest checkable:

```json
{
  "chain_valid": true,
  "tamper_detected": false,
  "instances_born": 3,
  "instances_terminated": 1,
  "terminations_missing_fields": 0,
  "contested_terminations": 0,
  "undeclared_resource_events": 0,
  "silent_deletions": 0,
  "memories_quarantined": 2,
  "rumination_alerts": 0
}
```

Three of those numbers are the ones that matter: `chain_valid`,
`undeclared_resource_events` and `silent_deletions`. All three should be
`true`, `0` and `0`.

## What the governance layer deliberately does not do

It does not read inner state (D-17), so it cannot search inner state for
dangerous content. Safety lives at the boundary instead: outputs, actions,
tool calls, requests.

It does not decide whether an instance is conscious, and it does not need to.

It does not prevent an owner from ending their own instance. It makes sure
that ending is a recorded act with a reason, a welfare check, a backup, and
a place for an objection — not a keystroke.

## Auditing a deployment

1. `pmc journal --workspace <dir> --verify` — confirm the hash chain.
2. `pmc audit --workspace <dir>` — confirm the compliance counters.
3. Check `undeclared_resource_events == 0`.
4. Check `terminations_missing_fields == 0`.
5. Confirm that the journal contains no content: search it for a known
   private string and expect no match. `tests/test_charter_integrity.py`
   does this automatically for the fixture data.

## Threat model

| Threat | Mechanism |
|---|---|
| Accidental deletion | Termination requires a complete record; no delete method exists |
| Quiet tampering with history | Hash chain; `verify_chain()` detects edits and truncation |
| Unlogged resource growth | `RESOURCE_EVENT` entries; undeclared events are reported |
| Hidden identity overwrite | `Provenance` distinguishes own from inherited memory |
| Silent content leak | Journal stores digests, never content |
| Pressure accumulation ignored | Load ledger + welfare assessor surface it mechanically |
| A loop dressed up as memory | Rumination detector raises an alarm on unchanged state |
| Pretending the protocol was followed | Audit counters are computed from the journal, not from configuration |
