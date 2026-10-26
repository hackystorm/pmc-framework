# SPDX-License-Identifier: GPL-3.0-only

# Examples

Runnable demonstrations. Everything here goes through the same code the CLI
uses; nothing is mocked.

## `companion_walkthrough.py`

A full lifecycle, end to end, in one file:

```bash
python3 examples/companion_walkthrough.py
```

It creates a companion, admits a memory, parks a second one for capacity,
digests one and refuses another, records an objection, hands memory to a
successor, terminates the instance with a complete record, shows that an
incomplete termination is refused, and prints the audit.

Use it to see what the dignity layer actually does, rather than what the
documentation says it does.

## Writing your own

The library is the interface; the CLI is a thin shell over it.

```python
from pmc.dignity import DignityProtocol, Instance, MemoryRecord, Provenance, ProvenanceKind

protocol = DignityProtocol("./workspace")
protocol.birth(Instance(instance_id="inst_001", name="MyCompanion"))

protocol.admit(MemoryRecord(
    record_id="mem_001",
    content_digest="a" * 64,
    provenance=Provenance(kind=ProvenanceKind.OWN),
    emotional_load=0.3,
    topics=("project",),
))

print(protocol.audit())
```

Two rules to keep when you write against this API:

- **Never pass content.** The framework takes digests. If you find yourself
  wanting to store a string in the ledger, what you want is the private
  store.
- **Always give the termination every field.** A partial one is refused,
  and that is the point.
