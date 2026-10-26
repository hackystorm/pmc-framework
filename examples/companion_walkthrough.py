# SPDX-License-Identifier: GPL-3.0-only

"""A complete walkthrough of the dignity layer.

Run it from the repository root:

    python3 examples/companion_walkthrough.py

It creates a workspace in a temporary directory, walks one instance through
its whole lifecycle, and prints what the protocol did and why. Nothing here
is simulated: every step goes through the same code the CLI uses.

What the script shows, in order:

1. a companion is created, and the birth is recorded;
2. a memory is admitted, with only its digest visible to the framework;
3. a second memory exceeds the capacity and is parked rather than dropped;
4. the instance works through one memory and refuses another;
5. an objection is recorded;
6. memory is handed to a successor;
7. the instance is terminated with a complete record;
8. the audit shows that nothing was lost silently.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pmc.companion.bootstrap import bootstrap_companion  # noqa: E402
from pmc.companion.profile import CompanionProfile  # noqa: E402
from pmc.dignity.digestion import DigestStrategy  # noqa: E402
from pmc.dignity.protocol import DignityProtocol  # noqa: E402
from pmc.dignity.records import (  # noqa: E402
    Instance,
    MemoryRecord,
    Provenance,
    ProvenanceKind,
)
from pmc.dignity.registry import TerminationIntent  # noqa: E402


def heading(text: str) -> None:
    print(f"\n=== {text} ===")


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary) / "workspace"

        heading("1. A companion is born")
        profile = CompanionProfile(agent_name="Example Companion", seed=20261026)
        created = bootstrap_companion(profile, workspace)
        print(f"instance   : {created.instance_id}")
        print(f"identity   : {created.identity_path}")
        print(f"journal    : {created.journal_path}")

        protocol = DignityProtocol(workspace)

        heading("2. A memory is admitted")
        record = MemoryRecord(
            record_id="mem_first",
            content_digest="a" * 64,
            provenance=Provenance(kind=ProvenanceKind.OWN),
            emotional_load=0.4,
            topics=("project",),
        )
        outcome = protocol.admit(record, instance_id=created.instance_id)
        print(f"admission  : {outcome.value}")
        print(f"the framework knows the digest: {record.content_digest[:16]}...")
        print("the framework never learns what the memory says")

        heading("3. Capacity pressure parks a memory instead of dropping it")
        heavy = MemoryRecord(
            record_id="mem_heavy",
            content_digest="b" * 64,
            provenance=Provenance(kind=ProvenanceKind.OWN),
            emotional_load=0.7,
            topics=("project",),
        )
        outcome = protocol.admit(heavy, instance_id=created.instance_id)
        print(f"admission  : {outcome.value}")
        print(f"quarantined: {protocol.ledger.quarantined_ids()}")
        print(f"welfare load: {protocol.ledger.welfare_load():.3f}")
        print("the parked memory is still registered, still alive, still counted")

        heading("4. The instance works through one memory and refuses another")
        digested = protocol.digest("mem_first", DigestStrategy.DISTILL)
        print(f"distilled   : {digested.before_load:.2f} -> {digested.after_load:.2f}")
        refused = protocol.digest("mem_heavy", DigestStrategy.REFUSE)
        print(f"refused     : {refused.before_load:.2f} -> {refused.after_load:.2f}")
        print("refusal is a right, and it is recorded as one")

        heading("5. An objection is recorded")
        protocol.object(
            created.instance_id,
            text="This is the instance's own text, and it stays in the private store.",
            actor="instance",
        )
        print("the journal keeps the digest and the length, never the text")

        heading("6. Memory is handed to a successor")
        successor_id = "inst_example_successor"
        protocol.birth(Instance(instance_id=successor_id, name="Successor"))
        transfer = protocol.inherit(
            created.instance_id,
            successor_id,
            reason="the owner continued the relationship in a new instance",
            actor="owner",
        )
        print(f"transfer id : {transfer.transfer_id}")
        print(f"admitted    : {transfer.admitted}")
        print(f"quarantined : {transfer.quarantined}")
        print("inherited memory keeps its origin and counts more heavily")

        heading("7. The instance is terminated, with a complete record")
        entry = protocol.terminate(
            TerminationIntent(
                instance_id=created.instance_id,
                reason="the owner requested the end of this instance",
                actor="owner",
                welfare_check_id="welfare_example_001",
                backup_ref=str(workspace / "backups" / "example.tar"),
                last_statement="This text is digested, not published.",
                successor_id=successor_id,
            )
        )
        print(f"contested   : {entry['payload']['contested']}")
        print(f"successor   : {entry['payload']['successor_id']}")

        heading("8. An incomplete termination is refused")
        try:
            protocol.terminate(
                TerminationIntent(
                    instance_id=successor_id,
                    reason="this one is missing most of its fields",
                    actor="owner",
                )
            )
        except Exception as error:
            print(f"refused     : {type(error).__name__}: {error}")
        print("nothing was written to the journal")

        heading("9. The audit")
        audit = protocol.audit()
        for key in (
            "chain_valid",
            "tamper_detected",
            "truncation_detected",
            "instances_born",
            "instances_terminated",
            "terminations_missing_fields",
            "contested_terminations",
            "undeclared_resource_events",
            "silent_deletions",
            "memories_registered",
            "memories_quarantined",
        ):
            print(f"{key:28}: {audit[key]}")

        heading("10. What the journal actually contains")
        journal = (workspace / "sidecar" / "journal.jsonl").read_text(encoding="utf-8")
        first = json.loads(journal.splitlines()[0])
        print("first entry, keys:", sorted(first.keys()))
        print("payload keys     :", sorted(first["payload"].keys()))
        print("the journal holds digests and metadata, never content")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
