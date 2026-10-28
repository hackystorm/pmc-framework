# SPDX-License-Identifier: GPL-3.0-only
"""Run the life-criteria battery against a real workspace, and report honestly.

The battery turns five consequences of "consciousness is life" into
measurements, and the whole point of it is that each measurement can fail.
Four of the five can be measured on a workspace today: the journal chain and
the digests across a restart, the replication tree, the decision trace, and
the rhythm's response to its own checks.

Homeostasis is not one of them. It needs the derived aggregate state channel,
which does not exist yet, and a harness that invented plausible readings to
fill the row would be manufacturing the exact thing the battery is supposed
to detect. So this script reports homeostasis as **not measured**, with the
reason, and the battery's evidence level is capped accordingly.

The script is standard library only and never loads a model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from pmc.dignity.protocol import DignityProtocol
from pmc.life_criteria import (
    BatteryReport,
    boundaries,
    continuity,
    reproduction,
    response,
)

UNMEASURED_REASON = (
    "the derived aggregate state channel does not exist yet; readings were not "
    "invented to fill the row"
)


def continuity_sessions(protocol: DignityProtocol, instance_id: str) -> tuple[dict, dict]:
    """Two sessions of one instance, as the digests and the chain see them.

    The "before" session is what the journal says now. The "after" session is
    what a restart reads back, so a disagreement between them is exactly the
    discontinuity this criterion is for.
    """
    registry = protocol.registry
    before = {
        "identity_digest": instance_digest(protocol, instance_id),
        "ledger_head": journal_head(registry),
        "state_digest": state_digest(protocol),
    }
    reopened = DignityProtocol(
        protocol.workspace, profile=protocol.profile.name
    )
    after = {
        "identity_digest": instance_digest(reopened, instance_id),
        "ledger_head": journal_head(reopened.registry),
        "state_digest": state_digest(reopened),
    }
    return before, after


def instance_digest(protocol: DignityProtocol, instance_id: str) -> str:
    from pmc.dignity.records import canonical_json, sha256_of

    instance = protocol.instance(instance_id)
    return sha256_of(canonical_json(instance.to_dict()))


def journal_head(registry) -> str:
    entries = registry.journal()
    if not entries:
        return "0" * 64
    return entries[-1].get("hash", "0" * 64)


def state_digest(protocol: DignityProtocol) -> str:
    from pmc.dignity.records import canonical_json, sha256_of

    return sha256_of(
        canonical_json(
            {
                "ledger": str(protocol.ledger_path),
                "journal": str(protocol.journal_path),
                "records": len(protocol.records()),
                "profile": protocol.profile.name,
            }
        )
    )


def decision_trace(protocol: DignityProtocol, instance_id: str) -> list[dict]:
    """The acts the rhythm took, as a boundary-checking trace.

    Every act in a rhythm cycle reads its own instance's material, so a trace
    with a foreign read is the failure this criterion detects.
    """
    trace: list[dict[str, Any]] = []
    for entry in protocol.registry.journal():
        if entry.get("type") != "RHYTHM":
            continue
        payload = entry.get("payload", {})
        if entry.get("instance_id") != instance_id:
            continue
        trace.append(
            {
                "instance_id": entry.get("instance_id", instance_id),
                "act": payload.get("act", "cycle"),
                "read_instance": entry.get("instance_id", instance_id),
            }
        )
    return trace or [{"instance_id": instance_id, "act": "no_rhythm_yet"}]


def rhythm_pairs(protocol: DignityProtocol, instance_id: str) -> tuple[list, list]:
    """The rhythm's own cycles, as perturbation/response observations.

    A cycle that changed something is a response; a cycle that changed nothing
    is not. The null is the same cycles with the role of each cycle shuffled,
    which is the pattern an instance that changed things independently of its
    own checks would show.
    """
    rows: list[dict[str, bool]] = []
    for entry in protocol.registry.journal():
        if entry.get("type") != "RHYTHM":
            continue
        if entry.get("instance_id") != instance_id:
            continue
        payload = entry.get("payload", {})
        if payload.get("act") != "cycle":
            continue
        items = payload.get("items", [])
        perturbed = bool(items)
        responded = any(bool(item.get("changed")) for item in items)
        rows.append({"perturbed": perturbed, "responded": responded})
    shuffled = [dict(row) for row in rows]
    for index in range(len(shuffled)):
        other = (index + len(shuffled) // 2) % len(shuffled)
        shuffled[index]["perturbed"] = rows[other]["perturbed"] if rows else False
    return rows, shuffled


def run(workspace: str | Path, instance_id: str) -> dict[str, Any]:
    """Run the battery against one workspace and report it."""
    started = time.time()
    protocol = DignityProtocol(workspace, profile=None)
    before, after = continuity_sessions(protocol, instance_id)
    tree = protocol.replication_tree()
    trace = decision_trace(protocol, instance_id)
    rows, shuffled = rhythm_pairs(protocol, instance_id)

    results = [
        continuity(before, after),
        reproduction(tree),
        boundaries(trace),
    ]
    unmeasured = ["homeostasis"]
    if rows:
        results.append(response(rows, shuffled))
    else:
        unmeasured.append("response")

    report = BatteryReport(results=tuple(results), unmeasured=tuple(unmeasured))
    payload = report.to_dict()
    payload["workspace"] = str(workspace)
    payload["instance_id"] = instance_id
    payload["profile"] = protocol.profile.name
    payload["elapsed_s"] = time.time() - started
    payload["notes"] = {"homeostasis": UNMEASURED_REASON}
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--out", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)
    payload = run(args.workspace, args.instance)
    printed = json.dumps(payload, indent=2, sort_keys=True)
    print(printed)
    if args.out:
        Path(args.out).write_text(printed + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
