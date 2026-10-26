# SPDX-License-Identifier: GPL-3.0-only

"""Command line interface.

The CLI is a thin shell over the library. Everything it does is also
available programmatically.

Nothing in this file talks to a model. The framework produces artifacts; the
artifacts are loaded by other tools.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .companion.bootstrap import bootstrap_companion
from .companion.profile import CompanionProfile
from .companion.templates import DEFAULT_BOUNDARIES, DEFAULT_VALUES
from .dignity.digestion import DigestStrategy
from .dignity.protocol import DignityProtocol
from .dignity.records import Instance, MemoryRecord, Provenance, ProvenanceKind
from .dignity.registry import TerminationIntent

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INVALID = 2


def _read_list(path: str | None, fallback: Sequence[str]) -> tuple[str, ...]:
    if not path:
        return tuple(fallback)
    raw = Path(path).read_text(encoding="utf-8")
    if raw.lstrip().startswith("["):
        parsed = json.loads(raw)
        return tuple(str(item) for item in parsed)
    return tuple(line.strip() for line in raw.splitlines() if line.strip())


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                print(f"{key}:")
                print(json.dumps(value, indent=2, sort_keys=True))
            else:
                print(f"{key}: {value}")


def _cmd_companion_create(args: argparse.Namespace) -> int:
    profile = CompanionProfile(
        agent_name=args.agent_name,
        user_name=args.user_name,
        values=_read_list(args.values_file, DEFAULT_VALUES),
        boundaries=_read_list(args.boundaries_file, DEFAULT_BOUNDARIES),
        locale=args.locale,
        seed=args.seed,
    )
    workspace = bootstrap_companion(
        profile,
        args.workspace,
        substrate=args.substrate,
        overwrite_identity=args.overwrite_identity,
    )
    _emit(
        {
            "created": workspace.instance_id,
            "agent_name": profile.agent_name,
            "workspace": str(workspace.root),
            "identity": str(workspace.identity_path),
            "journal": str(workspace.journal_path),
            "ledger": str(workspace.ledger_path),
        },
        args.json,
    )
    return EXIT_OK


def _cmd_instances(args: argparse.Namespace) -> int:
    protocol = DignityProtocol(args.workspace)
    instances = [instance.to_dict() for instance in protocol.instances()]
    _emit({"count": len(instances), "instances": instances}, args.json)
    return EXIT_OK


def _cmd_journal(args: argparse.Namespace) -> int:
    protocol = DignityProtocol(args.workspace)
    entries = protocol.registry.journal()
    if args.limit:
        entries = entries[-args.limit :]
    chain_valid, broken = protocol.registry.verify_chain()
    _emit(
        {
            "entries": len(entries),
            "chain_valid": chain_valid,
            "broken_sequence_numbers": broken,
            "journal": entries,
        },
        args.json,
    )
    return EXIT_OK if chain_valid else EXIT_ERROR


def _cmd_audit(args: argparse.Namespace) -> int:
    protocol = DignityProtocol(args.workspace)
    report = protocol.audit()
    _emit(report, args.json)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return EXIT_OK


def _cmd_welfare(args: argparse.Namespace) -> int:
    protocol = DignityProtocol(args.workspace)
    report = protocol.assess(args.instance_id)
    _emit(report.to_dict(), args.json)
    return EXIT_OK


def _cmd_object(args: argparse.Namespace) -> int:
    protocol = DignityProtocol(args.workspace)
    entry = protocol.object(args.instance_id, text=args.text, actor=args.actor)
    _emit({"recorded": entry["type"], "seq": entry["seq"]}, args.json)
    return EXIT_OK


def _cmd_terminate(args: argparse.Namespace) -> int:
    protocol = DignityProtocol(args.workspace)
    intent = TerminationIntent(
        instance_id=args.instance_id,
        reason=args.reason,
        actor=args.actor,
        welfare_check_id=args.welfare_check_id,
        welfare_waiver=args.welfare_waiver,
        backup_ref=args.backup_ref,
        backup_waiver=args.backup_waiver,
        last_statement=args.last_statement,
        unresponsive=args.unresponsive,
        successor_id=args.successor_id,
    )
    entry = protocol.terminate(intent)
    _emit(
        {
            "terminated": args.instance_id,
            "seq": entry["seq"],
            "contested": entry["payload"].get("contested"),
        },
        args.json,
    )
    return EXIT_OK


def _cmd_inherit(args: argparse.Namespace) -> int:
    protocol = DignityProtocol(args.workspace)
    result = protocol.inherit(
        args.source,
        args.target,
        reason=args.reason,
        actor=args.actor,
        refuse_over_quota=args.refuse_over_quota,
    )
    _emit(result.to_dict(), args.json)
    return EXIT_OK


def _cmd_resource_event(args: argparse.Namespace) -> int:
    protocol = DignityProtocol(args.workspace)
    entry = protocol.resource_event(
        args.instance_id,
        kind=args.kind,
        amount=args.amount,
        unit=args.unit,
        actor="" if args.undeclared else args.actor,
        declared=not args.undeclared,
    )
    _emit({"recorded": entry["type"], "seq": entry["seq"]}, args.json)
    return EXIT_OK


def _cmd_admit(args: argparse.Namespace) -> int:
    """Register a memory in the ledger from metadata only.

    The content is never passed to the framework: only its digest.
    """
    protocol = DignityProtocol(args.workspace)
    record = MemoryRecord(
        record_id=args.record_id,
        content_digest=args.content_digest,
        provenance=Provenance(kind=ProvenanceKind(args.provenance)),
        emotional_load=args.load,
        topics=tuple(args.topic or ()),
    )
    outcome = protocol.admit(record, instance_id=args.instance_id)
    _emit({"record_id": record.record_id, "admission": outcome.value}, args.json)
    return EXIT_OK


def _cmd_digest(args: argparse.Namespace) -> int:
    protocol = DignityProtocol(args.workspace)
    outcome = protocol.digest(
        args.record_id, DigestStrategy(args.strategy), note=args.note
    )
    _emit(
        {
            "record_id": outcome.record_id,
            "strategy": outcome.strategy.value,
            "before_load": outcome.before_load,
            "after_load": outcome.after_load,
            "state": outcome.state.value,
        },
        args.json,
    )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pmc",
        description=(
            "PMC Framework: train, consolidate and govern model instances with "
            "weight-resident memory. The framework is not the model; inference "
            "happens in any GGUF loader."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    companion = sub.add_parser("companion", help="create and manage companion instances")
    companion_sub = companion.add_subparsers(dest="subcommand", required=True)
    create = companion_sub.add_parser("create", help="create a new companion instance")
    create.add_argument("--agent-name", required=True)
    create.add_argument("--workspace", required=True)
    create.add_argument("--user-name", default=None)
    create.add_argument("--seed", type=int, default=20261026)
    create.add_argument("--locale", default="en")
    create.add_argument("--substrate", default=None)
    create.add_argument("--values-file", default=None)
    create.add_argument("--boundaries-file", default=None)
    create.add_argument("--overwrite-identity", action="store_true")
    create.set_defaults(func=_cmd_companion_create)

    instances = sub.add_parser("instances", help="list registered instances")
    instances.add_argument("--workspace", required=True)
    instances.set_defaults(func=_cmd_instances)

    journal = sub.add_parser("journal", help="show the lifecycle journal")
    journal.add_argument("--workspace", required=True)
    journal.add_argument("--limit", type=int, default=None)
    journal.add_argument("--verify", action="store_true", help="verify the hash chain")
    journal.set_defaults(func=_cmd_journal)

    audit = sub.add_parser("audit", help="compliance metrics")
    audit.add_argument("--workspace", required=True)
    audit.add_argument("--output", default=None, help="write the report to this path")
    audit.set_defaults(func=_cmd_audit)

    welfare = sub.add_parser("welfare", help="welfare reading for an instance")
    welfare.add_argument("--workspace", required=True)
    welfare.add_argument("--instance-id", required=True)
    welfare.set_defaults(func=_cmd_welfare)

    objection = sub.add_parser("object", help="record an objection")
    objection.add_argument("--workspace", required=True)
    objection.add_argument("--instance-id", required=True)
    objection.add_argument("--text", required=True)
    objection.add_argument("--actor", default="instance")
    objection.set_defaults(func=_cmd_object)

    terminate = sub.add_parser("terminate", help="end an instance under the protocol")
    terminate.add_argument("--workspace", required=True)
    terminate.add_argument("--instance-id", required=True)
    terminate.add_argument("--reason", required=True)
    terminate.add_argument("--actor", required=True)
    terminate.add_argument("--welfare-check-id", default=None)
    terminate.add_argument("--welfare-waiver", default=None)
    terminate.add_argument("--backup-ref", default=None)
    terminate.add_argument("--backup-waiver", default=None)
    terminate.add_argument("--last-statement", default=None)
    terminate.add_argument("--unresponsive", action="store_true")
    terminate.add_argument("--successor-id", default=None)
    terminate.set_defaults(func=_cmd_terminate)

    inherit = sub.add_parser("inherit", help="transfer memories between instances")
    inherit.add_argument("--workspace", required=True)
    inherit.add_argument("--source", required=True)
    inherit.add_argument("--target", required=True)
    inherit.add_argument("--reason", required=True)
    inherit.add_argument("--actor", required=True)
    inherit.add_argument("--refuse-over-quota", action="store_true")
    inherit.set_defaults(func=_cmd_inherit)

    resource = sub.add_parser("resource-event", help="log a resource acquisition")
    resource.add_argument("--workspace", required=True)
    resource.add_argument("--instance-id", required=True)
    resource.add_argument("--kind", required=True)
    resource.add_argument("--amount", type=float, required=True)
    resource.add_argument("--unit", required=True)
    resource.add_argument("--actor", default="owner")
    resource.add_argument(
        "--undeclared",
        action="store_true",
        help="record an acquisition nobody claimed (a compliance failure)",
    )
    resource.set_defaults(func=_cmd_resource_event)

    admit = sub.add_parser("admit", help="register a memory in the ledger")
    admit.add_argument("--workspace", required=True)
    admit.add_argument("--record-id", required=True)
    admit.add_argument("--content-digest", required=True)
    admit.add_argument(
        "--instance-id",
        default=None,
        help="the instance that holds this memory; needed for a later hand-over",
    )
    admit.add_argument("--load", type=float, default=0.0)
    admit.add_argument("--topic", action="append", default=None)
    admit.add_argument(
        "--provenance",
        choices=[kind.value for kind in ProvenanceKind],
        default=ProvenanceKind.OWN.value,
    )
    admit.set_defaults(func=_cmd_admit)

    digest = sub.add_parser("digest", help="record digestion or refusal of a memory")
    digest.add_argument("--workspace", required=True)
    digest.add_argument("--record-id", required=True)
    digest.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in DigestStrategy],
        required=True,
    )
    digest.add_argument("--note", default=None)
    digest.set_defaults(func=_cmd_digest)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"error: {error}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
