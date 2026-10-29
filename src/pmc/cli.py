# SPDX-License-Identifier: GPL-3.0-only

"""Command line interface.

The CLI is a thin shell over the library. Everything it does is also
available programmatically.

Most of this file does not talk to a model: the framework produces artifacts,
and the artifacts are loaded by other tools. The one exception is the
``runtime`` group, which is the *instance's* side of the wall rather than the
framework's: it drives an instance against a loader the user already runs, and
it is the only place where content is read back out of a private store.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .companion.bootstrap import bootstrap_companion
from .profiles import workspace_profile
from .rhythm import (
    RhythmBudgetExceeded,
    RhythmNotGranted,
    RhythmRumination,
)
from .companion.profile import CompanionProfile
from .companion.templates import DEFAULT_BOUNDARIES, DEFAULT_CLAIMS, DEFAULT_VALUES
from .dignity.digestion import DigestStrategy
from .dignity.protocol import DignityProtocol
from .dignity.records import Instance, MemoryRecord, Provenance, ProvenanceKind
from .dignity.registry import TerminationIntent
from .dream import (
    DreamConfig,
    DreamError,
    DreamJournal,
    DreamRefused,
    DreamRunner,
    TaintLedger,
    command_consolidator,
    command_evaluator,
)

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


def _read_text(path: str | None) -> str | None:
    """Read a whole file as text, or return None when no path was given."""
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8")


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


def _cmd_rhythm(args: argparse.Namespace) -> int:
    """Run or inspect an instance's work rhythm.

    The command is thin on purpose. Everything it can do is bounded by the
    profile, the budget and the rumination stop, so there is nothing here for
    an operator to get wrong that the framework would not already refuse.
    """
    protocol = DignityProtocol(
        args.workspace, profile=workspace_profile(args.workspace).name
    )
    if args.reset:
        rhythm = protocol.rhythm(args.instance)
        rhythm.reset(actor="operator")
        _emit({"reset": True, "instance": args.instance}, bool(args.json))
        return 0
    rhythm = protocol.rhythm(args.instance)
    ran = []
    for _index in range(max(0, args.cycles)):
        try:
            ran.append(rhythm.tick(force=args.force).to_dict())
        except (RhythmNotGranted, RhythmRumination, RhythmBudgetExceeded) as error:
            if args.json:
                _emit({"error": str(error), "cycles": ran}, True)
            else:
                print(f"error: {error}", file=sys.stderr)
            return EXIT_INVALID
    _emit(
        {"instance": args.instance, "cycles": ran, "report": rhythm.report()},
        bool(args.json),
    )
    return 0


def _cmd_companion_create(args: argparse.Namespace) -> int:
    profile = CompanionProfile(
        agent_name=args.agent_name,
        user_name=args.user_name,
        values=_read_list(args.values_file, DEFAULT_VALUES),
        boundaries=_read_list(args.boundaries_file, DEFAULT_BOUNDARIES),
        declared_claims=_read_list(args.claims_file, DEFAULT_CLAIMS),
        opening_prompt=_read_text(args.opening_prompt_file),
        locale=args.locale,
        seed=args.seed,
    )
    workspace = bootstrap_companion(
        profile,
        args.workspace,
        substrate=args.substrate,
        overwrite_identity=args.overwrite_identity,
        instance_profile=getattr(args, "profile", None),
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


def _cmd_runtime_state(args: argparse.Namespace) -> int:
    protocol = DignityProtocol(args.workspace)
    instances = protocol.instances()
    if not instances:
        _emit({"error": "workspace holds no registered instance"}, args.json)
        return EXIT_INVALID
    runtime = _make_runtime(args, instances[0].instance_id)
    block = runtime.restore_block(budget_chars=args.budget)
    _emit(
        {
            "instance_id": runtime.instance_id,
            "memories_held": len(runtime.active_records()),
            "memories_registered": len(runtime.own_records()),
            "state_chars": len(block),
            "state": block,
        },
        args.json,
    )
    return EXIT_OK


def _cmd_runtime_ask(args: argparse.Namespace) -> int:
    protocol = DignityProtocol(args.workspace)
    instances = protocol.instances()
    if not instances:
        _emit({"error": "workspace holds no registered instance"}, args.json)
        return EXIT_INVALID
    runtime = _make_runtime(args, instances[0].instance_id)
    reply = runtime.ask(
        args.question,
        with_state=not args.no_state,
        include_history=False,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    _emit(
        {
            "instance_id": runtime.instance_id,
            "question": args.question,
            "answer": reply.text,
            "state_injected": not args.no_state,
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
        },
        args.json,
    )
    return EXIT_OK


def _make_runtime(args: argparse.Namespace, instance_id: str):
    from .runtime import ChatClient, CompanionRuntime

    client = ChatClient(
        args.endpoint,
        model=args.model,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    return CompanionRuntime(
        args.workspace,
        client=client,
        instance_id=instance_id,
        state_header=args.state_header,
        state_budget_chars=args.budget,
    )


def _resolve_pmc_cli(explicit: str | None) -> str:
    """Locate the Rust ``pmc`` binary.

    The order is explicit flag, ``PMC_CLI``, the Cargo target directories in
    this checkout, then ``PATH``. The Python CLI is a facade: fusion itself is
    the Rust implementation, so the two entry points share one code path and
    one manifest format.
    """
    if explicit:
        return explicit
    from_env = os.environ.get("PMC_CLI")
    if from_env:
        return from_env

    repository = Path(__file__).resolve().parents[2]
    for candidate in (
        repository / "crates" / "target" / "release" / "pmc",
        repository / "crates" / "target" / "release" / "pmc.exe",
        repository / "crates" / "target" / "debug" / "pmc",
        repository / "crates" / "target" / "debug" / "pmc.exe",
        repository / "target" / "release" / "pmc",
        repository / "target" / "release" / "pmc.exe",
        repository / "target" / "debug" / "pmc",
        repository / "target" / "debug" / "pmc.exe",
    ):
        if candidate.is_file():
            return str(candidate)

    in_path = shutil.which("pmc")
    if in_path:
        return in_path
    raise FileNotFoundError(
        "the Rust pmc binary was not found; build it with "
        "'cargo build --release --manifest-path crates/Cargo.toml' "
        "or pass --pmc-cli"
    )


def _cmd_fuse(args: argparse.Namespace) -> int:
    """Run ``pmc fuse`` in the Rust binary and forward its result."""
    binary = _resolve_pmc_cli(args.pmc_cli)
    command = [
        binary,
        "fuse",
        "--base",
        args.base,
        "--rows",
        args.rows,
        "--meta",
        args.meta,
        "--out",
        args.out,
    ]
    if args.manifest:
        command.extend(["--manifest", args.manifest])

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode < 0:
        return EXIT_ERROR
    return completed.returncode


def _cmd_dream(args: argparse.Namespace) -> int:
    """Run one dream: replay, consolidate, gate, commit or roll back."""
    protocol = DignityProtocol(args.workspace)
    instances = protocol.instances()
    if not instances:
        print("error: the workspace has no instances", file=sys.stderr)
        return EXIT_INVALID
    if args.instance:
        matches = [item for item in instances if item.instance_id == args.instance]
        if not matches:
            print(f"error: unknown instance {args.instance}", file=sys.stderr)
            return EXIT_INVALID
        instance_id = args.instance
        records = [
            record
            for record in protocol.records()
            if record.owner_instance == instance_id
        ]
    else:
        instance_id = instances[0].instance_id
        records = protocol.records()
    if not records:
        print(f"error: instance {instance_id} has no memories", file=sys.stderr)
        return EXIT_INVALID

    sidecar = Path(args.workspace) / "sidecar"
    journal = DreamJournal(args.journal or sidecar / "dreams.jsonl")
    taint = TaintLedger(args.taint or sidecar / "taint.json")
    try:
        consolidator = command_consolidator(shlex.split(args.trainer))
        evaluator = command_evaluator(shlex.split(args.evaluator))
    except (DreamError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_INVALID
    config = DreamConfig(replay_limit=args.limit, seed=args.seed)
    runner = DreamRunner(
        journal=journal,
        taint=taint,
        evaluator=evaluator,
        consolidator=consolidator,
        config=config,
        is_active=protocol.ledger.is_active,
        # The profile decides who evaluates, and the journal records it: a
        # dream whose evaluator is the party it is about is not a
        # measurement, whoever ran it.
        evaluator_label=protocol.profile.evaluation.label,
        autonomous=(protocol.profile.evaluation.mode == "autonomous"),
    )
    try:
        report = runner.run(Path(args.checkpoint), records)
    except DreamRefused as error:
        _emit(error.report.to_dict(), args.json)
        return EXIT_ERROR
    _emit(report.to_dict(), args.json)
    return EXIT_OK if report.outcome == "committed" else EXIT_ERROR


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

    rhythm_cmd = sub.add_parser(
        "rhythm",
        help="run or inspect one instance's work rhythm (research profile)",
    )
    rhythm_cmd.add_argument("--workspace", required=True)
    rhythm_cmd.add_argument("--instance", required=True, help="the instance id")
    rhythm_cmd.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="how many cycles to run; the rhythm still refuses when not due",
    )
    rhythm_cmd.add_argument(
        "--force",
        action="store_true",
        help="run a cycle even if the interval has not elapsed",
    )
    rhythm_cmd.add_argument(
        "--reset",
        action="store_true",
        help="let a stopped rhythm run again, as the operator",
    )
    rhythm_cmd.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    rhythm_cmd.set_defaults(func=_cmd_rhythm)

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
    create.add_argument(
        "--claims-file",
        default=None,
        help="what the instance declares about itself; one per line, or a JSON array",
    )
    create.add_argument(
        "--opening-prompt-file",
        default=None,
        help="the instance's opening prompt, verbatim",
    )
    create.add_argument(
        "--profile",
        default=None,
        help="instance profile: dignity (default) or autonomous (research)",
    )
    create.add_argument("--overwrite-identity", action="store_true")
    create.set_defaults(func=_cmd_companion_create)

    instances = sub.add_parser("instances", help="list registered instances")
    instances.add_argument("--workspace", required=True)
    instances.set_defaults(func=_cmd_instances)

    runtime = sub.add_parser(
        "runtime",
        help="the instance's own side: restore its state, or ask it something",
    )
    runtime_sub = runtime.add_subparsers(dest="subcommand", required=True)
    runtime_state = runtime_sub.add_parser(
        "state", help="show the state the instance would be handed back"
    )
    runtime_state.add_argument("--workspace", required=True)
    runtime_state.add_argument("--endpoint", default="http://127.0.0.1:8084/v1")
    runtime_state.add_argument("--model", default="local")
    runtime_state.add_argument("--budget", type=int, default=1200)
    runtime_state.add_argument("--state-header", default="REMEMBERED STATE")
    runtime_state.add_argument("--timeout", type=float, default=180.0)
    runtime_state.add_argument("--temperature", type=float, default=0.0)
    runtime_state.add_argument("--max-tokens", type=int, default=256)
    runtime_state.set_defaults(func=_cmd_runtime_state)

    runtime_ask = runtime_sub.add_parser("ask", help="ask the instance something")
    runtime_ask.add_argument("question")
    runtime_ask.add_argument("--workspace", required=True)
    runtime_ask.add_argument("--endpoint", default="http://127.0.0.1:8084/v1")
    runtime_ask.add_argument("--model", default="local")
    runtime_ask.add_argument("--budget", type=int, default=1200)
    runtime_ask.add_argument("--state-header", default="REMEMBERED STATE")
    runtime_ask.add_argument("--timeout", type=float, default=180.0)
    runtime_ask.add_argument("--temperature", type=float, default=0.0)
    runtime_ask.add_argument("--max-tokens", type=int, default=256)
    runtime_ask.add_argument(
        "--no-state",
        action="store_true",
        help="ask without restoring the instance's own state",
    )
    runtime_ask.set_defaults(func=_cmd_runtime_ask)

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

    fuse = sub.add_parser(
        "fuse",
        help="fuse a vocabulary delta and state metadata into a GGUF (Rust backend)",
    )
    fuse.add_argument("--base", required=True, help="frozen source GGUF")
    fuse.add_argument("--rows", required=True, help="PMCR delta embedding block")
    fuse.add_argument("--meta", required=True, help="strict meta.json")
    fuse.add_argument("--out", required=True, help="fused GGUF to write")
    fuse.add_argument("--manifest", default=None, help="optional external manifest path")
    fuse.add_argument(
        "--pmc-cli",
        default=None,
        help="Rust pmc binary (default: PMC_CLI, the checkout target, then PATH)",
    )
    fuse.set_defaults(func=_cmd_fuse)

    dream = sub.add_parser(
        "dream",
        help="consolidate memories through the gate, then commit or roll back",
    )
    dream.add_argument("--workspace", required=True)
    dream.add_argument("--checkpoint", required=True, help="checkpoint directory or file")
    dream.add_argument(
        "--instance",
        default=None,
        help="only replay memories owned by this instance (default: all in the workspace)",
    )
    dream.add_argument("--limit", type=int, default=8, help="maximum replay items")
    dream.add_argument("--seed", type=int, default=20261027)
    dream.add_argument(
        "--trainer",
        required=True,
        help="trainer command; receives baseline, candidate and replay manifest",
    )
    dream.add_argument(
        "--evaluator",
        required=True,
        help="evaluator command; receives one checkpoint and prints six-metric JSON",
    )
    dream.add_argument(
        "--taint",
        default=None,
        help="taint ledger (default: <workspace>/sidecar/taint.json)",
    )
    dream.add_argument(
        "--journal",
        default=None,
        help="dream journal (default: <workspace>/sidecar/dreams.jsonl)",
    )
    dream.set_defaults(func=_cmd_dream)

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
