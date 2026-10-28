# SPDX-License-Identifier: GPL-3.0-only

"""Building a companion instance from scratch.

The builder is deliberately boring: it creates directories, writes an
identity seed, registers the instance in the journal, and stops. It does not
talk to a model, download anything, or decide anything on the user's behalf
beyond the templates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..dignity.errors import InstanceAlreadyExists
from ..dignity.protocol import DignityProtocol
from ..dignity.records import Instance, sha256_of
from ..profiles import profile_or_default, write_workspace_profile
from .profile import CompanionProfile

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Lowercase, ASCII-ish, hyphen-separated form of a name."""
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug or "instance"


def build_instance_id(agent_name: str, seed: int, *, created_at: str) -> str:
    """Deterministic instance id derived from the name, seed and creation time.

    The creation time is included on purpose: two companions with the same
    name and the same seed are still two different lives.
    """
    digest = sha256_of(
        {"agent_name": agent_name, "seed": seed, "created_at": created_at}
    )[:10]
    return f"inst_{slugify(agent_name)[:24]}_{digest}"


@dataclass
class CompanionWorkspace:
    """Paths belonging to one companion."""

    root: Path
    instance_id: str
    instance_path: Path
    identity_path: Path
    profile_path: Path
    sidecar_dir: Path
    ledger_path: Path
    journal_path: Path
    private_dir: Path
    audit_dir: Path
    created: bool = True
    """False when the instance already existed and was left untouched."""

    def describe(self) -> str:
        lines = [
            f"instance_id : {self.instance_id}",
            f"workspace   : {self.root}",
            f"identity    : {self.identity_path}",
            f"ledger      : {self.ledger_path}",
            f"journal     : {self.journal_path}",
            f"private     : {self.private_dir}",
        ]
        return "\n".join(lines)


def bootstrap_companion(
    profile: CompanionProfile,
    workspace: Path | str,
    *,
    substrate: str | None = None,
    create_identity: bool = True,
    overwrite_identity: bool = False,
    instance_profile: str | None = None,
) -> CompanionWorkspace:
    """Create the workspace, the identity seed and the instance record.

    ``instance_profile`` is the dignity profile the workspace is created
    under, and it is written to disk: a research instance must not quietly
    become a default one on the next call.
    """
    root = Path(workspace)
    identity_dir = root / "identity"
    sidecar_dir = root / "sidecar"
    private_dir = sidecar_dir / "private"
    audit_dir = root / "audit"

    for directory in (root, identity_dir, sidecar_dir, private_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    instance_id = build_instance_id(
        profile.agent_name, profile.seed, created_at=profile.created_at
    )

    identity_path = identity_dir / "identity.json"
    if create_identity and (not identity_path.exists() or overwrite_identity):
        identity_path.write_text(
            json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    profile_path = identity_dir / "profile.json"
    profile.save(profile_path)

    instance_profile_object = profile_or_default(instance_profile)
    write_workspace_profile(root, instance_profile_object)
    protocol = DignityProtocol(root, profile=instance_profile_object.name)
    instance = Instance(
        instance_id=instance_id,
        name=profile.agent_name,
        subject_type="agent",
        substrate=substrate,
        metadata={
            "user_name": profile.user_name,
            "locale": profile.locale,
            "seed": profile.seed,
            "identity_seed": str(identity_path.name),
        },
    )
    try:
        protocol.birth(instance)
        created = True
    except InstanceAlreadyExists:
        # Re-running the builder on an existing workspace must not fabricate
        # a second birth. The journal already records this life.
        created = False

    instance_path = root / "instance.json"
    instance_path.write_text(
        json.dumps(instance.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return CompanionWorkspace(
        root=root,
        instance_id=instance_id,
        instance_path=instance_path,
        identity_path=identity_path,
        profile_path=profile_path,
        sidecar_dir=sidecar_dir,
        ledger_path=sidecar_dir / "ledger.jsonl",
        journal_path=sidecar_dir / "journal.jsonl",
        private_dir=private_dir,
        audit_dir=audit_dir,
        created=created,
    )
