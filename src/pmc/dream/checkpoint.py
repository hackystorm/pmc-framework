# SPDX-License-Identifier: GPL-3.0-only

"""Checkpoint snapshots, digests and atomic replacement.

A checkpoint is treated as an opaque directory (the trainer writes its own
files inside it) or, for small tests, an opaque file. The dream pipeline never
parses the checkpoint format. The digest covers the relative names, sizes and
SHA-256 of every regular file, so a rollback can be checked byte for byte
without depending on the trainer that produced the checkpoint.

Symlinks are rejected. A checkpoint is data, not a way to make the digest
cover a file outside the directory.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from ..dignity.records import sha256_of


class CheckpointError(RuntimeError):
    """The checkpoint is missing, malformed or cannot be replaced safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(root: Path) -> list[dict[str, Any]]:
    if root.is_file():
        return [
            {
                "path": root.name,
                "bytes": root.stat().st_size,
                "sha256": _sha256_file(root),
            }
        ]
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise CheckpointError(f"checkpoint contains a symlink: {path.name}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise CheckpointError(f"checkpoint contains a non-regular entry: {path.name}")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def checkpoint_digest(path: Path | str) -> str:
    """Return a stable digest over every file in a checkpoint."""
    root = Path(path)
    if root.is_symlink():
        raise CheckpointError(f"checkpoint is a symlink: {root.name}")
    if not root.exists():
        raise CheckpointError(f"checkpoint does not exist: {root.name}")
    if root.is_file():
        kind = "file"
    elif root.is_dir():
        kind = "dir"
    else:
        raise CheckpointError(f"checkpoint is not a file or directory: {root.name}")
    return sha256_of({"kind": kind, "files": _rows(root)})


def remove_path(path: Path) -> None:
    """Remove one checkpoint path without following a symlink."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def copy_checkpoint(source: Path | str, target: Path | str) -> None:
    """Copy a checkpoint directory or file.

    The copy is byte-identical; timestamps and permissions are not part of the
    checkpoint digest.
    """
    origin = Path(source)
    destination = Path(target)
    if origin.is_symlink():
        raise CheckpointError(f"checkpoint is a symlink: {origin.name}")
    checkpoint_digest(origin)
    if origin.is_dir():
        if destination.exists() or destination.is_symlink():
            remove_path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(origin, destination)
    elif origin.is_file():
        if destination.exists() or destination.is_symlink():
            remove_path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
    else:
        raise CheckpointError(f"checkpoint does not exist: {origin.name}")


def restore_checkpoint(checkpoint: Path | str, snapshot: Path | str) -> str:
    """Replace ``checkpoint`` with ``snapshot`` and return the restored digest."""
    target = Path(checkpoint)
    if target.exists() or target.is_symlink():
        remove_path(target)
    copy_checkpoint(snapshot, target)
    return checkpoint_digest(target)


def atomic_commit(checkpoint: Path | str, candidate: Path | str) -> str:
    """Replace ``checkpoint`` with ``candidate`` and return the committed digest.

    The candidate must be on the same filesystem as the checkpoint. The old
    checkpoint is moved aside first; if the second rename fails, it is moved
    back. A failed commit never leaves a half-replaced checkpoint.
    """
    target = Path(checkpoint)
    new = Path(candidate)
    if not new.exists():
        raise CheckpointError("the candidate checkpoint does not exist")
    backup = target.parent / f".{target.name}.old-{uuid.uuid4().hex[:12]}"
    had_old = target.exists() or target.is_symlink()
    try:
        if had_old:
            os.replace(target, backup)
        os.replace(new, target)
    except OSError as error:
        if had_old and not target.exists() and backup.exists():
            os.replace(backup, target)
        raise CheckpointError("the checkpoint could not be replaced") from error
    if had_old and backup.exists():
        remove_path(backup)
    return checkpoint_digest(target)
