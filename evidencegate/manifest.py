from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def manifest_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


class ManifestConflictError(FileExistsError):
    pass


@dataclass(frozen=True)
class ManifestArtifact:
    path: str
    sha256: str
    bytes_written: int


def write_immutable_manifest(path: str | os.PathLike[str], value: Any) -> ManifestArtifact:
    """Atomically write canonical JSON, allowing only byte-identical rewrites."""
    target = Path(path)
    payload = canonical_json(value)
    digest = hashlib.sha256(payload).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = target.read_bytes()
        if existing != payload:
            raise ManifestConflictError(f"manifest_exists_with_different_content:{target}")
        return ManifestArtifact(str(target), digest, len(payload))
    temporary = target.with_name(f".{target.name}.{digest}.tmp")
    try:
        if temporary.exists() and temporary.read_bytes() != payload:
            raise ManifestConflictError(f"temporary_manifest_conflict:{temporary}")
        temporary.write_bytes(payload)
        try:
            os.replace(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise ManifestConflictError(f"manifest_exists_with_different_content:{target}")
    finally:
        if temporary.exists():
            temporary.unlink()
    return ManifestArtifact(str(target), digest, len(payload))
