from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .manifest import manifest_sha256


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_files(root: str | Path) -> tuple[Path, ...]:
    base = Path(root)
    files = (
        path for path in base.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(base).parts
        and path.name != "model-snapshot-manifest.json"
    )
    # Path ordering follows host filesystem semantics (case-insensitive on
    # Windows, case-sensitive on Linux).  Sort canonical POSIX-relative names
    # so identical snapshots receive the same identity on every platform.
    return tuple(sorted(files, key=lambda path: path.relative_to(base).as_posix()))


def build_model_snapshot_manifest(root: str | Path, repo_id: str, revision: str) -> dict[str, Any]:
    base = Path(root)
    files = [
        {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in snapshot_files(base)
    ]
    if not files or not any(item["path"].endswith(".safetensors") for item in files):
        raise ValueError("model_snapshot_requires_safetensors")
    manifest: dict[str, Any] = {
        "schema_version": "evidencegate.g1.model-snapshot.v1",
        "repo_id": repo_id,
        "revision": revision,
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def validate_model_snapshot(root: str | Path, manifest: dict[str, Any]) -> None:
    base = Path(root)
    material = dict(manifest)
    identity = material.pop("manifest_sha256", None)
    if identity != manifest_sha256(material):
        raise ValueError("model_snapshot_manifest_identity_mismatch")
    actual_paths = {path.relative_to(base).as_posix() for path in snapshot_files(base)}
    declared_paths = {item["path"] for item in manifest["files"]}
    if actual_paths != declared_paths:
        raise ValueError("model_snapshot_file_set_mismatch")
    for item in manifest["files"]:
        path = base / item["path"]
        if path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"model_snapshot_file_mismatch:{item['path']}")


def build_model_cache_index(models: Mapping[str, str | Path]) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    for family in sorted(models):
        root = Path(models[family])
        manifest_path = root / "model-snapshot-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_model_snapshot(root, manifest)
        entries[family] = {
            "path": root.as_posix(),
            "manifest_path": manifest_path.as_posix(),
            "manifest_file_sha256": file_sha256(manifest_path),
            "snapshot_manifest_sha256": manifest["manifest_sha256"],
            "repo_id": manifest["repo_id"],
            "revision": manifest["revision"],
        }
    if not entries:
        raise ValueError("model_cache_index_requires_models")
    index: dict[str, Any] = {
        "schema_version": "evidencegate.g1.model-cache-index.v1",
        "models": entries,
    }
    index["manifest_sha256"] = manifest_sha256(index)
    return index


def validate_model_cache_index(index: Mapping[str, Any]) -> None:
    material = dict(index)
    identity = material.pop("manifest_sha256", None)
    if identity != manifest_sha256(material):
        raise ValueError("model_cache_index_identity_mismatch")
    if index.get("schema_version") != "evidencegate.g1.model-cache-index.v1":
        raise ValueError("model_cache_index_schema_invalid")
    models = index.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError("model_cache_index_requires_models")
    for family, entry in models.items():
        root = Path(entry["path"])
        manifest_path = Path(entry["manifest_path"])
        if manifest_path != root / "model-snapshot-manifest.json":
            raise ValueError(f"model_cache_manifest_path_invalid:{family}")
        if file_sha256(manifest_path) != entry["manifest_file_sha256"]:
            raise ValueError(f"model_cache_manifest_file_mismatch:{family}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["manifest_sha256"] != entry["snapshot_manifest_sha256"]:
            raise ValueError(f"model_cache_snapshot_identity_mismatch:{family}")
        if manifest["repo_id"] != entry["repo_id"] or manifest["revision"] != entry["revision"]:
            raise ValueError(f"model_cache_repository_identity_mismatch:{family}")
        validate_model_snapshot(root, manifest)
