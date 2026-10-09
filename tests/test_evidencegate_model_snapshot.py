from pathlib import Path

import pytest

from evidencegate.model_snapshot import (
    build_model_cache_index,
    build_model_snapshot_manifest,
    validate_model_cache_index,
    validate_model_snapshot,
)
from evidencegate.manifest import write_immutable_manifest


def test_model_snapshot_manifest_binds_every_file(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    manifest = build_model_snapshot_manifest(tmp_path, "org/model", "a" * 40)
    validate_model_snapshot(tmp_path, manifest)
    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file_mismatch"):
        validate_model_snapshot(tmp_path, manifest)


def test_model_snapshot_file_order_is_platform_independent(tmp_path: Path):
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    (tmp_path / "added_tokens.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    manifest = build_model_snapshot_manifest(tmp_path, "org/model", "a" * 40)

    assert [item["path"] for item in manifest["files"]] == [
        "README.md",
        "added_tokens.json",
        "model.safetensors",
    ]


def test_model_cache_index_binds_manifest_file_and_snapshot(tmp_path: Path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    manifest = build_model_snapshot_manifest(root, "org/model", "a" * 40)
    write_immutable_manifest(root / "model-snapshot-manifest.json", manifest)

    index = build_model_cache_index({"family": root})
    validate_model_cache_index(index)
    assert index["models"]["family"]["snapshot_manifest_sha256"] == manifest["manifest_sha256"]

    (root / "config.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="file_mismatch"):
        validate_model_cache_index(index)
