from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .manifest import manifest_sha256


IMAGE_CONDITIONS = {"native", "jpeg", "resize50", "random_crop50", "center_crop50"}
VIDEO_CONDITIONS = {"dense", "uniform50", "random50", "rgb_difference50", "codec_guided50", "bitrate_only", "resize50"}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_formal_manifest(scope_path: str | Path, config_path: str | Path) -> dict[str, Any]:
    scope_file = Path(scope_path)
    config_file = Path(config_path)
    scope_bytes = scope_file.read_bytes()
    config_bytes = config_file.read_bytes()
    scope = json.loads(scope_bytes)
    config = json.loads(config_bytes)
    config_material = dict(config)
    config_identity = config_material.pop("execution_config_sha256", None)
    if config_identity != manifest_sha256(config_material) or config.get("schema_version") != "evidencegate.g1.execution-config.v8":
        raise ValueError("formal_config_identity_or_schema_invalid")
    if hashlib.sha256(scope_bytes).hexdigest() != config["source_scope_sha256"]:
        raise ValueError("formal_scope_hash_mismatch")
    if scope.get("scope_verdict") != "BROAD":
        raise ValueError("formal_manifest_requires_broad_scope")
    aliases = config["runtime"]["formal_condition_resolution"]
    layers = {item["model_key"]: item for item in config["model_layers"]}
    units: list[dict[str, Any]] = []
    for source in scope["matrix"]["quality_units"]:
        modality = source["modality"]
        resolved = aliases.get(modality, {}).get(source["condition"], source["condition"])
        allowed = IMAGE_CONDITIONS if modality == "image" else VIDEO_CONDITIONS if modality == "video" else set()
        if resolved not in allowed:
            raise ValueError(f"formal_condition_unresolved:{modality}:{source['condition']}")
        layer = layers[source["model"]]
        units.append({
            **source,
            "resolved_condition": resolved,
            "model_layer_identity": layer["model_layer_identity"],
        })
    if len(units) != 986 or len({unit["unit_id"] for unit in units}) != 986:
        raise ValueError("formal_quality_unit_cardinality_invalid")
    family_counts = Counter(unit["family"] for unit in units)
    model_counts = Counter(unit["model"] for unit in units)
    if family_counts != Counter({"quality": 590, "deployment": 396}):
        raise ValueError("formal_family_cardinality_invalid")
    if model_counts != Counter({"0.5b-bf16": 590, "0.5b-nf4": 132, "3b-bf16": 132, "3b-nf4": 132}):
        raise ValueError("formal_model_cardinality_invalid")
    manifest: dict[str, Any] = {
        "schema_version": "evidencegate.g3.formal-quality-manifest.v1",
        "source_scope_file_sha256": hashlib.sha256(scope_bytes).hexdigest(),
        "execution_config_file_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "execution_config_sha256": config_identity,
        "codec_derivative_manifest_sha256": config["codec_derivative_manifest_sha256"],
        "model_cache_index_file_sha256": config["runtime"]["model_cache_index_sha256"],
        "g2_output_index_file_sha256": config["runtime"]["g2_output_index_sha256"],
        "prompt_adjustment_receipt_file_sha256": config["runtime"]["prompt_adjustment_receipt_sha256"],
        "quality_logic_units": 986,
        "quality_success_minimum": 937,
        "quality_terminal_coverage_required": 986,
        "retry_attempt_budget": 189,
        "units": units,
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def validate_formal_manifest(manifest: dict[str, Any]) -> None:
    material = dict(manifest)
    identity = material.pop("manifest_sha256", None)
    if identity != manifest_sha256(material):
        raise ValueError("formal_manifest_identity_mismatch")
    if manifest.get("schema_version") != "evidencegate.g3.formal-quality-manifest.v1":
        raise ValueError("formal_manifest_schema_invalid")
    units = manifest.get("units", [])
    if len(units) != 986 or len({unit["unit_id"] for unit in units}) != 986:
        raise ValueError("formal_manifest_unit_cardinality_invalid")
