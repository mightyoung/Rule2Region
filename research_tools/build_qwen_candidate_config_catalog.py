from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from research_tools.run_scene_adaptive_small_track_ablation import fixed_tiles
from research_tools.sceneconfig_e2_proxy_validation import frozen_panel


CATALOG_CONFIG_IDS = tuple(f"C{index:02d}" for index in range(6))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_catalog(*, width: int = 1280, height: int = 720, overlap_px: int = 64) -> dict[str, Any]:
    if width <= 0 or height <= 0:
        raise ValueError("catalog_reference_resolution_invalid")
    cards = {card.config_id: card for card in frozen_panel()}
    tiles = {tile.tile_id: tile for tile in fixed_tiles(width, height, overlap_px)}
    source_root = Path(__file__).resolve().parent
    source_paths = {
        "panel_source": source_root / "sceneconfig_e2_proxy_validation.py",
        "tile_source": source_root / "run_scene_adaptive_small_track_ablation.py",
    }
    configs = []
    for config_id in CATALOG_CONFIG_IDS:
        card = cards[config_id]
        tile_rows = []
        for tile_id in card.tiles:
            tile = tiles[tile_id]
            tile_rows.append({
                "tile_id": tile_id,
                "core_xyxy_px": list(tile.core.coords),
                "crop_xyxy_px": list(tile.crop.coords),
                "core_xyxy_normalized": [
                    tile.core.x1 / width,
                    tile.core.y1 / height,
                    tile.core.x2 / width,
                    tile.core.y2 / height,
                ],
            })
        configs.append({
            "config_id": config_id,
            "operator": {
                "policy": card.policy,
                "global_detector": "yolo11s_global",
                "expert_detector": "yolo26_static_roi",
                "selected_tiles": tile_rows,
                "tile_confidence": card.tile_confidence,
                "fusion": {"method": "deterministic_nms", "iou_threshold": 0.5},
            },
            "resource_evidence": {
                "tile_count": len(card.tiles),
                "tile_cost_ratio_proxy": len(card.tiles) / 4.0,
                "measured_p95_ms": None,
                "measured_peak_vram_mib": None,
                "note": "Proxy only; exact end-to-end replay is required before deployment decisions.",
            },
            "applicability": {
                "allowed_only_when": [
                    "registered fixed-camera geometry matches the catalog reference transform",
                    "historical evidence identifies small-target or occlusion risk inside every selected tile",
                ],
                "forbidden_when": [
                    "candidate lacks exact end-to-end replay on the target camera",
                    "per-camera precision safety gate fails",
                    "measured p95 latency exceeds the registered 3x YOLO11s cap",
                    "OOM or operator-fingerprint mismatch is observed",
                ],
            },
        })
    payload: dict[str, Any] = {
        "schema_version": "sceneconfig.qwen-candidate-config-catalog.v1",
        "catalog_id": "sceneconfig-c00-c05-operator-catalog-1280x720-v1",
        "evidence_boundary": "DEVELOPMENT_ONLY_OPERATOR_DEFINITION",
        "reference_geometry": {"width": width, "height": height, "overlap_px": overlap_px},
        "config_ids": list(CATALOG_CONFIG_IDS),
        "configs": configs,
        "compiler_authority": {
            "llm_may_rank_only": True,
            "llm_may_not_modify_parameters": True,
            "deterministic_compiler_retains_legality_and_deployment_authority": True,
        },
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
    }
    payload["catalog_sha256"] = _canonical_sha(payload)
    validate_catalog(payload)
    return payload


def validate_catalog(payload: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version", "catalog_id", "catalog_sha256", "evidence_boundary", "reference_geometry",
        "config_ids", "configs", "compiler_authority", "source_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("candidate_catalog_top_level_keys_invalid")
    if tuple(payload["config_ids"]) != CATALOG_CONFIG_IDS:
        raise ValueError("candidate_catalog_ids_invalid")
    if [row.get("config_id") for row in payload["configs"]] != list(CATALOG_CONFIG_IDS):
        raise ValueError("candidate_catalog_config_order_invalid")
    expected_sha = payload["catalog_sha256"]
    unsigned = dict(payload)
    unsigned.pop("catalog_sha256")
    if expected_sha != _canonical_sha(unsigned):
        raise ValueError("candidate_catalog_sha256_invalid")
    for row in payload["configs"]:
        if row["resource_evidence"]["measured_p95_ms"] is not None:
            raise ValueError("candidate_catalog_must_not_invent_latency")
        if row["resource_evidence"]["measured_peak_vram_mib"] is not None:
            raise ValueError("candidate_catalog_must_not_invent_vram")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--overlap-px", type=int, default=64)
    args = parser.parse_args()
    payload = build_catalog(width=args.width, height=args.height, overlap_px=args.overlap_px)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"catalog_id": payload["catalog_id"], "catalog_sha256": payload["catalog_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
