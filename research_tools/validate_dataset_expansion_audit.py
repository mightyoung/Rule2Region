"""Validate the metadata-only dataset expansion audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(inventory_path: Path, plan_path: Path) -> dict:
    data = json.loads(inventory_path.read_text(encoding="utf-8"))
    virat_six = data.get("virat", {}).get("recommended_six", [])
    choke_six = data.get("chokepoint", {}).get("recommended_six", [])
    checks = {
        "schema": data.get("schema_version") == "fieldpack.dataset-expansion-metadata-audit.v1",
        "metadata_only": data.get("raw_media_extracted") is False and data.get("model_inference_run") is False and data.get("ground_truth_values_read") is False,
        "virat_inventory_matches_source": data.get("virat", {}).get("video_member_count") == 226 and data.get("virat", {}).get("sequence_prefix_count") == 40 and data.get("virat", {}).get("physical_scene_count") == 6 and data.get("virat", {}).get("annotation_scene_count") == 11 and data.get("virat", {}).get("fully_annotation_mapped_video_count") == 226,
        "virat_identity_rule_recorded": "first four digits" in data.get("virat", {}).get("identity_rule", "") and data.get("virat", {}).get("official_reference_filename") == "VIRAT_Video_Dataset_Release2.0_Introduction_v1.0.pdf",
        "virat_gap_explicit": len(virat_six) == 3 and data.get("virat", {}).get("structural_candidate_count") == 3 and len(data.get("virat", {}).get("annotation_scenes_without_video_archive", [])) == 5,
        "chokepoint_inventory_complete": data.get("chokepoint", {}).get("scene_archive_count") == 18 and data.get("chokepoint", {}).get("xml_count") == 72 and data.get("chokepoint", {}).get("physical_view_count") == 12,
        "chokepoint_six_balanced": len(choke_six) == 6 and data.get("chokepoint", {}).get("shortlist_balance") == {"P1": 3, "P2": 3, "E": 3, "L": 3, "C1": 2, "C2": 2, "C3": 2},
        "chokepoint_location_dependence_recorded": data.get("chokepoint", {}).get("independent_location_count") == 2 and data.get("chokepoint", {}).get("independent_locations") == ["P1", "P2"],
        "structural_shortfall_recomputed": data.get("readiness", {}).get("structural_target_met") is False and data.get("readiness", {}).get("available_new_independent_physical_scenes") == 5 and data.get("readiness", {}).get("new_independent_scene_gap") == 3,
        "gpu_remains_closed": data.get("readiness", {}).get("gpu_experiment_allowed") is False and len(data.get("readiness", {}).get("blockers", [])) >= 7,
        "plan_present": plan_path.is_file() and plan_path.stat().st_size > 0,
    }
    passed = all(checks.values())
    return {
        "schema_version": "autoresearch.completion.v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "research_outcome": "existing_archives_insufficient_stop_before_gpu" if passed else "inventory_incomplete",
        "checks": checks,
        "summary": "现有归档包含 3 个新 VIRAT 场景和 2 个 ChokePoint 地点，共 5 个新地点；距离 8 个新地点还差 3 个，且新 VIRAT 媒体尚未完成授权、哈希和解码检查。" if passed else "数据扩展清单未通过完整性检查。",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.inventory, args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
