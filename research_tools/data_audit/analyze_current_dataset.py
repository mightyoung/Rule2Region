"""Consolidate current dataset evidence into one reproducible summary."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(inventory_path: Path, audit_path: Path, filtered_path: Path) -> dict[str, Any]:
    inventory = _read(inventory_path)
    audit = _read(audit_path)
    filtered = _read(filtered_path)
    rows = audit["videos"]
    included_ids = {row["source_video_id"] for row in filtered["videos"]}
    if len(rows) != 46 or len(included_ids) != 38:
        raise ValueError("unexpected_videos00_counts")

    scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene_rows[str(row["scene_id"])].append(row)

    scenes = []
    for scene_id in sorted(scene_rows):
        all_rows = scene_rows[scene_id]
        labeled = [row for row in all_rows if row["source_video_id"] in included_ids]
        annotation_rows = [row["annotation"] for row in labeled]
        total_annotation_rows = sum(item["row_count"] for item in annotation_rows)
        person_rows = sum(item["person_row_count"] for item in annotation_rows)
        labeled_frames = sum(row["media"]["frame_count"] for row in labeled)
        scenes.append(
            {
                "scene_id": f"VIRAT_S_{scene_id}",
                "media_video_count": len(all_rows),
                "labeled_video_count": len(labeled),
                "excluded_video_count": len(all_rows) - len(labeled),
                "media_hours": sum(row["media"]["duration_seconds"] for row in all_rows) / 3600,
                "labeled_hours": sum(row["media"]["duration_seconds"] for row in labeled) / 3600,
                "resolution_values": sorted(
                    {f'{row["media"]["width"]}x{row["media"]["height"]}' for row in all_rows}
                ),
                "fps_values": sorted({row["media"]["fps"] for row in all_rows}),
                "annotation_row_count": total_annotation_rows,
                "person_row_count": person_rows,
                "person_row_share": person_rows / total_annotation_rows if total_annotation_rows else None,
                "labeled_frame_count": labeled_frames,
                "average_person_boxes_per_frame": person_rows / labeled_frames if labeled_frames else None,
                "strict_frame_range_video_count": sum(
                    row["annotation"]["frame_indices_in_range"] is True for row in labeled
                ),
                "endpoint_equal_frame_count_video_count": sum(
                    row["annotation"]["frame_indices_in_range"] is False
                    and row["annotation"]["maximum_frame_index"] == row["media"]["frame_count"]
                    for row in labeled
                ),
            }
        )

    archive_video_count = int(inventory["virat"]["video_member_count"])
    corrected_labeled_count = archive_video_count - int(filtered["excluded_video_count"])
    available_new_scenes = [
        scene for scene in inventory["virat"]["scenes"] if scene["used_in_existing_development"] is False
    ]
    payload = {
        "schema_version": "fieldpack.current-dataset-analysis.v1",
        "evidence_date": "2026-08-20",
        "virat": {
            "archive_video_count": archive_video_count,
            "physical_scene_count": inventory["virat"]["physical_scene_count"],
            "used_development_scene_count": sum(
                scene["used_in_existing_development"] is True for scene in inventory["virat"]["scenes"]
            ),
            "available_new_scene_count": len(available_new_scenes),
            "videos00_media_video_count": len(rows),
            "videos00_labeled_video_count": len(included_ids),
            "videos00_excluded_video_count": len(rows) - len(included_ids),
            "videos00_label_match_rate": len(included_ids) / len(rows),
            "all_videos_corrected_labeled_count": corrected_labeled_count,
            "all_videos_corrected_label_match_rate": corrected_labeled_count / archive_video_count,
            "stale_inventory_reported_labeled_count": inventory["virat"]["fully_annotation_mapped_video_count"],
            "stale_inventory_overcount": inventory["virat"]["fully_annotation_mapped_video_count"]
            - corrected_labeled_count,
            "videos00_total_hours": sum(row["media"]["duration_seconds"] for row in rows) / 3600,
            "videos00_labeled_hours": sum(
                row["media"]["duration_seconds"] for row in rows if row["source_video_id"] in included_ids
            )
            / 3600,
            "videos00_all_media_decodable": all(
                row["media"]["metadata_valid"]
                and row["media"]["first_frame_decodable"]
                and row["media"]["last_frame_decodable"]
                for row in rows
            ),
            "scenes": scenes,
            "annotated_scenes_missing_video": inventory["virat"]["annotation_scenes_without_video_archive"],
        },
        "chokepoint": {
            "scene_archive_count": inventory["chokepoint"]["scene_archive_count"],
            "camera_view_count": inventory["chokepoint"]["physical_view_count"],
            "independent_location_count": inventory["chokepoint"]["independent_location_count"],
            "independent_locations": inventory["chokepoint"]["independent_locations"],
            "raw_media_audited": False,
            "annotation_mapping_after_extraction_verified": False,
        },
        "readiness": {
            "target_new_independent_scenes": 8,
            "structurally_available_new_scenes_or_locations": 5,
            "remaining_independent_scene_gap": 3,
            "media_checked_new_virat_scenes": 3,
            "experiment_ready_new_scenes": 0,
            "development_gpu_ready": False,
            "formal_gpu_ready": False,
            "blockers": [
                "annotation_endpoint_policy_not_frozen",
                "time_split_not_frozen",
                "chokepoint_media_not_audited",
                "three_independent_scenes_still_missing",
                "new_dataset_gpu_scope_not_authorized",
            ],
        },
        "recommended_tiers": [
            {
                "tier": "A",
                "name": "existing_six_video_pilot",
                "use": "existing engineering regression only",
            },
            {
                "tier": "B",
                "name": "videos00_labeled_38",
                "use": "freeze boundary and time split before development GPU",
            },
            {
                "tier": "C",
                "name": "chokepoint_p1_p2",
                "use": "media and annotation audit required before use",
            },
            {
                "tier": "D",
                "name": "virat_0400_0401_0500_0502_0503",
                "use": "unavailable until video archives are supplied",
            },
        ],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--filtered", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.inventory, args.audit, args.filtered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["readiness"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
