"""Build a metadata-only inventory for the next fixed-camera study."""
from __future__ import annotations

import argparse
import json
import re
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


VIRAT_VIDEO_RE = re.compile(r"^VIRAT_S_(\d{6})(?:_(\d{2})_(\d{6})_(\d{6}))?\.mp4$")
CHOKE_ARCHIVE_RE = re.compile(r"^(P[12][EL])_S(\d+)\.tar\.xz$")
CHOKE_XML_RE = re.compile(r"^(P[12][EL])_S(\d+)_C([123])(?:\.(\d+))?\.xml$")
USED_VIRAT_SCENES = frozenset({"VIRAT_S_0100", "VIRAT_S_0101", "VIRAT_S_0102"})
CHOKEPOINT_SHORTLIST = (
    "P1E_C1", "P1L_C2", "P2E_C3", "P2L_C1", "P1E_C3", "P2L_C2",
)


def _basenames(names: Iterable[str]) -> list[str]:
    return [Path(name).name for name in names if Path(name).name]


def _virat_inventory(video_zips: list[Path], annotations_zip: Path) -> dict[str, Any]:
    video_names: list[str] = []
    for video_zip in video_zips:
        with zipfile.ZipFile(video_zip) as archive:
            video_names.extend(name for name in _basenames(archive.namelist()) if name.lower().endswith(".mp4"))
    video_names = sorted(set(video_names))
    with zipfile.ZipFile(annotations_zip) as archive:
        annotation_stems = {
            name.removesuffix(".viratdata.objects.txt")
            for name in _basenames(archive.namelist())
            if name.endswith(".viratdata.objects.txt")
        }
    annotation_scenes = sorted({
        f"VIRAT_S_{match.group(1)}"
        for stem in annotation_stems
        if (match := re.match(r"^VIRAT_S_(\d{4})", stem)) is not None
    })

    scenes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sequence_prefixes: set[str] = set()
    unmatched_names: list[str] = []
    for name in video_names:
        match = VIRAT_VIDEO_RE.match(name)
        if match is None:
            unmatched_names.append(name)
            continue
        sequence_code, segment, start, end = match.groups()
        scene_id = f"VIRAT_S_{sequence_code[:4]}"
        sequence_id = f"VIRAT_S_{sequence_code}"
        sequence_prefixes.add(sequence_id)
        stem = name[:-4]
        duration = int(end) - int(start) if start is not None and end is not None else None
        scenes[scene_id].append({
            "video": name,
            "sequence_id": sequence_id,
            "segment": segment,
            "duration_seconds_from_name": duration,
            "objects_annotation_present": stem in annotation_stems,
        })

    rows: list[dict[str, Any]] = []
    for scene_id, clips in sorted(scenes.items()):
        known_durations = [int(row["duration_seconds_from_name"]) for row in clips if row["duration_seconds_from_name"] is not None]
        all_annotated = all(bool(row["objects_annotation_present"]) for row in clips)
        used = scene_id in USED_VIRAT_SCENES
        media_present_candidate = not used and all_annotated and len(clips) >= 1
        duration_confirmed_candidate = (
            not used
            and all_annotated
            and len(clips) >= 4
            and len(known_durations) == len(clips)
            and sum(known_durations) >= 120
        )
        rows.append({
            "scene_id": scene_id,
            "sequence_prefix_count": len({row["sequence_id"] for row in clips}),
            "clip_count": len(clips),
            "known_duration_clip_count": len(known_durations),
            "total_duration_seconds_from_names": sum(known_durations),
            "all_video_members_have_objects_annotation": all_annotated,
            "used_in_existing_development": used,
            "media_present_candidate": media_present_candidate,
            "duration_confirmed_candidate": duration_confirmed_candidate,
            "clips": clips,
        })
    candidates = [row for row in rows if row["media_present_candidate"]]
    candidates.sort(key=lambda row: (-int(row["clip_count"]), -int(row["total_duration_seconds_from_names"]), str(row["scene_id"])))
    return {
        "video_member_count": len(video_names),
        "video_archives": [{"name": path.name, "bytes": path.stat().st_size} for path in video_zips],
        "sequence_prefix_count": len(sequence_prefixes),
        "physical_scene_count": len(rows),
        "annotation_scene_count": len(annotation_scenes),
        "available_video_scenes": sorted(scenes),
        "annotation_scenes_without_video_archive": sorted(set(annotation_scenes) - set(scenes)),
        "identity_rule": "official Release 2 rule: first four digits XXYY identify a scene; remaining digits encode sequence/time",
        "official_reference_filename": "VIRAT_Video_Dataset_Release2.0_Introduction_v1.0.pdf",
        "unmatched_video_names": unmatched_names,
        "fully_annotation_mapped_video_count": sum(
            bool(clip["objects_annotation_present"]) for row in rows for clip in row["clips"]
        ),
        "candidate_rule": "unused physical scene with media present and all video members mapped to objects annotations",
        "structural_candidate_count": len(candidates),
        "duration_confirmed_candidate_count": sum(bool(row["duration_confirmed_candidate"]) for row in candidates),
        "recommended_six": [str(row["scene_id"]) for row in candidates[:6]],
        "scenes": rows,
    }


def _chokepoint_inventory(root: Path) -> dict[str, Any]:
    scene_archives: dict[str, set[int]] = defaultdict(set)
    for path in root.glob("*.tar.xz"):
        match = CHOKE_ARCHIVE_RE.match(path.name)
        if match:
            scene_archives[match.group(1)].add(int(match.group(2)))

    groundtruth = root / "groundtruth.tar.xz"
    with tarfile.open(groundtruth, mode="r:xz") as archive:
        xml_names = sorted(name for name in _basenames(member.name for member in archive.getmembers()) if name.endswith(".xml"))

    views: dict[str, dict[str, Any]] = {}
    unmatched_xml: list[str] = []
    blocks: dict[str, set[str]] = defaultdict(set)
    sessions: dict[str, set[int]] = defaultdict(set)
    for name in xml_names:
        match = CHOKE_XML_RE.match(name)
        if match is None:
            unmatched_xml.append(name)
            continue
        group, session_raw, camera, variant = match.groups()
        view_id = f"{group}_C{camera}"
        session = int(session_raw)
        block_id = f"S{session}" + (f".{variant}" if variant else "")
        blocks[view_id].add(block_id)
        sessions[view_id].add(session)
    for group in sorted(scene_archives):
        for camera in (1, 2, 3):
            view_id = f"{group}_C{camera}"
            archive_sessions = sorted(scene_archives[group])
            annotated_sessions = sorted(sessions.get(view_id, set()))
            views[view_id] = {
                "view_id": view_id,
                "portal": group[:2],
                "direction": group[2],
                "camera": camera,
                "archive_sessions": archive_sessions,
                "annotated_sessions": annotated_sessions,
                "annotated_block_count": len(blocks.get(view_id, set())),
                "structural_candidate": len(archive_sessions) >= 4 and len(annotated_sessions) >= 4,
            }
    candidates = [view_id for view_id, row in views.items() if row["structural_candidate"]]
    recommended = [view_id for view_id in CHOKEPOINT_SHORTLIST if view_id in candidates]
    return {
        "scene_archive_count": sum(len(rows) for rows in scene_archives.values()),
        "xml_count": len(xml_names),
        "unmatched_xml_names": unmatched_xml,
        "physical_view_count": len(views),
        "independent_location_count": len({row["portal"] for row in views.values()}),
        "independent_locations": sorted({row["portal"] for row in views.values()}),
        "independence_note": "C1/C2/C3 and E/L add fixed-camera views but remain clustered within portals P1 and P2",
        "structural_candidate_count": len(candidates),
        "recommended_six": recommended,
        "shortlist_balance": {"P1": 3, "P2": 3, "E": 3, "L": 3, "C1": 2, "C2": 2, "C3": 2},
        "views": [views[key] for key in sorted(views)],
    }


def build_inventory(*, virat_root: Path, chokepoint_root: Path) -> dict[str, Any]:
    video_zips = sorted(virat_root.glob("videos-*.zip"))
    virat = _virat_inventory(video_zips, virat_root / "annotations-and-docs.zip")
    chokepoint = _chokepoint_inventory(chokepoint_root)
    available_new_locations = int(virat["structural_candidate_count"]) + int(chokepoint["independent_location_count"])
    structural_target_met = available_new_locations >= 8
    return {
        "schema_version": "fieldpack.dataset-expansion-metadata-audit.v1",
        "evidence_status": "METADATA_ONLY_DEVELOPMENT_PLANNING",
        "raw_media_extracted": False,
        "model_inference_run": False,
        "ground_truth_values_read": False,
        "target": {"new_independent_physical_scenes": 8, "camera_view_stress_panel": 12},
        "virat": virat,
        "chokepoint": chokepoint,
        "readiness": {
            "structural_target_met": structural_target_met,
            "available_new_independent_physical_scenes": available_new_locations,
            "new_independent_scene_gap": max(0, 8 - available_new_locations),
            "gpu_experiment_allowed": False,
            "status": "STRUCTURALLY_SUFFICIENT_ADMISSION_NOT_CLOSED" if structural_target_met else "STRUCTURALLY_INSUFFICIENT_EXISTING_ARCHIVES",
            "blockers": ([
                "three_new_virat_scenes_have_media_but_require_hash_and_media_probe",
                "five_annotated_virat_scenes_have_no_video_archive_on_server",
                "chokepoint_has_12_views_but_only_2_independent_portal_locations",
                "processing_permission_not_sealed",
                "selected_archives_not_extracted_or_media_probed",
                "selected_frame_timeline_not_frozen",
                "selected_camera_annotation_mapping_not_fully_validated_after_extraction",
            ] if not structural_target_met else [
                "processing_permission_not_sealed",
                "selected_archives_not_extracted_or_media_probed",
                "selected_frame_timeline_not_frozen",
                "selected_camera_annotation_mapping_not_fully_validated_after_extraction",
            ]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--virat-root", type=Path, required=True)
    parser.add_argument("--chokepoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refuse_existing_output:{args.output}")
    result = build_inventory(virat_root=args.virat_root, chokepoint_root=args.chokepoint_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "virat_candidates": result["virat"]["structural_candidate_count"],
        "chokepoint_candidates": result["chokepoint"]["structural_candidate_count"],
        "status": result["readiness"]["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
