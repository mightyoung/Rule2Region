"""Audit official DIVA/KPF VIRAT annotations proposed as a separate supplement."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REPOSITORY = "https://gitlab.kitware.com/viratdata/viratannotations"
COMMIT = "f1eff0d7db0dc0b25e1c99951ac0907521f3a216"
FOUND = {
    "VIRAT_S_000200_02_000479_000635": "validate",
    "VIRAT_S_000200_05_001525_001575": "train",
    "VIRAT_S_000201_04_000682_000822": "validate",
    "VIRAT_S_000203_05_001122_001159": "train",
    "VIRAT_S_000205_03_000860_000922": "validate",
    "VIRAT_S_000205_05_001092_001124": "validate",
}
ABSENT = (
    "VIRAT_S_000203_09_001789_001842",
    "VIRAT_S_000207_05_001125_001193",
)
SUFFIXES = ("activities", "geom", "regions", "types")
GEOM_RE = re.compile(
    r"geom:\s*\{\s*id1:\s*(\d+),\s*id0:\s*\d+,\s*ts0:\s*(\d+),.*?g0:\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)"
)
TYPE_RE = re.compile(r"types:\s*\{\s*id1:\s*(\d+)\s*,\s*cset3:\s*\{\s*([A-Za-z_]+):")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_types(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TYPE_RE.search(line)
        if match:
            result[int(match.group(1))] = match.group(2)
    return result


def parse_geom(path: Path) -> list[tuple[int, int, int, int, int, int]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            match = GEOM_RE.search(line)
            if match:
                rows.append(tuple(map(int, match.groups())))
    return rows


def legacy_rows(path: Path) -> set[tuple[int, int, int, int, int, int, int]]:
    rows = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        columns = raw.split()
        if len(columns) != 8:
            continue
        track, _duration, frame, x, y, width, height, kind = map(int, columns)
        rows.add((track, frame, x, y, width, height, kind))
    return rows


def audit(root: Path, video_audit_path: Path, legacy_compare_path: Path) -> dict[str, Any]:
    video_audit = json.loads(video_audit_path.read_text(encoding="utf-8"))
    frame_counts = {row["source_video_id"]: row["media"]["frame_count"] for row in video_audit["videos"]}
    files = []
    annotations = []
    for source_id, split in FOUND.items():
        paths = {suffix: root / "raw" / split / f"{source_id}.{suffix}.yml" for suffix in SUFFIXES}
        if not all(path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"incomplete_kpf_group:{source_id}")
        types = parse_types(paths["types"])
        geom = parse_geom(paths["geom"])
        if not geom:
            raise ValueError(f"empty_geometry:{source_id}")
        unknown_tracks = sorted({row[0] for row in geom} - set(types))
        frame_max = max(row[1] for row in geom)
        frame_min = min(row[1] for row in geom)
        annotations.append(
            {
                "source_video_id": source_id,
                "split": split,
                "track_count": len({row[0] for row in geom}),
                "detection_count": len(geom),
                "person_track_count": sum(kind == "Person" for kind in types.values()),
                "person_detection_count": sum(types.get(row[0]) == "Person" for row in geom),
                "type_counts": dict(sorted(Counter(types.values()).items())),
                "minimum_frame": frame_min,
                "maximum_frame": frame_max,
                "video_frame_count": frame_counts[source_id],
                "all_frames_in_range": frame_min >= 0 and frame_max < frame_counts[source_id],
                "unknown_geometry_track_ids": unknown_tracks,
            }
        )
        for suffix, path in paths.items():
            files.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "source_url": (
                        "https://gitlab.kitware.com/api/v4/projects/4997/repository/files/"
                        f"{split}%2F{source_id}.{suffix}.yml/raw?ref={COMMIT}"
                    ),
                }
            )

    compare_id = "VIRAT_S_000200_03_000657_000899"
    compare_types = parse_types(root / "raw/train" / f"{compare_id}.types.yml")
    compare_geom = parse_geom(root / "raw/train" / f"{compare_id}.geom.yml")
    kpf_legacy_shape = set()
    for track, frame, x1, y1, x2, y2 in compare_geom:
        kind = compare_types.get(track)
        numeric = 1 if kind == "Person" else 2 if kind == "Vehicle" else None
        if numeric is not None:
            kpf_legacy_shape.add((track, frame, x1, y1, x2 - x1, y2 - y1, numeric))
    legacy = legacy_rows(legacy_compare_path)
    intersection = kpf_legacy_shape & legacy
    compatibility = {
        "source_video_id": compare_id,
        "legacy_row_count": len(legacy),
        "kpf_person_vehicle_row_count": len(kpf_legacy_shape),
        "exact_row_intersection_count": len(intersection),
        "legacy_exact_match_share": len(intersection) / len(legacy) if legacy else 0,
        "kpf_exact_match_share": len(intersection) / len(kpf_legacy_shape) if kpf_legacy_shape else 0,
        "same_annotation_schema": False,
    }
    aligned = [row for row in annotations if row["all_frames_in_range"]]
    mismatched = [row["source_video_id"] for row in annotations if not row["all_frames_in_range"]]
    return {
        "schema_version": "fieldpack.virat-official-diva-kpf-supplement.v1",
        "repository": REPOSITORY,
        "commit": COMMIT,
        "format": "KPF YAML from IARPA DIVA annotations",
        "requested_missing_video_count": 8,
        "official_kpf_group_found_count": len(FOUND),
        "official_kpf_group_absent_count": len(ABSENT),
        "absent_source_video_ids": list(ABSENT),
        "files": files,
        "annotations": annotations,
        "frame_aligned_group_count": len(aligned),
        "frame_mismatched_source_video_ids": mismatched,
        "all_downloaded_frames_in_video_range": all(row["all_frames_in_range"] for row in annotations),
        "compatibility_check": compatibility,
        "integration_status": "SEPARATE_SUPPLEMENT_DO_NOT_MIX_WITH_RELEASE2_OBJECTS",
        "gpu_experiment_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--video-audit", type=Path, required=True)
    parser.add_argument("--legacy-compare", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.video_audit, args.legacy_compare)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "found": result["official_kpf_group_found_count"],
        "absent": result["official_kpf_group_absent_count"],
        "compatibility": result["compatibility_check"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
