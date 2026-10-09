"""Validate the authorized VIRAT videos-00 media audit bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_ARCHIVE_SHA256 = "f1e0af1f21c48fab569afd2a33f9b2dff5ab4d3a3b819e460867f672dde1954a"
EXPECTED_SCENE_COUNTS = {"VIRAT_S_0000": 5, "VIRAT_S_0001": 2, "VIRAT_S_0002": 39}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _first(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _nested(data: dict[str, Any], *names: str) -> Any:
    for mapping in (data, data.get("summary", {}), data.get("checks", {}), data.get("safety", {})):
        if isinstance(mapping, dict):
            value = _first(mapping, *names)
            if value is not None:
                return value
    return None


def _video_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _first(data, "videos", "per_video", "video_results")
    if isinstance(rows, dict):
        rows = list(rows.values())
    return rows if isinstance(rows, list) and all(isinstance(row, dict) for row in rows) else []


def _scene_id(row: dict[str, Any]) -> str | None:
    explicit = _first(row, "scene_id", "scene")
    if isinstance(explicit, str):
        if re.fullmatch(r"\d{4}", explicit):
            return f"VIRAT_S_{explicit}"
        match = re.search(r"VIRAT_S_(\d{4})", explicit)
        if match:
            return f"VIRAT_S_{match.group(1)}"
    name = str(
        _first(row, "source_video_id", "video_id", "sequence_id", "filename", "path", "media_path") or ""
    )
    match = re.search(r"VIRAT_S_(\d{4})", name)
    return f"VIRAT_S_{match.group(1)}" if match else None


def _row_true(row: dict[str, Any], *names: str) -> bool:
    for mapping in (row, row.get("checks", {}), row.get("media", {}), row.get("annotation", {})):
        if isinstance(mapping, dict):
            value = _first(mapping, *names)
            if value is not None:
                return value is True
    return False


def _row_sha256(row: dict[str, Any]) -> Any:
    value = _first(row, "sha256", "media_sha256", "video_sha256")
    if value is not None:
        return value
    media = row.get("media", {})
    return media.get("sha256") if isinstance(media, dict) else None


def _manifest_valid(bundle_root: Path) -> tuple[bool, str]:
    manifest = bundle_root / "SHA256SUMS"
    if not manifest.is_file():
        return False, "not_present"
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw_line)
        if not match:
            return False, f"malformed_line_{line_number}"
        expected, relative_text = match.groups()
        relative = Path(relative_text.removeprefix("./"))
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] == ("extracted",):
            return False, f"unsafe_or_media_entry_{line_number}"
        target = bundle_root / relative
        if not target.is_file():
            return False, f"missing_{relative.as_posix()}"
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            return False, f"hash_mismatch_{relative.as_posix()}"
    return True, "verified"


def validate(result_path: Path, bundle_root: Path | None = None) -> dict[str, Any]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    rows = _video_rows(data)
    scene_counts = Counter(filter(None, (_scene_id(row) for row in rows)))
    archive = data.get("archive", {}) if isinstance(data.get("archive"), dict) else {}
    archive_sha = _nested(data, "archive_sha256", "source_archive_sha256") or archive.get("sha256")

    media_checks = all(
        _row_true(row, "opened", "media_opened", "video_opened")
        and _row_true(row, "metadata_valid")
        and _row_true(row, "first_frame_decodable", "first_frame_readable", "first_frame_ok")
        and _row_true(row, "last_frame_decodable", "last_frame_readable", "last_frame_ok")
        for row in rows
    )
    annotation_checks = all(
        _row_true(row, "annotation_matched", "annotation_mapped", "annotation_found", "has_annotation")
        and _row_true(row, "parse_ok")
        and _row_true(
            row,
            "frame_indices_in_range",
            "annotation_frame_bounds_valid",
            "annotation_bounds_valid",
            "frame_bounds_valid",
        )
        for row in rows
    )
    hashes_present = all(
        isinstance(_row_sha256(row), str)
        and SHA256_RE.fullmatch(_row_sha256(row)) is not None
        for row in rows
    )
    gpu_used = _nested(data, "gpu_used", "gpu")
    model_used = _nested(data, "model_inference_run", "model_inference", "model_called")
    manifest_ok, manifest_detail = _manifest_valid(bundle_root or result_path.parent)
    checks = {
        "schema": data.get("schema_version") == "fieldpack.virat-videos00-media-audit.v1",
        "archive_pinned": archive_sha == EXPECTED_ARCHIVE_SHA256,
        "three_scenes": _nested(data, "scene_count") == 3 and scene_counts == EXPECTED_SCENE_COUNTS,
        "forty_six_videos": _nested(data, "video_count") == 46 and len(rows) == 46,
        "all_checks_passed": _nested(data, "all_checks_passed") is True,
        "all_media_decodable": bool(rows) and media_checks,
        "all_annotations_mapped_and_in_range": bool(rows) and annotation_checks,
        "per_video_hashes_present": bool(rows) and hashes_present,
        "raw_media_extracted": _nested(data, "raw_media_extracted") is True,
        "no_gpu": gpu_used is False,
        "no_model_inference": model_used is False,
        "manifest": manifest_ok,
    }
    passed = all(checks.values())
    return {
        "schema_version": "fieldpack.virat-videos00-media-audit-validation.v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks": checks,
        "scene_video_counts": dict(sorted(scene_counts.items())),
        "manifest_detail": manifest_detail,
        "summary": "0000、0001、0002 三个场景的 46 个视频均完成媒体、标注和逐文件哈希检查。" if passed else "VIRAT videos-00 媒体检查结果未通过完整性校验。",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.result, args.bundle_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
