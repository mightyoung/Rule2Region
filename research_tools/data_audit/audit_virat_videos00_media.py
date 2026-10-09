"""Audit unpacked VIRAT videos-00 media and its object annotations.

This is a development-only integrity and decodability check.  It does not run
model inference and does not use a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "fieldpack.virat-videos00-media-audit.v1"
EXPECTED_SCENES = ("0000", "0001", "0002")
SCENE_PATTERN = re.compile(r"^VIRAT_S_(\d{4})\d{2}(?:_|$)")
ANNOTATION_SUFFIX = ".viratdata.objects.txt"
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_id_from_stem(stem: str) -> str:
    match = SCENE_PATTERN.match(stem)
    if match is None:
        raise ValueError(f"unrecognized_virat_video_name:{stem}")
    return match.group(1)


def annotation_index(root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(root.rglob(f"*{ANNOTATION_SUFFIX}")):
        stem = path.name.removesuffix(ANNOTATION_SUFFIX)
        if stem in indexed:
            raise ValueError(f"duplicate_annotation:{stem}")
        indexed[stem] = path
    return indexed


def parse_annotation(path: Path, *, total_frames: int) -> dict[str, Any]:
    row_count = 0
    person_row_count = 0
    track_ids: set[int] = set()
    frame_indices: list[int] = []
    malformed_lines: list[int] = []

    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, start=1):
            columns = raw.split()
            if not columns or raw.lstrip().startswith("#"):
                continue
            if len(columns) != 8:
                malformed_lines.append(line_number)
                continue
            try:
                track_id = int(columns[0])
                int(columns[1])
                frame_index = int(columns[2])
                float(columns[3])
                float(columns[4])
                float(columns[5])
                float(columns[6])
                object_type = int(columns[7])
            except ValueError:
                malformed_lines.append(line_number)
                continue
            row_count += 1
            track_ids.add(track_id)
            frame_indices.append(frame_index)
            if object_type == 1:
                person_row_count += 1

    minimum = min(frame_indices) if frame_indices else None
    maximum = max(frame_indices) if frame_indices else None
    parse_ok = not malformed_lines
    frames_in_range = parse_ok and (
        not frame_indices or (minimum is not None and minimum >= 0 and maximum is not None and maximum < total_frames)
    )
    return {
        "sha256": sha256_file(path),
        "row_count": row_count,
        "unique_trajectory_count": len(track_ids),
        "person_row_count": person_row_count,
        "minimum_frame_index": minimum,
        "maximum_frame_index": maximum,
        "malformed_line_count": len(malformed_lines),
        "malformed_line_numbers": malformed_lines,
        "parse_ok": parse_ok,
        "frame_indices_in_range": frames_in_range,
    }


def _fourcc_text(value: int) -> str:
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4)).rstrip("\x00")


def inspect_video(path: Path) -> dict[str, Any]:
    import cv2  # type: ignore

    capture = cv2.VideoCapture(str(path))
    try:
        opened = bool(capture.isOpened())
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = _fourcc_text(int(capture.get(cv2.CAP_PROP_FOURCC)))
        first_frame_decodable = False
        last_frame_decodable = False
        if opened and total_frames > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            first_frame_decodable = bool(capture.read()[0])
            capture.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            last_frame_decodable = bool(capture.read()[0])
    finally:
        capture.release()

    metadata_valid = opened and total_frames > 0 and fps > 0 and width > 0 and height > 0
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "opened": opened,
        "frame_count": total_frames,
        "fps": fps,
        "duration_seconds": total_frames / fps if fps > 0 else None,
        "width": width,
        "height": height,
        "fourcc": fourcc,
        "metadata_valid": metadata_valid,
        "first_frame_decodable": first_frame_decodable,
        "last_frame_decodable": last_frame_decodable,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def audit(
    *,
    media_root: Path,
    annotations_root: Path,
    output_json: Path,
    archive_path: str,
    archive_sha256: str,
) -> dict[str, Any]:
    if not media_root.is_dir():
        raise FileNotFoundError(f"media_root_missing:{media_root}")
    if not annotations_root.is_dir():
        raise FileNotFoundError(f"annotations_root_missing:{annotations_root}")
    if not SHA256_PATTERN.fullmatch(archive_sha256):
        raise ValueError("archive_sha256_must_be_64_hex_characters")

    videos = sorted(path for path in media_root.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4")
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in videos:
        grouped[scene_id_from_stem(path.stem)].append(path)
    observed_scenes = tuple(sorted(grouped))
    if observed_scenes != EXPECTED_SCENES:
        raise ValueError(f"unexpected_scene_set:expected={','.join(EXPECTED_SCENES)}:actual={','.join(observed_scenes)}")

    annotations = annotation_index(annotations_root)
    video_results: list[dict[str, Any]] = []
    for scene_id in EXPECTED_SCENES:
        for video_path in grouped[scene_id]:
            source_id = video_path.stem
            media = inspect_video(video_path)
            annotation_path = annotations.get(source_id)
            annotation_matched = annotation_path is not None
            annotation = (
                parse_annotation(annotation_path, total_frames=media["frame_count"])
                if annotation_path is not None
                else None
            )
            checks_passed = bool(
                media["metadata_valid"]
                and media["first_frame_decodable"]
                and media["last_frame_decodable"]
                and annotation_matched
                and annotation is not None
                and annotation["parse_ok"]
                and annotation["frame_indices_in_range"]
            )
            video_results.append(
                {
                    "scene_id": scene_id,
                    "source_video_id": source_id,
                    "media_relative_path": video_path.relative_to(media_root).as_posix(),
                    "annotation_relative_path": (
                        annotation_path.relative_to(annotations_root).as_posix() if annotation_path is not None else None
                    ),
                    "annotation_matched": annotation_matched,
                    "checks_passed": checks_passed,
                    "media": media,
                    "annotation": annotation,
                }
            )

    scene_results: list[dict[str, Any]] = []
    for scene_id in EXPECTED_SCENES:
        rows = [row for row in video_results if row["scene_id"] == scene_id]
        annotation_rows = [row["annotation"] for row in rows if row["annotation"] is not None]
        scene_results.append(
            {
                "scene_id": scene_id,
                "video_count": len(rows),
                "total_bytes": sum(row["media"]["size_bytes"] for row in rows),
                "total_frames": sum(row["media"]["frame_count"] for row in rows),
                "annotation_row_count": sum(row["row_count"] for row in annotation_rows),
                "unique_trajectory_count_sum": sum(row["unique_trajectory_count"] for row in annotation_rows),
                "person_row_count": sum(row["person_row_count"] for row in annotation_rows),
                "fps_values": sorted({row["media"]["fps"] for row in rows}),
                "resolutions": sorted({f'{row["media"]["width"]}x{row["media"]["height"]}' for row in rows}),
                "all_checks_passed": all(row["checks_passed"] for row in rows),
            }
        )

    annotations_present = [row["annotation"] for row in video_results if row["annotation"] is not None]
    all_checks_passed = all(row["checks_passed"] for row in video_results)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_scope": "DEVELOPMENT_ONLY",
        "raw_media_extracted": True,
        "gpu_used": False,
        "model_inference": False,
        "model_inference_run": False,
        "archive_sha256": archive_sha256.lower(),
        "scene_ids": list(observed_scenes),
        "scene_count": len(scene_results),
        "video_count": len(video_results),
        "all_checks_passed": all_checks_passed,
        "expected_scene_ids": list(EXPECTED_SCENES),
        "observed_scene_ids": list(observed_scenes),
        "source_archive": {"path": archive_path, "sha256": archive_sha256.lower()},
        "inputs": {"media_root": str(media_root), "annotations_root": str(annotations_root)},
        "summary": {
            "scene_count": len(scene_results),
            "video_count": len(video_results),
            "total_bytes": sum(row["media"]["size_bytes"] for row in video_results),
            "total_frames": sum(row["media"]["frame_count"] for row in video_results),
            "annotation_match_count": sum(row["annotation_matched"] for row in video_results),
            "annotation_row_count": sum(row["row_count"] for row in annotations_present),
            "unique_trajectory_count_sum": sum(row["unique_trajectory_count"] for row in annotations_present),
            "person_row_count": sum(row["person_row_count"] for row in annotations_present),
            "all_checks_passed": all_checks_passed,
        },
        "scenes": scene_results,
        "videos": video_results,
    }
    _atomic_write_json(output_json, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit unpacked VIRAT videos-00 media")
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--output-json", "--output", dest="output_json", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-path")
    parser.add_argument("--archive-sha256")
    args = parser.parse_args()
    if args.archive is None and args.archive_path is None:
        parser.error("one of --archive or --archive-path is required")
    archive_path = args.archive_path or str(args.archive)
    archive_sha256 = args.archive_sha256
    if archive_sha256 is None:
        if args.archive is None:
            parser.error("--archive-sha256 is required when --archive-path is used")
        archive_sha256 = sha256_file(args.archive)
    result = audit(
        media_root=args.media_root,
        annotations_root=args.annotations_root,
        output_json=args.output_json,
        archive_path=archive_path,
        archive_sha256=archive_sha256,
    )
    print(json.dumps(result["summary"], sort_keys=True))
    return 0 if result["summary"]["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
