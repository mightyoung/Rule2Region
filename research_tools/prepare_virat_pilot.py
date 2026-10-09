"""Prepare a deterministic, subset-scoped VIRAT Ground engineering pilot.

The tool deliberately selects sources without reading labels, extracts only the
selected video/annotation pairs, samples temporal-bin midpoint frames, and emits
MOT-format labels so the existing cascade runner can consume the pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


CAMERA_PATTERN = re.compile(r"^(VIRAT_S_\d{6})_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_spaced_indices(total: int, count: int) -> tuple[int, ...]:
    if total < count or count < 1:
        raise ValueError("insufficient_items_for_selection")
    return tuple(((2 * index + 1) * total - 1) // (2 * count) for index in range(count))


def select_video_members(
    members: Iterable[str],
    *,
    camera_count: int,
    videos_per_camera: int,
    excluded_cameras: Iterable[str] = (),
) -> tuple[tuple[str, str], ...]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for member in sorted(members):
        if not member.lower().endswith(".mp4"):
            continue
        source_id = PurePosixPath(member).stem
        match = CAMERA_PATTERN.match(source_id)
        if match:
            grouped[match.group(1)].append(member)
    excluded = set(excluded_cameras)
    eligible = sorted(
        camera
        for camera, videos in grouped.items()
        if len(videos) >= videos_per_camera and camera not in excluded
    )
    selected_cameras = [eligible[index] for index in evenly_spaced_indices(len(eligible), camera_count)]
    selected: list[tuple[str, str]] = []
    for camera in selected_cameras:
        videos = grouped[camera]
        selected.extend((camera, videos[index]) for index in evenly_spaced_indices(len(videos), videos_per_camera))
    return tuple(selected)


def annotation_member_map(members: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    suffix = ".viratdata.objects.txt"
    for member in sorted(members):
        name = PurePosixPath(member).name
        if name.endswith(suffix):
            source_id = name[: -len(suffix)]
            result.setdefault(source_id, member)
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    encoded = (json.dumps(unsigned, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    final = (json.dumps({**unsigned, "manifest_sha256": digest}, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != final:
            raise FileExistsError(f"immutable_manifest_differs:{path}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(final)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return digest


def _extract_member(archive: zipfile.ZipFile, member: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, target.open("xb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)


def _person_rows(annotation_text: str, selected: set[int]) -> dict[int, list[str]]:
    rows: dict[int, list[str]] = defaultdict(list)
    for line_number, raw in enumerate(annotation_text.splitlines(), start=1):
        columns = raw.split()
        if not columns or raw.lstrip().startswith("#"):
            continue
        if len(columns) != 8:
            raise ValueError(f"annotation_line_{line_number}_expected_8_columns")
        track_id, _duration, frame, x, y, width, height, object_type = columns
        frame_index = int(frame)
        if frame_index in selected and int(object_type) == 1 and float(width) > 0 and float(height) > 0:
            rows[frame_index].append(
                f"{frame_index + 1},{int(track_id) + 1},{float(x)},{float(y)},{float(width)},{float(height)},1,1,1"
            )
    return rows


def prepare(
    video_zip: Path,
    annotation_zip: Path,
    output: Path,
    *,
    camera_count: int,
    videos_per_camera: int,
    frames_per_video: int,
    excluded_cameras: Iterable[str] = (),
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refuse_existing_output:{output}")
    output.mkdir(parents=True)
    sources: list[dict[str, Any]] = []
    try:
        import cv2  # type: ignore

        with zipfile.ZipFile(video_zip) as videos, zipfile.ZipFile(annotation_zip) as annotations:
            excluded = tuple(sorted(set(excluded_cameras)))
            selected = select_video_members(
                videos.namelist(),
                camera_count=camera_count,
                videos_per_camera=videos_per_camera,
                excluded_cameras=excluded,
            )
            annotation_members = annotation_member_map(annotations.namelist())
            for camera_id, video_member in selected:
                source_id = PurePosixPath(video_member).stem
                annotation_member = annotation_members.get(source_id)
                if annotation_member is None:
                    raise FileNotFoundError(f"annotation_missing:{source_id}")
                sequence = output / "sequences" / source_id
                video_path = sequence / "source.mp4"
                annotation_path = sequence / "objects.txt"
                _extract_member(videos, video_member, video_path)
                _extract_member(annotations, annotation_member, annotation_path)
                capture = cv2.VideoCapture(str(video_path))
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if not capture.isOpened() or total_frames < frames_per_video or fps <= 0 or width <= 0 or height <= 0:
                    capture.release()
                    raise ValueError(f"invalid_media_metadata:{source_id}")
                frame_indices = evenly_spaced_indices(total_frames, frames_per_video)
                frames_dir = sequence / "img1"
                frames_dir.mkdir()
                for frame_index in frame_indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, image = capture.read()
                    if not ok:
                        capture.release()
                        raise IOError(f"frame_decode_failed:{source_id}:{frame_index}")
                    if not cv2.imwrite(str(frames_dir / f"{frame_index + 1:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        capture.release()
                        raise IOError(f"frame_write_failed:{source_id}:{frame_index}")
                capture.release()
                selected_set = set(frame_indices)
                gt_rows = _person_rows(annotation_path.read_text(encoding="utf-8"), selected_set)
                gt_path = sequence / "gt.txt"
                gt_path.write_text("\n".join(row for frame in frame_indices for row in gt_rows.get(frame, ())) + "\n", encoding="utf-8")
                sources.append(
                    {
                        "dataset_id": "virat-ground-release2",
                        "camera_id": camera_id,
                        "source_video_id": source_id,
                        "video_zip_member": video_member,
                        "annotation_zip_member": annotation_member,
                        "total_frames": total_frames,
                        "fps": fps,
                        "width": width,
                        "height": height,
                        "video_sha256": sha256_file(video_path),
                        "annotation_sha256": sha256_file(annotation_path),
                        "gt_sha256": sha256_file(gt_path),
                        "sampled_frame_indices": list(frame_indices),
                        "sampled_person_rows": sum(len(value) for value in gt_rows.values()),
                    }
                )
        manifest = {
            "schema_version": "fieldpack.virat-subset-pilot.v1",
            "dataset": "VIRAT Ground Release 2 videos-01 subset",
            "selection_policy": "label_blind_sorted_camera_and_video_equal_bin_midpoints",
            "excluded_camera_ids": list(excluded),
            "mot_compatibility_transform": "VIRAT object_id and source frame mapped to MOT track_id=object_id+1 and mot_frame=source_frame+1",
            "camera_count": camera_count,
            "videos_per_camera": videos_per_camera,
            "frames_per_video": frames_per_video,
            "sample_count": camera_count * videos_per_camera * frames_per_video,
            "source_archives": {
                "video_zip": {"path": str(video_zip), "sha256": sha256_file(video_zip)},
                "annotation_zip": {"path": str(annotation_zip), "sha256": sha256_file(annotation_zip)},
            },
            "sources": sources,
        }
        digest = _atomic_json(output / "pilot-manifest.json", manifest)
        return {"output": str(output), "manifest_sha256": digest, "sources": len(sources), "samples": manifest["sample_count"]}
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare deterministic VIRAT engineering pilot")
    parser.add_argument("--video-zip", type=Path, required=True)
    parser.add_argument("--annotation-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-count", type=int, default=3)
    parser.add_argument("--videos-per-camera", type=int, default=2)
    parser.add_argument("--frames-per-video", type=int, default=200)
    parser.add_argument(
        "--exclude-camera",
        action="append",
        default=[],
        help="Camera id excluded before label-blind midpoint selection; repeatable.",
    )
    args = parser.parse_args()
    receipt = prepare(
        args.video_zip,
        args.annotation_zip,
        args.output,
        camera_count=args.camera_count,
        videos_per_camera=args.videos_per_camera,
        frames_per_video=args.frames_per_video,
        excluded_cameras=args.exclude_camera,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
