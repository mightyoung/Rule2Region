"""Prepare label-blind contiguous VIRAT microclips for temporal proposal research."""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from research_tools.prepare_virat_pilot import (
    CAMERA_PATTERN,
    _atomic_json,
    _extract_member,
    _person_rows,
    annotation_member_map,
    evenly_spaced_indices,
    sha256_file,
)

CONFIRMATORY_FRAMES = 64
CONFIRMATORY_VAL_COUNT = 32
SAME_VIEW = {"VIRAT_S_010112": "VIRAT_S_010111"}
DEVELOPMENT_VAL_COUNT = 16


def freeze_time_split(sampled_frame_indices: Iterable[int], *, val_count: int) -> dict[str, list[int]]:
    if val_count <= DEVELOPMENT_VAL_COUNT:
        raise ValueError("confirmatory_val_count_must_exceed_development_16")
    frames = [int(index) + 1 for index in sampled_frame_indices]
    if len(frames) <= val_count:
        raise ValueError("clip_too_short_for_confirmatory_split")
    return {
        "frames": frames,
        "history_frames": frames[:-val_count],
        "validation_frames": frames[-val_count:],
    }


def write_confirmatory_split(
    output: Path,
    sources: list[dict[str, Any]],
    *,
    val_count: int,
    sealed_camera: str,
) -> dict[str, Any]:
    sequences: dict[str, Any] = {}
    for row in sources:
        camera = str(row["camera_id"])
        sequence = str(row["source_video_id"])
        if sealed_camera in camera or sealed_camera in sequence:
            raise ValueError("sealed_camera_present")
        split = freeze_time_split(row["sampled_frame_indices"], val_count=val_count)
        sequences[sequence] = {
            "camera_id": camera,
            "history_frames": split["history_frames"],
            "validation_frames": split["validation_frames"],
        }
    payload = {
        "schema_version": "fieldpack.virat-h1-confirmatory-split.v1",
        "frames_per_video": CONFIRMATORY_FRAMES,
        "validation_frame_count": int(val_count),
        "selection_policy": "label_blind_contiguous_midpoint_then_tail_validation",
        "sealed_camera_id": sealed_camera,
        "same_view": dict(SAME_VIEW),
        "pack_status": "SEALED_SPLIT_BEFORE_STACK",
        "official_h1": False,
        "sequences": sequences,
    }
    digest = _atomic_json(output / "split-manifest.json", payload)
    (output / "split-manifest.sha256").write_text(digest + "\n", encoding="utf-8")
    return {"path": str(output / "split-manifest.json"), "sha256": digest, "sequence_count": len(sequences)}


def contiguous_midpoint_indices(total: int, count: int) -> tuple[int, ...]:
    if total < count or count < 1:
        raise ValueError("insufficient_frames_for_microclip")
    start = max(0, min(total - count, (total - count) // 2))
    return tuple(range(start, start + count))


def select_for_cameras(
    members: Iterable[str], cameras: Iterable[str], *, videos_per_camera: int
) -> tuple[tuple[str, str], ...]:
    requested = tuple(sorted(set(cameras)))
    grouped: dict[str, list[str]] = {camera: [] for camera in requested}
    for member in sorted(members):
        if not member.lower().endswith(".mp4"):
            continue
        match = CAMERA_PATTERN.match(PurePosixPath(member).stem)
        if match and match.group(1) in grouped:
            grouped[match.group(1)].append(member)
    selected: list[tuple[str, str]] = []
    for camera in requested:
        videos = grouped[camera]
        if len(videos) < videos_per_camera:
            raise ValueError(f"camera_has_insufficient_videos:{camera}")
        selected.extend(
            (camera, videos[index])
            for index in evenly_spaced_indices(len(videos), videos_per_camera)
        )
    return tuple(selected)


def prepare(
    video_zip: Path,
    annotation_zip: Path,
    output: Path,
    *,
    cameras: Iterable[str],
    sealed_camera: str,
    videos_per_camera: int,
    frames_per_video: int,
    val_count: int | None = None,
) -> dict[str, Any]:
    development_cameras = tuple(sorted(set(cameras)))
    if sealed_camera in development_cameras:
        raise ValueError("sealed_camera_in_development_set")
    if val_count is not None:
        if frames_per_video != CONFIRMATORY_FRAMES or val_count != CONFIRMATORY_VAL_COUNT:
            raise ValueError("confirmatory_pack_must_use_frozen_64_32")
    if output.exists():
        raise FileExistsError(f"refuse_existing_output:{output}")
    output.mkdir(parents=True)
    sources: list[dict[str, Any]] = []
    try:
        import cv2  # type: ignore

        with zipfile.ZipFile(video_zip) as videos, zipfile.ZipFile(annotation_zip) as annotations:
            selected = select_for_cameras(videos.namelist(), development_cameras, videos_per_camera=videos_per_camera)
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
                if not capture.isOpened() or fps <= 0 or width <= 0 or height <= 0:
                    capture.release()
                    raise ValueError(f"invalid_media_metadata:{source_id}")
                frame_indices = contiguous_midpoint_indices(total_frames, frames_per_video)
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
                sources.append({
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
                })
        manifest = {
            "schema_version": "fieldpack.virat-dcrq-microclips.v1",
            "dataset": "VIRAT Ground Release 2 videos-01 subset",
            "selection_policy": "label_blind_fixed_development_cameras_equal_bin_video_midpoints_contiguous_frame_midpoint",
            "development_camera_ids": list(development_cameras),
            "sealed_confirmation_camera_id": sealed_camera,
            "sealed_camera_annotations_read": False,
            "videos_per_camera": videos_per_camera,
            "frames_per_video": frames_per_video,
            "validation_frame_count": val_count,
            "sample_count": len(development_cameras) * videos_per_camera * frames_per_video,
            "sources": sources,
            "source_archives": {
                "video_zip": {"path": str(video_zip), "sha256": sha256_file(video_zip)},
                "annotation_zip": {"path": str(annotation_zip), "sha256": sha256_file(annotation_zip)},
            },
        }
        digest = _atomic_json(output / "pilot-manifest.json", manifest)
        result = {"output": str(output), "manifest_sha256": digest, "sources": len(sources), "samples": manifest["sample_count"]}
        if val_count is not None:
            result["split"] = write_confirmatory_split(
                output, sources, val_count=val_count, sealed_camera=sealed_camera
            )
        return result
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare VIRAT contiguous DCRQ microclips")
    parser.add_argument("--video-zip", type=Path, required=True)
    parser.add_argument("--annotation-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", action="append", required=True)
    parser.add_argument("--sealed-camera", required=True)
    parser.add_argument("--videos-per-camera", type=int, default=2)
    parser.add_argument("--frames-per-video", type=int, default=32)
    parser.add_argument("--val-count", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.video_zip, args.annotation_zip, args.output,
        cameras=args.camera, sealed_camera=args.sealed_camera,
        videos_per_camera=args.videos_per_camera, frames_per_video=args.frames_per_video,
        val_count=args.val_count,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
