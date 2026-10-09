from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from PIL import Image


DecodeFrames = Callable[[Path, tuple[int, ...], Mapping[int, Path]], dict[str, int]]
_IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _verify_sha256(observed: str, expected: str, label: str) -> None:
    if observed.lower() != expected.lower():
        raise ValueError(f"sha256_mismatch:{label}:expected={expected}:observed={observed}")


def _resolve_unique(root: Path, basename: str) -> Path:
    direct = root / basename
    if direct.is_file():
        return direct
    matches = sorted(path for path in root.rglob(basename) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"source_missing:{basename}")
    if len(matches) != 1:
        raise ValueError(f"source_ambiguous:{basename}")
    return matches[0]


def frozen_frames(scene: Mapping[str, Any]) -> tuple[tuple[int, str, str], ...]:
    rows: list[tuple[int, str, str]] = []
    seen: set[int] = set()
    timeline_frame_count = int(scene.get("timeline_frame_count", scene["frame_count"]))
    for segment in scene["segments"]:
        segment_id = str(segment["id"])
        role = "history" if segment_id.startswith("H") else "future" if segment_id.startswith("F") else ""
        if not role:
            raise ValueError(f"unknown_segment_role:{segment_id}")
        start, end = int(segment["start"]), int(segment["end"])
        if start < 0 or end < start or end >= timeline_frame_count:
            raise ValueError(f"invalid_segment_bounds:{scene['scene_id']}:{segment_id}")
        for frame in range(start, end + 1):
            if frame in seen:
                raise ValueError(f"overlapping_frozen_frame:{scene['scene_id']}:{frame}")
            seen.add(frame)
            rows.append((frame, segment_id, role))
    return tuple(rows)


def _validate_frozen_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("status") != "frozen":
        raise ValueError("input_manifest_not_frozen")
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("input_manifest_scenes_missing")
    expected_scene_count = manifest.get("scene_count")
    if expected_scene_count is not None and int(expected_scene_count) != len(scenes):
        raise ValueError("input_manifest_scene_count_mismatch")
    expected_segments = manifest.get("segment_count_per_scene")
    expected_frames = manifest.get("frames_per_segment")
    for scene in scenes:
        segments = scene.get("segments", [])
        if expected_segments is not None and len(segments) != int(expected_segments):
            raise ValueError(f"input_manifest_segment_count_mismatch:{scene['scene_id']}")
        if expected_frames is not None:
            for segment in segments:
                observed = int(segment["end"]) - int(segment["start"]) + 1
                if observed != int(expected_frames):
                    raise ValueError(f"input_manifest_segment_length_mismatch:{scene['scene_id']}:{segment['id']}")
        frozen_frames(scene)


def parse_virat_objects(
    text: str,
    selected_frames: Iterable[int],
    *,
    frame_count: int,
    source_name: str,
) -> dict[int, list[dict[str, Any]]]:
    selected = set(selected_frames)
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for line_number, raw in enumerate(text.splitlines(), start=1):
        columns = raw.split()
        if not columns or raw.lstrip().startswith("#"):
            continue
        if len(columns) != 8:
            raise ValueError(f"virat_annotation_columns:{source_name}:{line_number}")
        track_id, _duration, frame, x, y, width, height, object_type = columns
        frame_index = int(frame)
        if frame_index == frame_count:
            continue
        if frame_index < 0 or frame_index > frame_count:
            raise ValueError(f"virat_annotation_frame_out_of_range:{source_name}:{line_number}:{frame_index}")
        box = [float(x), float(y), float(width), float(height)]
        if frame_index in selected and int(object_type) == 1 and box[2] > 0 and box[3] > 0:
            result[frame_index].append({
                "track_id": int(track_id),
                "bbox_xywh": box,
                "source_annotation": source_name,
            })
    return dict(result)


def _eye_points(person: ET.Element) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for element in (person, *person.iter()):
        label = element.tag.rsplit("}", 1)[-1]
        lowered = label.lower()
        attrs = element.attrib
        if "eye" in lowered and "x" in attrs and "y" in attrs:
            result.append({"label": label, "x": float(attrs["x"]), "y": float(attrs["y"])})
        for prefix in ("leftEye", "rightEye", "left_eye", "right_eye"):
            x_key, y_key = f"{prefix}_x", f"{prefix}_y"
            if x_key in attrs and y_key in attrs:
                result.append({"label": prefix, "x": float(attrs[x_key]), "y": float(attrs[y_key])})
    unique: dict[tuple[str, float, float], dict[str, Any]] = {}
    for point in result:
        unique[(point["label"], point["x"], point["y"])] = point
    return list(unique.values())


def parse_chokepoint_xml(
    payload: bytes,
    selected_frames: Iterable[int],
    *,
    timeline_frame_count: int,
    source_name: str,
) -> dict[int, list[dict[str, Any]]]:
    selected = set(selected_frames)
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    root = ET.fromstring(payload)
    for frame_element in root.iter("frame"):
        raw_number = frame_element.get("number", frame_element.get("id"))
        if raw_number is None:
            raise ValueError(f"chokepoint_frame_number_missing:{source_name}")
        frame_index = int(raw_number)
        if frame_index < 0 or frame_index >= timeline_frame_count:
            raise ValueError(f"chokepoint_frame_out_of_range:{source_name}:{frame_index}")
        if frame_index not in selected:
            continue
        people = list(frame_element.iter("person"))
        if not people:
            people = list(frame_element.iter("object"))
        for person in people:
            raw_track = person.get("id", person.get("track_id", "-1"))
            result[frame_index].append({
                "person_id": str(raw_track),
                "eye_points": _eye_points(person),
                "bbox_xywh": None,
                "source_annotation": source_name,
            })
    return dict(result)


def _default_decode_frames(
    media_path: Path,
    frame_indices: tuple[int, ...],
    destinations: Mapping[int, Path],
) -> dict[str, int]:
    import cv2  # type: ignore

    capture = cv2.VideoCapture(str(media_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"media_open_failed:{media_path.name}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                raise IOError(f"frame_decode_failed:{media_path.name}:{frame_index}")
            destination = destinations[frame_index]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise IOError(f"frame_write_failed:{media_path.name}:{frame_index}")
        return {"width": width, "height": height}
    finally:
        capture.release()


def _tar_member_by_basename(archive: tarfile.TarFile, basename: str) -> tarfile.TarInfo:
    matches = [member for member in archive.getmembers() if member.isfile() and Path(member.name).name == basename]
    if len(matches) != 1:
        raise FileNotFoundError(f"archive_member_count:{basename}:{len(matches)}")
    return matches[0]


def _copy_tar_member_to_temp(
    archive_path: Path,
    member_basename: str,
    expected_sha256: str,
) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(suffix=".tar.xz")
    os.close(descriptor)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        with tarfile.open(archive_path, mode="r:xz") as archive:
            member = _tar_member_by_basename(archive, member_basename)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"archive_member_unreadable:{member_basename}")
            with source, temporary.open("wb") as destination:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    destination.write(chunk)
        _verify_sha256(digest.hexdigest(), expected_sha256, member_basename)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_groundtruth_xmls(
    groundtruth_archive: Path,
    names: Iterable[str],
    expected_hashes: Iterable[str],
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    with tarfile.open(groundtruth_archive, mode="r:xz") as archive:
        for name, expected in zip(names, expected_hashes, strict=True):
            member = _tar_member_by_basename(archive, name)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"archive_member_unreadable:{name}")
            with stream:
                payload = stream.read()
            _verify_sha256(_sha256_bytes(payload), str(expected), name)
            result[name] = payload
    return result


def _explicit_choke_members(scene: Mapping[str, Any], selected: Iterable[int]) -> dict[int, str]:
    payload = scene.get("media_members")
    template = scene.get("member_name_template")
    if scene.get("media_timeline") != "sparse_jpeg_names_on_original_frame_axis":
        raise ValueError(f"chokepoint_sparse_timeline_contract_required:{scene['scene_id']}")
    if payload is None and isinstance(template, str):
        try:
            mapping = {int(frame): template.format(frame_index=int(frame)) for frame in selected}
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError(f"chokepoint_member_template_invalid:{scene['scene_id']}") from error
    elif isinstance(payload, dict):
        mapping = {int(frame): str(member) for frame, member in payload.items()}
    elif isinstance(payload, list):
        mapping = {
            int(row.get("frame_index", row.get("frame"))): str(row.get("member", row.get("name")))
            for row in payload
        }
    else:
        raise ValueError(f"chokepoint_media_members_invalid:{scene['scene_id']}")
    required = set(selected)
    missing = sorted(required - mapping.keys())
    if missing:
        raise ValueError(f"chokepoint_frozen_media_member_missing:{scene['scene_id']}:{missing[0]}")
    selected_mapping = {frame: mapping[frame] for frame in required}
    if len(set(selected_mapping.values())) != len(selected_mapping):
        raise ValueError(f"chokepoint_duplicate_media_member:{scene['scene_id']}")
    return selected_mapping


def _extract_choke_frames(
    nested_path: Path,
    selected_members: Mapping[int, str],
    destinations: Mapping[int, Path],
    expected_member_count: int,
) -> dict[str, int]:
    dimensions: set[tuple[int, int]] = set()
    with tarfile.open(nested_path, mode="r:xz") as archive:
        image_members = [
            member for member in archive.getmembers()
            if member.isfile() and Path(member.name).suffix.lower() in _IMAGE_SUFFIXES
        ]
        if len(image_members) != expected_member_count:
            raise ValueError(
                f"chokepoint_member_count_mismatch:expected={expected_member_count}:observed={len(image_members)}"
            )
        by_name = {member.name: member for member in image_members}
        for frame_index, member_name in selected_members.items():
            member = by_name.get(member_name)
            if member is None:
                raise FileNotFoundError(f"chokepoint_frozen_member_missing:{member_name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"chokepoint_frame_unreadable:{member.name}")
            with source:
                payload = source.read()
            try:
                with Image.open(BytesIO(payload)) as image:
                    dimensions.add((int(image.width), int(image.height)))
                    image.verify()
            except Exception as error:
                raise ValueError(f"chokepoint_frame_decode_failed:{member.name}") from error
            destination = destinations[frame_index]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                output.write(payload)
    if len(dimensions) != 1:
        raise ValueError(f"chokepoint_frame_dimensions_inconsistent:{sorted(dimensions)}")
    width, height = dimensions.pop()
    if width <= 0 or height <= 0:
        raise ValueError("chokepoint_frame_dimensions_invalid")
    return {"width": width, "height": height}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(
    manifest_path: Path,
    *,
    virat_media_roots: Mapping[str, Path],
    virat_annotations_root: Path,
    chokepoint_root: Path,
    output: Path,
    decode_frames: DecodeFrames | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refuse_existing_output:{output}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    _validate_frozen_manifest(manifest)
    decoder = decode_frames or _default_decode_frames
    prepared_sources: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    temporary_nested: list[Path] = []

    try:
        for scene in manifest["scenes"]:
            dataset = str(scene["dataset"])
            if dataset == "VIRAT":
                source_key = str(scene["source"]).removesuffix(".zip")
                media_root = virat_media_roots.get(source_key)
                if media_root is None:
                    raise ValueError(f"virat_media_root_missing:{source_key}")
                media_path = _resolve_unique(media_root, str(scene["media"]))
                annotation_path = _resolve_unique(virat_annotations_root, str(scene["annotation"]))
                _verify_sha256(sha256_file(media_path), str(scene["media_sha256"]), str(scene["media"]))
                _verify_sha256(sha256_file(annotation_path), str(scene["annotation_sha256"]), str(scene["annotation"]))
                pending.append({"scene": scene, "media_path": media_path, "annotation_path": annotation_path})
            elif dataset == "ChokePoint":
                outer_path = _resolve_unique(chokepoint_root, str(scene["source"]))
                nested = _copy_tar_member_to_temp(outer_path, str(scene["media"]), str(scene["media_sha256"]))
                temporary_nested.append(nested)
                xmls = _read_groundtruth_xmls(
                    _resolve_unique(chokepoint_root, "groundtruth.tar.xz"),
                    [str(value) for value in scene["annotations"]],
                    [str(value) for value in scene["annotation_sha256"]],
                )
                pending.append({"scene": scene, "nested_path": nested, "xmls": xmls})
            else:
                raise ValueError(f"unsupported_dataset:{dataset}")

        output.mkdir(parents=True)
        for item in pending:
            scene = item["scene"]
            scene_id = str(scene["scene_id"])
            frame_rows = frozen_frames(scene)
            indices = tuple(frame for frame, _segment, _role in frame_rows)
            frame_metadata = {frame: (segment, role) for frame, segment, role in frame_rows}
            scene_root = output / "scenes" / scene_id
            destinations = {frame: scene_root / "frames" / f"{frame:08d}.jpg" for frame in indices}
            if scene["dataset"] == "VIRAT":
                dimensions = decoder(item["media_path"], indices, destinations)
                boxes = parse_virat_objects(
                    item["annotation_path"].read_text(encoding="utf-8"), indices,
                    frame_count=int(scene["frame_count"]), source_name=str(scene["annotation"]),
                )
                annotation_hashes: Any = str(scene["annotation_sha256"])
            else:
                selected_members = _explicit_choke_members(scene, indices)
                dimensions = _extract_choke_frames(
                    item["nested_path"], selected_members, destinations, int(scene["media_member_count"])
                )
                boxes: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for name, payload in item["xmls"].items():
                    parsed = parse_chokepoint_xml(
                        payload, indices,
                        timeline_frame_count=int(scene["timeline_frame_count"]), source_name=name,
                    )
                    for frame, persons in parsed.items():
                        boxes[frame].extend(persons)
                boxes = dict(boxes)
                annotation_hashes = list(scene["annotation_sha256"])
            truth = {
                "schema_version": "fieldpack.fixed-camera-truth.v1",
                "scene_id": scene_id,
                "dataset": scene["dataset"],
                "frame_index_base": 0,
                "annotation_contract": (
                    "person_bbox_xywh" if scene["dataset"] == "VIRAT"
                    else "frame_person_count_and_eye_points_without_person_bbox"
                ),
                "frames": [
                    {
                        "frame_index": frame,
                        "segment_id": frame_metadata[frame][0],
                        "role": frame_metadata[frame][1],
                        "truth_count": len(boxes.get(frame, [])),
                        "truth_boxes": (
                            [
                                {"track_id": person["track_id"], "bbox_xywh": person["bbox_xywh"]}
                                for person in boxes.get(frame, [])
                            ]
                            if scene["dataset"] == "VIRAT" else []
                        ),
                        "truth_points": (
                            [
                                {
                                    "track_id": person["track_id"],
                                    "x": person["bbox_xywh"][0] + person["bbox_xywh"][2] / 2,
                                    "y": person["bbox_xywh"][1] + person["bbox_xywh"][3] / 2,
                                }
                                for person in boxes.get(frame, [])
                            ]
                            if scene["dataset"] == "VIRAT" else [
                                {
                                    "person_id": person["person_id"],
                                    "x": sum(point["x"] for point in person["eye_points"]) / len(person["eye_points"]),
                                    "y": sum(point["y"] for point in person["eye_points"]) / len(person["eye_points"]),
                                }
                                for person in boxes.get(frame, []) if person["eye_points"]
                            ]
                        ),
                    }
                    for frame in indices
                ],
            }
            truth_path = scene_root / "truth.json"
            _write_json(truth_path, truth)
            prepared_sources.append({
                "scene_id": scene_id,
                "dataset": scene["dataset"],
                "source": scene["source"],
                "media": scene["media"],
                "media_sha256": scene["media_sha256"],
                "annotation": scene.get("annotation", scene.get("annotations")),
                "annotation_sha256": annotation_hashes,
                "frame_count": scene["frame_count"],
                "timeline_frame_count": scene.get("timeline_frame_count", scene["frame_count"]),
                "media_member_count": scene.get("media_member_count"),
                "fps": scene["fps"],
                "width": dimensions["width"],
                "height": dimensions["height"],
                "prepared_frame_count": len(indices),
                "history_frame_count": sum(role == "history" for _, _, role in frame_rows),
                "future_frame_count": sum(role == "future" for _, _, role in frame_rows),
                "metric_family": "box_iou" if scene["dataset"] == "VIRAT" else "count_only_eye_point_focus",
                "truth_sha256": sha256_file(truth_path),
                "prepared_frames": [
                    {
                        "frame_index": frame,
                        "segment_id": frame_metadata[frame][0],
                        "role": frame_metadata[frame][1],
                        "path": str(destinations[frame].relative_to(output)).replace("\\", "/"),
                        "sha256": sha256_file(destinations[frame]),
                    }
                    for frame in indices
                ],
                "segments": [
                    {"id": row["id"], "role": "history" if str(row["id"]).startswith("H") else "future", "start": row["start"], "end": row["end"]}
                    for row in scene["segments"]
                ],
            })
        prepared_manifest = {
            "schema_version": "fieldpack.prepared-fixed-camera.v1",
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256_bytes(manifest_bytes),
            "source_manifest_schema_version": manifest["schema_version"],
            "frame_index_base": 0,
            "role_contract": {"H": "history", "F": "future"},
            "source_hashes_verified": True,
            "unselected_source_files_opened": False,
            "unselected_media_payloads_materialized": False,
            "scene_count": len(prepared_sources),
            "prepared_frame_count": sum(row["prepared_frame_count"] for row in prepared_sources),
            "scenes": prepared_sources,
        }
        prepared_path = output / "prepared-manifest.json"
        _write_json(prepared_path, prepared_manifest)
        return {
            "output": str(output),
            "prepared_manifest": str(prepared_path),
            "prepared_manifest_sha256": sha256_file(prepared_path),
            "scene_count": len(prepared_sources),
            "prepared_frame_count": prepared_manifest["prepared_frame_count"],
        }
    except Exception:
        if output.exists():
            shutil.rmtree(output)
        raise
    finally:
        for path in temporary_nested:
            path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the frozen expanded fixed-camera split")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--virat-videos-00-root", type=Path, required=True)
    parser.add_argument("--virat-videos-05-root", type=Path, required=True)
    parser.add_argument("--virat-annotations-root", type=Path, required=True)
    parser.add_argument("--chokepoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(
        args.manifest,
        virat_media_roots={"videos-00": args.virat_videos_00_root, "videos-05": args.virat_videos_05_root},
        virat_annotations_root=args.virat_annotations_root,
        chokepoint_root=args.chokepoint_root,
        output=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
