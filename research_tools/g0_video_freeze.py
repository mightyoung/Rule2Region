from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SEQUENCES = ("02", "04", "05", "09", "10", "11", "13")
STATIC_SEQUENCES = frozenset({"02", "04", "09"})
WINDOW_SECONDS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def seal_manifest(path: Path, manifest: dict[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    digest = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    payload = canonical_bytes({**unsigned, "manifest_sha256": digest})
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable manifest differs: {path}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return digest


def window_frame_ranges(
    frame_count: int, fps: int, window_seconds: int = WINDOW_SECONDS
) -> tuple[tuple[int, int], ...]:
    if frame_count <= 0 or fps <= 0 or window_seconds <= 0:
        raise ValueError("frame_count_fps_and_window_seconds_must_be_positive")
    frames_per_window = fps * window_seconds
    return tuple(
        (start, start + frames_per_window)
        for start in range(1, frame_count + 1, frames_per_window)
        if start + frames_per_window - 1 <= frame_count
    )


def load_person_tracks(gt_path: Path) -> dict[int, set[int]]:
    """Load valid pedestrian identities using the MOTChallenge train-GT contract."""

    by_frame: dict[int, set[int]] = defaultdict(set)
    with gt_path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if len(row) < 9:
                raise ValueError(f"{gt_path}:{line_number}: expected at least 9 fields")
            frame = int(float(row[0]))
            track_id = int(float(row[1]))
            width = float(row[4])
            height = float(row[5])
            confidence = int(float(row[6]))
            class_id = int(float(row[7]))
            visibility = float(row[8])
            if confidence == 1 and class_id == 1 and visibility > 0 and width > 0 and height > 0:
                by_frame[frame].add(track_id)
    return by_frame


def label_window(
    tracks_by_frame: dict[int, set[int]], start_frame: int, end_frame_exclusive: int
) -> dict[str, Any]:
    candidates = [
        (len(tracks_by_frame.get(frame, set())), frame, tracks_by_frame.get(frame, set()))
        for frame in range(start_frame, end_frame_exclusive)
    ]
    count, peak_frame, track_ids = max(candidates, key=lambda item: (item[0], -item[1]))
    return {
        "count": count,
        "peak_frame": peak_frame,
        "peak_track_ids": sorted(track_ids),
    }


def frame_tree_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        item_hash = sha256_file(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(item_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def proportional_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    if total <= 0 or total > sum(counts.values()):
        raise ValueError("invalid_deployment_total")
    denominator = sum(counts.values())
    raw = {key: total * value / denominator for key, value in counts.items()}
    quotas = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(quotas.values())
    ranking = sorted(counts, key=lambda key: (-(raw[key] - quotas[key]), key))
    for key in ranking[:remaining]:
        quotas[key] += 1
    if any(quotas[key] > counts[key] for key in counts):
        raise ValueError("deployment_quota_exceeds_sequence_windows")
    return quotas


def choose_deployment_ids(
    windows: list[dict[str, Any]], *, seed: int, total: int = 24
) -> set[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        grouped[window["sequence_id"]].append(window)
    quotas = proportional_quotas(
        {sequence: len(values) for sequence, values in grouped.items()}, total
    )
    selected: set[str] = set()
    for sequence in sorted(grouped):
        ranked = sorted(
            grouped[sequence],
            key=lambda window: hashlib.sha256(
                f"{seed}|{window['sample_id']}".encode("utf-8")
            ).hexdigest(),
        )
        selected.update(window["sample_id"] for window in ranked[: quotas[sequence]])
    return selected


def build_video_manifest(mot17_root: Path, *, seed: int) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    sequence_receipts: dict[str, Any] = {}
    for sequence_id in SEQUENCES:
        sequence_name = f"MOT17-{sequence_id}-DPM"
        sequence_root = mot17_root / "train" / sequence_name
        parser = configparser.ConfigParser()
        seqinfo_path = sequence_root / "seqinfo.ini"
        if not parser.read(seqinfo_path, encoding="utf-8"):
            raise FileNotFoundError(seqinfo_path)
        fps = parser.getint("Sequence", "frameRate")
        frame_count = parser.getint("Sequence", "seqLength")
        width = parser.getint("Sequence", "imWidth")
        height = parser.getint("Sequence", "imHeight")
        ranges = window_frame_ranges(frame_count, fps)
        gt_path = sequence_root / "gt" / "gt.txt"
        tracks = load_person_tracks(gt_path)
        sequence_receipts[sequence_id] = {
            "sequence_name": sequence_name,
            "source_suffix": "DPM",
            "seqinfo_path": str(seqinfo_path.resolve()),
            "seqinfo_sha256": sha256_file(seqinfo_path),
            "gt_path": str(gt_path.resolve()),
            "gt_sha256": sha256_file(gt_path),
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "full_nonoverlapping_windows": len(ranges),
        }
        for window_index, (start_frame, end_frame) in enumerate(ranges):
            frame_paths = [
                sequence_root / "img1" / f"{frame:06d}.jpg"
                for frame in range(start_frame, end_frame)
            ]
            missing = [path for path in frame_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(missing[0])
            sample_id = f"MOT17-{sequence_id}:4s:{window_index:02d}"
            windows.append(
                {
                    "sample_id": sample_id,
                    "sequence_id": f"MOT17-{sequence_id}",
                    "camera_motion_stratum": (
                        "STATIC_PRIMARY"
                        if sequence_id in STATIC_SEQUENCES
                        else "MOVING_CAMERA_STRESS"
                    ),
                    "start_frame_inclusive": start_frame,
                    "end_frame_exclusive": end_frame,
                    "start_seconds": (start_frame - 1) / fps,
                    "end_seconds": (end_frame - 1) / fps,
                    "frame_count": len(frame_paths),
                    "frame_tree_sha256": frame_tree_hash(frame_paths),
                    "first_frame": frame_paths[0].name,
                    "last_frame": frame_paths[-1].name,
                    "label_contract": "conf=1,class=1,visibility>0,positive_bbox; peak unique tracks",
                    "label": label_window(tracks, start_frame, end_frame),
                    "deployment_member": False,
                }
            )

    deployment_ids = choose_deployment_ids(windows, seed=seed)
    for window in windows:
        window["deployment_member"] = window["sample_id"] in deployment_ids
    strata = defaultdict(int)
    for window in windows:
        strata[window["camera_motion_stratum"]] += 1
    if len(windows) != 50 or strata != {
        "STATIC_PRIMARY": 17,
        "MOVING_CAMERA_STRESS": 33,
    }:
        raise AssertionError(f"unexpected_window_contract:{len(windows)}:{dict(strata)}")
    if len(deployment_ids) != 24:
        raise AssertionError("deployment_subset_must_have_24_windows")
    return {
        "schema_version": "evidencegate.g0.video-scope.v1",
        "seed": seed,
        "dataset": "MOT17 train",
        "source_detector_suffix": "DPM",
        "window_seconds": WINDOW_SECONDS,
        "selection_rule": "all complete natural non-overlapping windows; no label enrichment",
        "camera_motion_policy": "17 STATIC_PRIMARY and 33 MOVING_CAMERA_STRESS; report separately",
        "formal_samples": 50,
        "deployment_samples": 24,
        "strata": dict(sorted(strata.items())),
        "sequence_receipts": sequence_receipts,
        "samples": windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze EvidenceGate G0 MOT17 scope.")
    parser.add_argument("--mot17-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    arguments = parser.parse_args()
    manifest = build_video_manifest(arguments.mot17_root, seed=arguments.seed)
    digest = seal_manifest(arguments.output, manifest)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "manifest_sha256": digest,
                "formal_samples": manifest["formal_samples"],
                "deployment_samples": manifest["deployment_samples"],
                "strata": manifest["strata"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
