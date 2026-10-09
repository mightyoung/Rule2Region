"""Select unstitched high-box frames after YOLO11s+YOLO26 and keep native coordinates."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS

OBSERVER_PRIORITY = (
    "C15",
    "C14",
    "C13",
    "C12",
    "C11",
    "C10",
    "C09",
    "C08",
    "C07",
    "C06",
    "C05",
    "C23",
    "C22",
    "C21",
    "C20",
    "C19",
    "C18",
    "C17",
    "C16",
    "C04",
    "C03",
    "C02",
    "C01",
    "C00",
    "B0",
)
SOURCE_COLORS = {
    "yolo11s_global": (255, 215, 0),
    "default": (0, 200, 255),
}


@dataclass(frozen=True)
class DenseFrame:
    frame: int
    sequence_id: str
    camera_id: str
    merged_count: int
    tile_raw_count: int
    global_count: int
    merged: tuple[dict[str, Any], ...]
    tile_raw: tuple[dict[str, Any], ...]
    global_boxes: tuple[dict[str, Any], ...]

    def as_manifest(self, *, overlay_name: str, clean_name: str) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "sequence_id": self.sequence_id,
            "camera_id": self.camera_id,
            "merged_count": self.merged_count,
            "tile_raw_count": self.tile_raw_count,
            "global_count": self.global_count,
            "overlay_name": overlay_name,
            "clean_name": clean_name,
        }


def choose_observer_config(config_ids: Iterable[str]) -> str:
    available = set(config_ids)
    for config_id in OBSERVER_PRIORITY:
        if config_id in available:
            return config_id
    raise ValueError("no_observer_config_in_prediction_ledger")


def score_frame(row: Mapping[str, Any]) -> DenseFrame:
    merged = tuple(row.get("merged") or ())
    tile_raw = tuple(row.get("tile_raw") or ())
    global_boxes = tuple(row.get("global") or ())
    return DenseFrame(
        frame=int(row["frame"]),
        sequence_id=str(row.get("sequence_id") or ""),
        camera_id=str(row.get("camera_id") or ""),
        merged_count=len(merged),
        tile_raw_count=len(tile_raw),
        global_count=len(global_boxes),
        merged=merged,
        tile_raw=tile_raw,
        global_boxes=global_boxes,
    )


def select_dense_frames(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_frames: set[int],
    k: int,
) -> tuple[DenseFrame, ...]:
    if k < 1:
        raise ValueError("k_must_be_positive")
    scored = [score_frame(row) for row in rows if int(row["frame"]) in allowed_frames]
    if len(scored) < k:
        raise ValueError("insufficient_calibration_frames")
    ranked = sorted(
        scored,
        key=lambda row: (-row.merged_count, -row.tile_raw_count, -row.global_count, row.frame),
    )
    selected = tuple(ranked[:k])
    if len({row.frame for row in selected}) != k:
        raise ValueError("dense_frame_selection_not_unique")
    return selected


def _load_prediction_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows[(payload["camera_id"], payload["config_id"])] = payload
    return rows


def select_from_prediction_ledger(
    *,
    predictions_path: Path,
    frame_split: Mapping[str, Any],
    camera_id: str,
    k: int,
) -> dict[str, Any]:
    if camera_id not in EXPECTED_CAMERAS:
        raise ValueError(f"camera_not_in_development_set:{camera_id}")
    rows = _load_prediction_rows(predictions_path)
    available = [config_id for camera, config_id in rows if camera == camera_id]
    observer = choose_observer_config(available)
    payload = rows[(camera_id, observer)]
    camera_split = frame_split["cameras"][camera_id]
    allowed = {int(value) for value in camera_split["calibration_frames"]}
    forbidden = {int(value) for value in camera_split.get("evaluation_frames") or ()}
    if allowed & forbidden:
        raise ValueError(f"calibration_evaluation_overlap:{camera_id}")
    selected = select_dense_frames(payload["frames"], allowed_frames=allowed, k=k)
    if any(row.frame in forbidden for row in selected):
        raise ValueError(f"evaluation_frame_selected:{camera_id}")
    return {
        "camera_id": camera_id,
        "observer_config_id": observer,
        "sequence_id": camera_split["sequence_id"],
        "frames": selected,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_color(source: str) -> tuple[int, int, int]:
    if source.startswith("yolo11s"):
        return SOURCE_COLORS["yolo11s_global"]
    return SOURCE_COLORS["default"]


def materialize_native_scene_pack(
    *,
    source_path: Path,
    output_clean: Path,
    output_overlay: Path,
    boxes: Sequence[Mapping[str, Any]],
    box_id_prefix: str,
) -> dict[str, Any]:
    if not source_path.is_file():
        raise FileNotFoundError(f"missing_native_frame:{source_path}")
    with Image.open(source_path) as raw:
        image = raw.convert("RGB")
        width, height = image.size
        clean = image.copy()
        overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    drawn: list[dict[str, Any]] = []
    for index, item in enumerate(boxes, start=1):
        box = item["box"]
        x1, y1, x2, y2 = (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))
        if min(x1, y1, x2, y2) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("box_must_be_finite_positive_rectangle")
        color = _source_color(str(item.get("source") or ""))
        box_id = f"{box_id_prefix}{index:02d}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{box_id} {float(item.get('confidence') or 0.0):.2f}"
        draw.rectangle((x1, max(0, y1 - 16), x1 + 8 * len(label), max(0, y1 - 16) + 16), fill=(0, 0, 0))
        draw.text((x1 + 2, max(0, y1 - 16)), label, fill=color, font=font)
        drawn.append(
            {
                "box_id": box_id,
                "xyxy": [x1, y1, x2, y2],
                "confidence": float(item.get("confidence") or 0.0),
                "source": item.get("source"),
            }
        )
    output_clean.parent.mkdir(parents=True, exist_ok=True)
    output_overlay.parent.mkdir(parents=True, exist_ok=True)
    clean.save(output_clean, format="JPEG", quality=95, subsampling=0)
    overlay.save(output_overlay, format="JPEG", quality=95, subsampling=0)
    return {
        "width": width,
        "height": height,
        "transform": "identity",
        "collage": False,
        "box_count": len(drawn),
        "boxes": drawn,
        "clean_sha256": _sha256_file(output_clean),
        "overlay_sha256": _sha256_file(output_overlay),
        "source_sha256": _sha256_file(source_path),
    }


def build_camera_pack(
    *,
    camera_id: str,
    observer_config_id: str,
    sequence_id: str,
    frames: Sequence[DenseFrame],
    pilot_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    camera_dir = output_dir / camera_id
    camera_dir.mkdir(parents=True, exist_ok=True)
    manifest_frames: list[dict[str, Any]] = []
    for rank, row in enumerate(frames, start=1):
        source = pilot_root / "sequences" / sequence_id / "img1" / f"{row.frame:06d}.jpg"
        clean_name = f"{camera_id}/{row.frame:06d}.clean.jpg"
        overlay_name = f"{camera_id}/{row.frame:06d}.overlay.jpg"
        receipt = materialize_native_scene_pack(
            source_path=source,
            output_clean=output_dir / clean_name,
            output_overlay=output_dir / overlay_name,
            boxes=row.merged,
            box_id_prefix=f"f{row.frame}b",
        )
        manifest_frames.append(
            {
                **row.as_manifest(overlay_name=overlay_name, clean_name=clean_name),
                "rank": rank,
                "source_path": source.as_posix(),
                **receipt,
            }
        )
    return {
        "camera_id": camera_id,
        "observer_config_id": observer_config_id,
        "sequence_id": sequence_id,
        "selection_rule": "yolo11s_yolo26_merged_box_count_desc",
        "collage_used": False,
        "frames": manifest_frames,
    }


def build_selection(
    *,
    predictions_path: Path,
    frame_split_path: Path,
    pilot_root: Path,
    output_dir: Path,
    k: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    output_dir.mkdir(parents=True)
    frame_split = json.loads(frame_split_path.read_text(encoding="utf-8"))
    cameras: dict[str, Any] = {}
    for camera_id in EXPECTED_CAMERAS:
        selected = select_from_prediction_ledger(
            predictions_path=predictions_path,
            frame_split=frame_split,
            camera_id=camera_id,
            k=k,
        )
        cameras[camera_id] = build_camera_pack(
            camera_id=camera_id,
            observer_config_id=selected["observer_config_id"],
            sequence_id=selected["sequence_id"],
            frames=selected["frames"],
            pilot_root=pilot_root,
            output_dir=output_dir,
        )
    payload = {
        "schema_version": "sceneconfig.dense-native-frame-selection.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "holdout_opened": False,
        "sealed_camera_used": False,
        "collage_used": False,
        "selection_rule": "yolo11s_yolo26_merged_box_count_desc",
        "k": k,
        "predictions_sha256": _sha256_file(predictions_path),
        "frame_split_sha256": _sha256_file(frame_split_path),
        "cameras": cameras,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--frame-split", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    payload = build_selection(
        predictions_path=args.predictions,
        frame_split_path=args.frame_split,
        pilot_root=args.pilot_root,
        output_dir=args.output_dir,
        k=args.k,
    )
    print(
        json.dumps(
            {
                "collage_used": payload["collage_used"],
                "k": payload["k"],
                "cameras": {
                    camera: {
                        "observer_config_id": row["observer_config_id"],
                        "frames": [item["frame"] for item in row["frames"]],
                    }
                    for camera, row in payload["cameras"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
