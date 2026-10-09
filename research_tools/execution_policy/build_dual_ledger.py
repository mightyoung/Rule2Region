"""Build aligned full-frame YOLO11s / YOLO26 ledgers. Missing 26 is regenerated."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from research_tools.cascade_experiment import Box, Prediction
from research_tools.execution_policy.ledger import FrameLedger, align_ledgers
from research_tools.route_a_clip_commissioning import load_clips


CONF = 0.25
IMGSZ = 640


def _bgr(path: Path) -> Any:
    import numpy as np

    rgb = np.asarray(Image.open(path).convert("RGB"))
    return np.ascontiguousarray(rgb[..., ::-1])


def _predict(model: Any, image: Any, frame: int, source: str, device: str) -> tuple[Prediction, ...]:
    result = model.predict(
        source=image,
        device=device,
        imgsz=IMGSZ,
        verbose=False,
        conf=CONF,
        end2end=False,
    )[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()
    rows = []
    for coords, score, cls in zip(
        boxes.xyxy.detach().cpu().tolist(),
        boxes.conf.detach().cpu().tolist(),
        boxes.cls.detach().cpu().tolist(),
    ):
        if int(cls) != 0:
            continue
        rows.append(Prediction(frame, Box(*map(float, coords)), float(score), source))
    return tuple(rows)


def _predict_26(model: Any, image: Any, frame: int, device: str) -> tuple[Prediction, ...]:
    return _predict(model, image, frame, "yolo26", device)


def build(
    *,
    cache_root: Path,
    pilot_root: Path,
    output_path: Path,
    yolo26_path: Path,
    yolo11s_path: Path | None = None,
    device: str = "0",
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refuse_existing_output:{output_path}")
    from ultralytics import YOLO  # type: ignore

    clips = load_clips(cache_root, pilot_root)
    model_26 = YOLO(str(yolo26_path))
    model_11 = YOLO(str(yolo11s_path)) if yolo11s_path is not None else None
    sequences: dict[str, Any] = {}
    for sequence, clip in sorted(clips.items()):
        yolo11s: dict[int, tuple[Prediction, ...]] = dict(clip["global_by_frame"])
        yolo26: dict[int, tuple[Prediction, ...]] = {}
        for frame in clip["frames"]:
            image = _bgr(pilot_root / "sequences" / sequence / "img1" / f"{frame:06d}.jpg")
            yolo26[frame] = _predict_26(model_26, image, frame, device)
            if model_11 is not None:
                yolo11s[frame] = _predict(model_11, image, frame, "yolo11s", device)
        if model_11 is None and not yolo11s:
            raise ValueError("yolo11s_boxes_missing_pass_yolo11s_weights")
        ledgers = align_ledgers(clip["frames"], yolo11s, yolo26)
        if any("merged" in row.to_json() for row in ledgers):
            raise ValueError("mixed_box_list_forbidden")
        sequences[sequence] = {
            "camera_id": clip["camera_id"],
            "frames": [row.to_json() for row in ledgers],
            "history_frames": list(clip["history_frames"]),
            "validation_frames": list(clip["validation_frames"]),
        }
    payload = {
        "schema_version": "sceneconfig.execution-policy.dual-ledger.v1",
        "yolo26_full_rerun": True,
        "yolo11s_full_rerun": yolo11s_path is not None,
        "conf": CONF,
        "imgsz": IMGSZ,
        "mixed_lists": False,
        "sealed_camera_used": False,
        "clip_count": len(sequences),
        "sequences": sequences,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--yolo26", type=Path, required=True)
    parser.add_argument("--yolo11s", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    result = build(
        cache_root=args.cache_root,
        pilot_root=args.pilot_root,
        output_path=args.output,
        yolo26_path=args.yolo26,
        yolo11s_path=args.yolo11s,
        device=args.device,
    )
    print(json.dumps({
        "clip_count": result["clip_count"],
        "yolo26_full_rerun": True,
        "yolo11s_full_rerun": bool(result.get("yolo11s_full_rerun")),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
