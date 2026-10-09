"""One-image contract smoke for pinned YOLO26 detect and depth checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _predict(model: Any, image: Path, *, device: str, imgsz: int, end2end: bool | None = None) -> tuple[Any, float]:
    import torch

    kwargs: dict[str, Any] = {
        "source": str(image),
        "device": device,
        "imgsz": imgsz,
        "verbose": False,
        "conf": 0.25,
    }
    if end2end is not None:
        kwargs["end2end"] = end2end
    model.predict(**kwargs)  # unmeasured warm-up
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = model.predict(**kwargs)[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000.0


def run(image: Path, detect_checkpoint: Path, depth_checkpoint: Path, *, device: str = "0") -> dict[str, Any]:
    import ultralytics
    from ultralytics import YOLO

    detect_one_to_one = YOLO(detect_checkpoint)
    detect_one_to_many = YOLO(detect_checkpoint)
    depth = YOLO(depth_checkpoint)
    head = detect_one_to_many.model.model[-1]
    one_to_many_available = head.cv2 is not None and head.cv3 is not None
    head_weight_difference = None
    if one_to_many_available:
        one_to_many_weight = next(head.cv2.parameters()).detach().cpu()
        one_to_one_weight = next(head.one2one_cv2.parameters()).detach().cpu()
        head_weight_difference = float((one_to_many_weight - one_to_one_weight).abs().mean())
    one_to_one, one_to_one_ms = _predict(detect_one_to_one, image, device=device, imgsz=640, end2end=True)
    one_to_one_head_state = bool(detect_one_to_one.model.model[-1].end2end)
    one_to_one_backend_state = bool(detect_one_to_one.predictor.model.end2end)
    one_to_many = None
    one_to_many_ms = None
    one_to_many_head_state = None
    one_to_many_backend_state = None
    if one_to_many_available:
        one_to_many, one_to_many_ms = _predict(detect_one_to_many, image, device=device, imgsz=640, end2end=False)
        one_to_many_head_state = bool(detect_one_to_many.model.model[-1].end2end)
        one_to_many_backend_state = bool(detect_one_to_many.predictor.model.end2end)
    depth_result, depth_ms = _predict(depth, image, device=device, imgsz=768)
    depth_data = depth_result.depth.data.detach().cpu()
    one_to_one_xyxy = one_to_one.boxes.xyxy.detach().cpu()
    one_to_many_xyxy = one_to_many.boxes.xyxy.detach().cpu() if one_to_many is not None else None
    payload = {
        "schema_version": "fieldpack.yolo26-depth-contract-smoke.v1",
        "ultralytics_version": ultralytics.__version__,
        "detect_task": detect_one_to_one.task,
        "depth_task": depth.task,
        "one_to_one_boxes": len(one_to_one.boxes),
        "one_to_many_available": one_to_many_available,
        "one_to_many_unavailable_reason": None if one_to_many_available else "OFFICIAL_DEPLOY_CHECKPOINT_CV2_CV3_ABSENT",
        "one_to_many_boxes": len(one_to_many.boxes) if one_to_many is not None else None,
        "one_to_one_head_state": one_to_one_head_state,
        "one_to_one_backend_state": one_to_one_backend_state,
        "one_to_many_head_state": one_to_many_head_state,
        "one_to_many_backend_state": one_to_many_backend_state,
        "head_weight_mean_absolute_difference": head_weight_difference,
        "output_boxes_identical": None if one_to_many_xyxy is None else bool(
            one_to_one_xyxy.shape == one_to_many_xyxy.shape and (one_to_one_xyxy == one_to_many_xyxy).all()
        ),
        "depth_has_boxes": getattr(depth_result, "boxes", None) is not None,
        "depth_shape": list(depth_data.shape),
        "latency_ms": {
            "one_to_one": one_to_one_ms,
            "one_to_many": one_to_many_ms,
            "depth": depth_ms,
        },
        "checkpoint_sha256": {
            "detect": _sha256(detect_checkpoint),
            "depth": _sha256(depth_checkpoint),
        },
        "image_sha256": _sha256(image),
    }
    assert payload["ultralytics_version"] == "8.4.118"
    assert payload["detect_task"] == "detect" and payload["depth_task"] == "depth"
    assert payload["depth_has_boxes"] is False and len(payload["depth_shape"]) == 2
    assert payload["one_to_one_head_state"] is True and payload["one_to_one_backend_state"] is True
    if payload["one_to_many_available"]:
        assert payload["one_to_many_head_state"] is False and payload["one_to_many_backend_state"] is False
        assert payload["head_weight_mean_absolute_difference"] > 0
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--detect", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refuse_existing_output:{args.output}")
    payload = run(args.image, args.detect, args.depth, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("one_to_one_boxes", "one_to_many_available", "one_to_many_boxes", "depth_has_boxes", "depth_shape", "latency_ms")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
