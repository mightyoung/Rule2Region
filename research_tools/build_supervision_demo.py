"""Build offline, read-only supervision videos from frozen cascade receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research_tools.cascade_experiment import parse_mot17_gt


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_table(receipt: dict[str, Any]) -> list[list[Any]]:
    columns = ("count_mae", "ap50", "recall", "small_object_recall", "budget_fraction", "latency_p50_ms", "latency_p95_ms", "throughput_fps")
    return [[name, *[round(float(values.get(column, 0.0)), 4) for column in columns]] for name, values in sorted(receipt["conditions"].items())]


def _detections(rows: Sequence[Sequence[float]], confidences: Sequence[float] | None = None) -> Any:
    import supervision as sv  # type: ignore

    xyxy = np.asarray(rows, dtype=np.float32).reshape((-1, 4))
    confidence = None if confidences is None else np.asarray(confidences, dtype=np.float32)
    return sv.Detections(xyxy=xyxy, confidence=confidence)


def _annotate(frame: np.ndarray, gt_rows: Sequence[Any], prediction_rows: Sequence[dict[str, Any]], selected_roi: dict[str, Any] | None, labels: dict[str, str]) -> np.ndarray:
    import cv2  # type: ignore
    import supervision as sv  # type: ignore

    canvas = frame.copy()
    gt_boxes = [item.box.coords for item in gt_rows]
    pred_boxes = [[row["box"][key] for key in ("x1", "y1", "x2", "y2")] for row in prediction_rows]
    pred_conf = [float(row["confidence"]) for row in prediction_rows]
    if gt_boxes:
        canvas = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2, color_lookup=sv.ColorLookup.INDEX).annotate(canvas, _detections(gt_boxes))
    if pred_boxes:
        detections = _detections(pred_boxes, pred_conf)
        canvas = sv.BoxAnnotator(color=sv.Color.from_hex("#8B5CF6"), thickness=2, color_lookup=sv.ColorLookup.INDEX).annotate(canvas, detections)
        pred_labels = [f"person {value:.2f}" for value in pred_conf]
        canvas = sv.LabelAnnotator(color=sv.Color.from_hex("#8B5CF6"), color_lookup=sv.ColorLookup.INDEX, text_color=sv.Color.WHITE).annotate(canvas, detections, pred_labels)
    if selected_roi is not None:
        core_xyxy = selected_roi.get("core_xyxy", selected_roi.get("xyxy"))
        roi = _detections([core_xyxy])
        canvas = sv.BoxAnnotator(color=sv.Color.YELLOW, thickness=3, color_lookup=sv.ColorLookup.INDEX).annotate(canvas, roi)
        crop = selected_roi.get("crop_xyxy")
        if crop is not None:
            x1, y1, x2, y2 = map(int, crop)
            dash = 16
            for x in range(x1, x2, dash * 2):
                cv2.line(canvas, (x, y1), (min(x + dash, x2), y1), (0, 255, 255), 2)
                cv2.line(canvas, (x, y2), (min(x + dash, x2), y2), (0, 255, 255), 2)
            for y in range(y1, y2, dash * 2):
                cv2.line(canvas, (x1, y), (x1, min(y + dash, y2)), (0, 255, 255), 2)
                cv2.line(canvas, (x2, y), (x2, min(y + dash, y2)), (0, 255, 255), 2)
    lines = [labels["title"], labels["counts"], labels["latency"]]
    if selected_roi is not None and "crop_area_ratio" in selected_roi:
        scale = selected_roi.get("resize_scale_xy", (0.0, 0.0))
        lines.append(
            f"ROI core {100*selected_roi['core_area_ratio']:.1f}% | crop {100*selected_roi['crop_area_ratio']:.1f}% | scale {scale[0]:.2f}x/{scale[1]:.2f}x"
        )
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (18, 30 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (18, 30 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def build_video(receipt_path: Path, sequence_dir: Path, condition: str, output: Path, *, fps: float = 4.0) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
        import supervision  # noqa: F401  # type: ignore
    except ImportError as exc:
        raise RuntimeError("supervision_demo_dependencies_required") from exc
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if condition not in receipt.get("receipts", {}):
        raise KeyError(f"condition_missing:{condition}")
    rows = receipt["receipts"][condition]
    truth = parse_mot17_gt(sequence_dir / "gt.txt")
    by_truth: dict[int, list[Any]] = {}
    for item in truth:
        by_truth.setdefault(item.frame, []).append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for row in rows:
            frame_index = int(row["frame"])
            frame_path = sequence_dir / "img1" / f"{frame_index:06d}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise FileNotFoundError(frame_path)
            labels = {
                "title": f"{receipt['sequence_id']} | {condition} | frame {frame_index}",
                "counts": f"GT {len(by_truth.get(frame_index, ()))} | Pred {len(row['predictions'])} | upgraded {bool(row.get('upgraded'))}",
                "latency": f"amortized compute {float(row.get('amortized_compute_ms', 0.0)):.2f} ms",
            }
            canvas = _annotate(frame, by_truth.get(frame_index, ()), row["predictions"], row.get("selected_roi"), labels)
            if writer is None:
                height, width = canvas.shape[:2]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise IOError(f"video_writer_open_failed:{output}")
            writer.write(canvas)
    finally:
        if writer is not None:
            writer.release()
    if not output.is_file() or output.stat().st_size == 0:
        raise IOError("demo_video_not_created")
    return {"sequence_id": receipt["sequence_id"], "condition": condition, "video": str(output), "video_sha256": sha256_file(output), "frames": len(rows), "summary": receipt["conditions"][condition], "condition_table": condition_table(receipt)}


def write_index(path: Path, assets: Sequence[dict[str, Any]], *, pilot_manifest_sha256: str) -> dict[str, Any]:
    payload = {
        "schema_version": "fieldpack.supervision-demo-index.v1",
        "evidence_status": "ENGINEERING_PILOT_NOT_FORMAL_STATISTICS",
        "pilot_manifest_sha256": pilot_manifest_sha256,
        "assets": list(assets),
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return {**payload, "index_sha256": hashlib.sha256(encoded).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build supervision visualization from frozen result")
    parser.add_argument("--receipt", type=Path, required=True); parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path); parser.add_argument("--pilot-manifest-sha256", default="")
    args = parser.parse_args()
    asset = build_video(args.receipt, args.sequence_dir, args.condition, args.output)
    if args.index:
        write_index(args.index, (asset,), pilot_manifest_sha256=args.pilot_manifest_sha256)
    print(json.dumps(asset, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
