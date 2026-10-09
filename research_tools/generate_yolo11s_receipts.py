"""Generate immutable YOLO11s-only prediction receipts for proposal research."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from research_tools.cascade_experiment import UltralyticsYOLOAdapter
from research_tools.run_cascade_successor import write_experiment_receipt


def generate_receipt(
    frames_dir: Path,
    model_path: Path,
    *,
    sequence_id: str,
    device: str = "0",
    imgsz: int = 640,
    batch_size: int = 8,
) -> dict[str, object]:
    frame_ids = tuple(sorted(int(path.stem) for path in frames_dir.glob("*.jpg") if path.stem.isdigit()))
    if not frame_ids:
        raise ValueError("no_frames_for_yolo11s_receipt")
    detector = UltralyticsYOLOAdapter(model_path, device=device, name="yolo11s", imgsz=imgsz)
    first = np.asarray(Image.open(frames_dir / f"{frame_ids[0]:06d}.jpg").convert("RGB"))
    detector.predict(first, frame_index=-1)
    rows: list[dict[str, object]] = []
    elapsed_ms = 0.0
    for start in range(0, len(frame_ids), batch_size):
        chunk_ids = frame_ids[start : start + batch_size]
        frames = [
            np.asarray(Image.open(frames_dir / f"{frame_id:06d}.jpg").convert("RGB"))
            for frame_id in chunk_ids
        ]
        began = time.perf_counter()
        batches = detector.predict_batch(frames, frame_indices=chunk_ids)
        elapsed_ms += (time.perf_counter() - began) * 1000.0
        rows.extend(
            {
                "frame": frame_id,
                "predictions": [asdict(prediction) for prediction in predictions],
            }
            for frame_id, predictions in zip(chunk_ids, batches)
        )
    return {
        "phase": "proposal-source",
        "sequence_partition": "independent_confirmation",
        "sequence_id": sequence_id,
        "frames": len(frame_ids),
        "global_imgsz": imgsz,
        "strong_model_called": False,
        "conditions": {
            "yolo11s_global": {
                "frames": len(frame_ids),
                "model_compute_total_ms": elapsed_ms,
                "timing_scope": "MODEL_CALLS_ONLY_NOT_END_TO_END",
            }
        },
        "receipts": {"yolo11s_global": rows},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate YOLO11s-only proposal receipts")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--small-model", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    payload = generate_receipt(
        args.frames_dir,
        args.small_model,
        sequence_id=args.sequence_id,
        device=args.device,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
    )
    digest = write_experiment_receipt(args.output, payload)
    print(json.dumps({"output": str(args.output), "sha256": digest, "frames": payload["frames"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
