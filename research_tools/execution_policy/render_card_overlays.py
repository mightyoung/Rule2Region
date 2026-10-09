"""Draw submitted policy-card regions on observer JPEGs with supervision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


COLORS = {
    "yolo11s": "#22D3EE",
    "yolo26": "#F59E0B",
}


def _detections(rows: list[list[float]]):
    import supervision as sv

    xyxy = np.asarray(rows, dtype=np.float32).reshape((-1, 4))
    return sv.Detections(xyxy=xyxy)


def annotate_card(image_bgr: np.ndarray, card: dict[str, Any], title: str, *, only_yolo26: bool) -> np.ndarray:
    import cv2
    import supervision as sv

    canvas = image_bgr.copy()
    drawn = 0
    for region in card.get("regions") or ():
        model = str(region.get("model") or "yolo11s")
        if only_yolo26 and model != "yolo26":
            continue
        color = sv.Color.from_hex(COLORS.get(model, "#F43F5E"))
        xyxy = [float(value) for value in region["xyxy"]]
        det = _detections([xyxy])
        canvas = sv.BoxAnnotator(color=color, thickness=3, color_lookup=sv.ColorLookup.INDEX).annotate(canvas, det)
        label = f"{region.get('region_id')} {model} {region.get('conf')}"
        canvas = sv.LabelAnnotator(
            color=color,
            color_lookup=sv.ColorLookup.INDEX,
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1,
        ).annotate(canvas, det, [label])
        drawn += 1
    if only_yolo26 and drawn == 0:
        title = f"{title} (no yolo26 region)"
    cv2.putText(canvas, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(canvas, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def render(result_path: Path, output_dir: Path, *, only_yolo26: bool) -> list[Path]:
    import cv2

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    rows = payload.get("per_camera") or {}
    if rows:
        items = sorted(rows.items())
    else:
        items = sorted(payload["per_clip"].items())
    for key, row in items:
        card = row.get("intention_card")
        if not card:
            continue
        image_path = Path(row["observer_image"]["path"])
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"observer_image_unreadable:{image_path}")
        camera = str(row.get("camera_id") or key)
        title = f"{camera} {row.get('observer_sequence', key)} f{int(row['observer_frame']):06d} {row.get('verdict', '')}".strip()
        canvas = annotate_card(frame, card, title, only_yolo26=only_yolo26)
        dest = output_dir / f"{camera}.card.png"
        if not cv2.imwrite(str(dest), canvas):
            raise OSError(f"write_failed:{dest}")
        written.append(dest)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only-yolo26", action="store_true")
    args = parser.parse_args()
    paths = render(args.result, args.output_dir, only_yolo26=args.only_yolo26)
    print(json.dumps({"count": len(paths), "files": [str(path) for path in paths]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
