"""Residual commissioning arm R1: frozen causal reject on the eight-clip ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_tools.cascade_experiment import Box, Prediction
from research_tools.route_a_clip_commissioning import load_clips, score_condition
from research_tools.sceneconfig_e4_m0_deployability import select_m0_configs


RULE_ID = "residual.reject.v1"
CONF_MAX = 0.35
HEIGHT_MAX = 28.0
SUPPORT_IOU = 0.30
PRIMARY_EVAL = {
    "VIRAT_S_010004": "VIRAT_S_010004_01_000163_000188",
    "VIRAT_S_010204": "VIRAT_S_010204_03_000606_000632",
}


def box_height(box: Box) -> float:
    return float(box.y2 - box.y1)


def has_causal_support(box: Box, previous: Sequence[Prediction], *, iou: float = SUPPORT_IOU) -> bool:
    return any(box.iou(row.box) >= iou for row in previous)


def reject_box(pred: Prediction, previous: Sequence[Prediction]) -> bool:
    return (
        pred.confidence < CONF_MAX
        and box_height(pred.box) < HEIGHT_MAX
        and not has_causal_support(pred.box, previous)
    )


def apply_reject(
    by_frame: Mapping[int, Sequence[Prediction]],
    frames: Sequence[int],
) -> dict[int, tuple[Prediction, ...]]:
    output: dict[int, tuple[Prediction, ...]] = {}
    previous: tuple[Prediction, ...] = ()
    for frame in frames:
        current = tuple(by_frame.get(frame, ()))
        kept = tuple(row for row in current if not reject_box(row, previous))
        output[frame] = kept
        previous = current
    return output


def ingest_top6_tiles(
    path: Path,
    selected: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[int, list[Prediction]]]:
    store: dict[str, dict[int, list[Prediction]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            camera = row.get("camera_id")
            if camera not in selected or row.get("config_id") != selected[camera]["config_id"]:
                continue
            for frame in row.get("frames") or ():
                sequence = str(frame.get("sequence_id") or "")
                frame_id = int(frame["frame"])
                bucket = store.setdefault(sequence, {}).setdefault(frame_id, [])
                for item in frame.get("tile_raw") or ():
                    box = item["box"]
                    bucket.append(
                        Prediction(
                            frame_id,
                            Box(float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])),
                            float(item.get("confidence") or 0.0),
                            str(item.get("source") or "yolo26_tile"),
                        )
                    )
    return store


def merge_sources(
    clip: Mapping[str, Any],
    tiles: Mapping[int, Sequence[Prediction]] | None,
) -> dict[int, tuple[Prediction, ...]]:
    merged: dict[int, tuple[Prediction, ...]] = {}
    for frame in clip["frames"]:
        rows = list(clip["global_by_frame"].get(frame, ()))
        if tiles:
            rows.extend(tiles.get(frame, ()))
        merged[frame] = tuple(rows)
    return merged


def run(
    *,
    cache_root: Path,
    pilot_root: Path,
    proxy_path: Path,
    output_path: Path,
    top6_path: Path | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refuse_existing_output:{output_path}")
    clips = load_clips(cache_root, pilot_root)
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    selected = select_m0_configs(proxy)
    tiles = ingest_top6_tiles(top6_path, selected) if top6_path else {}
    per_clip = {}
    for sequence, clip in sorted(clips.items()):
        sources = merge_sources(clip, tiles.get(sequence))
        rejected = apply_reject(sources, clip["frames"])
        val_only = {frame: rejected[frame] for frame in clip["validation_frames"]}
        scored = score_condition(clip, val_only, tile_cost_ratio=0.0)
        dropped = sum(
            len(sources[frame]) - len(rejected[frame]) for frame in clip["validation_frames"]
        )
        camera_clips = sorted(
            item["sequence_id"] for item in clips.values() if item["camera_id"] == clip["camera_id"]
        )
        per_clip[sequence] = {
            "camera_id": clip["camera_id"],
            "sequence_id": sequence,
            "role": "development_eval" if sequence == camera_clips[0] else "transfer",
            "rule_id": RULE_ID,
            "tiles_used": sequence in tiles,
            "val_boxes_dropped": dropped,
            **scored,
        }
    primary = [
        per_clip[PRIMARY_EVAL["VIRAT_S_010004"]],
        per_clip[PRIMARY_EVAL["VIRAT_S_010204"]],
    ]
    both_reject = all(row["verdict"] != "pass" for row in primary)
    result = {
        "schema_version": "sceneconfig.residual-reject.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "arm": "R1_ledger_reject",
        "rule_id": RULE_ID,
        "rule": {"confidence_lt": CONF_MAX, "height_lt": HEIGHT_MAX, "support_iou": SUPPORT_IOU, "causal": True},
        "qwen_called": False,
        "query_run": False,
        "holdout_opened": False,
        "confirm_opened": False,
        "sealed_camera_used": False,
        "clip_count": len(per_clip),
        "pass_count": sum(row["verdict"] == "pass" for row in per_clip.values()),
        "h1": {
            "primary_clips": list(PRIMARY_EVAL.values()),
            "both_legal_reject": both_reject,
            "passed": False,
            "reason": "both_primary_eval_clips_failed_or_held" if both_reject else "confirmation_closed",
        },
        "per_clip": per_clip,
        "next_gate": "STOP_R1_LEGAL_REJECT" if both_reject else "OPTIONAL_R2_QUERY_AFTER_RECEIPT",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top6-predictions", type=Path)
    args = parser.parse_args()
    result = run(
        cache_root=args.cache_root,
        pilot_root=args.pilot_root,
        proxy_path=args.proxy,
        output_path=args.output,
        top6_path=args.top6_predictions,
    )
    print(json.dumps({"pass_count": result["pass_count"], "h1": result["h1"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
