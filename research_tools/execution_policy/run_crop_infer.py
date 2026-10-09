"""Run exclusive per-region crop inference on a frozen E4 card set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research_tools.execution_policy.card import latency_proxy, parse_policy_card
from research_tools.execution_policy.compile import apply_card, h1_pass
from research_tools.execution_policy.crop_infer import infer_card_frame, infer_full_frame, load_bgr
from research_tools.execution_policy.ledger import frame_ledger_from_json
from research_tools.execution_policy.run_e4 import load_dual_ledger, view_id
from research_tools.execution_policy.score import _arm_metrics
from research_tools.residual_compiler import PRIMARY_EVAL
from research_tools.route_a_clip_commissioning import load_clips


FULL_CONF = 0.25


def load_view_cards(result: dict[str, Any]) -> dict[str, Any]:
    cards = {}
    for view, row in result["per_camera"].items():
        payload = row.get("intention_card")
        if not payload:
            continue
        cards[view] = parse_policy_card(payload)
    if not cards:
        raise ValueError("no_intention_cards")
    return cards


def _pack(arm: dict[str, float], *, time_ms: float | None = None, proxy: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = dict(arm)
    if time_ms is not None:
        out["time_ms_mean"] = time_ms
    if proxy is not None:
        out["latency_proxy"] = proxy
    return out


def score_split(
    card,
    ledgers,
    frames,
    truth,
    compiled_by_frame,
    *,
    compiled_time_ms: float,
    b26_time_ms: float,
    scoring: str,
) -> dict[str, Any]:
    b11 = {frame: ledgers[frame].yolo11s for frame in frames}
    b26 = {frame: ledgers[frame].yolo26 for frame in frames}
    filtered = {frame: apply_card(card, ledgers[frame]).for_metrics() for frame in frames}
    b11_m = _arm_metrics(frames, b11, truth)
    b26_m = _arm_metrics(frames, b26, truth)
    crop_m = _arm_metrics(frames, compiled_by_frame, truth)
    filter_m = _arm_metrics(frames, filtered, truth)
    proxy = latency_proxy(card)
    gate = h1_pass(
        compiled_mae=crop_m["mae"],
        b11_mae=b11_m["mae"],
        precision_change_vs_b11=crop_m["precision"] - b11_m["precision"],
        compiled_proxy=proxy,
        compiled_time_ms=compiled_time_ms,
        b26_time_ms=b26_time_ms,
    )
    return {
        "scoring": scoring,
        "b11": _pack(b11_m),
        "b26": _pack(b26_m, time_ms=b26_time_ms),
        "ledger_filter": _pack(filter_m, proxy=proxy),
        "crop_infer": _pack(crop_m, time_ms=compiled_time_ms, proxy=proxy),
        "h1": gate,
        "mae_change_vs_b11": gate["mae_change_vs_b11"],
        "mae_change_vs_b26": (0.0 if b26_m["mae"] == 0 else (b26_m["mae"] - crop_m["mae"]) / b26_m["mae"]),
        "precision_change_vs_b11": crop_m["precision"] - b11_m["precision"],
        "precision_change_vs_b26": crop_m["precision"] - b26_m["precision"],
        "time_ratio_vs_b26": (compiled_time_ms / b26_time_ms) if b26_time_ms else None,
    }


def run(
    *,
    cache_root: Path,
    pilot_root: Path,
    dual_ledger_path: Path,
    e4_result: Path,
    output_dir: Path,
    yolo11s: Path,
    yolo26: Path,
    device: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    output_dir.mkdir(parents=True)
    from ultralytics import YOLO  # type: ignore

    clips = load_clips(cache_root, pilot_root)
    dual = load_dual_ledger(dual_ledger_path)
    cards = load_view_cards(json.loads(e4_result.read_text(encoding="utf-8")))
    models = {"yolo11s": YOLO(str(yolo11s)), "yolo26": YOLO(str(yolo26))}
    warmup = next(iter(clips.values()))
    warm_frame = int(warmup["history_frames"][0])
    warm_path = pilot_root / "sequences" / warmup["sequence_id"] / "img1" / f"{warm_frame:06d}.jpg"
    warm_bgr = load_bgr(warm_path)
    infer_full_frame(models["yolo26"], warm_bgr, warm_frame, conf=FULL_CONF, source="yolo26", device=device)
    infer_full_frame(models["yolo11s"], warm_bgr, warm_frame, conf=FULL_CONF, source="yolo11s", device=device)

    per_clip: dict[str, Any] = {}
    for sequence, clip in sorted(clips.items()):
        view = view_id(str(clip["camera_id"]))
        card = cards.get(view)
        if card is None:
            per_clip[sequence] = {"camera_id": clip["camera_id"], "view_id": view, "skipped": "no_card"}
            continue
        ledgers = dual[sequence]
        splits: dict[str, Any] = {}
        for split_name, frames in (
            ("history", list(clip["history_frames"])),
            ("validation", list(clip["validation_frames"])),
        ):
            compiled: dict[int, tuple] = {}
            crop_times: list[float] = []
            b26_times: list[float] = []
            for frame in frames:
                image = pilot_root / "sequences" / sequence / "img1" / f"{frame:06d}.jpg"
                bgr = load_bgr(image)
                parted, crop_ms = infer_card_frame(models, card, bgr, frame, device=device)
                _, full_ms = infer_full_frame(
                    models["yolo26"], bgr, frame, conf=FULL_CONF, source="yolo26", device=device
                )
                compiled[frame] = parted.for_metrics()
                crop_times.append(crop_ms)
                b26_times.append(full_ms)
            crop_mean = sum(crop_times) / len(crop_times)
            b26_mean = sum(b26_times) / len(b26_times)
            scored = score_split(
                card,
                ledgers,
                frames,
                clip["truth"],
                compiled,
                compiled_time_ms=crop_mean,
                b26_time_ms=b26_mean,
                scoring="crop_infer",
            )
            splits[split_name] = scored
        per_clip[sequence] = {
            "camera_id": clip["camera_id"],
            "view_id": view,
            "card": {
                "regions": [
                    {
                        "region_id": row.region_id,
                        "xyxy": list(row.box.coords),
                        "model": row.model,
                        "conf": row.conf,
                        "role": row.role,
                    }
                    for row in card.regions
                ]
            },
            "history": splits["history"],
            "validation": splits["validation"],
            "verdict": splits["validation"]["h1"]["verdict"],
        }

    primary = [per_clip[PRIMARY_EVAL["VIRAT_S_010004"]], per_clip[PRIMARY_EVAL["VIRAT_S_010204"]]]
    both_reject = all(row.get("verdict") != "pass" for row in primary)
    result = {
        "schema_version": "sceneconfig.execution-policy.crop-infer.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "official_h1": False,
        "e4_source": str(e4_result),
        "val_score": "crop_infer",
        "history_score": "crop_infer_and_ledger_filter",
        "uncovered_inferred": False,
        "image_resized_before_detect": False,
        "imgsz": 640,
        "sealed_camera_used": False,
        "clip_count": len(per_clip),
        "pass_count": sum(row.get("verdict") == "pass" for row in per_clip.values()),
        "h1": {
            "primary_clips": list(PRIMARY_EVAL.values()),
            "both_legal_reject": both_reject,
            "passed": not both_reject and all(row.get("verdict") == "pass" for row in primary),
            "reason": "development_crop_infer_not_official_h1",
        },
        "per_clip": per_clip,
        "next_gate": "SEAL_LONGER_CLIPS_BEFORE_OFFICIAL_H1",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--dual-ledger", type=Path, required=True)
    parser.add_argument("--e4-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yolo11s", type=Path, required=True)
    parser.add_argument("--yolo26", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    result = run(
        cache_root=args.cache_root,
        pilot_root=args.pilot_root,
        dual_ledger_path=args.dual_ledger,
        e4_result=args.e4_result,
        output_dir=args.output_dir,
        yolo11s=args.yolo11s,
        yolo26=args.yolo26,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "pass_count": result["pass_count"],
                "official_h1": False,
                "h1": result["h1"],
                "val_score": "crop_infer",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
