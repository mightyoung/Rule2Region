"""Residual commissioning arm R3: history-only VLM box vote, one compiled threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from research_tools.cascade_experiment import Box, Prediction
from research_tools.residual_compiler import PRIMARY_EVAL, box_height, has_causal_support
from research_tools.residual_query import expand_roi
from research_tools.route_a_clip_commissioning import load_clips, score_condition
from research_tools.visual_input_health import parse_mtmd_log


RULE_ID = "residual.vlm-vote.v1"
MAX_VOTES_PER_FRAME = 1
HOLD_RATIO = 0.30
SUPPORT_IOU = 0.30


def box_id_for(sequence: str, frame: int) -> str:
    parts = sequence.split("_")
    camera = parts[2] if len(parts) > 2 else sequence[-6:]
    clip = parts[3] if len(parts) > 3 else "xx"
    return f"b{camera}_{clip}_{frame:06d}"


def vote_prompt(box_id: str, camera_id: str, confidence: float) -> str:
    return (
        f"You judge one detector crop for fixed camera {camera_id}. "
        "This image is a native crop of one box, not a collage or full frame. "
        f"box_id must be exactly {box_id}. "
        f"Ledger confidence is {confidence:.3f}. "
        "Look only inside this crop. judgment must be PERSON, FP, or UNSURE. "
        "Do not output config_id, tiles, thresholds, or polygons. Return only JSON."
    )


def select_history_targets(clip: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    jobs: list[dict[str, Any]] = []
    history = set(clip["history_frames"])
    validation = set(clip["validation_frames"])
    if history & validation:
        raise ValueError("history_validation_overlap")
    for frame in clip["history_frames"]:
        rows = list(clip["global_by_frame"].get(frame, ()))
        if not rows:
            continue
        ranked = sorted(rows, key=lambda row: (row.confidence, box_height(row.box), row.box.coords))
        chosen = ranked[:MAX_VOTES_PER_FRAME]
        for pred in chosen:
            jobs.append(
                {
                    "box_id": box_id_for(clip["sequence_id"], frame),
                    "camera_id": clip["camera_id"],
                    "sequence_id": clip["sequence_id"],
                    "frame": frame,
                    "confidence": float(pred.confidence),
                    "height": box_height(pred.box),
                    "xyxy": list(pred.box.coords),
                    "split": "history",
                }
            )
    if any(job["frame"] in validation for job in jobs):
        raise ValueError("validation_frame_selected_for_vlm")
    return tuple(jobs)


def parse_vote(text: str, box_id: str) -> dict[str, Any]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict) or set(payload) != {"box_id", "judgment"}:
        raise ValueError("vote_keys_invalid")
    if payload["box_id"] != box_id:
        raise ValueError("vote_box_id_mismatch")
    if payload["judgment"] not in {"PERSON", "FP", "UNSURE"}:
        raise ValueError("vote_judgment_invalid")
    return {"box_id": box_id, "judgment": payload["judgment"], "parse_ok": True}


def fit_tau(votes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usable = [row for row in votes if row.get("parse_ok") and row.get("judgment") in {"PERSON", "FP"}]
    fps = [float(row["confidence"]) for row in usable if row["judgment"] == "FP"]
    persons = [float(row["confidence"]) for row in usable if row["judgment"] == "PERSON"]
    if not fps or not persons:
        return {"tau": None, "reason": "insufficient_labels", "n_fp": len(fps), "n_person": len(persons)}
    if max(fps) >= min(persons):
        return {"tau": None, "reason": "overlapping_confidences", "n_fp": len(fps), "n_person": len(persons)}
    tau = (max(fps) + min(persons)) / 2.0
    return {"tau": tau, "reason": "separated", "n_fp": len(fps), "n_person": len(persons)}


def clip_visual_hold(votes: Sequence[Mapping[str, Any]]) -> bool:
    if not votes:
        return True
    bad = sum(1 for row in votes if row.get("token_position_warning") or not row.get("parse_ok"))
    return bad / len(votes) >= HOLD_RATIO


def apply_tau(
    by_frame: Mapping[int, Sequence[Prediction]],
    frames: Sequence[int],
    tau: float | None,
) -> dict[int, tuple[Prediction, ...]]:
    output: dict[int, tuple[Prediction, ...]] = {}
    previous: tuple[Prediction, ...] = ()
    for frame in frames:
        current = tuple(by_frame.get(frame, ()))
        if tau is None:
            kept = current
        else:
            kept = tuple(
                row
                for row in current
                if not (row.confidence < tau and not has_causal_support(row.box, previous, iou=SUPPORT_IOU))
            )
        output[frame] = kept
        previous = current
    return output


def _write_crop(source: Path, xyxy: Sequence[float], dest: Path) -> None:
    roi = expand_roi(Box(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])))
    with Image.open(source) as raw:
        cut = raw.convert("RGB").crop((int(roi.x1), int(roi.y1), int(roi.x2), int(roi.y2)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    cut.save(dest, format="JPEG", quality=95, subsampling=0)


def prepare(*, cache_root: Path, pilot_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    clips = load_clips(cache_root, pilot_root)
    output_dir.mkdir(parents=True)
    jobs: list[dict[str, Any]] = []
    for sequence, clip in sorted(clips.items()):
        for job in select_history_targets(clip):
            image = output_dir / "crops" / sequence / f"{job['box_id']}.jpg"
            prompt = output_dir / "prompts" / f"{job['box_id']}.txt"
            source = pilot_root / "sequences" / sequence / "img1" / f"{job['frame']:06d}.jpg"
            if job["frame"] in set(clip["validation_frames"]):
                raise ValueError("validation_frame_selected_for_vlm")
            _write_crop(source, job["xyxy"], image)
            prompt.parent.mkdir(parents=True, exist_ok=True)
            prompt.write_text(vote_prompt(job["box_id"], job["camera_id"], job["confidence"]), encoding="utf-8")
            record = {
                **job,
                "image": image.as_posix(),
                "prompt": prompt.as_posix(),
            }
            jobs.append(record)
    (output_dir / "jobs.json").write_text(json.dumps(jobs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tsv = output_dir / "jobs.tsv"
    tsv.write_text(
        "".join(f"{row['box_id']}\t{row['image']}\t{row['prompt']}\n" for row in jobs),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "sceneconfig.residual-vlm-vote.prepare.v1",
        "rule_id": RULE_ID,
        "job_count": len(jobs),
        "val_images_sent_to_vlm": False,
        "collage_used": False,
        "jobs": jobs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def ingest_votes(jobs: Sequence[Mapping[str, Any]], response_dir: Path) -> list[dict[str, Any]]:
    votes: list[dict[str, Any]] = []
    for job in jobs:
        box_id = job["box_id"]
        response = response_dir / f"{box_id}.json"
        log_path = response_dir / f"{box_id}.log"
        parsed_log = parse_mtmd_log(log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else "")
        try:
            vote = parse_vote(response.read_text(encoding="utf-8"), box_id)
        except (OSError, ValueError, json.JSONDecodeError):
            vote = {"box_id": box_id, "judgment": "UNSURE", "parse_ok": False}
        votes.append(
            {
                **dict(job),
                **vote,
                "token_position_warning": bool(parsed_log.get("token_position_warning")),
                "encoder_ran": bool(parsed_log.get("encoder_ran")),
            }
        )
    return votes


def finalize(
    *,
    cache_root: Path,
    pilot_root: Path,
    prepare_dir: Path,
    response_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refuse_existing_output:{output_path}")
    jobs = json.loads((prepare_dir / "jobs.json").read_text(encoding="utf-8"))
    if any(row.get("split") != "history" for row in jobs):
        raise ValueError("non_history_job_in_prepare")
    votes = ingest_votes(jobs, response_dir)
    clips = load_clips(cache_root, pilot_root)
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for vote in votes:
        by_sequence.setdefault(vote["sequence_id"], []).append(vote)
    per_clip: dict[str, Any] = {}
    any_token_warning = any(row["token_position_warning"] for row in votes)
    for sequence, clip in sorted(clips.items()):
        clip_votes = by_sequence.get(sequence, [])
        hold = clip_visual_hold(clip_votes)
        fitted = {"tau": None, "reason": "visual_hold", "n_fp": 0, "n_person": 0} if hold else fit_tau(clip_votes)
        compiled = apply_tau(clip["global_by_frame"], clip["frames"], fitted["tau"])
        val_only = {frame: compiled[frame] for frame in clip["validation_frames"]}
        dropped = sum(
            len(clip["global_by_frame"].get(frame, ())) - len(val_only[frame]) for frame in clip["validation_frames"]
        )
        scored = score_condition(clip, val_only, tile_cost_ratio=0.0)
        camera_clips = sorted(
            item["sequence_id"] for item in clips.values() if item["camera_id"] == clip["camera_id"]
        )
        per_clip[sequence] = {
            "camera_id": clip["camera_id"],
            "sequence_id": sequence,
            "role": "development_eval" if sequence == camera_clips[0] else "transfer",
            "rule_id": RULE_ID,
            "vote_count": len(clip_votes),
            "visual_hold": hold,
            "fit": fitted,
            "val_boxes_dropped": dropped,
            "fp_votes": sum(row["judgment"] == "FP" and row.get("parse_ok") for row in clip_votes),
            "person_votes": sum(row["judgment"] == "PERSON" and row.get("parse_ok") for row in clip_votes),
            "unsure_or_invalid": sum(row["judgment"] == "UNSURE" or not row.get("parse_ok") for row in clip_votes),
            **scored,
        }
    primary = [
        per_clip[PRIMARY_EVAL["VIRAT_S_010004"]],
        per_clip[PRIMARY_EVAL["VIRAT_S_010204"]],
    ]
    both_reject = all(row["verdict"] != "pass" for row in primary)
    result = {
        "schema_version": "sceneconfig.residual-vlm-vote.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "arm": "R3_vlm_box_vote",
        "rule_id": RULE_ID,
        "qwen_called": True,
        "query_run": False,
        "r1_thresholds_changed": False,
        "val_images_sent_to_vlm": False,
        "collage_used": False,
        "token_position_warning_any": any_token_warning,
        "holdout_opened": False,
        "confirm_opened": False,
        "sealed_camera_used": False,
        "clip_count": len(per_clip),
        "pass_count": sum(row["verdict"] == "pass" for row in per_clip.values()),
        "h1": {
            "primary_clips": list(PRIMARY_EVAL.values()),
            "both_legal_reject": both_reject,
            "passed": not both_reject and all(row["verdict"] == "pass" for row in primary),
            "reason": "both_primary_eval_clips_failed_or_held" if both_reject else "confirmation_closed",
            "scoring_arm": "VLM_HISTORY_TAU",
        },
        "per_clip": per_clip,
        "votes": [
            {
                "box_id": row["box_id"],
                "sequence_id": row["sequence_id"],
                "frame": row["frame"],
                "judgment": row["judgment"],
                "parse_ok": row["parse_ok"],
                "token_position_warning": row["token_position_warning"],
                "confidence": row["confidence"],
            }
            for row in votes
        ],
        "next_gate": "STOP_R3_LEGAL_REJECT" if both_reject else "STOP_WRITE_R3_RECEIPT_CONFIRMATION_CLOSED",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_cmd = sub.add_parser("prepare")
    prepare_cmd.add_argument("--cache-root", type=Path, required=True)
    prepare_cmd.add_argument("--pilot-root", type=Path, required=True)
    prepare_cmd.add_argument("--output-dir", type=Path, required=True)
    final_cmd = sub.add_parser("finalize")
    final_cmd.add_argument("--cache-root", type=Path, required=True)
    final_cmd.add_argument("--pilot-root", type=Path, required=True)
    final_cmd.add_argument("--prepare-dir", type=Path, required=True)
    final_cmd.add_argument("--response-dir", type=Path, required=True)
    final_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(cache_root=args.cache_root, pilot_root=args.pilot_root, output_dir=args.output_dir)
        print(json.dumps({"job_count": manifest["job_count"], "val_images_sent_to_vlm": False}, sort_keys=True))
        return 0
    result = finalize(
        cache_root=args.cache_root,
        pilot_root=args.pilot_root,
        prepare_dir=args.prepare_dir,
        response_dir=args.response_dir,
        output_path=args.output,
    )
    print(json.dumps({"pass_count": result["pass_count"], "h1": result["h1"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
