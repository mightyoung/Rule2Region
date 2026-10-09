"""Parallel agents: one full scene frame + text config + that frame's boxes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_tools.cascade_experiment import Box
from research_tools.parallel_agent_review import _remote, parse_receipt_name, receipt_name
from research_tools.roi_structure_protocol import (
    EXPECTED_CAMERAS,
    PRIMARY_CAMERAS,
    _owned_cell,
    apply_patch,
    assign_disposition,
    evaluate_camera,
    parse_structure,
)
from research_tools.visual_input_health import parse_mtmd_log


REMOTE_SOURCE = "/srv/project/fieldpack-g0/source"
REMOTE_HOME = "/srv/project"
BIN = "/srv/project/sop-compliance-demo/runtime/llama.cpp-master/build-mtmd/bin/llama-mtmd-cli"
MODEL = "/srv/project/models/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf"
MMPROJ = "/srv/project/models/Qwen3.5-9B-Q4_K_M/mmproj-BF16.gguf"
PRIOR_MEDIA = "/srv/project-assets/fieldpack-g0/sceneconfig-roi-structure-valgate-20260817-v1/prepare/media"
SCHEMA_REL = "research_tools/remote/frame_review_schema.json"
N_PREDICT = 768

FRAME_JOBS = (
    ("A01", "VIRAT_S_010004", 249, "C15", 1),
    ("A02", "VIRAT_S_010111", 1327, "C16", 0),
    ("A03", "VIRAT_S_010112", 878, "C16", 1),
    ("A04", "VIRAT_S_010204", 194, "C19", 0),
)


def frame_review_prompt(camera_id: str, frame: int) -> str:
    return (
        f"You review one native unstitched 1280x720 scene frame of camera {camera_id}, frame {frame}. "
        "The attached image is the full scene, not a crop and not a collage. "
        "CONFIG and BOXES are separate text files whose contents follow this paragraph. "
        "BOXES use this same frame's pixel coordinates. "
        "Assign at most four cells from t0,t1,t2,t3 to INCLUDE, EXCLUDE, or EXPERT. At most two EXPERT cells. "
        "suspect_box_ids must be a subset of box_id values listed in BOXES. "
        "Do not output config_id, thresholds, or pixel polygons. "
        "Return only JSON with keys abstain, cells, suspect_box_ids, rationale."
    )


def parse_frame_review(text: str, allowed_box_ids: set[str]) -> dict[str, Any]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict) or set(payload) != {"abstain", "cells", "suspect_box_ids", "rationale"}:
        raise ValueError("frame_review_keys_invalid")
    structure = parse_structure(json.dumps({"abstain": payload["abstain"], "cells": payload["cells"]}))
    suspects = payload["suspect_box_ids"]
    if not isinstance(suspects, list) or not all(isinstance(item, str) for item in suspects):
        raise ValueError("suspect_box_ids_invalid")
    if len(suspects) != len(set(suspects)):
        raise ValueError("suspect_box_ids_duplicate")
    leaked = [item for item in suspects if item not in allowed_box_ids]
    if leaked:
        raise ValueError("suspect_box_id_not_in_frame_boxes")
    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        raise ValueError("rationale_required")
    return {
        "abstain": structure["abstain"],
        "cells": structure["cells"],
        "suspect_box_ids": tuple(suspects),
        "rationale": payload["rationale"].strip(),
    }


def boxes_from_observer_frame(row: Mapping[str, Any], frame: int) -> tuple[dict[str, Any], ...]:
    match = next((item for item in row["frames"] if int(item["frame"]) == frame), None)
    if match is None:
        raise ValueError(f"frame_missing_from_observer:{frame}")
    boxes = []
    for index, det in enumerate(match.get("merged") or match.get("global") or (), start=1):
        box = det["box"]
        owned = _owned_cell(Box(float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])), 1280, 720)
        boxes.append(
            {
                "box_id": f"f{frame}b{index:02d}",
                "cell": owned,
                "source": str(det.get("source") or "detector"),
                "confidence": round(float(det.get("confidence") or 0.0), 3),
                "xyxy": [round(float(box[key]), 1) for key in ("x1", "y1", "x2", "y2")],
            }
        )
    return tuple(boxes)


def prepare_jobs(*, predictions_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                index[(row["camera_id"], row["config_id"])] = row
    jobs_dir = output_dir / "jobs"
    inputs_dir = output_dir / "inputs"
    jobs_dir.mkdir(parents=True)
    written = []
    for agent_id, camera, frame, observer, gpu in FRAME_JOBS:
        boxes = boxes_from_observer_frame(index[(camera, observer)], frame)
        bundle = inputs_dir / agent_id
        bundle.mkdir(parents=True)
        config = {
            "camera_id": camera,
            "frame": frame,
            "observer_config_id": observer,
            "image_kind": "native_full_frame_1280x720",
            "tiles": ["t0", "t1", "t2", "t3"],
            "tile_rule": "quadrants, ownership by box center",
            "note": "Config is text-only. Image is the unstitched full scene. Boxes are a separate file.",
        }
        (bundle / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (bundle / "boxes.json").write_text(json.dumps({"frame": frame, "boxes": list(boxes)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prompt = (
            frame_review_prompt(camera, frame)
            + "\n\n----- FILE config.json -----\n"
            + (bundle / "config.json").read_text(encoding="utf-8")
            + "\n----- FILE boxes.json -----\n"
            + (bundle / "boxes.json").read_text(encoding="utf-8")
        )
        (bundle / "prompt.txt").write_text(prompt, encoding="utf-8")
        scene = f"{PRIOR_MEDIA}/{camera}/{frame:06d}.clean.jpg"
        job = {
            "agent_id": agent_id,
            "camera_id": camera,
            "frame_id": f"f{frame}",
            "frame": frame,
            "gpu": gpu,
            "observer_config_id": observer,
            "scene_path": scene,
            "config_path": (bundle / "config.json").as_posix(),
            "boxes_path": (bundle / "boxes.json").as_posix(),
            "prompt_path": (bundle / "prompt.txt").as_posix(),
            "box_ids": [row["box_id"] for row in boxes],
            "input_files": ["scene", "config.json", "boxes.json"],
        }
        job_path = jobs_dir / f"{agent_id}.json"
        job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(job)
    return written


def run_one(job: Mapping[str, Any], output_dir: Path, *, when: datetime | None = None) -> dict[str, Any]:
    agent_id = str(job["agent_id"])
    frame_id = str(job["frame_id"])
    name = receipt_name(agent_id, frame_id, when)
    dest = output_dir / name
    if dest.exists():
        raise FileExistsError(f"refuse_existing_output:{dest}")
    output_dir.mkdir(parents=True, exist_ok=True)
    remote_rel = f"fieldpack-g0/tmp-parallel-fullframe/{agent_id}_{frame_id}"
    remote_abs = f"{REMOTE_HOME}/{remote_rel}"
    put_prompt = _remote(["put", str(job["prompt_path"]), f"{remote_rel}/prompt.txt"], timeout=60)
    put_config = _remote(["put", str(job["config_path"]), f"{remote_rel}/config.json"], timeout=60)
    put_boxes = _remote(["put", str(job["boxes_path"]), f"{remote_rel}/boxes.json"], timeout=60)
    schema = f"{REMOTE_SOURCE}/{SCHEMA_REL}"
    command = (
        f"set -euo pipefail; "
        f"test -f {job['scene_path']} && test -f {remote_abs}/prompt.txt && test -f {remote_abs}/config.json && test -f {remote_abs}/boxes.json; "
        f"CUDA_VISIBLE_DEVICES={int(job['gpu'])} {BIN} -m {MODEL} --mmproj {MMPROJ} --image {job['scene_path']} "
        f"--ctx-size 8192 --perf -p \"$(cat {remote_abs}/prompt.txt)\" -n {N_PREDICT} -ngl 999 --temp 0 "
        f"--json-schema-file {schema} > {remote_abs}/review.json 2> {remote_abs}/review.log"
    )
    executed = _remote(["exec", "--timeout", "300", command], timeout=320)
    raw_dir = output_dir / f".{agent_id}_{frame_id}.raw"
    raw_dir.mkdir(exist_ok=True)
    _remote(["get", f"{remote_rel}/review.json", str(raw_dir / "review.json")], timeout=60)
    _remote(["get", f"{remote_rel}/review.log", str(raw_dir / "review.log")], timeout=60)
    errors: list[str] = []
    if put_prompt.returncode or put_config.returncode or put_boxes.returncode:
        errors.append("upload_failed")
    if executed.returncode != 0:
        errors.append(f"remote_exit:{executed.returncode}")
    review = None
    try:
        review = parse_frame_review((raw_dir / "review.json").read_text(encoding="utf-8"), set(job["box_ids"]))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"review:{exc}")
    log = parse_mtmd_log((raw_dir / "review.log").read_text(encoding="utf-8", errors="replace") if (raw_dir / "review.log").is_file() else "")
    receipt = {
        "schema_version": "sceneconfig.fullframe-agent-review.receipt.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "agent_id": agent_id,
        "frame_id": frame_id,
        "camera_id": job["camera_id"],
        "gpu": job["gpu"],
        "input_files": {
            "scene": job["scene_path"],
            "config": job["config_path"],
            "boxes": job["boxes_path"],
        },
        "input_kind": "full_frame_plus_text_config_plus_frame_boxes",
        "collage_used": False,
        "full_frame_used": True,
        "confirm_opened": False,
        "holdout_opened": False,
        "review": review,
        "visual_runtime": log,
        "errors": errors,
        "receipt_name": name,
        "remote_exit": executed.returncode,
        "n_predict": N_PREDICT,
    }
    dest.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def disposition_from_review(review: Mapping[str, Any] | None, boxes: Sequence[Mapping[str, Any]], parse_error: str | None) -> dict[str, Any]:
    if parse_error or review is None or review.get("abstain"):
        return {"level": "L4", "patch": None, "reason": parse_error or "abstain"}
    frame_ids = sorted({int(row["box_id"][1:].split("b")[0]) for row in boxes})
    suspect_frames = sorted({int(box_id[1:].split("b")[0]) for box_id in review["suspect_box_ids"]})
    cells = sorted({row["cell"] for row in boxes if row["box_id"] in set(review["suspect_box_ids"])})
    payload = {
        "level": "L3" if len(cells) == 1 else "L1",
        "suspect_frame_ids": tuple(suspect_frames),
        "patch": {"cell": cells[0], "type": "EXCLUDE"} if len(cells) == 1 else None,
        "rationale": review["rationale"],
    }
    return assign_disposition(payload, history_frames=frame_ids)


def aggregate_and_score(
    receipts: Sequence[Path],
    *,
    jobs: Sequence[Mapping[str, Any]],
    predictions_path: Path,
    seal_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refuse_existing_output:{output_path}")
    job_by_agent = {row["agent_id"]: row for row in jobs}
    parsed_receipts = []
    for path in sorted(receipts):
        ident = parse_receipt_name(path.name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("agent_id") != ident["agent_id"] or payload.get("frame_id") != ident["frame_id"]:
            raise ValueError(f"receipt_identity_mismatch:{path.name}")
        parsed_receipts.append(payload)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    index: dict[tuple[str, str], dict[str, Any]] = {}
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                index[(row["camera_id"], row["config_id"])] = row
    per_camera: dict[str, Any] = {}
    for row in parsed_receipts:
        job = job_by_agent[row["agent_id"]]
        camera = row["camera_id"]
        assets = seal["cameras"][camera]
        review = row.get("review")
        errors = list(row.get("errors") or [])
        boxes = json.loads(Path(job["boxes_path"]).read_text(encoding="utf-8"))["boxes"]
        structure = {"abstain": True, "cells": ()} if not review else {"abstain": review["abstain"], "cells": review["cells"]}
        disposition = disposition_from_review(review, boxes, ";".join(errors) or None)
        if disposition["level"] == "L3":
            try:
                apply_patch(structure.get("cells") or (), disposition.get("patch"))
            except ValueError as exc:
                disposition = {"level": "L4", "patch": None, "reason": str(exc)}
        observer = assets["observer_config_id"]
        observer_row = index[(camera, observer)]
        baseline_row = index.get((camera, "B0"), observer_row)
        # Local scoring cannot read remote GT. Record compiled recipe only; gain is filled later if present.
        per_camera[camera] = {
            "agent_id": row["agent_id"],
            "frame_id": row["frame_id"],
            "structure": structure,
            "disposition": disposition,
            "errors": errors,
            "token_position_warning": bool((row.get("visual_runtime") or {}).get("token_position_warning")),
            "proposal_frames": assets["proposal_frames"],
            "split": assets["split"],
            "observer_config_id": observer,
            "sequence_id": assets["sequence_id"],
        }
    result = {
        "schema_version": "sceneconfig.fullframe-agent-review.aggregate.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "aggregator": "main_agent",
        "input_kind": "full_frame_plus_text_config_plus_frame_boxes",
        "agent_count": len(parsed_receipts),
        "receipts": [row["receipt_name"] for row in parsed_receipts],
        "per_camera": per_camera,
        "collage_used": False,
        "full_frame_used": True,
        "confirm_opened": False,
        "holdout_opened": False,
        "sealed_camera_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def evaluate_gain(
    aggregate_path: Path,
    *,
    predictions_path: Path,
    seal_path: Path,
    pilot_root: Path,
    proxy_path: Path | None,
    output_path: Path,
    comparators: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refuse_existing_output:{output_path}")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    index: dict[tuple[str, str], dict[str, Any]] = {}
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                index[(row["camera_id"], row["config_id"])] = row
    proxy = json.loads(proxy_path.read_text(encoding="utf-8")) if proxy_path else None
    per_camera = {}
    for camera in EXPECTED_CAMERAS:
        assets = aggregate["per_camera"][camera]
        seal_cam = seal["cameras"][camera]
        observer = seal_cam["observer_config_id"]
        proxy_scores = None
        if proxy:
            proxy_scores = {row["config_id"]: float(row["proxy_utility"]) for row in proxy["per_camera"][camera]}
        evaluated = evaluate_camera(
            camera_id=camera,
            observer_row=index[(camera, observer)],
            baseline_row=index.get((camera, "B0"), index[(camera, observer)]),
            split={key: tuple(value) for key, value in seal_cam["split"].items()},
            structure=assets["structure"],
            disposition=assets["disposition"],
            gt_path=pilot_root / "sequences" / seal_cam["sequence_id"] / "gt.txt",
            width=1280,
            height=720,
            proxy_scores=proxy_scores,
        )
        evaluated["structure"] = assets["structure"]
        evaluated["disposition"] = assets["disposition"]
        evaluated["agent_id"] = assets["agent_id"]
        per_camera[camera] = evaluated
    primary = [per_camera[camera] for camera in PRIMARY_CAMERAS]
    both_reject = all(row["verdict"] != "pass" for row in primary)
    result = {
        "schema_version": "sceneconfig.fullframe-agent-review.gain.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "input_kind": "full_frame_plus_text_config_plus_frame_boxes",
        "collage_used": False,
        "confirm_opened": False,
        "holdout_opened": False,
        "h_b1": {
            "primary_cameras": list(PRIMARY_CAMERAS),
            "primary_pass_count": sum(row["verdict"] == "pass" for row in primary),
            "both_legal_reject": both_reject,
            "passed": False,
            "reason": "both_primary_cameras_failed_or_held" if both_reject else "confirmation_closed",
        },
        "per_camera": per_camera,
        "comparators": comparators or {},
        "next_gate": "STOP_WRITE_LEGAL_REJECT" if both_reject else "OPEN_CONFIRM_ONLY_IF_NOT_BOTH_PRIMARY_REJECT",
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare-jobs")
    prep.add_argument("--predictions", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    run = sub.add_parser("run-one")
    run.add_argument("--job", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--receipt-dir", type=Path, required=True)
    agg.add_argument("--jobs-dir", type=Path, required=True)
    agg.add_argument("--predictions", type=Path, required=True)
    agg.add_argument("--seal", type=Path, required=True)
    agg.add_argument("--output", type=Path, required=True)
    gain = sub.add_parser("evaluate-gain")
    gain.add_argument("--aggregate", type=Path, required=True)
    gain.add_argument("--predictions", type=Path, required=True)
    gain.add_argument("--seal", type=Path, required=True)
    gain.add_argument("--pilot-root", type=Path, required=True)
    gain.add_argument("--proxy", type=Path)
    gain.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare-jobs":
        jobs = prepare_jobs(predictions_path=args.predictions, output_dir=args.output_dir)
        print(json.dumps({"jobs": [row["agent_id"] for row in jobs], "box_counts": {row["agent_id"]: len(row["box_ids"]) for row in jobs}}))
        return 0
    if args.action == "run-one":
        receipt = run_one(json.loads(args.job.read_text(encoding="utf-8")), args.output_dir)
        print(json.dumps({"receipt_name": receipt["receipt_name"], "errors": receipt["errors"]}))
        return 0 if not receipt["errors"] else 1
    if args.action == "aggregate":
        jobs = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((args.jobs_dir).glob("A*.json"))]
        result = aggregate_and_score(
            list(args.receipt_dir.glob("A*_f*_*.json")),
            jobs=jobs,
            predictions_path=args.predictions,
            seal_path=args.seal,
            output_path=args.output,
        )
        print(json.dumps({"agent_count": result["agent_count"], "cameras": list(result["per_camera"])}))
        return 0
    evaluate_gain(
        args.aggregate,
        predictions_path=args.predictions,
        seal_path=args.seal,
        pilot_root=args.pilot_root,
        proxy_path=args.proxy,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
