"""Four parallel single-image review agents, then one aggregator."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_tools.visual_input_health import (
    compile_structure_from_judgments,
    parse_crop_judgment,
    parse_crop_suspect,
    parse_mtmd_log,
)


AGENT_RE = re.compile(r"^A\d{2}$")
FRAME_RE = re.compile(r"^f\d+$")
RECEIPT_RE = re.compile(r"^(A\d{2})_(f\d+)_(\d{8}T\d{6}Z)\.json$")


def receipt_name(agent_id: str, frame_id: str, when: datetime | None = None) -> str:
    if not AGENT_RE.fullmatch(agent_id):
        raise ValueError("agent_id_invalid")
    if not FRAME_RE.fullmatch(frame_id):
        raise ValueError("frame_id_invalid")
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{agent_id}_{frame_id}_{stamp}.json"


def parse_receipt_name(name: str) -> dict[str, str]:
    match = RECEIPT_RE.fullmatch(name)
    if not match:
        raise ValueError("receipt_name_invalid")
    return {"agent_id": match.group(1), "frame_id": match.group(2), "timestamp": match.group(3)}


def _remote(args: Sequence[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "research_tools.remote.secure_remote", "--connection-file", ".env", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_one(job: Mapping[str, Any], output_dir: Path, *, when: datetime | None = None) -> dict[str, Any]:
    agent_id = str(job["agent_id"])
    frame_id = str(job["frame_id"])
    box_id = str(job["box_id"])
    gpu = int(job["gpu"])
    name = receipt_name(agent_id, frame_id, when)
    dest = output_dir / name
    if dest.exists():
        raise FileExistsError(f"refuse_existing_output:{dest}")
    output_dir.mkdir(parents=True, exist_ok=True)
    remote_stem = f"fieldpack-g0/tmp-parallel-review/{agent_id}_{frame_id}"
    remote_abs = f"/srv/project/{remote_stem}"
    image = job["remote_image"]
    judge_prompt = job["remote_judgment_prompt"]
    suspect_prompt = job["remote_suspect_prompt"]
    schema_judge = "/srv/project/fieldpack-g0/source/research_tools/remote/crop_judgment_schema.json"
    schema_suspect = "/srv/project/fieldpack-g0/source/research_tools/remote/crop_suspect_schema.json"
    bin_path = "/srv/project/sop-compliance-demo/runtime/llama.cpp-master/build-mtmd/bin/llama-mtmd-cli"
    model = "/srv/project/models/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf"
    mmproj = "/srv/project/models/Qwen3.5-9B-Q4_K_M/mmproj-BF16.gguf"
    command = (
        f"set -euo pipefail; mkdir -p {remote_abs}; "
        f"test -f {image} && test -f {judge_prompt} && test -f {suspect_prompt}; "
        f"CUDA_VISIBLE_DEVICES={gpu} {bin_path} -m {model} --mmproj {mmproj} --image {image} "
        f"--ctx-size 8192 --perf -p \"$(cat {judge_prompt})\" -n 160 -ngl 999 --temp 0 "
        f"--json-schema-file {schema_judge} > {remote_abs}/judgment.json 2> {remote_abs}/judgment.log; "
        f"CUDA_VISIBLE_DEVICES={gpu} {bin_path} -m {model} --mmproj {mmproj} --image {image} "
        f"--ctx-size 8192 --perf -p \"$(cat {suspect_prompt})\" -n 128 -ngl 999 --temp 0 "
        f"--json-schema-file {schema_suspect} > {remote_abs}/suspect.json 2> {remote_abs}/suspect.log"
    )
    executed = _remote(["exec", "--timeout", "240", command], timeout=260)
    local_tmp = output_dir / f".{agent_id}_{frame_id}.raw"
    local_tmp.mkdir(exist_ok=True)
    files = {
        "judgment.json": local_tmp / "judgment.json",
        "suspect.json": local_tmp / "suspect.json",
        "judgment.log": local_tmp / "judgment.log",
        "suspect.log": local_tmp / "suspect.log",
    }
    for remote_name, local_path in files.items():
        _remote(["get", f"{remote_stem}/{remote_name}", str(local_path)], timeout=60)
    errors: list[str] = []
    if executed.returncode != 0:
        errors.append(f"remote_exit:{executed.returncode}")
    judgment = None
    suspect = None
    try:
        judgment = parse_crop_judgment(files["judgment.json"].read_text(encoding="utf-8"), box_id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"judgment:{exc}")
    try:
        suspect = parse_crop_suspect(files["suspect.json"].read_text(encoding="utf-8"), box_id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"suspect:{exc}")
    judge_log = parse_mtmd_log(files["judgment.log"].read_text(encoding="utf-8", errors="replace") if files["judgment.log"].is_file() else "")
    suspect_log = parse_mtmd_log(files["suspect.log"].read_text(encoding="utf-8", errors="replace") if files["suspect.log"].is_file() else "")
    receipt = {
        "schema_version": "sceneconfig.parallel-agent-review.receipt.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "agent_id": agent_id,
        "frame_id": frame_id,
        "box_id": box_id,
        "camera_id": job["camera_id"],
        "gpu": gpu,
        "preconfig": job["preconfig"],
        "input_kind": "one_crop_plus_text_preconfig",
        "collage_used": False,
        "full_frame_used": False,
        "confirm_opened": False,
        "holdout_opened": False,
        "judgment": judgment,
        "suspect": suspect,
        "visual_runtime": {"judgment": judge_log, "suspect": suspect_log},
        "errors": errors,
        "receipt_name": name,
        "remote_exit": executed.returncode,
    }
    dest.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def aggregate_receipts(paths: Sequence[Path], output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refuse_existing_output:{output_path}")
    receipts = []
    for path in sorted(paths):
        parsed = parse_receipt_name(path.name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("agent_id") != parsed["agent_id"] or payload.get("frame_id") != parsed["frame_id"]:
            raise ValueError(f"receipt_identity_mismatch:{path.name}")
        receipts.append(payload)
    agent_ids = [row["agent_id"] for row in receipts]
    if len(agent_ids) != len(set(agent_ids)):
        raise ValueError("duplicate_agent_id")
    crops = []
    judgments = []
    suspects = []
    for row in receipts:
        if not row.get("judgment"):
            continue
        crops.append(
            {
                "box_id": row["box_id"],
                "frame": int(row["frame_id"][1:]),
                "cell": row["preconfig"]["cell"],
                "source": row["preconfig"]["source"],
                "camera_id": row["camera_id"],
            }
        )
        judgments.append(row["judgment"])
        if row.get("suspect") is True:
            suspects.append(row["box_id"])
    by_camera: dict[str, dict[str, Any]] = {}
    for crop, judgment in zip(crops, judgments):
        camera = crop["camera_id"]
        by_camera.setdefault(camera, {"crops": [], "judgments": []})
        by_camera[camera]["crops"].append(crop)
        by_camera[camera]["judgments"].append(judgment)
    compiled = {
        camera: compile_structure_from_judgments(bundle["crops"], bundle["judgments"])
        for camera, bundle in by_camera.items()
    }
    result = {
        "schema_version": "sceneconfig.parallel-agent-review.aggregate.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "aggregator": "main_agent",
        "agent_count": len(receipts),
        "receipts": [row["receipt_name"] for row in receipts],
        "per_agent": [
            {
                "agent_id": row["agent_id"],
                "frame_id": row["frame_id"],
                "camera_id": row["camera_id"],
                "box_id": row["box_id"],
                "judgment": row.get("judgment"),
                "suspect": row.get("suspect"),
                "errors": row.get("errors") or [],
                "token_position_warning": bool(
                    (row.get("visual_runtime") or {}).get("judgment", {}).get("token_position_warning")
                ),
            }
            for row in receipts
        ],
        "compiled_by_camera": compiled,
        "suspect_box_ids": suspects,
        "collage_used": False,
        "confirm_opened": False,
        "holdout_opened": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def default_jobs() -> list[dict[str, Any]]:
    root = "/srv/project-assets/fieldpack-g0/sceneconfig-visual-health-boxcrop-20260817-v2/boxcrop/media"
    rows = (
        ("A01", "VIRAT_S_010004", "f249", "f249b01", "t2", "yolo26_hr_tile:t2", 1),
        ("A02", "VIRAT_S_010111", "f1327", "f1327b01", "t0", "yolo26_hr_tile:t0", 0),
        ("A03", "VIRAT_S_010112", "f878", "f878b01", "t3", "yolo11s_global", 1),
        ("A04", "VIRAT_S_010204", "f194", "f194b01", "t3", "yolo26_hr_tile:t3", 0),
    )
    jobs = []
    for agent_id, camera, frame_id, box_id, cell, source, gpu in rows:
        media = f"{root}/{camera}"
        jobs.append(
            {
                "agent_id": agent_id,
                "camera_id": camera,
                "frame_id": frame_id,
                "box_id": box_id,
                "gpu": gpu,
                "remote_image": f"{media}/{box_id}.jpg",
                "remote_judgment_prompt": f"{media}/{box_id}.judgment-prompt.txt",
                "remote_suspect_prompt": f"{media}/{box_id}.suspect-prompt.txt",
                "preconfig": {
                    "camera_id": camera,
                    "box_id": box_id,
                    "frame": int(frame_id[1:]),
                    "cell": cell,
                    "source": source,
                    "note": "Single native detector crop. Config is text-only. Not a collage.",
                },
            }
        )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    write_jobs = sub.add_parser("write-jobs")
    write_jobs.add_argument("--output-dir", type=Path, required=True)
    run = sub.add_parser("run-one")
    run.add_argument("--job", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--receipt-dir", type=Path, required=True)
    agg.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "write-jobs":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for job in default_jobs():
            path = args.output_dir / f"{job['agent_id']}.json"
            if path.exists():
                raise FileExistsError(f"refuse_existing_output:{path}")
            path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            written.append(path.name)
        print(json.dumps({"jobs": written}))
        return 0
    if args.action == "run-one":
        receipt = run_one(json.loads(args.job.read_text(encoding="utf-8")), args.output_dir)
        print(json.dumps({"receipt_name": receipt["receipt_name"], "errors": receipt["errors"]}))
        return 0 if not receipt["errors"] else 1
    receipts = list(args.receipt_dir.glob("A*_f*_*.json"))
    result = aggregate_receipts(receipts, args.output)
    print(json.dumps({"agent_count": result["agent_count"], "receipts": result["receipts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
