"""Fail-closed visual-token health probes and box-crop VLM inputs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from research_tools.cascade_experiment import Box
from research_tools.roi_structure_protocol import (
    EXPECTED_CAMERAS,
    PRIMARY_CAMERAS,
    _owned_cell,
    _sha256_file,
    apply_patch,
    assign_disposition,
    evaluate_camera,
)
from research_tools.select_dense_detection_frames import choose_observer_config


CHUNK_RE = re.compile(
    r"encoding mtmd batch, n_chunks\s*=\s*(\d+)\s*\(done\s*=\s*(\d+),\s*total\s*=\s*(\d+)\)"
)
ENCODE_MS_RE = re.compile(r"mtmd batch encoding done in (\d+) ms")
TOKEN_POS_RE = re.compile(
    r"non-consecutive token position (\d+) after (\d+) for sequence \d+ with (\d+) new tokens"
)
DEPRECATED_RE = re.compile(r"only last value will be used")
PROMPT_TOKENS_RE = re.compile(r"prompt eval time\s*=.*?/\s*(\d+)\s+tokens", re.I)
MARKERS = ("K7Q", "A1X", "M3Z", "R8C", "T2N", "V5H", "P4D", "Z9Q")
MAX_CROPS = 6
CROP_PAD = 16
MIN_CROP = 32


def parse_mtmd_log(text: str) -> dict[str, Any]:
    chunks = [
        {"n_chunks": int(n), "done": int(done), "total": int(total)}
        for n, done, total in CHUNK_RE.findall(text)
    ]
    inserts = [
        {"position": int(pos), "after": int(after), "new_tokens": int(new)}
        for pos, after, new in TOKEN_POS_RE.findall(text)
    ]
    totals = [row["total"] for row in chunks]
    return {
        "encoder_ran": bool(chunks),
        "encode_events": len(chunks),
        "encode_ms": [int(value) for value in ENCODE_MS_RE.findall(text)],
        "max_total_chunks": max(totals) if totals else 0,
        "chunk_progress": chunks,
        "token_position_warning": bool(TOKEN_POS_RE.search(text)),
        "token_position_events": len(inserts),
        "token_inserts": inserts,
        "inserted_visual_tokens": sum(row["new_tokens"] for row in inserts if row["new_tokens"] >= 32),
        "deprecated_repeat_image": bool(DEPRECATED_RE.search(text)),
        "prompt_tokens": int(PROMPT_TOKENS_RE.findall(text)[-1]) if PROMPT_TOKENS_RE.findall(text) else None,
    }


def score_visual_health(
    parsed: Mapping[str, Any],
    *,
    expected_images: int,
    identity_ok: bool | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not parsed["encoder_ran"]:
        reasons.append("vision_encoder_did_not_run")
    if parsed["token_position_warning"]:
        reasons.append("non_consecutive_token_position")
    if expected_images > 1 and parsed["deprecated_repeat_image"]:
        reasons.append("repeated_image_flag_ambiguous")
    if parsed["max_total_chunks"] == 0:
        reasons.append("no_mtmd_chunks")
    # One 1280x720 frame with min-tokens=1024 can tile into many chunks. Chunk
    # count therefore cannot prove N files were ingested.
    if expected_images > 1 and parsed["max_total_chunks"] in {expected_images, expected_images + 1}:
        reasons.append("chunk_count_compatible_with_n_images_or_single_image_tiles")
    if identity_ok is False:
        reasons.append("painted_marker_not_recovered")
    if identity_ok is True and not parsed["token_position_warning"] and not parsed["deprecated_repeat_image"]:
        binding = "healthy"
    elif identity_ok is True:
        binding = "encoder_ok_binding_unhealthy"
    elif identity_ok is False:
        binding = "identity_failed"
    else:
        binding = "encoder_only_unverified"
    return {
        "binding": binding,
        "healthy": binding == "healthy",
        "reasons": reasons,
        "expected_images": expected_images,
        "identity_ok": identity_ok,
    }


def parse_identity(text: str) -> dict[str, Any]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("identity_not_object")
    markers = payload.get("markers_seen")
    if not isinstance(markers, list) or not all(isinstance(item, str) for item in markers):
        raise ValueError("markers_seen_invalid")
    return {
        "markers_seen": tuple(item.strip().upper() for item in markers if item.strip()),
        "first_marker": str(payload.get("first_marker") or "").strip().upper() or None,
        "last_marker": str(payload.get("last_marker") or "").strip().upper() or None,
    }


def identity_matches(parsed: Mapping[str, Any], expected: Sequence[str]) -> bool:
    expected_u = tuple(item.upper() for item in expected)
    seen = parsed["markers_seen"]
    if expected_u and set(seen) == set(expected_u):
        if len(expected_u) == 1:
            return True
        return parsed["first_marker"] == expected_u[0] and parsed["last_marker"] == expected_u[-1]
    return False


def paint_marker(image: Image.Image, marker: str) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        except OSError:
            font = ImageFont.load_default()
    label = f"FRAME {marker}"
    draw.rectangle((12, 12, 250, 78), fill=(255, 255, 255), outline=(0, 0, 0), width=4)
    draw.text((22, 22), label, fill=(0, 0, 0), font=font)
    return output


def crop_detection(
    image: Image.Image,
    box: Mapping[str, float],
    *,
    pad: int = CROP_PAD,
    min_size: int = MIN_CROP,
) -> Image.Image:
    width, height = image.size
    x1 = max(0, int(box["x1"]) - pad)
    y1 = max(0, int(box["y1"]) - pad)
    x2 = min(width, int(box["x2"]) + pad)
    y2 = min(height, int(box["y2"]) + pad)
    if x2 - x1 < min_size:
        extra = min_size - (x2 - x1)
        x1 = max(0, x1 - extra // 2)
        x2 = min(width, x1 + min_size)
    if y2 - y1 < min_size:
        extra = min_size - (y2 - y1)
        y1 = max(0, y1 - extra // 2)
        y2 = min(height, y1 + min_size)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("empty_detection_crop")
    return image.convert("RGB").crop((x1, y1, x2, y2))


def select_box_crops(
    frames: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    k: int = MAX_CROPS,
) -> tuple[dict[str, Any], ...]:
    ranked: list[dict[str, Any]] = []
    for frame in frames:
        frame_id = int(frame["frame"])
        for index, row in enumerate(frame.get("merged") or frame.get("global") or (), start=1):
            box = row["box"] if "box" in row else row
            owned = _owned_cell(Box(float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])), width, height)
            ranked.append(
                {
                    "box_id": f"f{frame_id}b{index:02d}",
                    "frame": frame_id,
                    "cell": owned,
                    "source": str(row.get("source") or "detector"),
                    "confidence": float(row.get("confidence") or 0.0),
                    "xyxy": [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])],
                    "area": max(0.0, (float(box["x2"]) - float(box["x1"])) * (float(box["y2"]) - float(box["y1"]))),
                }
            )
    ranked.sort(key=lambda row: (-row["confidence"], -row["area"], row["frame"], row["box_id"]))
    chosen: list[dict[str, Any]] = []
    seen_cells: set[str] = set()
    for row in ranked:
        if row["cell"] not in seen_cells:
            chosen.append(row)
            seen_cells.add(row["cell"])
        if len(chosen) == k:
            return tuple(chosen)
    for row in ranked:
        if row in chosen:
            continue
        chosen.append(row)
        if len(chosen) == k:
            break
    return tuple(chosen)


def parse_crop_judgment(text: str, box_id: str) -> dict[str, Any]:
    payload = json.loads(text.strip())
    required = {"box_id", "judgment", "cell_advice", "rationale"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("crop_judgment_keys_invalid")
    if payload["box_id"] != box_id:
        raise ValueError("crop_judgment_box_id_mismatch")
    if payload["judgment"] not in {"PERSON", "FP", "UNSURE"}:
        raise ValueError("crop_judgment_invalid")
    if payload["cell_advice"] not in {"INCLUDE", "EXCLUDE", "EXPERT", "NONE"}:
        raise ValueError("crop_advice_invalid")
    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        raise ValueError("crop_rationale_required")
    return {
        "box_id": box_id,
        "judgment": payload["judgment"],
        "cell_advice": payload["cell_advice"],
        "rationale": payload["rationale"].strip(),
    }


def compile_structure_from_judgments(
    crops: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {row["box_id"]: row for row in crops}
    by_cell: dict[str, list[dict[str, Any]]] = {cell: [] for cell in ("t0", "t1", "t2", "t3")}
    for row in judgments:
        crop = by_id.get(row["box_id"])
        if crop is None:
            raise ValueError("judgment_for_unknown_crop")
        merged = {**crop, **row}
        by_cell[crop["cell"]].append(merged)
    cells: list[dict[str, str]] = []
    expert_ranks: list[tuple[int, str]] = []
    for cell, rows in by_cell.items():
        if not rows:
            continue
        persons = sum(row["judgment"] == "PERSON" for row in rows)
        fps = sum(row["judgment"] == "FP" for row in rows)
        advice = [row["cell_advice"] for row in rows if row["cell_advice"] != "NONE"]
        if fps > persons and fps > 0:
            kind = "EXCLUDE"
        elif persons > 0 and any("tile" in row["source"] for row in rows if row["judgment"] == "PERSON"):
            kind = "EXPERT"
        elif persons > 0:
            kind = "INCLUDE"
        elif advice.count("EXCLUDE") > advice.count("INCLUDE"):
            kind = "EXCLUDE"
        else:
            continue
        if kind == "EXPERT":
            expert_ranks.append((persons, cell))
        cells.append({"cell": cell, "type": kind})
    expert_ranks.sort(reverse=True)
    if len([row for row in cells if row["type"] == "EXPERT"]) > 2:
        keep = {cell for _, cell in expert_ranks[:2]}
        for row in cells:
            if row["type"] == "EXPERT" and row["cell"] not in keep:
                row["type"] = "INCLUDE"
    if not cells:
        return {"abstain": True, "cells": ()}
    return {"abstain": False, "cells": tuple(cells)}


def health_identity_prompt(expected_count: int) -> str:
    return (
        "Read only the attached native image(s). Each image has a painted white label "
        "at the top-left that says FRAME followed by a three-character code. "
        f"There should be {expected_count} image(s). "
        "Return only JSON with keys markers_seen (all codes in left-to-right input order), "
        "first_marker, last_marker. Do not invent codes that are not painted."
    )


def crop_judgment_prompt(camera_id: str, crop: Mapping[str, Any]) -> str:
    preconfig = {
        "camera_id": camera_id,
        "box_id": crop["box_id"],
        "frame": crop["frame"],
        "cell": crop["cell"],
        "source": crop["source"],
        "confidence": round(float(crop["confidence"]), 3),
        "xyxy": [round(value, 1) for value in crop["xyxy"]],
        "note": "This image is a native crop of one detector box, not a collage or full frame.",
    }
    return (
        f"You judge one detector crop for fixed camera {camera_id}. "
        f"Preconfigured ledger fields: {json.dumps(preconfig, sort_keys=True)}. "
        "Look only inside this crop. judgment must be PERSON, FP, or UNSURE. "
        "cell_advice must be INCLUDE, EXCLUDE, EXPERT, or NONE. "
        f"box_id must be exactly {crop['box_id']}. "
        "Do not output config_id, thresholds, or pixel polygons. Return only JSON."
    )


def crop_suspect_prompt(camera_id: str, crop: Mapping[str, Any], cells: Sequence[Mapping[str, str]]) -> str:
    return (
        f"You review one detector crop for camera {camera_id}. "
        f"Compiled structure so far: {json.dumps(list(cells))}. "
        f"Preconfigured fields: {json.dumps({'box_id': crop['box_id'], 'frame': crop['frame'], 'cell': crop['cell'], 'source': crop['source']}, sort_keys=True)}. "
        "This image is a native crop of that box, not a full frame. "
        'Return only JSON: {"box_id":"%s","suspect":true,"rationale":"..."} '
        "suspect is true only if this crop is a stable false positive or a clearly empty/non-person box."
        % crop["box_id"]
    )


def prepare_marked_images(sources: Sequence[Path], output_dir: Path, markers: Sequence[str]) -> list[Path]:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    output_dir.mkdir(parents=True)
    paths: list[Path] = []
    for source, marker in zip(sources, markers, strict=True):
        with Image.open(source) as raw:
            marked = paint_marker(raw, marker)
        dest = output_dir / f"{marker}.jpg"
        marked.save(dest, format="JPEG", quality=95, subsampling=0)
        paths.append(dest)
    return paths


def evaluate_health_case(
    *,
    case_id: str,
    expected_markers: Sequence[str],
    expected_images: int,
    response_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    parsed_log = parse_mtmd_log(log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else "")
    identity_ok: bool | None
    identity: dict[str, Any] | None
    try:
        identity = parse_identity(response_path.read_text(encoding="utf-8"))
        identity_ok = identity_matches(identity, expected_markers)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        identity = {"error": str(exc)}
        identity_ok = False
    score = score_visual_health(parsed_log, expected_images=expected_images, identity_ok=identity_ok)
    return {
        "case_id": case_id,
        "expected_markers": list(expected_markers),
        "identity": identity,
        "log": parsed_log,
        "score": score,
    }


def prepare_camera_crops(
    *,
    camera_id: str,
    observer_row: Mapping[str, Any],
    proposal_frames: Sequence[int],
    sequence_id: str,
    pilot_root: Path,
    output_dir: Path,
    k: int = MAX_CROPS,
) -> dict[str, Any]:
    allowed = set(proposal_frames)
    frames = [frame for frame in observer_row["frames"] if int(frame["frame"]) in allowed]
    crops = select_box_crops(frames, width=1280, height=720, k=k)
    camera_dir = output_dir / camera_id
    camera_dir.mkdir(parents=True, exist_ok=True)
    materialized = []
    for crop in crops:
        source = pilot_root / "sequences" / sequence_id / "img1" / f"{crop['frame']:06d}.jpg"
        with Image.open(source) as raw:
            cut = crop_detection(raw, {"x1": crop["xyxy"][0], "y1": crop["xyxy"][1], "x2": crop["xyxy"][2], "y2": crop["xyxy"][3]})
        dest = camera_dir / f"{crop['box_id']}.jpg"
        cut.save(dest, format="JPEG", quality=95, subsampling=0)
        item = dict(crop)
        item["path"] = dest.as_posix()
        item["sha256"] = _sha256_file(dest)
        item["width"], item["height"] = cut.size
        materialized.append(item)
        (camera_dir / f"{crop['box_id']}.judgment-prompt.txt").write_text(
            crop_judgment_prompt(camera_id, item), encoding="utf-8"
        )
        (camera_dir / f"{crop['box_id']}.suspect-prompt.txt").write_text(
            crop_suspect_prompt(camera_id, item, ()), encoding="utf-8"
        )
    return {
        "camera_id": camera_id,
        "crops": materialized,
        "collage_used": False,
        "full_frame_used": False,
        "input_kind": "detection_crop_plus_preconfig",
    }


def review_disposition_from_suspects(
    crops: Sequence[Mapping[str, Any]],
    suspect_box_ids: Sequence[str],
    proposal_frames: Sequence[int],
) -> dict[str, Any]:
    allowed_ids = {row["box_id"] for row in crops}
    leaked = [box_id for box_id in suspect_box_ids if box_id not in allowed_ids]
    if leaked:
        return {"level": "L4", "patch": None, "reason": "review_cited_unknown_crop"}
    suspect_frames = sorted({row["frame"] for row in crops if row["box_id"] in set(suspect_box_ids)})
    review = {
        "level": "L1",
        "suspect_frame_ids": tuple(suspect_frames),
        "patch": None,
        "rationale": "compiled_from_per_crop_suspect_votes",
    }
    cells = sorted({row["cell"] for row in crops if row["box_id"] in set(suspect_box_ids)})
    if len(cells) == 1 and suspect_frames:
        review = {
            "level": "L3",
            "suspect_frame_ids": tuple(suspect_frames),
            "patch": {"cell": cells[0], "type": "EXCLUDE"},
            "rationale": "one_cell_suspect_crops",
        }
    return assign_disposition(review, history_frames=proposal_frames)


def parse_crop_suspect(text: str, box_id: str) -> bool:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict) or set(payload) != {"box_id", "suspect", "rationale"}:
        raise ValueError("crop_suspect_keys_invalid")
    if payload["box_id"] != box_id or not isinstance(payload["suspect"], bool):
        raise ValueError("crop_suspect_invalid")
    return payload["suspect"]


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prep_health = sub.add_parser("prepare-health")
    prep_health.add_argument("--sources", type=Path, nargs="+", required=True)
    prep_health.add_argument("--output-dir", type=Path, required=True)
    prep_crops = sub.add_parser("prepare-boxcrop")
    prep_crops.add_argument("--seal", type=Path, required=True)
    prep_crops.add_argument("--predictions", type=Path, required=True)
    prep_crops.add_argument("--pilot-root", type=Path, required=True)
    prep_crops.add_argument("--output-dir", type=Path, required=True)
    health = sub.add_parser("score-health")
    health.add_argument("--manifest", type=Path, required=True)
    health.add_argument("--response-dir", type=Path, required=True)
    health.add_argument("--output", type=Path, required=True)
    judge = sub.add_parser("compile-boxcrop")
    judge.add_argument("--prepare", type=Path, required=True)
    judge.add_argument("--response-dir", type=Path, required=True)
    judge.add_argument("--predictions", type=Path, required=True)
    judge.add_argument("--pilot-root", type=Path, required=True)
    judge.add_argument("--proxy", type=Path)
    judge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare-health":
        if len(args.sources) != 8:
            raise ValueError("health_probe_requires_eight_source_images")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths = prepare_marked_images(args.sources, args.output_dir / "marked", MARKERS)
        manifest = {
            "schema_version": "sceneconfig.visual-input-health.manifest.v1",
            "cases": [
                {"case_id": "single_default", "images": [paths[0].as_posix()], "expected_markers": [MARKERS[0]], "expected_images": 1, "image_flag": "single", "min_tokens": None},
                {"case_id": "single_min1024", "images": [paths[0].as_posix()], "expected_markers": [MARKERS[0]], "expected_images": 1, "image_flag": "single", "min_tokens": 1024},
                {"case_id": "repeat_8_min1024", "images": [path.as_posix() for path in paths], "expected_markers": list(MARKERS), "expected_images": 8, "image_flag": "repeat", "min_tokens": 1024},
                {"case_id": "comma_8_min1024", "images": [path.as_posix() for path in paths], "expected_markers": list(MARKERS), "expected_images": 8, "image_flag": "comma", "min_tokens": 1024},
                {"case_id": "comma_8_default", "images": [path.as_posix() for path in paths], "expected_markers": list(MARKERS), "expected_images": 8, "image_flag": "comma", "min_tokens": None},
            ],
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prompt_dir = args.output_dir / "prompts"
        prompt_dir.mkdir()
        lists = args.output_dir / "image-lists"
        lists.mkdir()
        for row in manifest["cases"]:
            (prompt_dir / f"{row['case_id']}.txt").write_text(health_identity_prompt(row["expected_images"]), encoding="utf-8")
            (lists / f"{row['case_id']}.flag").write_text(row["image_flag"] + "\n", encoding="utf-8")
            (lists / f"{row['case_id']}.min_tokens").write_text("" if row["min_tokens"] is None else str(row["min_tokens"]), encoding="utf-8")
            (lists / f"{row['case_id']}.images").write_text("\n".join(row["images"]) + "\n", encoding="utf-8")
        print(json.dumps({"cases": [row["case_id"] for row in manifest["cases"]]}))
        return 0
    if args.action == "prepare-boxcrop":
        if args.output_dir.exists():
            raise FileExistsError(f"refuse_existing_output:{args.output_dir}")
        args.output_dir.mkdir(parents=True)
        seal = json.loads(args.seal.read_text(encoding="utf-8"))
        if seal.get("confirm_opened") is not False:
            raise ValueError("confirm_already_opened")
        index: dict[tuple[str, str], dict[str, Any]] = {}
        with args.predictions.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    index[(row["camera_id"], row["config_id"])] = row
        cameras = {}
        for camera in EXPECTED_CAMERAS:
            assets = seal["cameras"][camera]
            observer = assets.get("observer_config_id") or choose_observer_config(
                config_id for cam, config_id in index if cam == camera
            )
            packed = prepare_camera_crops(
                camera_id=camera,
                observer_row=index[(camera, observer)],
                proposal_frames=assets["proposal_frames"],
                sequence_id=assets["sequence_id"],
                pilot_root=args.pilot_root,
                output_dir=args.output_dir / "media",
            )
            packed["observer_config_id"] = observer
            packed["proposal_frames"] = assets["proposal_frames"]
            packed["split"] = assets["split"]
            packed["sequence_id"] = assets["sequence_id"]
            cameras[camera] = packed
        payload = {
            "schema_version": "sceneconfig.boxcrop-valgate.prepare.v1",
            "evidence_status": "DEVELOPMENT_ONLY",
            "collage_used": False,
            "full_frame_used": False,
            "confirm_opened": False,
            "holdout_opened": False,
            "sealed_camera_used": False,
            "input_kind": "detection_crop_plus_preconfig",
            "cameras": cameras,
        }
        (args.output_dir / "PREPARE.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"cameras": list(cameras), "crops": {camera: len(row["crops"]) for camera, row in cameras.items()}}, sort_keys=True))
        return 0
    if args.action == "score-health":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        cases = []
        for row in manifest["cases"]:
            cases.append(
                evaluate_health_case(
                    case_id=row["case_id"],
                    expected_markers=row["expected_markers"],
                    expected_images=int(row["expected_images"]),
                    response_path=args.response_dir / f"{row['case_id']}.json",
                    log_path=args.response_dir / f"{row['case_id']}.log",
                )
            )
        single = next((case for case in cases if case["case_id"].startswith("single_")), None)
        payload = {
            "schema_version": "sceneconfig.visual-input-health.v1",
            "evidence_status": "DEVELOPMENT_ONLY",
            "cases": cases,
            "single_image_identity_ok": bool(single and single["score"]["identity_ok"]),
            "any_healthy_binding": any(case["score"]["healthy"] for case in cases),
            "recommend_per_crop_calls": True,
        }
        if args.output.exists():
            raise FileExistsError(f"refuse_existing_output:{args.output}")
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"single_image_identity_ok": payload["single_image_identity_ok"], "any_healthy_binding": payload["any_healthy_binding"]}))
        return 0
    prepare = json.loads(args.prepare.read_text(encoding="utf-8"))
    index: dict[tuple[str, str], dict[str, Any]] = {}
    with args.predictions.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                index[(row["camera_id"], row["config_id"])] = row
    proxy = json.loads(args.proxy.read_text(encoding="utf-8")) if args.proxy else None
    per_camera = {}
    for camera in EXPECTED_CAMERAS:
        assets = prepare["cameras"][camera]
        judgments = []
        errors = []
        for crop in assets["crops"]:
            path = args.response_dir / f"{camera}.{crop['box_id']}.judgment.json"
            try:
                judgments.append(parse_crop_judgment(path.read_text(encoding="utf-8"), crop["box_id"]))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(str(exc))
        structure = compile_structure_from_judgments(assets["crops"], judgments) if not errors else {"abstain": True, "cells": ()}
        suspects = []
        for crop in assets["crops"]:
            path = args.response_dir / f"{camera}.{crop['box_id']}.suspect.json"
            try:
                if parse_crop_suspect(path.read_text(encoding="utf-8"), crop["box_id"]):
                    suspects.append(crop["box_id"])
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(str(exc))
        disposition = (
            {"level": "L4", "patch": None, "reason": ";".join(errors)}
            if errors
            else review_disposition_from_suspects(assets["crops"], suspects, assets["proposal_frames"])
        )
        if disposition["level"] == "L3":
            try:
                apply_patch(structure.get("cells") or (), disposition.get("patch"))
            except ValueError as exc:
                disposition = {"level": "L4", "patch": None, "reason": str(exc)}
        observer = assets["observer_config_id"]
        observer_row = index[(camera, observer)]
        baseline_row = index.get((camera, "B0"), observer_row)
        proxy_scores = None
        if proxy:
            proxy_scores = {row["config_id"]: float(row["proxy_utility"]) for row in proxy["per_camera"][camera]}
        evaluated = evaluate_camera(
            camera_id=camera,
            observer_row=observer_row,
            baseline_row=baseline_row,
            split={key: tuple(value) for key, value in assets["split"].items()},
            structure=structure,
            disposition=disposition,
            gt_path=args.pilot_root / "sequences" / assets["sequence_id"] / "gt.txt",
            width=1280,
            height=720,
            proxy_scores=proxy_scores,
        )
        evaluated["structure"] = structure
        evaluated["disposition"] = disposition
        evaluated["proposal_frames"] = assets["proposal_frames"]
        evaluated["crops"] = [{"box_id": row["box_id"], "cell": row["cell"], "source": row["source"]} for row in assets["crops"]]
        evaluated["judgments"] = judgments
        evaluated["input_kind"] = "detection_crop_plus_preconfig"
        per_camera[camera] = evaluated
    primary = [per_camera[camera] for camera in PRIMARY_CAMERAS]
    both_reject = all(row["verdict"] != "pass" for row in primary)
    result = {
        "schema_version": "sceneconfig.boxcrop-valgate.result.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "visual_protocol": "detection_crop_plus_preconfig",
        "collage_used": False,
        "full_frame_used": False,
        "holdout_opened": False,
        "confirm_opened": False,
        "sealed_camera_used": False,
        "h_b1": {
            "primary_cameras": list(PRIMARY_CAMERAS),
            "primary_pass_count": sum(row["verdict"] == "pass" for row in primary),
            "both_legal_reject": both_reject,
            "passed": False,
            "reason": "both_primary_cameras_failed_or_held" if both_reject else "confirmation_closed_until_both_primary_verdicts_sealed",
        },
        "per_camera": per_camera,
        "next_gate": "STOP_WRITE_LEGAL_REJECT" if both_reject else "OPEN_CONFIRM_ONLY_IF_NOT_BOTH_PRIMARY_REJECT",
    }
    if args.output.exists():
        raise FileExistsError(f"refuse_existing_output:{args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"next_gate": result["next_gate"], "h_b1": result["h_b1"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
