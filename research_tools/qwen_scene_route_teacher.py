"""Offline Qwen scene teacher for fixed per-camera high-resolution tile routes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS, SEALED_CAMERA
from research_tools.run_missed_person_proposal_baseline import _camera

TILE_IDS = ("t0", "t1", "t2", "t3")
RISK_REASONS = (
    "far_small_person_zone", "occlusion_zone", "perspective_scale_transition",
    "background_false_positive_zone", "no_special_risk",
)
PROMPT = """You are an offline scene-routing teacher, not a people counter.
The image is a 2x2 temporal contact sheet from one fixed camera. Every panel is
marked with spatial tiles t0 top-left, t1 top-right, t2 bottom-left, t3 bottom-right.
Select only tiles that repeatedly need a high-resolution small-person detector.
Do not select a tile merely because any person is present. Return exactly one JSON
object and no markdown: {"complex_tiles":["t0"],"reasons":{"t0":["far_small_person_zone"]}}
Allowed reasons: far_small_person_zone, occlusion_zone, perspective_scale_transition,
background_false_positive_zone, no_special_risk. Use empty collections if none qualify.
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_route_output(text: str) -> dict[str, Any]:
    """Accept only the exact bounded route contract; never infer defaults."""
    try:
        payload = json.loads(str(text).strip())
    except json.JSONDecodeError as exc:
        raise ValueError("qwen_route_output_not_exact_json") from exc
    if not isinstance(payload, dict) or set(payload) != {"complex_tiles", "reasons"}:
        raise ValueError("qwen_route_output_keys_invalid")
    tiles, reasons = payload["complex_tiles"], payload["reasons"]
    if not isinstance(tiles, list) or not all(isinstance(tile, str) for tile in tiles):
        raise ValueError("qwen_complex_tiles_invalid")
    if len(tiles) != len(set(tiles)) or any(tile not in TILE_IDS for tile in tiles):
        raise ValueError("qwen_complex_tiles_out_of_contract")
    if tiles != sorted(tiles, key=TILE_IDS.index):
        raise ValueError("qwen_complex_tiles_not_canonical")
    if not isinstance(reasons, dict) or set(reasons) != set(tiles):
        raise ValueError("qwen_reasons_must_match_tiles")
    normalized: dict[str, list[str]] = {}
    for tile in tiles:
        values = reasons[tile]
        if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
            raise ValueError("qwen_reason_list_invalid")
        if len(values) != len(set(values)) or any(value not in RISK_REASONS for value in values):
            raise ValueError("qwen_reason_out_of_contract")
        if "no_special_risk" in values:
            raise ValueError("qwen_selected_tile_cannot_have_no_special_risk")
        normalized[tile] = list(values)
    return {"complex_tiles": list(tiles), "reasons": normalized}


def make_contact_sheet(image_paths: Iterable[Path], output: Path, *, panel_size: tuple[int, int] = (640, 360)) -> dict[str, Any]:
    paths = tuple(Path(path) for path in image_paths)
    if len(paths) != 4 or any(not path.is_file() for path in paths):
        raise ValueError("exactly_four_existing_history_frames_required")
    width, height = panel_size
    sheet = Image.new("RGB", (width * 2, height * 2), "black")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            panel = source.convert("RGB").resize(panel_size, Image.Resampling.LANCZOS)
        x0, y0 = (index % 2) * width, (index // 2) * height
        sheet.paste(panel, (x0, y0))
        draw.line((x0 + width // 2, y0, x0 + width // 2, y0 + height), fill="yellow", width=2)
        draw.line((x0, y0 + height // 2, x0 + width, y0 + height // 2), fill="yellow", width=2)
        labels = (("t0", 8, 8), ("t1", width // 2 + 8, 8), ("t2", 8, height // 2 + 8), ("t3", width // 2 + 8, height // 2 + 8))
        for label, dx, dy in labels:
            draw.rectangle((x0 + dx - 3, y0 + dy - 3, x0 + dx + 25, y0 + dy + 13), fill="black")
            draw.text((x0 + dx, y0 + dy), label, fill="yellow")
        draw.text((x0 + 8, y0 + height - 20), f"history frame {path.stem}", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=92, subsampling=0)
    return {"contact_sheet_sha256": sha256_file(output), "history_image_sha256": [sha256_file(path) for path in paths]}


def history_frame_manifest(cache_root: Path, pilot_root: Path) -> dict[str, list[Path]]:
    """Select four deterministic historical frames per camera without opening GT."""
    manifest = json.loads((cache_root / "cache-manifest.json").read_text(encoding="utf-8"))
    by_camera: dict[str, list[str]] = {camera: [] for camera in EXPECTED_CAMERAS}
    for row in manifest["sequences"]:
        sequence = row["sequence_id"]
        camera = _camera(sequence)
        if camera == SEALED_CAMERA:
            raise ValueError("sealed_camera_present")
        by_camera.setdefault(camera, []).append(sequence)
    if tuple(sorted(by_camera)) != EXPECTED_CAMERAS or any(len(rows) != 2 for rows in by_camera.values()):
        raise ValueError("expected_two_sequences_per_development_camera")
    output: dict[str, list[Path]] = {}
    for camera, sequences in sorted(by_camera.items()):
        history_sequence = sorted(sequences)[1]
        row = next(item for item in manifest["sequences"] if item["sequence_id"] == history_sequence)
        frames = np.load(cache_root / row["cache"])["frame_ids"].astype(int).tolist()
        positions = [round(index * (len(frames) - 1) / 3) for index in range(4)]
        output[camera] = [pilot_root / "sequences" / history_sequence / "img1" / f"{frames[position]:06d}.jpg" for position in positions]
    return output


def seal_profile(responses: dict[str, str], contact_receipts: dict[str, dict[str, Any]], *, model: Path, mmproj: Path, offline_ms: float) -> dict[str, Any]:
    if tuple(sorted(responses)) != EXPECTED_CAMERAS or set(contact_receipts) != set(responses):
        raise ValueError("profile_requires_all_development_cameras")
    return {
        "schema_version": "human-count.qwen-scene-route-profile.v1", "status": "passed",
        "teacher_role": "OFFLINE_HISTORY_ONLY_STATIC_TILE_ROUTER", "history_gt_used": False,
        "evaluation_frames_used": False, "sealed_camera_used": False, "online_qwen_called": False,
        "prompt": PROMPT, "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "model_sha256": sha256_file(model), "mmproj_sha256": sha256_file(mmproj),
        "offline_teacher_ms": float(offline_ms), "contact_sheets": contact_receipts,
        "cameras": {camera: parse_route_output(responses[camera]) for camera in EXPECTED_CAMERAS},
    }


def load_profile(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version": "human-count.qwen-scene-route-profile.v1", "status": "passed", "history_gt_used": False, "evaluation_frames_used": False, "sealed_camera_used": False, "online_qwen_called": False}
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("qwen_profile_provenance_invalid")
    cameras = payload.get("cameras")
    if not isinstance(cameras, dict) or tuple(sorted(cameras)) != EXPECTED_CAMERAS:
        raise ValueError("qwen_profile_camera_set_invalid")
    payload["cameras"] = {camera: parse_route_output(json.dumps(cameras[camera])) for camera in EXPECTED_CAMERAS}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare history-only scene-routing contact sheets")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{args.output_dir}")
    args.output_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {"schema_version": "human-count.qwen-scene-contact-sheets.v1", "prompt": PROMPT, "cameras": {}}
    for camera, paths in history_frame_manifest(args.cache_root, args.pilot_root).items():
        output = args.output_dir / f"{camera}.jpg"
        manifest["cameras"][camera] = {"image": output.name, **make_contact_sheet(paths, output)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "cameras": len(manifest["cameras"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
