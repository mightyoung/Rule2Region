"""E4: VLM fills the first card; text main agent may patch once per later round."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from research_tools.execution_policy.card import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    CardInvalid,
    PolicyCard,
    apply_patch,
    card_to_dict,
    merge_overlapping_regions,
    parse_policy_card,
)
from research_tools.cascade_experiment import Prediction
from research_tools.execution_policy.browse import FileBrowser, PathEscapesContext
from research_tools.execution_policy.context import write_agent_context
from research_tools.execution_policy.ledger import FrameLedger, frame_ledger_from_json
from research_tools.execution_policy.prompts import VISUAL_SYSTEM_PROMPT
from research_tools.execution_policy.rules import COUNTING_RULE_CARD
from research_tools.execution_policy.score import compare_camera, compare_three
from research_tools.residual_compiler import PRIMARY_EVAL
from research_tools.route_a_clip_commissioning import load_clips
from research_tools.visual_input_health import parse_mtmd_log


K = 3
MAX_TOOL_ROUNDS = 8
FEWSHOT_PER_STREAM = 4
# Same parking-lot camera, different VIRAT session prefixes.
SAME_VIEW = {"VIRAT_S_010112": "VIRAT_S_010111"}


def view_id(camera_id: str) -> str:
    return SAME_VIEW.get(str(camera_id), str(camera_id))


def load_dual_ledger(path: Path) -> dict[str, dict[int, FrameLedger]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    store: dict[str, dict[int, FrameLedger]] = {}
    for sequence, row in payload["sequences"].items():
        store[sequence] = {}
        for item in row["frames"]:
            ledger = frame_ledger_from_json(item)
            store[sequence][ledger.frame] = ledger
    return store


def pick_observer(history: list[int], ledgers: Mapping[int, FrameLedger]) -> int:
    def key(frame: int) -> tuple[int, int]:
        row = ledgers[frame]
        return (len(row.yolo11s) + len(row.yolo26), -frame)

    return max(history, key=key)


def read_observer_image(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"observer_image_missing:{path}")
    with Image.open(path) as image:
        width, height = image.size
        mode = str(image.mode)
        fmt = str(image.format or "")
    if width <= 0 or height <= 0:
        raise ValueError(f"observer_image_empty:{path}")
    return {
        "path": str(path),
        "filename": path.name,
        "decoded_width": int(width),
        "decoded_height": int(height),
        "file_bytes": path.stat().st_size,
        "mode": mode,
        "format": fmt,
        "resized_before_vlm": False,
        "collage": False,
    }


def require_compiler_frame(info: Mapping[str, Any]) -> None:
    if int(info["decoded_width"]) != int(FRAME_WIDTH) or int(info["decoded_height"]) != int(FRAME_HEIGHT):
        raise ValueError(
            "observer_size_mismatch:"
            f"{info['decoded_width']}x{info['decoded_height']}"
            f"!={int(FRAME_WIDTH)}x{int(FRAME_HEIGHT)}"
        )


def fewshot_xyxy(preds: Sequence[Prediction], *, limit: int = FEWSHOT_PER_STREAM) -> list[list[int]]:
    ranked = sorted(
        preds,
        key=lambda pred: (-float(pred.confidence), pred.box.x1, pred.box.y1, pred.box.x2, pred.box.y2),
    )
    return [
        [int(round(pred.box.x1)), int(round(pred.box.y1)), int(round(pred.box.x2)), int(round(pred.box.y2))]
        for pred in ranked[:limit]
    ]


def other_history_fewshot(
    history: Sequence[int],
    observer: int,
    ledgers: Mapping[int, FrameLedger],
) -> dict[str, Any]:
    candidates = [frame for frame in history if frame != observer]
    if not candidates:
        return {"frame": None, "yolo11s": [], "yolo26": []}

    def key(frame: int) -> tuple[int, int]:
        row = ledgers[frame]
        return (len(row.yolo11s) + len(row.yolo26), -frame)

    frame = max(candidates, key=key)
    return {
        "frame": frame,
        "yolo11s": fewshot_xyxy(ledgers[frame].yolo11s),
        "yolo26": fewshot_xyxy(ledgers[frame].yolo26),
    }


def group_clips_by_camera(clips: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for sequence, clip in sorted(clips.items()):
        groups.setdefault(view_id(str(clip["camera_id"])), []).append(sequence)
    return groups


def subagent_prompt(
    camera_id: str,
    observer: int,
    image_info: Mapping[str, Any],
    context_root: Path,
    *,
    observer_sequence: str = "",
    sequences: Sequence[str] | None = None,
    camera_ids: Sequence[str] | None = None,
) -> str:
    width = int(image_info["decoded_width"])
    height = int(image_info["decoded_height"])
    seqs = list(sequences or ())
    prefixes = list(camera_ids or ())
    seq_line = f"History sequences of this camera view: {', '.join(seqs)}. " if seqs else ""
    prefix_line = (
        f"VIRAT prefixes on this same physical view: {', '.join(prefixes)}. "
        if prefixes
        else ""
    )
    same_view = (
        "VIRAT_S_010111 and VIRAT_S_010112 are the same parking-lot camera; submit one card for both. "
        if {"VIRAT_S_010111", "VIRAT_S_010112"} <= set(prefixes)
        else ""
    )
    obs_seq = f" from {observer_sequence}" if observer_sequence else ""
    return (
        f"{COUNTING_RULE_CARD}\n"
        f"Camera view {camera_id}. One card for this physical camera view, shared by every clip of this view. "
        f"{prefix_line}{same_view}{seq_line}"
        f"This image is history frame {observer}{obs_seq}, one native JPEG, not a collage. "
        f"Attached --image file: {image_info['filename']}, {image_info['file_bytes']} bytes, "
        f"decoded pixels {width}x{height}, mode={image_info['mode']}, format={image_info['format']}. "
        "We did not crop, resize, pad, or overlay boxes before --image. "
        f"Original camera-frame extract size equals this file: {width}x{height}. "
        f"Compiler legal space is {int(FRAME_WIDTH)}x{int(FRAME_HEIGHT)} pixels, origin top-left. "
        f"xyxy is integer pixel coordinates: 0 <= x1 < x2 <= {width} and 0 <= y1 < y2 <= {height}. "
        "Do not use 0-1000 or 0-999 normalized coordinates. "
        "Observer-frame detector counts and observer-frame box xyxy are withheld. "
        f"Readable context directory: {context_root}. "
        "Load skills with list_skills and load_skill before reading history_boxes. "
        "Submit one camera-level card from aggregated history across all sequences of this camera "
        "(11s, 26, and history human labels), not a per-clip or per-frame annotation. "
        "Regions must not overlap. If two candidate regions overlap, merge them into one union box. "
        "Do not read validation frames."
    )


def _ok_count(tools_log: Sequence[Mapping[str, Any]], action: str, name: str) -> int:
    return sum(
        1
        for row in tools_log
        if row.get("ok")
        and str(row.get("action") or "") == action
        and name in {str(row.get("path") or ""), str(row.get("skill") or "")}
    )


def next_turn_prompt(
    base: str,
    tools_log: list[dict[str, Any]],
    latest: Mapping[str, Any],
    browser: FileBrowser,
    tail: str,
) -> str:
    memory = browser.memory()
    latest_out: dict[str, Any] = dict(latest)
    latest_path = str(latest.get("path") or "")
    latest_skill = str(latest.get("skill") or "")
    omit_text = False
    if latest_path and latest_path in memory["read_boxes"]:
        omit_text = _ok_count(tools_log, "read_file", latest_path) > 1
    if latest_skill and latest_skill in memory["loaded_skills"]:
        omit_text = omit_text or _ok_count(tools_log, "load_skill", latest_skill) > 1
    if omit_text:
        latest_out.pop("text", None)
        latest_out["text_omitted"] = "already_in_memory"
    latest_out.pop("listed_files", None)
    compact = {
        "loaded_skills": sorted(browser.loaded),
        "listed_files": memory["listed_files"],
        "read_boxes": sorted(memory["read_boxes"]),
    }
    return (
        f"{base}\n"
        f"Loaded skills:\n{json.dumps(compact['loaded_skills'], ensure_ascii=False)}\n"
        f"Listed files:\n{json.dumps(compact['listed_files'], ensure_ascii=False)}\n"
        f"Read boxes:\n{json.dumps(compact['read_boxes'], ensure_ascii=False)}\n"
        f"Previous tool calls:\n{json.dumps(tools_log, ensure_ascii=False)}\n"
        f"Latest tool result:\n{json.dumps(latest_out, ensure_ascii=False)}\n"
        f"{tail}"
    )


def _as_card_payload(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    action = payload.get("action")
    if action == "submit_card" or (action in {None, "submit_card"} and "regions" in payload):
        return payload
    return None


def run_visual_agent(
    *,
    bin_path: str,
    model: str,
    mmproj: str,
    image: Path,
    prompt: str,
    schema: Path,
    browser: FileBrowser,
    out_dir: Path,
    sequence: str,
    gpu: str,
) -> tuple[dict[str, Any] | None, bool, list[dict[str, Any]]]:
    tools_log: list[dict[str, Any]] = []
    current = prompt
    warning = False
    card_payload: dict[str, Any] | None = None
    for step in range(MAX_TOOL_ROUNDS):
        turn_json = out_dir / f"{sequence}.r0.t{step}.json"
        turn_log = out_dir / f"{sequence}.r0.t{step}.log"
        visual_log = call_visual(bin_path, model, mmproj, image, current, schema, turn_json, turn_log, gpu)
        warning = warning or bool(visual_log.get("token_position_warning"))
        try:
            payload = _parse_json(turn_json)
        except (OSError, ValueError, json.JSONDecodeError):
            tools_log.append({"step": step, "parse_ok": False})
            break
        card_payload = dict(_as_card_payload(payload) or {}) or None
        if card_payload is not None:
            used_files = any(row.get("action") in {"list_dir", "read_file"} and row.get("ok") for row in tools_log)
            missing = browser.missing_for_submit()
            if missing or step == 0 or not used_files:
                bounce = {
                    "ok": False,
                    "error": "submit_card_blocked",
                    "missing_skills": missing,
                    "need_file_read": not used_files,
                    "hint": "load_skill file-browse and count-schedule, then read a history_boxes file.",
                }
                tools_log.append({"step": step, "action": "submit_card", "ok": False, "rejected": True, "missing_skills": missing})
                current = next_turn_prompt(
                    prompt,
                    tools_log,
                    bounce,
                    browser,
                    "Load required skills, list_dir, and read a listed history_boxes file before submit_card.",
                )
                card_payload = None
                continue
            tools_log.append({"step": step, "action": "submit_card"})
            break
        action = str(payload.get("action") or "")
        rel = str(payload.get("path") or ".")
        try:
            result = browser.call(action, rel)
        except PathEscapesContext as exc:
            result = {"ok": False, "error": str(exc)}
        tools_log.append({"step": step, "action": action, "path": rel, "ok": result.get("ok")})
        if any(path.startswith("history_boxes/") for path in browser.read_boxes):
            tail = (
                "You already have history_boxes/_focus.json. submit_card now. "
                "If proposed_regions is non-empty, submit those regions and merge any overlap. "
                "If proposed_regions is empty, submit one small far-ground yolo26 0.25 count region "
                "from the visible walkable ground. Do not read another file."
            )
        else:
            tail = (
                "Only read_file paths in Listed files. Observer-frame names are not listed. "
                "Load skills, list_dir history_boxes, read history_boxes/_focus.json, then submit_card."
            )
        current = next_turn_prompt(prompt, tools_log, result, browser, tail)
    (out_dir / f"{sequence}.r0.tools.json").write_text(
        json.dumps(tools_log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return card_payload, warning, tools_log


def main_agent_prompt(camera_id: str, card: PolicyCard, receipt: Mapping[str, Any]) -> str:
    return (
        f"{COUNTING_RULE_CARD}\n"
        f"Camera {camera_id}. You are text-only. Current card: {json.dumps(card_to_dict(card), sort_keys=True)}. "
        f"History receipt numbers: {json.dumps(receipt, sort_keys=True)}. "
        "Change at most one field: stop, set_conf, set_model, or drop_region. "
        "Do not enable peak/night/stride. Return only JSON."
    )


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout, proc.stderr


def call_visual(bin_path: str, model: str, mmproj: str, image: Path, prompt: str, schema: Path, out_json: Path, out_log: Path, gpu: str) -> dict[str, Any]:
    code, stdout, stderr = _run(
        [
            bin_path, "-m", model, "--mmproj", mmproj, "--image", str(image),
            "--ctx-size", "8192", "--temp", "0", "-n", "256", "-ngl", "999",
            "--json-schema-file", str(schema), "-sys", VISUAL_SYSTEM_PROMPT, "-p", prompt,
        ]
    )
    out_json.write_text(stdout, encoding="utf-8")
    out_log.write_text(stderr, encoding="utf-8")
    parsed = parse_mtmd_log(stderr)
    parsed["exit_code"] = code
    return parsed


def call_text(bin_path: str, model: str, prompt: str, schema: Path, out_json: Path, out_log: Path, gpu: str) -> dict[str, Any]:
    code, stdout, stderr = _run(
        [
            bin_path, "-m", model, "--ctx-size", "8192", "--temp", "0", "-n", "128",
            "-ngl", "999", "--single-turn", "--simple-io", "--no-display-prompt",
            "--json-schema-file", str(schema), "-p", prompt,
        ]
    )
    out_json.write_text(stdout, encoding="utf-8")
    out_log.write_text(stderr, encoding="utf-8")
    return {"exit_code": code, "token_position_warning": False}


def _parse_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    return json.loads(text)


def run(
    *,
    cache_root: Path,
    pilot_root: Path,
    dual_ledger_path: Path,
    output_dir: Path,
    visual_bin: Path,
    text_bin: Path,
    model: Path,
    mmproj: Path,
    card_schema: Path,
    patch_schema: Path,
    agent_schema: Path,
    gpu: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    output_dir.mkdir(parents=True)
    responses = output_dir / "responses"
    responses.mkdir()
    contexts = output_dir / "context"
    contexts.mkdir()
    clips = load_clips(cache_root, pilot_root)
    dual = load_dual_ledger(dual_ledger_path)
    per_clip: dict[str, Any] = {}
    per_camera: dict[str, Any] = {}
    warning_any = False
    grouped = group_clips_by_camera(clips)
    for camera_id, camera_sequences in grouped.items():
        sources: list[dict[str, Any]] = []
        best: tuple[tuple[int, int, str], str, int] | None = None
        for sequence in camera_sequences:
            clip = clips[sequence]
            ledgers = dual[sequence]
            history = list(clip["history_frames"])
            validation = list(clip["validation_frames"])
            if any(frame not in ledgers for frame in history + validation):
                raise ValueError(f"ledger_frame_missing:{sequence}")
            observer_candidate = pick_observer(history, ledgers)
            if observer_candidate in set(validation):
                raise ValueError("validation_frame_selected_for_vlm")
            row = ledgers[observer_candidate]
            rank = (len(row.yolo11s) + len(row.yolo26), -observer_candidate, sequence)
            if best is None or rank > best[0]:
                best = (rank, sequence, observer_candidate)
            sources.append(
                {
                    "sequence": sequence,
                    "camera_id": camera_id,
                    "history": history,
                    "validation": validation,
                    "ledgers": ledgers,
                    "truth": clip["truth"],
                }
            )
        if best is None:
            raise ValueError(f"camera_has_no_observer:{camera_id}")
        observer_sequence, observer = best[1], best[2]
        image = pilot_root / "sequences" / observer_sequence / "img1" / f"{observer:06d}.jpg"
        image_info = read_observer_image(image)
        require_compiler_frame(image_info)
        context_meta = write_agent_context(
            contexts / camera_id,
            camera_id=camera_id,
            observer_sequence=observer_sequence,
            observer=observer,
            sources=sources,
        )
        prompt = subagent_prompt(
            camera_id,
            observer,
            image_info,
            Path(context_meta["root"]),
            observer_sequence=observer_sequence,
            sequences=camera_sequences,
            camera_ids=sorted({str(clips[seq]["camera_id"]) for seq in camera_sequences}),
        )
        (responses / f"{camera_id}.r0.system.txt").write_text(VISUAL_SYSTEM_PROMPT, encoding="utf-8")
        (responses / f"{camera_id}.r0.prompt.txt").write_text(prompt, encoding="utf-8")
        browser = FileBrowser(Path(context_meta["root"]))
        card_payload, visual_warning, tools_log = run_visual_agent(
            bin_path=str(visual_bin),
            model=str(model),
            mmproj=str(mmproj),
            image=image,
            prompt=prompt,
            schema=agent_schema,
            browser=browser,
            out_dir=responses,
            sequence=camera_id,
            gpu=gpu,
        )
        warning_any = warning_any or visual_warning
        if card_payload is not None:
            (responses / f"{camera_id}.r0.json").write_text(
                json.dumps(card_payload, indent=2) + "\n", encoding="utf-8"
            )
        try:
            if card_payload is None:
                raise CardInvalid("no_submit_card")
            card = parse_policy_card(merge_overlapping_regions(card_payload))
            parse_ok = True
        except (OSError, ValueError, json.JSONDecodeError, CardInvalid):
            card = None
            parse_ok = False
        rounds: list[dict[str, Any]] = []
        frozen = None
        if card is None:
            rounds.append({"round": 0, "parse_ok": False, "legal": False})
        else:
            for index in range(K):
                if index > 0:
                    call_text(
                        str(text_bin), str(model),
                        main_agent_prompt(
                            camera_id,
                            card,
                            {"round": index - 1, **rounds[-1].get("h1", {})},
                        ),
                        patch_schema,
                        responses / f"{camera_id}.r{index}.json",
                        responses / f"{camera_id}.r{index}.log",
                        gpu,
                    )
                    try:
                        patch = _parse_json(responses / f"{camera_id}.r{index}.json")
                        if patch.get("action") == "stop":
                            frozen = card
                            rounds.append({"round": index, "parse_ok": True, "action": "stop", "h1": rounds[-1].get("h1")})
                            break
                        card = apply_patch(card, patch)
                        patch_ok = True
                    except (OSError, ValueError, json.JSONDecodeError, CardInvalid):
                        patch_ok = False
                    if not patch_ok:
                        rounds.append({"round": index, "parse_ok": False, "legal": False, "h1": rounds[-1].get("h1")})
                        continue
                scored = compare_camera(card, sources, compiled_time_ms=0.4, b26_time_ms=1.0)
                rounds.append({"round": index, "parse_ok": True, "legal": True, "h1": scored["h1"], "compiled": scored["compiled"]})
                if scored["h1"]["passed"]:
                    frozen = card
                    break
            if frozen is None and card is not None:
                frozen = card
        shared_card = card_to_dict(frozen) if frozen else None
        view_cameras = sorted({str(clips[seq]["camera_id"]) for seq in camera_sequences})
        per_camera[camera_id] = {
            "camera_id": camera_id,
            "view_id": camera_id,
            "camera_ids": view_cameras,
            "sequences": list(camera_sequences),
            "observer_sequence": observer_sequence,
            "observer_frame": observer,
            "observer_image": image_info,
            "context": context_meta,
            "tool_calls": tools_log,
            "round0_parse_ok": parse_ok,
            "rounds": rounds,
            "intention_card": shared_card,
            "regions_merged_on_overlap": True,
        }
        for sequence in camera_sequences:
            clip = clips[sequence]
            val = None
            hist = None
            if frozen is not None:
                val = compare_three(
                    frozen, dual[sequence], list(clip["validation_frames"]), clip["truth"],
                    compiled_time_ms=0.4, b26_time_ms=1.0,
                )
                val["val_score"] = "ledger_filter_s1_not_crop"
                hist = compare_three(
                    frozen, dual[sequence], list(clip["history_frames"]), clip["truth"],
                    compiled_time_ms=0.4, b26_time_ms=1.0,
                )
            per_clip[sequence] = {
                "camera_id": clip["camera_id"],
                "view_id": camera_id,
                "sequence_id": sequence,
                "role": "development_eval" if sequence == camera_sequences[0] else "transfer",
                "shared_camera_card": True,
                "observer_sequence": observer_sequence,
                "observer_frame": observer,
                "observer_image": image_info,
                "context": context_meta,
                "tool_calls": tools_log if sequence == observer_sequence else [],
                "round0_parse_ok": parse_ok,
                "rounds": rounds,
                "clip_history": hist["h1"] if hist else None,
                "intention_card": shared_card,
                "validation": val["h1"] if val else {"passed": False, "verdict": "fail"},
                "val_score": "ledger_filter_s1_not_crop",
                "verdict": (val["h1"]["verdict"] if val else "fail"),
            }
    primary = [per_clip[PRIMARY_EVAL["VIRAT_S_010004"]], per_clip[PRIMARY_EVAL["VIRAT_S_010204"]]]
    both_reject = all(row["verdict"] != "pass" for row in primary)
    result = {
        "schema_version": "sceneconfig.execution-policy.e4.v18",
        "evidence_status": "DEVELOPMENT_ONLY",
        "arm": "E4_vlm_schedule_probe",
        "prompt_includes_image_size": True,
        "prompt_includes_rule_card": True,
        "prompt_includes_fewshot_boxes": False,
        "observer_boxes_in_prompt": False,
        "prompt_includes_context_dir": True,
        "agent_file_tools": True,
        "system_prompt_used": True,
        "skill_mechanism": True,
        "list_then_read_only": True,
        "history_boxes_summary_listing": True,
        "first_read_keeps_text": True,
        "block_history_reread": True,
        "skill_path_aliases": True,
        "accumulate_skills_and_boxes": True,
        "history_digest": True,
        "history_focus_26_not_11s_gt": True,
        "camera_level_regions": True,
        "camera_id_grouped": True,
        "one_card_per_camera": True,
        "same_view_map": dict(SAME_VIEW),
        "regions_merged_on_overlap": True,
        "required_skills": ["count-schedule", "file-browse"],
        "image_resized_before_vlm": False,
        "official_h1": False,
        "history_score": "ledger_filter_s1",
        "val_score": "ledger_filter_s1_not_crop",
        "qwen_called": True,
        "val_images_sent_to_vlm": False,
        "collage_used": False,
        "token_position_warning_any": warning_any,
        "holdout_opened": False,
        "confirm_opened": False,
        "sealed_camera_used": False,
        "camera_count": len(per_camera),
        "view_count": len(per_camera),
        "virat_camera_count": len({str(clip["camera_id"]) for clip in clips.values()}),
        "vlm_calls": len(per_camera),
        "clip_count": len(per_clip),
        "pass_count": sum(row["verdict"] == "pass" for row in per_clip.values()),
        "h1": {
            "primary_clips": list(PRIMARY_EVAL.values()),
            "both_legal_reject": both_reject,
            "passed": not both_reject and all(row["verdict"] == "pass" for row in primary),
            "reason": "development_probe_not_official_h1",
        },
        "per_camera": per_camera,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-bin", type=Path, required=True)
    parser.add_argument("--text-bin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--card-schema", type=Path, required=True)
    parser.add_argument("--patch-schema", type=Path, required=True)
    parser.add_argument("--agent-schema", type=Path, required=True)
    parser.add_argument("--gpu", default="1")
    args = parser.parse_args()
    result = run(
        cache_root=args.cache_root,
        pilot_root=args.pilot_root,
        dual_ledger_path=args.dual_ledger,
        output_dir=args.output_dir,
        visual_bin=args.visual_bin,
        text_bin=args.text_bin,
        model=args.model,
        mmproj=args.mmproj,
        card_schema=args.card_schema,
        patch_schema=args.patch_schema,
        agent_schema=args.agent_schema,
        gpu=args.gpu,
    )
    print(json.dumps({"pass_count": result["pass_count"], "official_h1": False, "h1": result["h1"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
