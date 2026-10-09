"""Prepare and evaluate the development-only SceneConfig E3 Qwen gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS


BUSINESS_CARD = {
    "task": "frame_level_person_counting",
    "false_negative_cost": "high",
    "false_positive_cost": "medium",
    "duplicate_count_cost": "high",
    "latency_ratio_cap": 3.0,
    "fallback": "yolo11s_global",
}


def parse_query_response(text: str, legal_ids: set[str], prior_ids: set[str] | None = None) -> dict[str, Any]:
    prior_ids = prior_ids or set()
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("qwen_query_not_exact_json") from exc
    if not isinstance(payload, dict) or set(payload) != {"config_ids", "rationale"}:
        raise ValueError("qwen_query_keys_invalid")
    ids, rationale = payload["config_ids"], payload["rationale"]
    if not isinstance(ids, list) or len(ids) != 3 or not all(isinstance(value, str) for value in ids):
        raise ValueError("qwen_query_requires_exactly_three_ids")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("qwen_query_rationale_required")
    rows = []
    seen = set(prior_ids)
    for value in ids:
        status = "legal"
        if value not in legal_ids:
            status = "out_of_panel"
        elif value in seen:
            status = "duplicate"
        rows.append({"config_id": value, "status": status})
        seen.add(value)
    return {"config_ids": ids, "rationale": rationale.strip(), "slots": rows}


def _compact_card(card: dict[str, Any]) -> str:
    tiles = ",".join(card["tiles"]) or "none"
    return f'{card["config_id"]}: tiles={tiles}; tile_conf={card["tile_confidence"]:.2f}; cost={len(card["tiles"])/4:.2f}'


def prepare_prompts(proxy_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    output_dir.mkdir(parents=True)
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    if proxy.get("holdout_opened") is not False or proxy.get("evaluation_gt_used_for_proxy") is not False:
        raise ValueError("proxy_ledger_not_label_blind")
    cards = proxy["panel"]
    legal = [card["config_id"] for card in cards]
    if legal != [f"C{index:02d}" for index in range(24)]:
        raise ValueError("expected_frozen_contiguous_panel")
    panel_text = "\n".join(_compact_card(card) for card in cards)
    for camera in EXPECTED_CAMERAS:
        prompt = f"""You are the offline business, algorithm, and deployment-strategy expert for fixed camera {camera}.
The attached contact sheet shows four historical moments. In every panel t0 is top-left, t1 top-right, t2 bottom-left, t3 bottom-right.
Business contract: {json.dumps(BUSINESS_CARD, sort_keys=True)}.
YOLO11s global always runs. A listed configuration additionally runs YOLO26 on its static tiles. Lower confidence can recover small people but may add false positives. Cost is relative to all four tiles.
Select exactly three distinct configurations for the first replay round. Balance missed-person recovery, false positives, duplicate counts, and the 3x cap. You may select only a config_id below; do not invent parameters.
Frozen panel:\n{panel_text}\nReturn only the required JSON object."""
        (output_dir / f"{camera}.round1.txt").write_text(prompt, encoding="utf-8")
    manifest = {
        "schema_version": "sceneconfig.e3-prompt-manifest.v1",
        "visual_protocol": "contact_sheet",
        "collage_used": True,
        "evidence_status": "DEVELOPMENT_ONLY",
        "holdout_opened": False,
        "business_card": BUSINESS_CARD,
        "legal_config_ids": legal,
        "proxy_ledger_sha256": hashlib.sha256(proxy_path.read_bytes()).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_dense_native_prompts(proxy_path: Path, selection_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    output_dir.mkdir(parents=True)
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if proxy.get("holdout_opened") is not False or proxy.get("evaluation_gt_used_for_proxy") is not False:
        raise ValueError("proxy_ledger_not_label_blind")
    if selection.get("collage_used") is not False:
        raise ValueError("selection_must_forbid_collage")
    if selection.get("holdout_opened", False) is not False:
        raise ValueError("selection_opened_holdout")
    if set(selection.get("cameras") or {}) != set(EXPECTED_CAMERAS):
        raise ValueError("selection_cameras_incomplete")
    cards = proxy["panel"]
    legal = [card["config_id"] for card in cards]
    if legal != [f"C{index:02d}" for index in range(24)]:
        raise ValueError("expected_frozen_contiguous_panel")
    panel_text = "\n".join(_compact_card(card) for card in cards)
    for camera in EXPECTED_CAMERAS:
        frames = selection["cameras"][camera]["frames"]
        observer = selection["cameras"][camera]["observer_config_id"]
        image_lines = []
        for index, row in enumerate(frames, start=1):
            image_lines.append(
                f"Image {index}: native unstitched frame {row['frame']} "
                f"with {row['merged_count']} YOLO11s+YOLO26 merged boxes from observer {observer}."
            )
        prompt = f"""You are the offline business, algorithm, and deployment-strategy expert for fixed camera {camera}.
The attached images are separate native 1280x720 scene frames, not a contact sheet or collage. Each image is one complete camera view. Detection boxes drawn on an image use that same frame's pixel coordinates. Do not treat the images as quadrants of one picture. Do not output pixel coordinates.
Frames were chosen after YOLO11s+YOLO26 finished, by merged box count descending. No random or uniform time sampling was used.
{chr(10).join(image_lines)}
Business contract: {json.dumps(BUSINESS_CARD, sort_keys=True)}.
YOLO11s global always runs. A listed configuration additionally runs YOLO26 on its static tiles. Lower confidence can recover small people but may add false positives. Cost is relative to all four tiles.
Select exactly three distinct configurations for the first replay round. Balance missed-person recovery, false positives, duplicate counts, and the 3x cap. You may select only a config_id below; do not invent parameters.
Frozen panel:\n{panel_text}\nReturn only the required JSON object."""
        (output_dir / f"{camera}.round1.txt").write_text(prompt, encoding="utf-8")
    manifest = {
        "schema_version": "sceneconfig.e3-prompt-manifest.v2",
        "visual_protocol": "dense_native_frames",
        "collage_used": False,
        "selection_rule": "yolo11s_yolo26_merged_box_count_desc",
        "evidence_status": "DEVELOPMENT_ONLY",
        "holdout_opened": False,
        "business_card": BUSINESS_CARD,
        "legal_config_ids": legal,
        "proxy_ledger_sha256": hashlib.sha256(proxy_path.read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_round2(proxy_path: Path, prompt_dir: Path, response_dir: Path) -> None:
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    legal = {card["config_id"] for card in proxy["panel"]}
    for camera in EXPECTED_CAMERAS:
        first_text = (response_dir / f"{camera}.round1.json").read_text(encoding="utf-8")
        first = parse_query_response(first_text, legal)
        feedback_by_id = {row["config_id"]: row for row in proxy["per_camera"][camera]}
        feedback = [feedback_by_id[row["config_id"]] for row in first["slots"] if row["status"] == "legal"]
        prompt = (prompt_dir / f"{camera}.round1.txt").read_text(encoding="utf-8")
        prompt += f"""\n\nRound 1 response was: {first_text.strip()}
Label-blind replay feedback for legal selections: {json.dumps(feedback, separators=(',', ':'), sort_keys=True)}
Select exactly three configurations for round 2. Previously selected IDs are invalid duplicates but still consume a slot. Use only frozen IDs and return only the required JSON object."""
        (prompt_dir / f"{camera}.round2.txt").write_text(prompt, encoding="utf-8")


def _normalized_regret(utilities: dict[str, float], queried: list[str]) -> float:
    best, worst = max(utilities.values()), min(utilities.values())
    legal_values = [utilities[value] for value in queried if value in utilities]
    achieved = max(legal_values) if legal_values else worst
    return (best - achieved) / (best - worst) if best > worst else 0.0


def _queries_to_95(utilities: dict[str, float], queried: list[str]) -> int:
    best, worst = max(utilities.values()), min(utilities.values())
    threshold = worst + .95 * (best - worst)
    for index, value in enumerate(queried, start=1):
        if value in utilities and utilities[value] >= threshold:
            return index
    return 7


def evaluate(
    proxy_path: Path,
    e2_path: Path,
    response_dir: Path,
    output_path: Path,
    *,
    selection_path: Path | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refuse_existing_output:{output_path}")
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    e2 = json.loads(e2_path.read_text(encoding="utf-8"))
    legal = {card["config_id"] for card in proxy["panel"]}
    per_camera = {}
    for camera in EXPECTED_CAMERAS:
        rows = e2["per_camera"][camera]["configs"]
        utilities = {row["config_id"]: float(row["true_utility"]) for row in rows}
        proxy_scores = {row["config_id"]: float(row["proxy_utility"]) for row in proxy["per_camera"][camera]}
        m0 = sorted(legal, key=lambda value: (-proxy_scores[value], value))[:6]
        first = parse_query_response((response_dir / f"{camera}.round1.json").read_text(encoding="utf-8"), legal)
        first_ids = [row["config_id"] for row in first["slots"]]
        second = parse_query_response(
            (response_dir / f"{camera}.round2.json").read_text(encoding="utf-8"), legal, set(first_ids)
        )
        qwen_slots = first["slots"] + second["slots"]
        qwen = [row["config_id"] for row in qwen_slots if row["status"] == "legal"]
        per_camera[camera] = {
            "m0_query_order": m0,
            "qwen_slots": qwen_slots,
            "qwen_legal_query_order": qwen,
            "qwen_legal_rate": len(qwen) / 6.0,
            "m0_normalized_simple_regret": _normalized_regret(utilities, m0),
            "qwen_normalized_simple_regret": _normalized_regret(utilities, qwen),
            "m0_queries_to_95pct_registered_utility": _queries_to_95(utilities, m0),
            "qwen_queries_to_95pct_registered_utility": _queries_to_95(utilities, qwen),
            "registered_best_config_id": max(utilities, key=lambda value: (utilities[value], value)),
        }
    regret_improvements = [
        row["m0_normalized_simple_regret"] - row["qwen_normalized_simple_regret"] for row in per_camera.values()
    ]
    m0_queries = [row["m0_queries_to_95pct_registered_utility"] for row in per_camera.values()]
    qwen_queries = [row["qwen_queries_to_95pct_registered_utility"] for row in per_camera.values()]
    query_reduction = 1.0 - float(np.median(qwen_queries)) / float(np.median(m0_queries))
    legal_rate = float(np.mean([row["qwen_legal_rate"] for row in per_camera.values()]))
    gate = {
        "median_normalized_simple_regret_improvement": float(np.median(regret_improvements)),
        "median_query_reduction_to_95pct": query_reduction,
        "mean_qwen_legal_rate": legal_rate,
        "requirements": {
            "regret_improvement_at_least": .05,
            "query_reduction_at_least": .20,
            "legal_rate_at_least": .95,
        },
    }
    passed = gate["median_normalized_simple_regret_improvement"] >= .05 and query_reduction >= .20 and legal_rate >= .95
    result = {
        "schema_version": "sceneconfig.e3-qwen-validation.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "passed": passed,
        "holdout_opened": False,
        "sealed_camera_used": False,
        "qwen_online_called": False,
        "query_budget": {"rounds": 2, "slots_per_round": 3, "total_slots": 6},
        "gate": gate,
        "per_camera": per_camera,
        "next_gate": "E4" if passed else "DELETE_OR_DOWNGRADE_QWEN_INCREMENTAL_CLAIM",
    }
    if selection_path is not None:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("collage_used") is not False:
            raise ValueError("selection_must_forbid_collage")
        result["schema_version"] = "sceneconfig.e3-qwen-validation.v2"
        result["visual_protocol"] = "dense_native_frames"
        result["collage_used"] = False
        result["selection_rule"] = "yolo11s_yolo26_merged_box_count_desc"
        result["selection_sha256"] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        for camera in EXPECTED_CAMERAS:
            camera_sel = selection["cameras"][camera]
            result["per_camera"][camera]["observer_config_id"] = camera_sel["observer_config_id"]
            result["per_camera"][camera]["selected_frames"] = [int(row["frame"]) for row in camera_sel["frames"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--proxy", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare_dense = sub.add_parser("prepare-dense-native")
    prepare_dense.add_argument("--proxy", type=Path, required=True)
    prepare_dense.add_argument("--selection", type=Path, required=True)
    prepare_dense.add_argument("--output-dir", type=Path, required=True)
    round2 = sub.add_parser("round2")
    round2.add_argument("--proxy", type=Path, required=True)
    round2.add_argument("--prompt-dir", type=Path, required=True)
    round2.add_argument("--response-dir", type=Path, required=True)
    report = sub.add_parser("evaluate")
    report.add_argument("--proxy", type=Path, required=True)
    report.add_argument("--e2", type=Path, required=True)
    report.add_argument("--response-dir", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare_prompts(args.proxy, args.output_dir)
    elif args.action == "prepare-dense-native":
        prepare_dense_native_prompts(args.proxy, args.selection, args.output_dir)
    elif args.action == "round2":
        prepare_round2(args.proxy, args.prompt_dir, args.response_dir)
    else:
        result = evaluate(
            args.proxy, args.e2, args.response_dir, args.output, selection_path=args.selection
        )
        print(json.dumps({"passed": result["passed"], "gate": result["gate"], "next_gate": result["next_gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
