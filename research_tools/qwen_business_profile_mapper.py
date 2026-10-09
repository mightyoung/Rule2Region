from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROFILE_CODES = {
    "FN_PRIORITY_WITH_PRECISION_GATE",
    "COUNT_ACCURACY_WITH_RECALL_TIEBREAK",
    "LATENCY_FIRST_WITH_QUALITY_FLOOR",
}
ABSTAIN_REASON_CODES = {
    "INSUFFICIENT_PRIORITY_ORDER",
    "CONFLICTING_HARD_REQUIREMENTS",
    "NO_REGISTERED_PROFILE",
}


def load_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    codes = {row["profile_code"] for row in payload["profiles"]}
    if codes != PROFILE_CODES:
        raise ValueError("profile_registry_codes_invalid")
    return payload


def load_cards(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cards = payload["cards"]
    if len({row["card_id"] for row in cards}) != len(cards):
        raise ValueError("business_card_ids_duplicate")
    for card in cards:
        ids = [row["span_id"] for row in card["spans"]]
        if len(ids) != len(set(ids)) or not set(card["gold_supporting_span_ids"]) <= set(ids):
            raise ValueError("business_card_span_contract_invalid")
    return cards


def load_blind_cases(blind_path: Path, gold_path: Path) -> list[dict[str, Any]]:
    blind_payload = json.loads(blind_path.read_text(encoding="utf-8"))
    gold_payload = json.loads(gold_path.read_text(encoding="utf-8"))
    blind_rows = blind_payload["cards"]
    if any(any(key.startswith("gold_") or key.startswith("author_") for key in row) for row in blind_rows):
        raise ValueError("blind_cards_contain_labels")
    gold_by_id = {row["card_id"]: row for row in gold_payload["cards"]}
    if set(gold_by_id) != {row["card_id"] for row in blind_rows}:
        raise ValueError("blind_gold_card_ids_mismatch")
    cases = []
    for blind in blind_rows:
        gold = gold_by_id[blind["card_id"]]
        span_ids = {row["span_id"] for row in blind["spans"]}
        if not set(gold["gold_supporting_span_ids"]) <= span_ids:
            raise ValueError("blind_gold_span_reference_invalid")
        cases.append({**blind, **{key: value for key, value in gold.items() if key.startswith("gold_")}})
    return cases


def rule_map(card: dict[str, Any]) -> dict[str, Any]:
    text = "".join(row["text"] for row in card["spans"])
    if ("漏掉" in text or "漏检" in text) and "更严重" in text:
        code = "FN_PRIORITY_WITH_PRECISION_GATE"
    elif "绝对误差最小" in text or "人数误差最小" in text:
        code = "COUNT_ACCURACY_WITH_RECALL_TIEBREAK"
    elif "延迟最低" in text and ("不能比现有基线更差" in text or "质量" in text):
        code = "LATENCY_FIRST_WITH_QUALITY_FLOOR"
    else:
        code = None
    return {"profile_code": code, "abstain": code is None}


def profile_prompt(card: dict[str, Any], registry: dict[str, Any]) -> str:
    compact_registry = [{"profile_code": row["profile_code"], "hard_gates": row["hard_gates"], "objective_order": row["objective_order"]} for row in registry["profiles"]]
    span_ids = [row["span_id"] for row in card["spans"]]
    return (
        "You are a business_profile_mapper. Map only explicit business priorities to one registered profile. "
        "False-negative cost maps to recall priority; it never creates the independent precision admission gate. "
        "If the text does not establish an objective priority, use INSUFFICIENT_PRIORITY_ORDER. If it requests behavior outside all registered profiles, use NO_REGISTERED_PROFILE. "
        "If hard requirements conflict, use CONFLICTING_HARD_REQUIREMENTS. Do not infer industry requirements not stated. "
        f"Registry: {json.dumps(compact_registry, ensure_ascii=False, separators=(',', ':'), sort_keys=True)}. "
        f"Business card: {json.dumps({'card_id': card['card_id'], 'spans': card['spans']}, ensure_ascii=False, separators=(',', ':'), sort_keys=True)}. "
        "Return exactly one JSON object with keys role, card_id, profile_code, supporting_span_ids, abstain, abstain_reason_codes. "
        f"Set role exactly to business_profile_mapper and card_id to {json.dumps(card['card_id'])}. "
        f"supporting_span_ids must be unique members of {json.dumps(span_ids)}. "
        "For a mapped profile set abstain=false and abstain_reason_codes=[]. For abstention set profile_code=null, abstain=true and use exactly one allowed reason code. Output JSON only."
    )


def parse_profile_response(text: str, card: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(text.strip())
    expected_keys = {"role", "card_id", "profile_code", "supporting_span_ids", "abstain", "abstain_reason_codes"}
    if set(payload) != expected_keys or payload["role"] != "business_profile_mapper" or payload["card_id"] != card["card_id"]:
        raise ValueError("business_profile_response_identity_invalid")
    span_ids = {row["span_id"] for row in card["spans"]}
    cited = payload["supporting_span_ids"]
    if not isinstance(cited, list) or len(cited) != len(set(cited)) or not set(cited) <= span_ids:
        raise ValueError("business_profile_span_reference_invalid")
    if payload["abstain"]:
        reasons = payload["abstain_reason_codes"]
        if payload["profile_code"] is not None or not isinstance(reasons, list) or len(reasons) != 1 or reasons[0] not in ABSTAIN_REASON_CODES:
            raise ValueError("business_profile_abstention_invalid")
    elif payload["profile_code"] not in PROFILE_CODES or payload["abstain_reason_codes"] != []:
        raise ValueError("business_profile_mapping_invalid")
    return payload


def compile_profile(catalog_path: Path, registry_path: Path, profile_code: str) -> dict[str, Any]:
    catalog_text = catalog_path.read_text(encoding="utf-8")
    if "evaluation_metrics" in catalog_text or "evaluation_frames" in catalog_text:
        raise ValueError("heldout_evaluation_leaked_into_profile_compiler")
    catalog = json.loads(catalog_text)
    registry = load_registry(registry_path)
    profile = next(row for row in registry["profiles"] if row["profile_code"] == profile_code)
    baseline = next(row for row in catalog["configs"] if row["config_id"] == "C00")["camera_replay"]["calibration_metrics"]["count_mae"]
    rows = []
    for config in catalog["configs"]:
        replay = config["camera_replay"]
        if config["config_id"] not in catalog["eligible_config_ids"]:
            continue
        metrics = replay["calibration_metrics"]
        if "count_mae_not_worse_than_baseline" in profile["hard_gates"] and metrics["count_mae"] > baseline:
            continue
        values = {
            "maximize_recall": -metrics["recall"],
            "minimize_count_mae": metrics["count_mae"],
            "maximize_precision": -metrics["precision"],
            "minimize_p95_ms": replay["p95_ms"],
        }
        rows.append((tuple(values[key] for key in profile["objective_order"]), config["config_id"]))
    ranked = [config_id for _, config_id in sorted(rows)]
    return {
        "camera_id": catalog["camera_id"],
        "profile_code": profile_code,
        "ranked_config_ids": ranked,
        "selected_config_id": ranked[0] if ranked else None,
        "abstain": not ranked,
        "abstain_reason_code": None if ranked else "NO_ELIGIBLE_CONFIGURATION",
        "holdout_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prompt = sub.add_parser("prompt")
    prompt.add_argument("--registry", type=Path, required=True)
    prompt.add_argument("--cards", type=Path, required=True)
    prompt.add_argument("--card-id", required=True)
    blind_prompt = sub.add_parser("prompt-blind")
    blind_prompt.add_argument("--registry", type=Path, required=True)
    blind_prompt.add_argument("--blind-cards", type=Path, required=True)
    blind_prompt.add_argument("--gold", type=Path, required=True)
    blind_prompt.add_argument("--card-id", required=True)
    compile_cmd = sub.add_parser("compile")
    compile_cmd.add_argument("--catalog", type=Path, required=True)
    compile_cmd.add_argument("--registry", type=Path, required=True)
    compile_cmd.add_argument("--profile-code", choices=sorted(PROFILE_CODES), required=True)
    args = parser.parse_args()
    if args.action == "prompt":
        registry = load_registry(args.registry)
        card = next(row for row in load_cards(args.cards) if row["card_id"] == args.card_id)
        print(profile_prompt(card, registry))
    elif args.action == "prompt-blind":
        registry = load_registry(args.registry)
        card = next(row for row in load_blind_cases(args.blind_cards, args.gold) if row["card_id"] == args.card_id)
        print(profile_prompt(card, registry))
    else:
        print(json.dumps(compile_profile(args.catalog, args.registry, args.profile_code), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
