"""Validate completion of the development-only SceneConfig E2/E3 research mission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(e2_path: Path, e3_path: Path) -> dict:
    e2 = json.loads(e2_path.read_text(encoding="utf-8"))
    e3 = json.loads(e3_path.read_text(encoding="utf-8"))
    errors = []
    for name, payload in (("E2", e2), ("E3", e3)):
        if payload.get("evidence_status") != "DEVELOPMENT_ONLY":
            errors.append(f"{name}_not_development_only")
        if payload.get("holdout_opened") is not False:
            errors.append(f"{name}_holdout_not_closed")
        if payload.get("sealed_camera_used") is not False:
            errors.append(f"{name}_sealed_camera_used")
    if e2.get("passed") is not True or e2.get("next_gate") != "E3":
        errors.append("E2_not_passed")
    if e3.get("query_budget") != {"rounds": 2, "slots_per_round": 3, "total_slots": 6}:
        errors.append("E3_query_contract_invalid")
    if e3.get("passed") is True:
        errors.append("E3_unexpected_pass_requires_E4")
    if e3.get("next_gate") != "DELETE_OR_DOWNGRADE_QWEN_INCREMENTAL_CLAIM":
        errors.append("E3_rejection_not_explicit")
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "validation_outcome": "VALIDATED_REJECTION_OF_QWEN_INCREMENTAL_CLAIM" if not errors else "INVALID",
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2", type=Path, required=True)
    parser.add_argument("--e3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.e2, args.e3)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
