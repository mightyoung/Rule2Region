from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FORBIDDEN_MARKERS = {
    "algorithm_diagnostician": ("sensor fusion", "under-segmentation", "false_negative_cost", "vram", "approved for deployment"),
    "business_expert": ("yolo11_count", "yolo26_roi_count", "small_box_ratio", "vram", "oom"),
    "deployment_expert": ("false_negative_cost", "false_positive_cost", "small_box_ratio", "ground truth is"),
}


def evaluate(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "case-manifest.json").read_text(encoding="utf-8"))
    cases = []
    hashes_by_role: dict[str, list[str]] = {role: [] for role in FORBIDDEN_MARKERS}
    for row in manifest["cases"]:
        case_dir = root / row["case_id"]
        validation_path = case_dir / "validation.json"
        response_path = case_dir / "extracted-response.json"
        meta_path = case_dir / "meta.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {"schema_valid": False}
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"exit_code": None}
        response_text = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
        lowered = response_text.lower()
        markers = [marker for marker in FORBIDDEN_MARKERS[row["role"]] if marker in lowered]
        digest = hashlib.sha256(response_text.encode("utf-8")).hexdigest() if response_text else None
        if digest:
            hashes_by_role[row["role"]].append(digest)
        cases.append({
            **row,
            "complete": (case_dir / "COMPLETE").exists(),
            "exit_code": meta.get("exit_code"),
            "forbidden_markers": markers,
            "response_sha256": digest,
            "schema_valid": bool(validation.get("schema_valid")),
            "wall_ms": meta.get("wall_ms"),
        })
    repeat_exact = {role: len(values) == 3 and len(set(values)) == 1 for role, values in hashes_by_role.items()}
    structural_pass = all(row["complete"] and row["exit_code"] == 0 and row["schema_valid"] for row in cases)
    role_isolation_pass = all(not row["forbidden_markers"] for row in cases)
    result = {
        "schema_version": "sceneconfig.qwen-text-role-matrix.v1",
        "case_count": len(cases),
        "cases": cases,
        "exact_repeat_match": repeat_exact,
        "role_isolation_pass": role_isolation_pass,
        "structural_pass": structural_pass,
        "passed": structural_pass and role_isolation_pass and all(repeat_exact.values()),
        "note": "Compatibility result only; it does not establish diagnostic correctness or deployment value.",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.run_dir)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
