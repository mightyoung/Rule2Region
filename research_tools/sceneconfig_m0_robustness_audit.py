"""Post-gate operational robustness audit for M0 development evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit(e4: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    rows = e4["per_camera"]
    precision_safe = sum(row["precision_change"] >= -.05 for row in rows.values())
    mae_noninferior = sum(row["relative_count_mae_improvement"] >= -.10 for row in rows.values())
    checks = {
        "registered_e4_gate_passed": e4.get("passed") is True,
        "all_cameras_precision_safe": precision_safe == len(rows),
        "all_cameras_count_mae_noninferior": mae_noninferior == len(rows),
        "roi_batch_invariant": batch.get("batch_invariant") is True,
    }
    return {
        "schema_version": "sceneconfig.m0-operational-robustness-audit.v1",
        "evidence_status": "DEVELOPMENT_ONLY_POST_HOC_DIAGNOSTIC",
        "registered_e4_passed": checks["registered_e4_gate_passed"],
        "operational_robustness_passed": all(checks.values()),
        "checks": checks,
        "precision_safe_camera_count": precision_safe,
        "count_mae_noninferior_camera_count": mae_noninferior,
        "camera_count": len(rows),
        "batch_count_disagreement_frame_rate": batch["count_disagreement_frame_rate"],
        "decision": "BLOCK_CONFIRMATION_ADMISSION_AND_FREEZE_CANONICAL_INFERENCE_OPERATOR" if not all(checks.values()) else "READY_FOR_CONFIRMATION_ADMISSION_REVIEW",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e4", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        json.loads(args.e4.read_text(encoding="utf-8")),
        json.loads(args.batch.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
