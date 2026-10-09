"""Validate decisive completion of the frozen YOLO26 detect/depth ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "comparison_complete": metrics.get("comparison_complete") is True,
        "fixed_evidence_status": metrics.get("evidence_status") == "BOUNDED_256_FRAME_YOLO26_DETECT_DEPTH_ABLATION",
        "four_development_cameras": metrics.get("development_camera_count") == 4,
        "exactly_256_frames": metrics.get("frame_count") == 256,
        "no_configuration_search": metrics.get("configuration_search") is False,
        "no_camera_leakage": metrics.get("camera_split_leakage") is False,
        "gt_not_used_online": metrics.get("gt_used_online") is False,
        "strong_model_not_called": metrics.get("strong_model_called") is False,
        "sealed_camera_not_used": metrics.get("sealed_camera_used") is False,
        "checkpoint_hashes_recorded": metrics.get("checkpoint_hashes_recorded") is True,
        "latency_reported": metrics.get("latency_reported") is True,
        "depth_contract_verified_no_boxes": metrics.get("yolo26_depth_has_boxes") is False,
        "numpy_color_contract_verified": metrics.get("numpy_input_color") == "BGR_CONVERTED_FROM_PIL_RGB",
    }
    passed = all(checks.values())
    detect = metrics.get("detect_replacement_recommended") is True
    depth = metrics.get("depth_aux_recommended") is True
    outcome = (
        "detect_and_depth_promoted" if detect and depth else
        "detect_promoted_depth_rejected" if detect else
        "depth_promoted_detect_rejected" if depth else
        "both_rejected"
    )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "research_outcome": outcome if passed else "incomplete",
        "detect_replacement_recommended": detect if passed else False,
        "depth_aux_recommended": depth if passed else False,
        "summary": f"YOLO26 spatial ablation complete; outcome={outcome}" if passed else "YOLO26 spatial ablation evidence incomplete",
        "checks": checks,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(json.loads(args.metrics.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
