"""Validate and close the autoresearch single-pass comparison."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_NAMES = {
    f"o{int(round(outside * 100)):02d}_f{int(round(focus * 100)):02d}"
    for outside in (0.15, 0.25, 0.35)
    for focus in (0.05, 0.10, 0.15, 0.25, 0.35)
}


def validate(result_path: Path, receipt_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checks = {
        "schema": result.get("schema_version") == "sceneconfig.execution-policy.single-pass-spatial-threshold.v1",
        "development_only": result.get("evidence_status") == "DEVELOPMENT_ONLY" and result.get("official_h1") is False,
        "single_model_call": result.get("model_calls_per_frame") == 1 and result.get("second_model_called") is False,
        "fixed_scope": result.get("candidate_count") == 15 and result.get("view_count") == 3 and result.get("clip_count") == 8,
        "sealed_camera_absent": result.get("sealed_camera_used") is False and "VIRAT_S_010113" not in json.dumps(result),
        "receipt_present": receipt_path.is_file() and receipt_path.stat().st_size > 0,
    }
    view_checks = []
    per_clip = result.get("per_clip", {})
    h1_checks = []
    for clip in per_clip.values():
        for phase in ("history", "validation"):
            phase_row = clip[phase]
            b11 = phase_row["b11_reference"]
            b11_mae = float(b11["mae"])
            b11_precision = float(b11["precision"])
            b26_time = float(phase_row["b26_time_ms_mean"])
            for candidate in phase_row["candidates"].values():
                h1 = candidate["h1"]
                mae = float(candidate["mae"])
                precision = float(candidate["precision"])
                time_ms = float(candidate["time_ms_mean"])
                mae_change = 0.0 if b11_mae == 0 else (b11_mae - mae) / b11_mae
                precision_change = precision - b11_precision
                expected_pass = (
                    float(h1["compiled_proxy"]) < 1.0
                    and time_ms < b26_time
                    and mae_change >= -0.05
                    and precision_change >= -0.05
                )
                h1_checks.append(
                    math.isclose(float(candidate["mae_change_vs_b11"]), mae_change, abs_tol=1e-12)
                    and math.isclose(float(candidate["precision_change_vs_b11"]), precision_change, abs_tol=1e-12)
                    and math.isclose(float(h1["mae_change_vs_b11"]), mae_change, abs_tol=1e-12)
                    and math.isclose(float(h1["precision_change_vs_b11"]), precision_change, abs_tol=1e-12)
                    and bool(h1["passed"]) == expected_pass
                )
    checks["all_h1_rows_recomputed"] = len(h1_checks) == 8 * 2 * 15 and all(h1_checks)

    for row in result.get("view_results", {}).values():
        candidates = row.get("candidates", [])
        names = {candidate.get("name") for candidate in candidates}
        eligible = [candidate for candidate in candidates if candidate.get("all_history_pass")]
        oracle = [candidate for candidate in candidates if candidate.get("all_validation_pass")]
        selected = row.get("history_selected")
        expected_selected = None
        if eligible:
            expected_selected = min(
                eligible,
                key=lambda candidate: (
                    float(candidate["history_mae_mean"]),
                    -float(candidate["history_precision_mean"]),
                    float(candidate["threshold_change"]),
                    str(candidate["name"]),
                ),
            )["name"]
        aggregate_checks = []
        for candidate in candidates:
            name = candidate["name"]
            history_rows = [per_clip[sequence]["history"]["candidates"][name] for sequence in row["sequences"]]
            validation_rows = [per_clip[sequence]["validation"]["candidates"][name] for sequence in row["sequences"]]
            mean = lambda values: sum(values) / len(values)
            aggregate_checks.append(
                bool(candidate["all_history_pass"]) == all(item["h1"]["passed"] for item in history_rows)
                and bool(candidate["all_validation_pass"]) == all(item["h1"]["passed"] for item in validation_rows)
                and math.isclose(float(candidate["history_mae_mean"]), mean([float(item["mae"]) for item in history_rows]), abs_tol=1e-12)
                and math.isclose(float(candidate["validation_mae_mean"]), mean([float(item["mae"]) for item in validation_rows]), abs_tol=1e-12)
                and math.isclose(float(candidate["history_precision_mean"]), mean([float(item["precision"]) for item in history_rows]), abs_tol=1e-12)
                and math.isclose(float(candidate["validation_precision_mean"]), mean([float(item["precision"]) for item in validation_rows]), abs_tol=1e-12)
            )
        view_checks.append(
            len(candidates) == 15
            and names == EXPECTED_NAMES
            and row.get("history_eligible_count") == len(eligible)
            and row.get("validation_oracle_count") == len(oracle)
            and ((selected or {}).get("name") if selected else None) == expected_selected
            and bool(row.get("selected_validation_pass")) == bool(selected and selected.get("all_validation_pass"))
            and all(aggregate_checks)
        )
    checks["all_views_recomputed"] = len(view_checks) == 3 and all(view_checks)
    summary = result.get("summary", {})
    views = list(result.get("view_results", {}).values())
    checks["summary_recomputed"] = (
        summary.get("history_selected_view_count") == sum(row.get("history_selected") is not None for row in views)
        and summary.get("selected_validation_pass_view_count") == sum(bool(row.get("selected_validation_pass")) for row in views)
        and summary.get("validation_oracle_possible_view_count") == sum(int(row.get("validation_oracle_count", 0)) > 0 for row in views)
    )
    passed = all(checks.values())
    transferred = int(result.get("summary", {}).get("selected_validation_pass_view_count", 0))
    return {
        "schema_version": "autoresearch.completion.v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "research_outcome": "positive_transfer" if transferred > 0 else "exhaustive_negative_stop",
        "checks": checks,
        "metrics": result.get("summary", {}),
        "summary": (
            f"比较完整；历史选出的配置在 {transferred}/3 个画面通过验证。"
            if passed else "比较产物未通过完整性核验。"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completion = validate(args.result, args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(completion, ensure_ascii=False, sort_keys=True))
    return 0 if completion["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
