from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


def summarize_predictions(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source_video_id"]), str(row["method"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (source_video_id, method), group in sorted(grouped.items()):
        valid = [
            row
            for row in group
            if row.get("status") == "passed"
            and isinstance(row.get("prediction"), int)
            and not isinstance(row.get("prediction"), bool)
            and row["prediction"] >= 0
        ]
        absolute_errors = [abs(row["prediction"] - row["label"]) for row in valid]
        signed_errors = [row["prediction"] - row["label"] for row in valid]
        summaries.append(
            {
                "source_video_id": source_video_id,
                "method": method,
                "windows": len(group),
                "valid_windows": len(valid),
                "valid_coverage": len(valid) / len(group) if group else 0.0,
                "mae_valid": statistics.fmean(absolute_errors)
                if absolute_errors
                else None,
                "bias_valid": statistics.fmean(signed_errors)
                if signed_errors
                else None,
                "exact_match_rate_valid": (
                    sum(error == 0 for error in absolute_errors) / len(absolute_errors)
                    if absolute_errors
                    else None
                ),
            }
        )
    return summaries


def evaluate_engineering_quality(
    rows: Iterable[dict[str, Any]],
    *,
    expected_windows_by_video: Mapping[str, Sequence[str]],
    method: str = "codec",
    baseline: str = "dense",
    minimum_completion: float = 0.95,
    median_delta_mae_limit: float = 0.5,
) -> dict[str, Any]:
    seen: set[tuple[str, str, str]] = set()
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        source_video_id = str(row["source_video_id"])
        row_method = str(row["method"])
        window_id = str(row["window_id"])
        key = (source_video_id, row_method, window_id)
        if key in seen:
            raise ValueError(f"duplicate result cell: {key}")
        seen.add(key)
        grouped[(source_video_id, row_method)][window_id] = row

    expected_manifest = {
        str(video): tuple(str(window) for window in windows)
        for video, windows in expected_windows_by_video.items()
    }
    for video, windows in expected_manifest.items():
        if not windows or len(set(windows)) != len(windows):
            raise ValueError(f"invalid expected window manifest for {video}")
    for source_video_id, _, window_id in seen:
        if source_video_id not in expected_manifest:
            raise ValueError(f"result video absent from manifest: {source_video_id}")
        if window_id not in expected_manifest[source_video_id]:
            raise ValueError(
                f"result window absent from manifest: {source_video_id}/{window_id}"
            )

    videos = sorted(expected_manifest)
    per_video: list[dict[str, Any]] = []
    for video in videos:
        method_rows = grouped.get((video, method), {})
        baseline_rows = grouped.get((video, baseline), {})
        expected_windows = expected_manifest[video]

        def valid(row: dict[str, Any] | None) -> bool:
            return bool(
                row is not None
                and row.get("status") == "passed"
                and isinstance(row.get("prediction"), int)
                and not isinstance(row.get("prediction"), bool)
                and row["prediction"] >= 0
            )

        paired_valid = [
            window
            for window in expected_windows
            if valid(method_rows.get(window))
            and valid(baseline_rows.get(window))
            and method_rows[window]["label"] == baseline_rows[window]["label"]
        ]
        completion = len(paired_valid) / len(expected_windows)
        delta_mae = None
        if len(paired_valid) == len(expected_windows):
            method_mae = statistics.fmean(
                abs(method_rows[window]["prediction"] - method_rows[window]["label"])
                for window in paired_valid
            )
            baseline_mae = statistics.fmean(
                abs(
                    baseline_rows[window]["prediction"] - baseline_rows[window]["label"]
                )
                for window in paired_valid
            )
            delta_mae = method_mae - baseline_mae
        per_video.append(
            {
                "source_video_id": video,
                "expected_windows": len(expected_windows),
                "paired_valid_windows": len(paired_valid),
                "completion": completion,
                "delta_mae": delta_mae,
            }
        )

    complete_deltas = [
        item["delta_mae"] for item in per_video if item["delta_mae"] is not None
    ]
    expected_total = sum(item["expected_windows"] for item in per_video)
    overall_completion = (
        sum(item["paired_valid_windows"] for item in per_video) / expected_total
        if expected_total
        else 0.0
    )
    median_delta = (
        statistics.median(complete_deltas)
        if per_video and len(complete_deltas) == len(per_video)
        else None
    )
    passed = bool(
        per_video
        and overall_completion >= minimum_completion
        and median_delta is not None
        and median_delta <= median_delta_mae_limit
    )
    return {
        "method": method,
        "baseline": baseline,
        "videos": per_video,
        "overall_completion": overall_completion,
        "median_delta_mae": median_delta,
        "passed": passed,
    }
