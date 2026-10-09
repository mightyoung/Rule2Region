"""Build the deterministic, fail-closed G4 EvidenceGate paper bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evidencegate.manifest import write_immutable_manifest
from evidencegate.terminals import TerminalState
from research_tools.g3_latency import expand_latency_attempts


QUALITY_UNITS = 986
QUALITY_SUCCESS_MINIMUM = 937
LATENCY_ATTEMPTS = 896
LATENCY_CELLS = 64
BOOTSTRAP_DRAWS = 2000
EXPECTED_FILES = {
    "g4-result.json",
    "g4-group-metrics.csv",
    "g4-attempts.csv",
    "g4-terminals.csv",
    "g4-quality-model-path-pareto.svg",
    "g4-stage-latency.svg",
    "g4-failure-cases.svg",
    "g4-output-manifest.json",
}
RESOURCE_FIELDS = (
    "encoded_bytes",
    "decoded_pixels",
    "model_pixels",
    "prefill_tokens",
    "visual_tokens",
    "vram_bytes",
    "wall_clock_ms",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"artifact_exists_with_different_content:{path}")
        return
    digest = hashlib.sha256(payload).hexdigest()
    temporary = path.with_name(f".{path.name}.{digest}.tmp")
    if temporary.exists() and temporary.read_bytes() != payload:
        raise FileExistsError(f"temporary_artifact_conflict:{temporary}")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_snapshot(summary_path: Path, recorded: object) -> Path:
    candidate = Path(str(recorded))
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(summary_path.parent / candidate)
    candidates.extend(
        [
            summary_path.parent / candidate.name,
            summary_path.parent / candidate.parent.name / candidate.name,
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    raise ValueError("g3_snapshot_missing")


def validate_g3_summary(
    path: Path, *, expected_schema: str, expected_snapshot_count: int
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Validate summary schema and its content-addressed JSONL snapshot."""
    summary = _read_json(path)
    if summary.get("schema_version") != expected_schema:
        raise ValueError("g3_summary_schema_invalid")
    snapshot_path = _resolve_snapshot(path, summary.get("snapshot_path", ""))
    digest = _sha(snapshot_path)
    if snapshot_path.stem != digest or summary.get("snapshot_sha256") != digest:
        raise ValueError("g3_snapshot_content_address_mismatch")
    records: list[dict[str, Any]] = []
    for line in snapshot_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError("g3_snapshot_record_invalid")
        records.append(item)
    # Snapshots preserve retries, so the frozen logical-unit/attempt count is
    # a lower bound. Exact terminal cardinality and retry chains are validated
    # later by _select_quality/_select_latency and _validate_attempt_provenance.
    if len(records) < expected_snapshot_count:
        raise ValueError("g3_snapshot_record_count_insufficient")
    return summary, records, digest


def _validate_lineage_and_matrix(
    *,
    quality_summary_path: Path,
    quality_summary: Mapping[str, Any],
    latency_summary: Mapping[str, Any],
    formal_manifest_path: Path,
    scope_path: Path,
    config_path: Path,
    quality: Sequence[Mapping[str, Any]],
    latency: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    formal, scope, config = (
        _read_json(formal_manifest_path),
        _read_json(scope_path),
        _read_json(config_path),
    )
    formal_hash, scope_hash, config_hash = (
        _sha(formal_manifest_path),
        _sha(scope_path),
        _sha(config_path),
    )
    if formal.get("execution_config_file_sha256") != config_hash:
        raise ValueError("g4_formal_config_lineage_mismatch")
    if formal.get("source_scope_file_sha256") != scope_hash:
        raise ValueError("g4_formal_scope_lineage_mismatch")
    if config.get("source_scope_sha256") != scope_hash:
        raise ValueError("g4_config_scope_lineage_mismatch")
    for summary in (quality_summary, latency_summary):
        if summary.get("config_file_sha256") != config_hash:
            raise ValueError("g4_summary_config_lineage_mismatch")
        if summary.get("formal_manifest_file_sha256") != formal_hash:
            raise ValueError("g4_summary_formal_lineage_mismatch")
        if summary.get("formal_manifest_sha256") != formal.get("manifest_sha256"):
            raise ValueError("g4_summary_formal_identity_mismatch")
    if latency_summary.get("quality_summary_file_sha256") != _sha(quality_summary_path):
        raise ValueError("g4_latency_quality_summary_lineage_mismatch")
    if quality_summary.get("selected_gpu_uuid") != latency_summary.get(
        "selected_gpu_uuid"
    ):
        raise ValueError("g4_quality_latency_gpu_identity_mismatch")
    expected_quality = {str(unit["unit_id"]): unit for unit in formal.get("units", [])}
    if len(expected_quality) != QUALITY_UNITS or set(expected_quality) != {
        str(record.get("unit_id")) for record in quality
    }:
        raise ValueError("g4_quality_frozen_unit_set_mismatch")
    quality_fields = (
        "unit_id",
        "model",
        "model_layer_identity",
        "modality",
        "condition",
        "resolved_condition",
        "sample_id",
        "sequence_id",
        "camera_motion_stratum",
        "deployment_member",
    )
    for record in quality:
        expected = expected_quality[str(record["unit_id"])]
        if any(record.get(field) != expected.get(field) for field in quality_fields):
            raise ValueError("g4_quality_frozen_identity_mismatch")
    expected_latency = {
        (str(item["unit_id"]), int(item["attempt_index"])): item
        for item in expand_latency_attempts(scope, config)
    }
    actual_latency = {
        (str(record.get("unit_id")), int(record.get("attempt_index", -1))): record
        for record in latency
    }
    if len(expected_latency) != LATENCY_ATTEMPTS or set(expected_latency) != set(
        actual_latency
    ):
        raise ValueError("g4_latency_frozen_attempt_set_mismatch")
    latency_fields = (
        "unit_id",
        "cell_id",
        "attempt_index",
        "model",
        "model_layer_identity",
        "modality",
        "sample_id",
        "resolved_condition",
        "repeat_index",
        "io_policy",
        "include_in_formal_latency",
        "latency_kind",
    )
    for key, record in actual_latency.items():
        expected = expected_latency[key]
        if any(record.get(field) != expected.get(field) for field in latency_fields):
            raise ValueError("g4_latency_frozen_identity_mismatch")
    return {"formal_manifest": formal_hash, "scope": scope_hash, "config": config_hash}


def _last(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    group = [dict(item) for item in records]
    if not group:
        raise ValueError("g3_terminal_missing")
    group.sort(key=lambda item: int(item.get("retry_index", -1)))
    indexes = [int(item.get("retry_index", -1)) for item in group]
    if indexes != list(range(len(group))):
        raise ValueError("g3_retry_sequence_invalid")
    attempt_ids = [item.get("attempt_id") for item in group]
    if any(not isinstance(value, str) or not value.strip() for value in attempt_ids):
        raise ValueError("g3_attempt_id_missing")
    if len(set(attempt_ids)) != len(group):
        raise ValueError("g3_attempt_id_duplicate")
    for index, item in enumerate(group):
        retry_of = item.get("retry_of")
        if index == 0 and retry_of not in (None, ""):
            raise ValueError("g3_initial_attempt_retry_of_invalid")
        if index > 0 and retry_of != attempt_ids[index - 1]:
            raise ValueError("g3_retry_chain_invalid")
    return group[-1]


def _validate_attempt_provenance(
    quality_records: Sequence[Mapping[str, Any]],
    latency_records: Sequence[Mapping[str, Any]],
) -> None:
    allowed = {state.value for state in TerminalState}
    all_records = list(quality_records) + list(latency_records)
    attempt_ids: list[str] = []
    for record in all_records:
        attempt_id = record.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id.strip():
            raise ValueError("g3_attempt_id_missing")
        attempt_ids.append(attempt_id)
        if record.get("terminal_state") not in allowed:
            raise ValueError("g3_terminal_state_invalid")
    if len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("g3_attempt_id_not_globally_unique")
    identity_fields = (
        "unit_id",
        "model",
        "model_layer_identity",
        "modality",
        "condition",
        "resolved_condition",
        "sample_id",
        "sequence_id",
        "camera_motion_stratum",
        "deployment_member",
        "config_file_sha256",
        "formal_manifest_file_sha256",
        "formal_manifest_sha256",
        "prompt_contract_sha256",
    )
    groups: dict[tuple[str, int | None], list[Mapping[str, Any]]] = defaultdict(list)
    for record in quality_records:
        groups[(str(record["unit_id"]), None)].append(record)
    for record in latency_records:
        groups[(str(record["unit_id"]), int(record.get("attempt_index", -1)))].append(
            record
        )
    for records in groups.values():
        ordered = sorted(records, key=lambda row: int(row.get("retry_index", -1)))
        _last(ordered)
        frozen = {field: ordered[0].get(field) for field in identity_fields}
        for record in ordered[1:]:
            if {field: record.get(field) for field in identity_fields} != frozen:
                raise ValueError("g3_retry_identity_changed")


def _select_quality(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        unit_id = str(record.get("unit_id", ""))
        if not unit_id:
            raise ValueError("quality_unit_id_missing")
        grouped[unit_id].append(record)
    selected = [_last(group) for _, group in sorted(grouped.items())]
    if len(selected) != QUALITY_UNITS:
        raise ValueError("quality_terminal_count_invalid")
    return selected


def _select_latency(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        unit_id = str(record.get("unit_id", ""))
        attempt_index = int(record.get("attempt_index", -1))
        if not unit_id or attempt_index < 0:
            raise ValueError("latency_attempt_identity_missing")
        grouped[(unit_id, attempt_index)].append(record)
    selected = [_last(group) for _, group in sorted(grouped.items())]
    if len(selected) != LATENCY_ATTEMPTS:
        raise ValueError("latency_terminal_count_invalid")
    if len({str(record.get("cell_id", "")) for record in selected}) != LATENCY_CELLS:
        raise ValueError("latency_cell_count_invalid")
    return selected


def _finite_nonnegative(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"resource_field_not_numeric:{field}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"resource_field_invalid:{field}")
    return numeric


def _resource_records(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result = record.get("result")
    if not isinstance(result, Mapping):
        if record.get("terminal_state") == "success":
            raise ValueError("successful_record_result_missing")
        return []
    ledger = result.get("resource_ledger")
    if not isinstance(ledger, Mapping):
        if record.get("terminal_state") == "success":
            raise ValueError("successful_record_resource_ledger_missing")
        return []
    if ledger.get("schema_version") != "evidencegate.g3.resource-ledger.v1":
        raise ValueError("resource_ledger_schema_invalid")
    rows = ledger.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("resource_ledger_records_invalid")
    output: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("resource_ledger_row_invalid")
        if not isinstance(row.get("stage"), str) or not str(row["stage"]).strip():
            raise ValueError("resource_ledger_stage_invalid")
        if not isinstance(row.get("combined"), bool):
            raise ValueError("resource_ledger_combined_invalid")
        for field in RESOURCE_FIELDS:
            if field not in row:
                raise ValueError(f"resource_field_missing:{field}")
            _finite_nonnegative(row[field], field)
        output.append(row)
    return output


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _model_precision(record: Mapping[str, Any]) -> tuple[str, str]:
    model = str(record.get("model", "unknown"))
    precision = model.rsplit("-", 1)[-1] if "-" in model else "unknown"
    return model, precision


def _condition(record: Mapping[str, Any]) -> str:
    return str(record.get("resolved_condition", record.get("condition", "unknown")))


def _group_tuple(record: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    model, precision = _model_precision(record)
    modality = str(record.get("modality", "unknown"))
    stratum = str(
        record.get("camera_motion_stratum")
        or ("IMAGE" if modality == "image" else "UNKNOWN")
    )
    return model, precision, modality, _condition(record), stratum


def _correct(record: Mapping[str, Any]) -> bool | None:
    if record.get("terminal_state") != "success":
        return None
    result = record.get("result")
    if not isinstance(result, Mapping) or not isinstance(result.get("correct"), bool):
        raise ValueError("successful_quality_correctness_missing")
    modality = record.get("modality")
    actual, expected = result.get("parsed_output"), result.get("expected")
    if modality == "image":
        if not isinstance(actual, bool) or not isinstance(expected, bool):
            raise ValueError("image_parsed_or_expected_invalid")
    elif modality == "video":
        if (
            not isinstance(actual, int)
            or isinstance(actual, bool)
            or not isinstance(expected, int)
            or isinstance(expected, bool)
            or not 0 <= actual <= 99
            or not 0 <= expected <= 99
        ):
            raise ValueError("video_parsed_or_expected_invalid")
    else:
        raise ValueError("quality_modality_invalid")
    recomputed = actual == expected
    if result["correct"] is not recomputed:
        raise ValueError("quality_correctness_mismatch")
    return recomputed


def _count_pair(record: Mapping[str, Any]) -> tuple[float, float] | None:
    if record.get("terminal_state") != "success":
        return None
    result = record.get("result")
    if not isinstance(result, Mapping):
        return None
    actual, expected = result.get("parsed_output"), result.get("expected")
    if (
        isinstance(actual, int)
        and not isinstance(actual, bool)
        and isinstance(expected, int)
        and not isinstance(expected, bool)
        and 0 <= actual <= 99
        and 0 <= expected <= 99
    ):
        return float(actual), float(expected)
    raise ValueError("video_parsed_or_expected_invalid")


def _latency_value(record: Mapping[str, Any]) -> float | None:
    result = record.get("result")
    value = result.get("model_path_seconds") if isinstance(result, Mapping) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _finite_nonnegative(value, "model_path_seconds")
    return None


def _cluster_bootstrap(
    clusters: Mapping[str, Sequence[float]],
    *,
    seed_text: str,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    means = [
        statistics.mean(clusters[key]) for key in sorted(clusters) if clusters[key]
    ]
    if not means:
        return {
            "clusters": 0,
            "draws": 0,
            "estimate": None,
            "lower_2_5": None,
            "upper_97_5": None,
        }
    state = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    estimates: list[float] = []
    for _ in range(draws):
        sample: list[float] = []
        for _ in means:
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            sample.append(means[state % len(means)])
        estimates.append(statistics.mean(sample))
    return {
        "clusters": len(means),
        "draws": draws,
        "estimate": statistics.mean(means),
        "lower_2_5": _quantile(estimates, 0.025),
        "upper_97_5": _quantile(estimates, 0.975),
    }


def _paired_effects(quality: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, str, str, str], Mapping[str, Any]] = {}
    for record in quality:
        if record.get("terminal_state") != "success":
            continue
        model, _, modality, condition, stratum = _group_tuple(record)
        sample_id = str(record.get("sample_id", ""))
        indexed[(model, modality, stratum, sample_id, condition)] = record
    conditions = sorted({_condition(record) for record in quality})
    effects: list[dict[str, Any]] = []
    for model, modality, stratum in sorted({key[:3] for key in indexed}):
        baseline = "native" if modality == "image" else "dense"
        sample_ids = sorted(
            {key[3] for key in indexed if key[:3] == (model, modality, stratum)}
        )
        for condition in conditions:
            if condition == baseline:
                continue
            clusters: dict[str, list[float]] = defaultdict(list)
            pairs = 0
            for sample_id in sample_ids:
                base = indexed.get((model, modality, stratum, sample_id, baseline))
                candidate = indexed.get(
                    (model, modality, stratum, sample_id, condition)
                )
                if base is None or candidate is None:
                    continue
                if modality == "image":
                    base_value, candidate_value = _correct(base), _correct(candidate)
                    if base_value is None or candidate_value is None:
                        continue
                    delta = float(candidate_value) - float(base_value)
                    cluster = sample_id
                else:
                    base_pair, candidate_pair = (
                        _count_pair(base),
                        _count_pair(candidate),
                    )
                    if base_pair is None or candidate_pair is None:
                        continue
                    delta = abs(candidate_pair[0] - candidate_pair[1]) - abs(
                        base_pair[0] - base_pair[1]
                    )
                    cluster = str(
                        candidate.get("sequence_id") or sample_id.split(":", 1)[0]
                    )
                clusters[cluster].append(delta)
                pairs += 1
            if not pairs:
                continue
            bootstrap = _cluster_bootstrap(
                clusters,
                seed_text=f"{model}|{modality}|{stratum}|{condition}|{baseline}",
            )
            effects.append(
                {
                    "model": model,
                    "modality": modality,
                    "camera_motion_stratum": stratum,
                    "condition": condition,
                    "baseline": baseline,
                    "effect": "accuracy_delta" if modality == "image" else "mae_delta",
                    "paired_units": pairs,
                    **bootstrap,
                }
            )
    return effects


def _group_metrics(
    quality: Sequence[Mapping[str, Any]], latency: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    quality_groups: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    latency_groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for record in quality:
        quality_groups[_group_tuple(record)].append(record)
    for record in latency:
        key = _group_tuple(record)
        latency_groups[key[:4]].append(record)
    rows: list[dict[str, Any]] = []
    keys = sorted(quality_groups)
    represented_latency_keys: set[tuple[str, str, str, str]] = set()
    for key in keys:
        qrows = quality_groups[key]
        latency_key = key[:4]
        represented_latency_keys.add(latency_key)
        lrows = latency_groups.get(latency_key, [])
        qsuccess = [row for row in qrows if row.get("terminal_state") == "success"]
        correctness = [
            value for row in qsuccess if (value := _correct(row)) is not None
        ]
        count_pairs = (
            [pair for row in qsuccess if (pair := _count_pair(row)) is not None]
            if key[2] == "video"
            else []
        )
        sequence_errors: dict[str, list[float]] = defaultdict(list)
        sequence_exact: dict[str, list[float]] = defaultdict(list)
        if key[2] == "video":
            for row in qsuccess:
                pair = _count_pair(row)
                if pair is None:
                    continue
                sequence = str(
                    row.get("sequence_id")
                    or str(row.get("sample_id", "")).split(":", 1)[0]
                )
                sequence_errors[sequence].append(abs(pair[0] - pair[1]))
                sequence_exact[sequence].append(float(pair[0] == pair[1]))
        sequence_mean_errors = [
            statistics.mean(sequence_errors[name]) for name in sorted(sequence_errors)
        ]
        sequence_mean_exact = [
            statistics.mean(sequence_exact[name]) for name in sorted(sequence_exact)
        ]
        steady = [
            value
            for row in lrows
            if row.get("terminal_state") == "success"
            and row.get("latency_kind") == "steady"
            if (value := _latency_value(row)) is not None
        ]
        resource_samples: dict[str, list[float]] = defaultdict(list)
        for record in qsuccess:
            per_record: Counter[str] = Counter()
            for resource in _resource_records(record):
                for field in RESOURCE_FIELDS:
                    per_record[field] += float(resource[field])
            for field in RESOURCE_FIELDS:
                resource_samples[field].append(per_record[field])
        stage_samples: dict[str, list[float]] = defaultdict(list)
        for record in lrows:
            if (
                record.get("terminal_state") != "success"
                or record.get("latency_kind") != "steady"
            ):
                continue
            for resource in _resource_records(record):
                stage_samples[str(resource.get("stage", "unknown"))].append(
                    float(resource["wall_clock_ms"])
                )
        rows.append(
            {
                "model": key[0],
                "precision": key[1],
                "modality": key[2],
                "condition": key[3],
                "camera_motion_stratum": key[4],
                "quality_terminals": len(qrows),
                "quality_successes": len(qsuccess),
                "quality_success_rate": _ratio(len(qsuccess), len(qrows)),
                "accuracy": _ratio(sum(correctness), len(correctness))
                if key[2] == "image"
                else None,
                "video_exact_count": sum(
                    actual == expected for actual, expected in count_pairs
                ),
                "video_scored": len(count_pairs),
                "video_exact_rate_sequence_equal": statistics.mean(sequence_mean_exact)
                if sequence_mean_exact
                else None,
                "video_mae": statistics.mean(sequence_mean_errors)
                if sequence_mean_errors
                else None,
                "video_sequence_clusters": len(sequence_mean_errors),
                "latency_stratum_scope": "shared_all_video"
                if key[2] == "video"
                else "image",
                "latency_steady_successes": len(steady),
                "latency_p50_seconds": statistics.median(steady) if steady else None,
                "latency_p95_seconds": _quantile(steady, 0.95),
                "resource_p50": {
                    field: statistics.median(values)
                    for field, values in sorted(resource_samples.items())
                    if values
                },
                "stage_p50_ms": {
                    stage: statistics.median(values)
                    for stage, values in sorted(stage_samples.items())
                    if values
                },
            }
        )
    for key in sorted(set(latency_groups) - represented_latency_keys):
        lrows = latency_groups[key]
        steady = [
            value
            for row in lrows
            if row.get("terminal_state") == "success"
            and row.get("latency_kind") == "steady"
            if (value := _latency_value(row)) is not None
        ]
        stage_samples: dict[str, list[float]] = defaultdict(list)
        for record in lrows:
            if (
                record.get("terminal_state") != "success"
                or record.get("latency_kind") != "steady"
            ):
                continue
            for resource in _resource_records(record):
                stage_samples[str(resource.get("stage", "unknown"))].append(
                    float(resource["wall_clock_ms"])
                )
        rows.append(
            {
                "model": key[0],
                "precision": key[1],
                "modality": key[2],
                "condition": key[3],
                "camera_motion_stratum": "LATENCY_ONLY",
                "quality_terminals": 0,
                "quality_successes": 0,
                "quality_success_rate": None,
                "accuracy": None,
                "video_exact_count": 0,
                "video_scored": 0,
                "video_exact_rate_sequence_equal": None,
                "video_mae": None,
                "video_sequence_clusters": 0,
                "latency_stratum_scope": "latency_only",
                "latency_steady_successes": len(steady),
                "latency_p50_seconds": statistics.median(steady) if steady else None,
                "latency_p95_seconds": _quantile(steady, 0.95),
                "resource_p50": {},
                "stage_p50_ms": {
                    stage: statistics.median(values)
                    for stage, values in sorted(stage_samples.items())
                    if values
                },
            }
        )
    return rows


def _validate_summary_counters(
    quality_summary: Mapping[str, Any],
    latency_summary: Mapping[str, Any],
    quality: Sequence[Mapping[str, Any]],
    latency: Sequence[Mapping[str, Any]],
) -> tuple[bool, bool, dict[str, int]]:
    qcounts = Counter(str(record.get("terminal_state")) for record in quality)
    lcounts = Counter(str(record.get("terminal_state")) for record in latency)
    qsuccess = qcounts["success"]
    if quality_summary.get("quality_logic_units") != QUALITY_UNITS:
        raise ValueError("quality_summary_unit_count_mismatch")
    if quality_summary.get("quality_success_minimum") != QUALITY_SUCCESS_MINIMUM:
        raise ValueError("quality_summary_success_minimum_mismatch")
    if quality_summary.get("quality_terminal_coverage") != QUALITY_UNITS:
        raise ValueError("quality_summary_coverage_mismatch")
    if quality_summary.get("quality_successes") != qsuccess:
        raise ValueError("quality_summary_success_mismatch")
    if quality_summary.get("terminal_counts") != dict(sorted(qcounts.items())):
        raise ValueError("quality_summary_terminal_counts_mismatch")
    quality_pass = qsuccess >= QUALITY_SUCCESS_MINIMUM
    if quality_summary.get("verdict") != ("PASS" if quality_pass else "FAIL"):
        raise ValueError("quality_summary_verdict_mismatch")
    if latency_summary.get("latency_attempts_expected") != LATENCY_ATTEMPTS:
        raise ValueError("latency_summary_attempt_count_mismatch")
    if latency_summary.get("latency_terminal_coverage") != LATENCY_ATTEMPTS:
        raise ValueError("latency_summary_coverage_mismatch")
    if latency_summary.get("latency_terminal_counts") != dict(sorted(lcounts.items())):
        raise ValueError("latency_summary_terminal_counts_mismatch")
    if latency_summary.get("terminal_verdict") != "PASS":
        raise ValueError("latency_summary_terminal_verdict_mismatch")
    if latency_summary.get("latency_cells_expected") != LATENCY_CELLS:
        raise ValueError("latency_summary_cell_count_mismatch")

    steady_by_cell: dict[str, list[float]] = defaultdict(list)
    expected_steady = Counter()
    for record in latency:
        if record.get("latency_kind") != "steady":
            continue
        cell_id = str(record.get("cell_id", ""))
        expected_steady[cell_id] += 1
        if record.get("terminal_state") != "success":
            continue
        value = _latency_value(record)
        if value is None:
            raise ValueError("latency_steady_timing_missing")
        steady_by_cell[cell_id].append(value)
    if len(expected_steady) != LATENCY_CELLS or set(expected_steady.values()) != {10}:
        raise ValueError("latency_steady_attempt_structure_mismatch")

    recomputed_metrics = []
    for cell_id in sorted(expected_steady):
        values = steady_by_cell.get(cell_id, [])
        recomputed_metrics.append(
            {
                "cell_id": cell_id,
                "steady_successes": len(values),
                "steady_attempts_expected": expected_steady[cell_id],
                "median_seconds": statistics.median(values) if values else None,
                "p95_seconds": _quantile(values, 0.95),
            }
        )
    complete_cells = sum(
        item["steady_successes"] == item["steady_attempts_expected"]
        for item in recomputed_metrics
    )
    if latency_summary.get("latency_cells_with_all_steady_successes") != complete_cells:
        raise ValueError("latency_summary_complete_cell_count_mismatch")
    if latency_summary.get("cell_metrics") != recomputed_metrics:
        raise ValueError("latency_summary_cell_metrics_mismatch")
    if (
        complete_cells != LATENCY_CELLS
        or latency_summary.get("metrics_completeness") != "COMPLETE"
    ):
        raise ValueError("latency_summary_metrics_incomplete")
    return (
        quality_pass,
        True,
        {
            "quality_successes": qsuccess,
            "latency_successes": lcounts["success"],
        },
    )


def _svg_document(title: str, body: str, description: str) -> bytes:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">'
        f"<title>{html.escape(title)}</title><desc>{html.escape(description)}</desc>"
        '<rect width="960" height="540" fill="white"/>'
        '<g font-family="Arial, sans-serif" fill="#111827">'
        f'<text x="40" y="36" font-size="22">{html.escape(title)}</text>{body}</g></svg>\n'
    ).encode("utf-8")


def _quality_pareto_svg(groups: Sequence[Mapping[str, Any]]) -> bytes:
    image_points = [
        row
        for row in groups
        if row.get("latency_p50_seconds") is not None
        and row.get("modality") == "image"
        and row.get("accuracy") is not None
    ]
    video_points = [
        row
        for row in groups
        if row.get("latency_p50_seconds") is not None
        and row.get("modality") == "video"
        and row.get("video_mae") is not None
    ]
    points = image_points + video_points
    if not points:
        body = '<text x="40" y="90" font-size="16">No matched quality/latency groups.</text>'
    else:
        xs = [float(row["latency_p50_seconds"]) for row in points]
        x_max = max(xs) or 1.0
        body_parts = [
            '<text x="80" y="68" font-size="14">Image accuracy</text>',
            '<line x1="80" y1="240" x2="920" y2="240" stroke="#374151"/>',
            '<line x1="80" y1="80" x2="80" y2="240" stroke="#374151"/>',
            '<text x="80" y="298" font-size="14">Video sequence-equal MAE (raw count units)</text>',
            '<line x1="80" y1="470" x2="920" y2="470" stroke="#374151"/>',
            '<line x1="80" y1="310" x2="80" y2="470" stroke="#374151"/>',
        ]
        for index, row in enumerate(image_points):
            x = 80 + 820 * float(row["latency_p50_seconds"]) / x_max
            y = 240 - 150 * float(row["accuracy"])
            label = f"{row['model']}:{row['condition']}:{row['camera_motion_stratum']}"
            color = "#2563eb" if index % 2 == 0 else "#dc2626"
            body_parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}"><title>{html.escape(label)}</title></circle>'
            )
        video_max = (
            max((float(row["video_mae"]) for row in video_points), default=1.0) or 1.0
        )
        for index, row in enumerate(video_points):
            x = 80 + 820 * float(row["latency_p50_seconds"]) / x_max
            y = 310 + 150 * float(row["video_mae"]) / video_max
            label = f"{row['model']}:{row['condition']}:{row['camera_motion_stratum']}:shared-latency"
            color = "#0f766e" if index % 2 == 0 else "#ea580c"
            body_parts.append(
                f'<rect x="{x - 5:.2f}" y="{y - 5:.2f}" width="10" height="10" fill="{color}"><title>{html.escape(label)}</title></rect>'
            )
        body = "".join(body_parts)
    return _svg_document(
        "Quality vs staged-warm model-path P50",
        body,
        "Not E2E latency. Image accuracy and video sequence-equal MAE use separate panels and native units; video strata share the same latency cells.",
    )


def _stage_latency_svg(groups: Sequence[Mapping[str, Any]]) -> bytes:
    stage_values: dict[str, list[float]] = defaultdict(list)
    for row in groups:
        for stage, value in dict(row.get("stage_p50_ms", {})).items():
            stage_values[str(stage)].append(float(value))
    items = sorted(
        ((stage, statistics.median(values)) for stage, values in stage_values.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:12]
    max_value = max((value for _, value in items), default=1.0)
    body_parts: list[str] = []
    for index, (stage, value) in enumerate(items):
        y = 70 + index * 36
        width = 700 * value / max_value
        body_parts.append(
            f'<text x="40" y="{y + 18}" font-size="12">{html.escape(stage)}</text>'
        )
        body_parts.append(
            f'<rect x="210" y="{y}" width="{width:.2f}" height="22" fill="#0f766e"><title>{value:.3f} ms</title></rect>'
        )
    if not items:
        body_parts.append(
            '<text x="40" y="90" font-size="16">No measured stage timings.</text>'
        )
    return _svg_document(
        "Stage latency",
        "".join(body_parts),
        "Median of group-level steady-attempt stage P50 wall-clock milliseconds.",
    )


def _failure_category(kind: str, record: Mapping[str, Any]) -> str | None:
    state = str(record.get("terminal_state"))
    if state != "success":
        return state
    if kind != "quality":
        return None
    result = record.get("result")
    if not isinstance(result, Mapping):
        return "missing_result"
    if record.get("modality") == "image":
        if _correct(record):
            return None
        return (
            "false_positive"
            if result.get("parsed_output") is True
            else "false_negative"
        )
    pair = _count_pair(record)
    if pair is None or pair[0] == pair[1]:
        return None
    return "undercount" if pair[0] < pair[1] else "overcount"


def _failure_cases(
    quality_records: Sequence[Mapping[str, Any]],
    latency_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for kind, records in (("quality", quality_records), ("latency", latency_records)):
        for record in records:
            category = _failure_category(kind, record)
            if category is None:
                continue
            cases.append(
                {
                    "kind": kind,
                    "attempt_id": record.get("attempt_id"),
                    "unit_id": record.get("unit_id"),
                    "retry_index": record.get("retry_index"),
                    "model": record.get("model"),
                    "modality": record.get("modality"),
                    "condition": _condition(record),
                    "sample_id": record.get("sample_id"),
                    "sequence_id": record.get("sequence_id"),
                    "camera_motion_stratum": record.get("camera_motion_stratum"),
                    "category": category,
                    "error": record.get("error"),
                }
            )
    return sorted(cases, key=lambda row: (str(row["category"]), str(row["attempt_id"])))


def _failure_cases_svg(cases: Sequence[Mapping[str, Any]]) -> bytes:
    counts = Counter(str(row["category"]) for row in cases)
    items = sorted(counts.items())
    max_value = max((value for _, value in items), default=1)
    body_parts: list[str] = []
    for index, (state, value) in enumerate(items):
        x = 60 + index * (520 / max(len(items), 1))
        height = 180 * value / max_value
        body_parts.append(
            f'<rect x="{x:.2f}" y="{270 - height:.2f}" width="46" height="{height:.2f}" fill="#7c3aed"><title>{html.escape(state)}: {value}</title></rect>'
        )
        body_parts.append(
            f'<text x="{x:.2f}" y="292" font-size="10">{html.escape(state)}</text>'
        )
    for index, case in enumerate(cases[:8]):
        label = f"{case['category']}: {case.get('sample_id') or case.get('unit_id')} / {case.get('condition')} / {case.get('attempt_id')}"
        body_parts.append(
            f'<text x="40" y="{330 + index * 22}" font-size="11">{html.escape(label)}</text>'
        )
    return _svg_document(
        "Traceable failure cases",
        "".join(body_parts),
        "Includes every failed retry and incorrect quality output; identifiers resolve to the all-attempt table.",
    )


def _csv_bytes(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def build_g4(
    quality_summary_path: str | Path,
    latency_summary_path: str | Path,
    formal_manifest_path: str | Path,
    scope_path: str | Path,
    execution_config_path: str | Path,
    gpu_budget_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build byte-reproducible tables, figures, statistics, and manifest."""
    qpath, lpath, formal_path, frozen_scope_path, config_path, budget_path, out = (
        Path(quality_summary_path),
        Path(latency_summary_path),
        Path(formal_manifest_path),
        Path(scope_path),
        Path(execution_config_path),
        Path(gpu_budget_path),
        Path(output_dir),
    )
    qsummary, qrecords, qhash = validate_g3_summary(
        qpath,
        expected_schema="evidencegate.g3.quality-summary.v1",
        expected_snapshot_count=QUALITY_UNITS,
    )
    lsummary, lrecords, lhash = validate_g3_summary(
        lpath,
        expected_schema="evidencegate.g3.latency-summary.v1",
        expected_snapshot_count=LATENCY_ATTEMPTS,
    )
    quality, latency = _select_quality(qrecords), _select_latency(lrecords)
    _validate_attempt_provenance(qrecords, lrecords)
    lineage = _validate_lineage_and_matrix(
        quality_summary_path=qpath,
        quality_summary=qsummary,
        latency_summary=lsummary,
        formal_manifest_path=formal_path,
        scope_path=frozen_scope_path,
        config_path=config_path,
        quality=quality,
        latency=latency,
    )
    quality_pass, latency_pass, terminal_counts = _validate_summary_counters(
        qsummary, lsummary, quality, latency
    )
    groups = _group_metrics(quality, latency)
    effects = _paired_effects(quality)
    failures = _failure_cases(qrecords, lrecords)
    config = _read_json(config_path)
    limits = config.get("limits")
    if not isinstance(limits, Mapping):
        raise ValueError("g4_config_limits_missing")
    declared_call_cap = int(limits.get("total_call_hard_cap", -1))
    declared_gpu_cap_seconds = float(limits.get("gpu_hour_hard_cap", math.nan)) * 3600
    if declared_call_cap != int(lsummary.get("total_call_hard_cap", -2)):
        raise ValueError("g4_declared_call_cap_mismatch")
    quality_calls = sum(
        record.get("terminal_state") != TerminalState.NOT_RUN_DUE_TO_HARD_STOP.value
        for record in qrecords
    )
    latency_calls = sum(
        record.get("terminal_state") != TerminalState.NOT_RUN_DUE_TO_HARD_STOP.value
        for record in lrecords
    )
    g2_calls = int(lsummary.get("g2_calls", -1))
    total_calls = g2_calls + quality_calls + latency_calls
    if qsummary.get("quality_attempts_started") != quality_calls:
        raise ValueError("g4_quality_raw_call_count_mismatch")
    if lsummary.get("quality_attempts_started") != quality_calls:
        raise ValueError("g4_latency_quality_call_count_mismatch")
    if lsummary.get("latency_attempts_started") != latency_calls:
        raise ValueError("g4_latency_raw_call_count_mismatch")
    if lsummary.get("total_calls_through_latency") != total_calls:
        raise ValueError("g4_total_raw_call_count_mismatch")
    budget = _read_json(budget_path)
    total_gpu_seconds = _finite_nonnegative(
        budget.get("consumed_seconds"), "gpu_budget_consumed_seconds"
    )
    summary_gpu_seconds = _finite_nonnegative(
        lsummary.get("gpu_seconds_consumed_g3_cumulative"),
        "summary_gpu_seconds_consumed_g3_cumulative",
    )
    if not math.isclose(
        total_gpu_seconds, summary_gpu_seconds, rel_tol=0, abs_tol=1e-6
    ):
        raise ValueError("g4_gpu_budget_summary_mismatch")
    budget_pass = (
        math.isfinite(total_gpu_seconds)
        and 0 <= total_gpu_seconds <= declared_gpu_cap_seconds <= 24 * 3600
        and 0 <= total_calls <= declared_call_cap <= 2276
    )
    if not budget_pass:
        raise ValueError("g3_budget_gate_failed")
    out.mkdir(parents=True, exist_ok=True)
    present = {path.name for path in out.iterdir() if path.is_file()}
    if present - EXPECTED_FILES:
        raise ValueError("g4_output_contains_unexpected_files")
    measurement_states = Counter(
        str(
            record["result"]["resource_ledger"].get(
                "visual_token_measurement", "unspecified"
            )
        )
        for record in quality + latency
        if record.get("terminal_state") == "success"
    )
    payload = {
        "schema_version": "evidencegate.g4.result.v3",
        "quality_summary_sha256": _sha(qpath),
        "latency_summary_sha256": _sha(lpath),
        "quality_snapshot_sha256": qhash,
        "latency_snapshot_sha256": lhash,
        "lineage": lineage,
        "gpu_budget_sha256": _sha(budget_path),
        "gates": {
            "quality": "PASS" if quality_pass else "FAIL",
            "latency_terminal_coverage": "PASS" if latency_pass else "FAIL",
            "budget": "PASS",
            "overall": "PASS" if quality_pass and latency_pass else "FAIL",
        },
        "terminal_counts": terminal_counts,
        "quality_terminals": len(quality),
        "latency_terminals": len(latency),
        "quality_raw_attempts": len(qrecords),
        "latency_raw_attempts": len(lrecords),
        "quality_calls_rebuilt": quality_calls,
        "latency_calls_rebuilt": latency_calls,
        "gpu_seconds_consumed_g3_cumulative": total_gpu_seconds,
        "total_calls_through_latency": total_calls,
        "latency_metrics_completeness": lsummary.get("metrics_completeness"),
        "group_metrics": groups,
        "paired_cluster_bootstrap": effects,
        "failure_cases": failures,
        "resource_measurement_states": dict(sorted(measurement_states.items())),
        "claim_boundaries": {
            "static_primary_and_moving_camera_stress_pooled": False,
            "video_sequence_equal_weighting": True,
            "image_cluster_key": "COCO image/sample_id",
            "video_cluster_key": "MOT sequence_id",
            "warmup_in_latency_metrics": False,
            "latency_scope": "staged_warm_model_path_not_e2e",
            "video_strata_share_latency_cells": True,
        },
    }
    write_immutable_manifest(out / "g4-result.json", payload)
    terminal_rows = [
        {
            "kind": kind,
            "unit_id": row.get("unit_id"),
            "attempt_index": "" if kind == "quality" else row.get("attempt_index"),
            "retry_index": row.get("retry_index"),
            "model": row.get("model"),
            "modality": row.get("modality"),
            "condition": _condition(row),
            "sample_id": row.get("sample_id"),
            "sequence_id": row.get("sequence_id"),
            "camera_motion_stratum": row.get("camera_motion_stratum"),
            "terminal_state": row.get("terminal_state"),
        }
        for kind, records in (("quality", quality), ("latency", latency))
        for row in records
    ]
    terminal_fields = (
        "kind",
        "unit_id",
        "attempt_index",
        "retry_index",
        "model",
        "modality",
        "condition",
        "sample_id",
        "sequence_id",
        "camera_motion_stratum",
        "terminal_state",
    )
    _write_immutable_bytes(
        out / "g4-terminals.csv", _csv_bytes(terminal_fields, terminal_rows)
    )
    selected_attempt_ids = {str(row["attempt_id"]) for row in quality + latency}
    failure_by_attempt = {str(row["attempt_id"]): row["category"] for row in failures}
    attempt_rows = [
        {
            "kind": kind,
            "attempt_id": row.get("attempt_id"),
            "unit_id": row.get("unit_id"),
            "attempt_index": row.get("attempt_index", ""),
            "retry_index": row.get("retry_index"),
            "retry_of": row.get("retry_of"),
            "selected_terminal": str(row.get("attempt_id")) in selected_attempt_ids,
            "model": row.get("model"),
            "modality": row.get("modality"),
            "condition": _condition(row),
            "sample_id": row.get("sample_id"),
            "sequence_id": row.get("sequence_id"),
            "camera_motion_stratum": row.get("camera_motion_stratum"),
            "terminal_state": row.get("terminal_state"),
            "failure_category": failure_by_attempt.get(str(row.get("attempt_id")), ""),
            "error": row.get("error"),
        }
        for kind, records in (("quality", qrecords), ("latency", lrecords))
        for row in records
    ]
    attempt_fields = (
        "kind",
        "attempt_id",
        "unit_id",
        "attempt_index",
        "retry_index",
        "retry_of",
        "selected_terminal",
        "model",
        "modality",
        "condition",
        "sample_id",
        "sequence_id",
        "camera_motion_stratum",
        "terminal_state",
        "failure_category",
        "error",
    )
    _write_immutable_bytes(
        out / "g4-attempts.csv", _csv_bytes(attempt_fields, attempt_rows)
    )
    group_fields = (
        "model",
        "precision",
        "modality",
        "condition",
        "camera_motion_stratum",
        "quality_terminals",
        "quality_successes",
        "quality_success_rate",
        "accuracy",
        "video_exact_count",
        "video_scored",
        "video_exact_rate_sequence_equal",
        "video_mae",
        "video_sequence_clusters",
        "latency_stratum_scope",
        "latency_steady_successes",
        "latency_p50_seconds",
        "latency_p95_seconds",
        "resource_p50",
        "stage_p50_ms",
    )
    csv_groups = [
        {
            **row,
            "resource_p50": json.dumps(
                row["resource_p50"], sort_keys=True, separators=(",", ":")
            ),
            "stage_p50_ms": json.dumps(
                row["stage_p50_ms"], sort_keys=True, separators=(",", ":")
            ),
        }
        for row in groups
    ]
    _write_immutable_bytes(
        out / "g4-group-metrics.csv", _csv_bytes(group_fields, csv_groups)
    )
    _write_immutable_bytes(
        out / "g4-quality-model-path-pareto.svg", _quality_pareto_svg(groups)
    )
    _write_immutable_bytes(out / "g4-stage-latency.svg", _stage_latency_svg(groups))
    _write_immutable_bytes(out / "g4-failure-cases.svg", _failure_cases_svg(failures))
    manifest = {
        "schema_version": "evidencegate.g4.output-manifest.v3",
        "artifacts": {
            path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
            for path in sorted(out.iterdir())
            if path.is_file() and path.name != "g4-output-manifest.json"
        },
        "source_snapshots": {"quality": qhash, "latency": lhash},
        "source_lineage": lineage,
        "gpu_budget_sha256": _sha(budget_path),
    }
    write_immutable_manifest(out / "g4-output-manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-summary", type=Path, required=True)
    parser.add_argument("--latency-summary", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--gpu-budget", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_g4(
        args.quality_summary,
        args.latency_summary,
        args.formal_manifest,
        args.scope,
        args.execution_config,
        args.gpu_budget,
        args.output_dir,
    )
