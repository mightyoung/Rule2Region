import hashlib
import json
from pathlib import Path

import pytest

from evidencegate.manifest import canonical_json
from research_tools.g4_build import (
    _cluster_bootstrap,
    _paired_effects,
    build_g4,
    validate_g3_summary,
)


def _snapshot(path: Path, records: list[dict], subdir: str) -> tuple[Path, str]:
    data = b"".join(canonical_json(item) for item in records)
    digest = hashlib.sha256(data).hexdigest()
    root = path / subdir
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{digest}.jsonl"
    target.write_bytes(data)
    return target, digest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resource(value: float = 1.0) -> dict:
    return {
        "schema_version": "evidencegate.g3.resource-ledger.v1",
        "visual_token_measurement": "unavailable_from_frozen_processor_api",
        "records": [
            {
                "stage": "vision_projector_prefill_generation_combined",
                "combined": True,
                "encoded_bytes": value,
                "decoded_pixels": value,
                "model_pixels": value,
                "prefill_tokens": value,
                "visual_tokens": 0,
                "vram_bytes": value,
                "wall_clock_ms": value,
            }
        ],
    }


def _quality_records(failures: int = 0) -> list[dict]:
    rows = []
    for index in range(986):
        modality = "image" if index < 240 else "video"
        condition = "native" if modality == "image" else "dense"
        state = "timeout" if index < failures else "success"
        expected = True if modality == "image" else 4
        parsed = True if modality == "image" else (4 if index % 2 else 3)
        result = None
        if state == "success":
            result = {
                "correct": parsed == expected,
                "expected": expected,
                "parsed_output": parsed,
                "resource_ledger": _resource(),
            }
        rows.append(
            {
                "attempt_id": f"q-{index}-0",
                "unit_id": f"quality:{index}",
                "retry_index": 0,
                "retry_of": None,
                "terminal_state": state,
                "model": "0.5b-bf16",
                "model_layer_identity": "layer-05b-bf16",
                "modality": modality,
                "condition": condition,
                "resolved_condition": condition,
                "sample_id": f"COCO_{index}.jpg"
                if modality == "image"
                else f"MOT17-02:4s:{index}",
                "sequence_id": None if modality == "image" else "MOT17-02",
                "camera_motion_stratum": None
                if modality == "image"
                else "STATIC_PRIMARY",
                "deployment_member": True,
                "result": result,
            }
        )
    return rows


def _latency_records() -> list[dict]:
    rows = []
    for cell in range(64):
        for attempt in range(14):
            modality = "image" if cell < 32 else "video"
            condition = "native" if modality == "image" else "dense"
            rows.append(
                {
                    "attempt_id": f"l-{cell}-{attempt}-0",
                    "unit_id": f"latency:cell-{cell}",
                    "cell_id": f"cell-{cell}",
                    "attempt_index": attempt,
                    "retry_index": 0,
                    "retry_of": None,
                    "latency_kind": "warmup" if attempt < 4 else "steady",
                    "terminal_state": "success",
                    "model": "0.5b-bf16",
                    "model_layer_identity": "layer-05b-bf16",
                    "modality": modality,
                    "resolved_condition": condition,
                    "sample_id": f"sample-{cell}",
                    "repeat_index": 0,
                    "io_policy": "staged_warm",
                    "include_in_formal_latency": True,
                    "result": {
                        "model_path_seconds": float(attempt),
                        "resource_ledger": _resource(float(attempt + 1)),
                    },
                }
            )
    return rows


def _bundle(
    path: Path, quality: list[dict], latency: list[dict]
) -> tuple[Path, Path, Path, Path, Path, Path]:
    path.mkdir(parents=True, exist_ok=True)
    frozen_attempts = []
    latency_cells = []
    for cell in range(64):
        modality = "image" if cell < 32 else "video"
        condition = "native" if modality == "image" else "dense"
        frozen_attempts.extend(
            {"unit_id": f"frozen-{cell}", "attempt_index": attempt}
            for attempt in range(14)
        )
        latency_cells.append(
            {
                "cell_id": f"cell-{cell}",
                "model_key": "0.5b-bf16",
                "model_layer_identity": "layer-05b-bf16",
                "modality": modality,
                "sample_id": f"sample-{cell}",
                "evidence_state": condition,
                "repeat_index": 0,
                "io_policy": "staged_warm",
                "include_in_formal_latency": True,
                "warmup_attempts": 4,
                "steady_attempts": 10,
            }
        )
    scope_path = path / "scope.json"
    scope_path.write_bytes(
        canonical_json({"matrix": {"latency_attempts": frozen_attempts}})
    )
    config = {
        "source_scope_sha256": _sha(scope_path),
        "latency_cells": latency_cells,
        "limits": {"total_call_hard_cap": 2107, "gpu_hour_hard_cap": 24},
    }
    config_path = path / "config.json"
    config_path.write_bytes(canonical_json(config))
    formal_units = [
        {
            field: row.get(field)
            for field in (
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
        }
        for row in quality
    ]
    formal = {
        "execution_config_file_sha256": _sha(config_path),
        "source_scope_file_sha256": _sha(scope_path),
        "manifest_sha256": "f" * 64,
        "units": formal_units,
    }
    formal_path = path / "formal.json"
    formal_path.write_bytes(canonical_json(formal))
    lineage = {
        "config_file_sha256": _sha(config_path),
        "formal_manifest_file_sha256": _sha(formal_path),
        "formal_manifest_sha256": formal["manifest_sha256"],
        "prompt_contract_sha256": "p" * 64,
    }
    for row in quality + latency:
        row.update(lineage)
    qsnapshot, qdigest = _snapshot(path, quality, "quality")
    lsnapshot, ldigest = _snapshot(path, latency, "latency")
    qterminal = {}
    for row in sorted(quality, key=lambda item: int(item["retry_index"])):
        qterminal[row["unit_id"]] = row
    qcounts: dict[str, int] = {}
    for row in qterminal.values():
        qcounts[row["terminal_state"]] = qcounts.get(row["terminal_state"], 0) + 1
    lcounts: dict[str, int] = {}
    for row in latency:
        lcounts[row["terminal_state"]] = lcounts.get(row["terminal_state"], 0) + 1
    qp, lp = path / "quality-summary.json", path / "latency-summary.json"
    qp.write_bytes(
        canonical_json(
            {
                "schema_version": "evidencegate.g3.quality-summary.v1",
                "snapshot_path": str(qsnapshot),
                "snapshot_sha256": qdigest,
                "quality_logic_units": 986,
                "quality_success_minimum": 937,
                "quality_terminal_coverage": 986,
                "quality_successes": qcounts.get("success", 0),
                "quality_attempts_started": len(quality),
                "terminal_counts": dict(sorted(qcounts.items())),
                "verdict": "PASS" if qcounts.get("success", 0) >= 937 else "FAIL",
                "selected_gpu_uuid": "GPU-test",
                **{
                    key: lineage[key]
                    for key in (
                        "config_file_sha256",
                        "formal_manifest_file_sha256",
                        "formal_manifest_sha256",
                    )
                },
            }
        )
    )
    lp.write_bytes(
        canonical_json(
            {
                "schema_version": "evidencegate.g3.latency-summary.v1",
                "snapshot_path": str(lsnapshot),
                "snapshot_sha256": ldigest,
                "latency_attempts_expected": 896,
                "latency_terminal_coverage": 896,
                "latency_terminal_counts": dict(sorted(lcounts.items())),
                "latency_cells_expected": 64,
                "latency_cells_with_all_steady_successes": 64,
                "cell_metrics": [
                    {
                        "cell_id": cell_id,
                        "steady_successes": 10,
                        "steady_attempts_expected": 10,
                        "median_seconds": 8.5,
                        "p95_seconds": 12.549999999999999,
                    }
                    for cell_id in sorted(f"cell-{cell}" for cell in range(64))
                ],
                "terminal_verdict": "PASS",
                "metrics_completeness": "COMPLETE",
                "gpu_seconds_consumed_g3_cumulative": 1000.0,
                "g2_calls": 36,
                "quality_attempts_started": len(quality),
                "latency_attempts_started": len(latency),
                "total_calls_through_latency": 36 + len(quality) + len(latency),
                "total_call_hard_cap": 2107,
                "quality_summary_file_sha256": _sha(qp),
                "selected_gpu_uuid": "GPU-test",
                **{
                    key: lineage[key]
                    for key in (
                        "config_file_sha256",
                        "formal_manifest_file_sha256",
                        "formal_manifest_sha256",
                    )
                },
            }
        )
    )
    budget_path = path / "gpu-budget.json"
    budget_path.write_bytes(canonical_json({"consumed_seconds": 1000.0}))
    return qp, lp, formal_path, scope_path, config_path, budget_path


def _build(paths: tuple[Path, Path, Path, Path, Path, Path], output: Path):
    return build_g4(*paths, output)


def test_g4_builds_lineage_bound_tables_figures_and_independent_rebuild(
    tmp_path: Path,
) -> None:
    quality, latency = _quality_records(failures=49), _latency_records()
    paths = _bundle(tmp_path / "source", quality, latency)
    manifest = _build(paths, tmp_path / "out-a")
    result = json.loads((tmp_path / "out-a/g4-result.json").read_text())
    assert result["gates"] == {
        "quality": "PASS",
        "latency_terminal_coverage": "PASS",
        "budget": "PASS",
        "overall": "PASS",
    }
    assert result["terminal_counts"]["quality_successes"] == 937
    assert (
        result["claim_boundaries"]["latency_scope"] == "staged_warm_model_path_not_e2e"
    )
    assert manifest["artifacts"]["g4-attempts.csv"]["sha256"]
    for name in (
        "g4-quality-model-path-pareto.svg",
        "g4-stage-latency.svg",
        "g4-failure-cases.svg",
    ):
        assert "<svg" in (tmp_path / "out-a" / name).read_text()
    _build(paths, tmp_path / "out-b")
    assert {
        path.name: path.read_bytes() for path in (tmp_path / "out-a").iterdir()
    } == {path.name: path.read_bytes() for path in (tmp_path / "out-b").iterdir()}


def test_video_expected_is_recomputed_and_sequence_is_clustered() -> None:
    records = []
    for sequence, dense, candidate in (("MOT17-02", 4, 5), ("MOT17-04", 4, 3)):
        for window in range(2):
            for condition, parsed in (("dense", dense), ("uniform50", candidate)):
                records.append(
                    {
                        "unit_id": f"{sequence}-{window}-{condition}",
                        "terminal_state": "success",
                        "model": "0.5b-bf16",
                        "modality": "video",
                        "condition": condition,
                        "sample_id": f"{sequence}:4s:{window}",
                        "sequence_id": sequence,
                        "camera_motion_stratum": "STATIC_PRIMARY",
                        "result": {
                            "correct": parsed == 4,
                            "expected": 4,
                            "parsed_output": parsed,
                        },
                    }
                )
    effects = _paired_effects(records)
    assert len(effects) == 1
    assert effects[0]["clusters"] == 2
    assert effects[0]["paired_units"] == 4
    assert effects[0]["estimate"] == 1.0


def test_cluster_bootstrap_aggregates_repeated_observations() -> None:
    result = _cluster_bootstrap(
        {"a": [0.0, 2.0], "b": [4.0, 6.0]}, seed_text="x", draws=100
    )
    assert result["clusters"] == 2
    assert result["estimate"] == 3.0


def test_g4_rejects_snapshot_hash_mismatch(tmp_path: Path) -> None:
    records = _quality_records()
    snapshot, digest = _snapshot(tmp_path, records, "quality")
    summary = tmp_path / "quality-summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": "evidencegate.g3.quality-summary.v1",
                "snapshot_path": str(snapshot),
                "snapshot_sha256": "0" * 64,
            }
        )
    )
    assert digest != "0" * 64
    with pytest.raises(ValueError, match="content_address"):
        validate_g3_summary(
            summary,
            expected_schema="evidencegate.g3.quality-summary.v1",
            expected_snapshot_count=986,
        )


def test_g4_rejects_lineage_correctness_resource_and_unknown_terminal(
    tmp_path: Path,
) -> None:
    quality, latency = _quality_records(), _latency_records()
    paths = _bundle(tmp_path / "lineage", quality, latency)
    bad_formal = json.loads(paths[2].read_text())
    bad_formal["source_scope_file_sha256"] = "0" * 64
    paths[2].write_bytes(canonical_json(bad_formal))
    with pytest.raises(ValueError, match="lineage"):
        _build(paths, tmp_path / "out-lineage")

    quality, latency = _quality_records(), _latency_records()
    quality[100]["result"]["correct"] = False
    paths = _bundle(tmp_path / "correctness", quality, latency)
    with pytest.raises(ValueError, match="correctness_mismatch"):
        _build(paths, tmp_path / "out-correctness")

    quality, latency = _quality_records(), _latency_records()
    quality[100]["result"]["resource_ledger"]["records"] = []
    paths = _bundle(tmp_path / "ledger", quality, latency)
    with pytest.raises(ValueError, match="resource_ledger_records_invalid"):
        _build(paths, tmp_path / "out-ledger")

    quality, latency = _quality_records(), _latency_records()
    quality[0]["terminal_state"] = "made_up"
    quality[0]["result"] = None
    paths = _bundle(tmp_path / "terminal", quality, latency)
    with pytest.raises(ValueError, match="terminal_state_invalid"):
        _build(paths, tmp_path / "out-terminal")


def test_g4_rejects_retry_identity_change_and_preserves_failed_retry(
    tmp_path: Path,
) -> None:
    quality, latency = _quality_records(), _latency_records()
    original = quality[0]
    retry = {
        **original,
        "attempt_id": "q-0-1",
        "retry_index": 1,
        "retry_of": original["attempt_id"],
        "terminal_state": "success",
    }
    retry["result"] = {
        "correct": True,
        "expected": True,
        "parsed_output": True,
        "resource_ledger": _resource(),
    }
    original["terminal_state"] = "timeout"
    original["result"] = None
    quality.append(retry)
    paths = _bundle(tmp_path / "retry", quality, latency)
    _build(paths, tmp_path / "out")
    attempts = (tmp_path / "out/g4-attempts.csv").read_text()
    assert "q-0-0" in attempts and "timeout" in attempts
    quality[-1]["sample_id"] = "changed.jpg"
    paths = _bundle(tmp_path / "retry-bad", quality, latency)
    with pytest.raises(ValueError, match="retry_identity_changed"):
        _build(paths, tmp_path / "out-bad")


def test_g4_rejects_raw_budget_and_unexpected_output(tmp_path: Path) -> None:
    quality, latency = _quality_records(), _latency_records()
    paths = _bundle(tmp_path / "source", quality, latency)
    payload = json.loads(paths[1].read_text())
    payload["total_calls_through_latency"] -= 1
    paths[1].write_bytes(canonical_json(payload))
    with pytest.raises(ValueError, match="raw_call_count"):
        _build(paths, tmp_path / "out")
    paths = _bundle(tmp_path / "second", _quality_records(), _latency_records())
    output = tmp_path / "out2"
    output.mkdir()
    (output / "foreign.txt").write_text("do not overwrite")
    with pytest.raises(ValueError, match="unexpected_files"):
        _build(paths, output)


def test_g4_rejects_missing_steady_latency_timing(tmp_path: Path) -> None:
    quality = _quality_records()
    latency = _latency_records()
    latency[4]["result"].pop("model_path_seconds")
    paths = _bundle(tmp_path / "missing-steady", quality, latency)

    with pytest.raises(ValueError, match="latency_steady_timing_missing"):
        _build(paths, tmp_path / "out-missing-steady")


def test_g4_rejects_incomplete_latency_cell_summary(tmp_path: Path) -> None:
    paths = _bundle(
        tmp_path / "incomplete-cells", _quality_records(), _latency_records()
    )
    latency_summary = paths[1]
    payload = json.loads(latency_summary.read_text(encoding="utf-8"))
    payload["latency_cells_with_all_steady_successes"] = 63
    latency_summary.write_bytes(canonical_json(payload))

    with pytest.raises(ValueError, match="complete_cell_count_mismatch"):
        _build(paths, tmp_path / "out-incomplete-cells")


def test_g4_fails_closed_when_latency_summary_is_absent(tmp_path: Path) -> None:
    paths = _bundle(
        tmp_path / "missing-latency", _quality_records(), _latency_records()
    )
    missing = tmp_path / "missing-latency" / "not-produced-latency-summary.json"

    with pytest.raises(FileNotFoundError):
        build_g4(paths[0], missing, *paths[2:], tmp_path / "out-missing-latency")
