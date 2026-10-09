from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research_tools.g3_failure_audit import build_failure_index


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(root: Path) -> dict[str, Path]:
    run = root / "run"
    attempts = run / "attempts"
    snapshots = run / "snapshots"
    model_load = run / "model-load"
    for path in (attempts, snapshots, model_load):
        path.mkdir(parents=True)

    config = root / "config.json"
    formal = root / "formal.json"
    admission = root / "admission.json"
    driver = root / "driver.log"
    _write_json(config, {"schema_version": "config.fixture"})
    _write_json(formal, {"manifest_sha256": "formal-identity"})
    _write_json(
        admission,
        {
            "verdict": "ADMITTED",
            "selected_gpu_uuid": "GPU-fixture",
            "selected_physical_index": 1,
            "devices": [
                {
                    "uuid": "GPU-fixture",
                    "physical_index": 1,
                    "memory_total_mib": 1024,
                }
            ],
        },
    )

    load_receipt = model_load / "load.json"
    _write_json(load_receipt, {"model": "fixture"})
    load_sha = _sha(load_receipt)
    records = [
        {
            "attempt_id": "g3q-quality-a-r0",
            "attempt_index": 0,
            "unit_id": "quality:a",
            "retry_index": 0,
            "retry_of": None,
            "terminal_state": "success",
            "gpu_wall_seconds": 1.25,
            "model_load_artifact_sha256": load_sha,
            "result": {"correct": True},
        },
        {
            "attempt_id": "g3q-quality-b-r0",
            "attempt_index": 0,
            "unit_id": "quality:b",
            "retry_index": 0,
            "retry_of": None,
            "terminal_state": "OOM",
            "gpu_wall_seconds": 2.0,
            "model_load_artifact_sha256": load_sha,
            "result": None,
            "error": "OutOfMemoryError:CUDA out of memory. Tried to allocate 2.00 GiB.",
        },
    ]
    wal = []
    for record in records:
        path = attempts / f"{record['attempt_id']}.json"
        _write_json(path, record)
        wal.append(
            {
                "event": "commit",
                "attempt_id": record["attempt_id"],
                "unit_id": record["unit_id"],
                "path": path.name,
                "sha256": _sha(path),
            }
        )
    (run / "commit.wal.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in wal),
        encoding="utf-8",
    )
    snapshot_content = "".join(
        json.dumps(item, sort_keys=True) + "\n" for item in records
    )
    snapshot_temporary = snapshots / "snapshot.partial"
    snapshot_temporary.write_bytes(snapshot_content.encode("utf-8"))
    snapshot_sha = _sha(snapshot_temporary)
    snapshot = snapshots / f"{snapshot_sha}.jsonl"
    snapshot_temporary.replace(snapshot)
    _write_json(
        run / "gpu-budget.json",
        {
            "consumed_seconds": 3.75,
            "reservation_id": None,
            "reserved_seconds": 0.0,
        },
    )
    summary = {
        "schema_version": "evidencegate.g3.quality-summary.v1",
        "verdict": "FAIL",
        "config_file_sha256": _sha(config),
        "formal_manifest_file_sha256": _sha(formal),
        "formal_manifest_sha256": "formal-identity",
        "gpu_admission_receipt_sha256": _sha(admission),
        "selected_gpu_uuid": "GPU-fixture",
        "selected_physical_index": 1,
        "snapshot_path": f"/remote/snapshots/{snapshot.name}",
        "snapshot_sha256": snapshot_sha,
        "quality_logic_units": 2,
        "quality_terminal_coverage_required": 2,
        "quality_attempts_started": 2,
        "quality_terminal_coverage": 2,
        "quality_successes": 1,
        "quality_retries_used": 0,
        "terminal_counts": {"OOM": 1, "success": 1},
        "scored_successes": 1,
        "correct_outputs": 1,
        "accuracy_on_successful_scored_units": 1.0,
        "gpu_seconds_consumed_g3": 3.75,
        "g2_calls": 36,
        "total_calls_through_quality": 38,
        "quality_success_minimum": 2,
    }
    _write_json(run / "g3-quality-summary.json", summary)
    driver.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "run": run,
        "config": config,
        "formal": formal,
        "admission": admission,
        "driver": driver,
    }


def _build(paths: dict[str, Path]) -> dict:
    return build_failure_index(
        run_root=paths["run"],
        gpu_admission_path=paths["admission"],
        driver_log_path=paths["driver"],
        formal_manifest_path=paths["formal"],
        execution_config_path=paths["config"],
    )


def test_build_failure_index_reconstructs_sealed_failure(tmp_path: Path) -> None:
    index = _build(_fixture(tmp_path))

    assert index["gate"]["quality_verdict"] == "FAIL"
    assert index["gate"]["success_shortfall"] == 1
    assert index["gate"]["latency_admitted"] is False
    assert index["reconstructed"]["attempt_gpu_wall_seconds"] == 3.25
    assert index["reconstructed"]["model_load_and_scheduler_gpu_seconds"] == 0.5
    assert index["failure_analysis"]["oom_by_model"] == {"": 1}
    assert (
        index["failure_analysis"]["oom_allocation_request_gib"][
            "requests_exceeding_selected_gpu_total_memory"
        ]
        == 1
    )
    assert len(index["lineage"]["audit_builder"]["sha256"]) == 64
    assert len(index["artifacts"]["attempts"]) == 2


def test_build_failure_index_rejects_wal_tampering(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    wal = paths["run"] / "commit.wal.jsonl"
    records = [
        json.loads(line) for line in wal.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["sha256"] = "0" * 64
    wal.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="wal_attempt_sha256_mismatch"):
        _build(paths)


def test_build_failure_index_rejects_invalid_terminal_enum(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    attempt = next((paths["run"] / "attempts").glob("*.json"))
    payload = json.loads(attempt.read_text(encoding="utf-8"))
    payload["terminal_state"] = "failed"
    _write_json(attempt, payload)

    with pytest.raises(ValueError, match="attempt_terminal_state_invalid"):
        _build(paths)


def test_build_failure_index_requires_frozen_coverage_requirement(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    summary = paths["run"] / "g3-quality-summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["quality_terminal_coverage_required"] = 3
    _write_json(summary, payload)

    with pytest.raises(
        ValueError, match="quality_terminal_coverage_requirement_mismatch"
    ):
        _build(paths)
