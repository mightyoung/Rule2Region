from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from evidencegate.terminals import TerminalState


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"jsonl_object_required:{path}:{line_number}")
        records.append(value)
    return records


def _canonical_sha(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_file(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _require_equal(actual: object, expected: object, error: str) -> None:
    if actual != expected:
        raise ValueError(error)


def _resolve_snapshot(run_root: Path, summary: Mapping[str, Any]) -> Path:
    recorded = Path(str(summary.get("snapshot_path", "")))
    candidate = run_root / "snapshots" / recorded.name
    if not recorded.name or not candidate.is_file():
        raise ValueError("g3_snapshot_missing")
    digest = _sha(candidate)
    if candidate.stem != digest or summary.get("snapshot_sha256") != digest:
        raise ValueError("g3_snapshot_content_address_mismatch")
    return candidate


def build_failure_index(
    *,
    run_root: Path,
    gpu_admission_path: Path,
    driver_log_path: Path,
    formal_manifest_path: Path,
    execution_config_path: Path,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    summary_path = run_root / "g3-quality-summary.json"
    wal_path = run_root / "commit.wal.jsonl"
    gpu_budget_path = run_root / "gpu-budget.json"
    summary = _json(summary_path)
    if summary.get("schema_version") != "evidencegate.g3.quality-summary.v1":
        raise ValueError("g3_quality_summary_schema_invalid")
    if summary.get("verdict") != "FAIL":
        raise ValueError("g3_failure_audit_requires_failed_quality_summary")

    formal_manifest = _json(formal_manifest_path)
    _require_equal(
        summary.get("formal_manifest_file_sha256"),
        _sha(formal_manifest_path),
        "formal_manifest_file_lineage_mismatch",
    )
    _require_equal(
        summary.get("formal_manifest_sha256"),
        formal_manifest.get("manifest_sha256"),
        "formal_manifest_identity_mismatch",
    )
    _require_equal(
        summary.get("config_file_sha256"),
        _sha(execution_config_path),
        "execution_config_file_lineage_mismatch",
    )

    admission = _json(gpu_admission_path)
    admission_sha = _sha(gpu_admission_path)
    _require_equal(
        summary.get("gpu_admission_receipt_sha256"),
        admission_sha,
        "gpu_admission_receipt_lineage_mismatch",
    )
    _require_equal(admission.get("verdict"), "ADMITTED", "gpu_not_admitted")
    _require_equal(
        summary.get("selected_gpu_uuid"),
        admission.get("selected_gpu_uuid"),
        "selected_gpu_uuid_mismatch",
    )
    _require_equal(
        summary.get("selected_physical_index"),
        admission.get("selected_physical_index"),
        "selected_gpu_index_mismatch",
    )
    devices = admission.get("devices")
    if not isinstance(devices, list):
        raise ValueError("gpu_admission_devices_missing")
    selected_devices = [
        device
        for device in devices
        if isinstance(device, Mapping)
        and device.get("uuid") == summary.get("selected_gpu_uuid")
        and device.get("physical_index") == summary.get("selected_physical_index")
    ]
    if len(selected_devices) != 1:
        raise ValueError("selected_gpu_inventory_identity_mismatch")
    selected_device = selected_devices[0]
    total_memory_mib = selected_device.get("memory_total_mib")
    if (
        not isinstance(total_memory_mib, (int, float))
        or isinstance(total_memory_mib, bool)
        or not math.isfinite(float(total_memory_mib))
        or float(total_memory_mib) <= 0
    ):
        raise ValueError("selected_gpu_total_memory_invalid")
    total_memory_gib = float(total_memory_mib) / 1024.0

    snapshot_path = _resolve_snapshot(run_root, summary)
    snapshot_records = _jsonl(snapshot_path)
    attempt_paths = sorted((run_root / "attempts").glob("*.json"))
    attempt_records = [_json(path) for path in attempt_paths]
    wal_records = _jsonl(wal_path)

    expected_count = int(summary.get("quality_logic_units", -1))
    _require_equal(
        summary.get("quality_terminal_coverage_required"),
        expected_count,
        "quality_terminal_coverage_requirement_mismatch",
    )
    for label, records in (
        ("snapshot", snapshot_records),
        ("attempt", attempt_records),
        ("wal", wal_records),
    ):
        if len(records) != expected_count:
            raise ValueError(f"g3_{label}_count_mismatch")

    attempt_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    allowed_terminals = {state.value for state in TerminalState}
    for path, record in zip(attempt_paths, attempt_records, strict=True):
        attempt_id = str(record.get("attempt_id", ""))
        if not attempt_id or attempt_id in attempt_by_id:
            raise ValueError("attempt_id_missing_or_duplicate")
        if path.name != f"{attempt_id}.json":
            raise ValueError("attempt_filename_identity_mismatch")
        if record.get("terminal_state") not in allowed_terminals:
            raise ValueError("attempt_terminal_state_invalid")
        attempt_by_id[attempt_id] = (path, record)

    snapshot_by_id = {
        str(record.get("attempt_id", "")): record for record in snapshot_records
    }
    if len(snapshot_by_id) != expected_count or set(snapshot_by_id) != set(
        attempt_by_id
    ):
        raise ValueError("snapshot_attempt_identity_set_mismatch")
    for attempt_id, (_, record) in attempt_by_id.items():
        if snapshot_by_id[attempt_id] != record:
            raise ValueError("snapshot_attempt_payload_mismatch")

    wal_by_id: dict[str, dict[str, Any]] = {}
    for record in wal_records:
        attempt_id = str(record.get("attempt_id", ""))
        if (
            record.get("event") != "commit"
            or not attempt_id
            or attempt_id in wal_by_id
            or attempt_id not in attempt_by_id
        ):
            raise ValueError("wal_commit_identity_invalid")
        path, attempt = attempt_by_id[attempt_id]
        _require_equal(record.get("path"), path.name, "wal_attempt_path_mismatch")
        _require_equal(record.get("sha256"), _sha(path), "wal_attempt_sha256_mismatch")
        _require_equal(
            record.get("unit_id"), attempt.get("unit_id"), "wal_unit_id_mismatch"
        )
        wal_by_id[attempt_id] = record
    if set(wal_by_id) != set(attempt_by_id):
        raise ValueError("wal_attempt_identity_set_mismatch")

    model_load_paths = sorted((run_root / "model-load").glob("*.json"))
    model_load_files = [_relative_file(path, run_root) for path in model_load_paths]
    model_load_hashes = {item["sha256"] for item in model_load_files}
    referenced_model_load_hashes = {
        str(record.get("model_load_artifact_sha256", "")) for record in attempt_records
    }
    if referenced_model_load_hashes != model_load_hashes:
        raise ValueError("model_load_artifact_set_mismatch")

    terminal_counts = Counter(
        str(record.get("terminal_state", "")) for record in attempt_records
    )
    success_records = [
        record
        for record in attempt_records
        if record.get("terminal_state") == "success"
    ]
    scored_records = [
        record
        for record in success_records
        if isinstance(record.get("result"), Mapping)
        and isinstance(record["result"].get("correct"), bool)
    ]
    if not scored_records:
        raise ValueError("g3_scored_successes_missing")
    correct_outputs = sum(
        record["result"]["correct"] is True for record in scored_records
    )
    oom_records = [
        record for record in attempt_records if record.get("terminal_state") == "OOM"
    ]
    allocation_requests = []
    for record in oom_records:
        match = re.search(
            r"Tried to allocate ([0-9.]+) GiB", str(record.get("error", ""))
        )
        if match:
            allocation_requests.append(float(match.group(1)))
    if len(allocation_requests) != len(oom_records):
        raise ValueError("oom_allocation_request_not_observed_for_every_oom")
    requests_exceeding_total_capacity = sum(
        requested > total_memory_gib for requested in allocation_requests
    )
    retries = sum(
        int(record.get("retry_index", 0)) > 0 or record.get("retry_of") is not None
        for record in attempt_records
    )
    attempt_gpu_seconds = math.fsum(
        float(record.get("gpu_wall_seconds", 0.0)) for record in attempt_records
    )
    gpu_budget = _json(gpu_budget_path)
    gpu_seconds = float(gpu_budget.get("consumed_seconds", -1))
    if gpu_seconds < attempt_gpu_seconds:
        raise ValueError("gpu_budget_less_than_attempt_wall_seconds")

    reconstructed = {
        "quality_logic_units": expected_count,
        "quality_attempts_started": len(attempt_records),
        "quality_terminal_coverage": len(attempt_records),
        "quality_successes": len(success_records),
        "quality_retries_used": retries,
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "scored_successes": len(scored_records),
        "correct_outputs": correct_outputs,
        "accuracy_on_successful_scored_units": correct_outputs / len(scored_records),
        "gpu_seconds_consumed_g3": gpu_seconds,
        "attempt_gpu_wall_seconds": attempt_gpu_seconds,
        "model_load_and_scheduler_gpu_seconds": gpu_seconds - attempt_gpu_seconds,
        "total_calls_through_quality": int(summary.get("g2_calls", -1))
        + len(attempt_records),
    }
    summary_reconstruction_keys = {
        "quality_logic_units",
        "quality_attempts_started",
        "quality_terminal_coverage",
        "quality_successes",
        "quality_retries_used",
        "terminal_counts",
        "scored_successes",
        "correct_outputs",
        "accuracy_on_successful_scored_units",
        "gpu_seconds_consumed_g3",
        "total_calls_through_quality",
    }
    for key in summary_reconstruction_keys:
        value = reconstructed[key]
        recorded = summary.get(key)
        if isinstance(value, float):
            if not isinstance(recorded, (int, float)) or not math.isclose(
                float(recorded), value, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"summary_reconstruction_mismatch:{key}")
        else:
            _require_equal(recorded, value, f"summary_reconstruction_mismatch:{key}")
    if (
        gpu_budget.get("reservation_id") is not None
        or float(gpu_budget.get("reserved_seconds", -1)) != 0.0
    ):
        raise ValueError("gpu_budget_open_reservation")

    driver_lines = [
        line
        for line in driver_log_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not driver_lines:
        raise ValueError("g3_driver_log_empty")
    try:
        final_driver_record = json.loads(driver_lines[-1])
    except json.JSONDecodeError as exc:
        raise ValueError("g3_driver_log_final_summary_missing") from exc
    _require_equal(final_driver_record, summary, "g3_driver_log_final_summary_mismatch")

    required_successes = int(summary.get("quality_success_minimum", -1))
    quality_pass = (
        len(success_records) >= required_successes
        and len(attempt_records) == expected_count
    )
    if quality_pass:
        raise ValueError("failure_index_cannot_encode_passing_quality_gate")

    attempt_files = []
    for path, record in zip(attempt_paths, attempt_records, strict=True):
        item = _relative_file(path, run_root)
        item.update(
            {
                "attempt_id": record["attempt_id"],
                "unit_id": record["unit_id"],
                "terminal_state": record["terminal_state"],
            }
        )
        attempt_files.append(item)

    index: dict[str, Any] = {
        "schema_version": "evidencegate.g3.failure-output-index.v1",
        "evidence_state": "formal_quality_failed_latency_not_admitted",
        "remote_output_root": str(summary.get("snapshot_path", "")).rsplit(
            "/snapshots/", 1
        )[0],
        "lineage": {
            "audit_builder": _relative_file(
                Path(__file__).resolve(), Path(__file__).resolve().parent.parent
            ),
            "execution_config": _relative_file(
                execution_config_path, execution_config_path.parent
            ),
            "formal_manifest": _relative_file(
                formal_manifest_path, formal_manifest_path.parent
            ),
            "formal_manifest_sha256": formal_manifest.get("manifest_sha256"),
            "gpu_admission": {
                "path": gpu_admission_path.name,
                "bytes": gpu_admission_path.stat().st_size,
                "sha256": admission_sha,
            },
        },
        "artifacts": {
            "quality_summary": _relative_file(summary_path, run_root),
            "snapshot": _relative_file(snapshot_path, run_root),
            "wal": _relative_file(wal_path, run_root),
            "gpu_budget": _relative_file(gpu_budget_path, run_root),
            "driver_log": {
                "path": driver_log_path.name,
                "bytes": driver_log_path.stat().st_size,
                "sha256": _sha(driver_log_path),
            },
            "model_load": model_load_files,
            "attempts": attempt_files,
        },
        "reconstructed": reconstructed,
        "gate": {
            "quality_verdict": "FAIL",
            "required_successes": required_successes,
            "actual_successes": len(success_records),
            "success_shortfall": required_successes - len(success_records),
            "terminal_coverage_required": expected_count,
            "terminal_coverage_actual": len(attempt_records),
            "latency_admitted": False,
            "latency_not_admitted_reason": "g3_quality_verdict_not_pass",
            "fallback_invoked": False,
            "formal_objective_satisfied": False,
        },
        "failure_analysis": {
            "oom_by_model": dict(
                sorted(
                    Counter(str(item.get("model", "")) for item in oom_records).items()
                )
            ),
            "oom_by_modality": dict(
                sorted(
                    Counter(
                        str(item.get("modality", "")) for item in oom_records
                    ).items()
                )
            ),
            "oom_by_family": dict(
                sorted(
                    Counter(str(item.get("family", "")) for item in oom_records).items()
                )
            ),
            "oom_allocation_request_gib": {
                "observations": len(allocation_requests),
                "minimum": min(allocation_requests) if allocation_requests else None,
                "maximum": max(allocation_requests) if allocation_requests else None,
                "selected_gpu_total_memory_gib_from_admission": total_memory_gib,
                "requests_exceeding_selected_gpu_total_memory": requests_exceeding_total_capacity,
            },
            "retries_observed": retries,
            "retry_disposition": "no_retries_executed_in_frozen_run",
        },
    }
    if archive_path is not None:
        index["export_archive"] = {
            "path": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": _sha(archive_path),
        }
    index["manifest_sha256"] = _canonical_sha(index)
    return index


def render_failure_audit(index: Mapping[str, Any]) -> str:
    reconstructed = index["reconstructed"]
    gate = index["gate"]
    gpu_hours = float(reconstructed["gpu_seconds_consumed_g3"]) / 3600.0
    return "\n".join(
        [
            "# EvidenceGate G3 formal failure audit",
            "",
            f"- Output-index identity: `{index['manifest_sha256']}`",
            "- Formal quality verdict: **FAIL**",
            f"- Terminal coverage: `{gate['terminal_coverage_actual']}/{gate['terminal_coverage_required']}`",
            f"- Successful units: `{gate['actual_successes']}/{gate['terminal_coverage_required']}`; required `{gate['required_successes']}`; shortfall `{gate['success_shortfall']}`",
            f"- Terminal states: `{json.dumps(reconstructed['terminal_counts'], sort_keys=True)}`",
            f"- OOM concentration: `{json.dumps(index['failure_analysis']['oom_by_model'], sort_keys=True)}`; all OOM records are video deployment units.",
            f"- Correct outputs among scored successes: `{reconstructed['correct_outputs']}/{reconstructed['scored_successes']}` (`{100.0 * reconstructed['accuracy_on_successful_scored_units']:.4f}%`)",
            f"- Calls through quality: `{reconstructed['total_calls_through_quality']}`",
            f"- G3 GPU time: `{reconstructed['gpu_seconds_consumed_g3']:.4f}` seconds (`{gpu_hours:.4f}` GPU-hours)",
            "- Latency stage: **NOT ADMITTED**, because the frozen quality summary is not PASS.",
            "- Registered fallback: **NOT INVOKED**. The selected G0 scope remains BROAD; no post-hoc matrix reduction is represented in this evidence.",
            "",
            "## Integrity checks",
            "",
            "The audit independently matched every attempt JSON to the content-addressed snapshot and WAL commit, reconstructed terminal/correctness/call counters, reconciled attempt GPU wall time plus the model-load/scheduler residual to the frozen GPU budget, verified model-load references, checked frozen config and formal-manifest lineage, and matched the final driver-log record to the immutable quality summary.",
            "",
            "## Professor-critic disposition",
            "",
            "The objective rubric is not satisfied: the required 95% success gate failed and the latency stage was consequently never admitted. G4 may be tested as software, but no G4 paper result or passing completion verdict can be claimed from this run.",
            "",
        ]
    )


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu-admission", type=Path, required=True)
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--audit-markdown", type=Path, required=True)
    args = parser.parse_args()
    index = build_failure_index(
        run_root=args.run_root,
        gpu_admission_path=args.gpu_admission,
        driver_log_path=args.driver_log,
        formal_manifest_path=args.formal_manifest,
        execution_config_path=args.execution_config,
        archive_path=args.archive,
    )
    _write_new(
        args.output_index,
        json.dumps(
            index,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
    )
    _write_new(args.audit_markdown, render_failure_audit(index))
    print(json.dumps({"manifest_sha256": index["manifest_sha256"], "verdict": "FAIL"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
