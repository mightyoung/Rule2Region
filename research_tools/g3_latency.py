from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from evidencegate.manifest import manifest_sha256, write_immutable_manifest
from evidencegate.runner import (
    AttemptArtifactStore,
    SingleScheduler,
    write_content_addressed_jsonl,
)
from evidencegate.terminals import TerminalState
from research_tools.g2_smoke import _generate, _load_layer
from research_tools.g3_formal import (
    RETRYABLE_STATES,
    FormalAssetLoader,
    _json,
    _prepare_unit,
    _record_state,
    _resource_ledger,
    _sha,
    _validate_gpu_admission,
    validate_quality_contract,
)


def expand_latency_attempts(
    scope: Mapping[str, Any], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    frozen_attempts = scope["matrix"]["latency_attempts"]
    cells = config["latency_cells"]
    if len(cells) != 64 or len(frozen_attempts) != 896:
        raise ValueError("latency_frozen_cardinality_invalid")
    expanded: list[dict[str, Any]] = []
    offset = 0
    for cell in cells:
        warmups = int(cell["warmup_attempts"])
        steady = int(cell["steady_attempts"])
        if warmups != 4 or steady != 10 or cell.get("io_policy") != "staged_warm":
            raise ValueError("latency_cell_attempt_or_io_policy_invalid")
        block = frozen_attempts[offset : offset + warmups + steady]
        offset += len(block)
        if (
            len(block) != 14
            or len({item["unit_id"] for item in block}) != 1
            or [int(item["attempt_index"]) for item in block] != list(range(14))
        ):
            raise ValueError(
                f"latency_cell_frozen_attempt_mapping_invalid:{cell['cell_id']}"
            )
        for item in block:
            attempt_index = int(item["attempt_index"])
            expanded.append(
                {
                    **item,
                    "frozen_plan_unit_id": item["unit_id"],
                    "unit_id": f"latency:{cell['cell_id']}",
                    "cell_id": cell["cell_id"],
                    "model": cell["model_key"],
                    "model_layer_identity": cell["model_layer_identity"],
                    "modality": cell["modality"],
                    "sample_id": cell["sample_id"],
                    "resolved_condition": cell["evidence_state"],
                    "repeat_index": int(cell["repeat_index"]),
                    "io_policy": cell["io_policy"],
                    "include_in_formal_latency": bool(
                        cell["include_in_formal_latency"]
                    ),
                    "latency_kind": "warmup" if attempt_index < warmups else "steady",
                }
            )
    if offset != 896:
        raise ValueError("latency_frozen_attempt_mapping_incomplete")
    return expanded


def validate_latency_contract(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    attempts = expand_latency_attempts(contract["scope"], contract["config"])
    for attempt in attempts:
        layer = contract["layers"].get(attempt["model"])
        if (
            layer is None
            or layer["model_layer_identity"] != attempt["model_layer_identity"]
        ):
            raise ValueError("latency_model_layer_identity_mismatch")
        modality = attempt["modality"]
        if modality == "image":
            if attempt["sample_id"] not in contract["image_samples"]:
                raise ValueError("latency_image_sample_not_frozen")
        elif modality == "video":
            if attempt["sample_id"] not in contract["video_samples"]:
                raise ValueError("latency_video_sample_not_frozen")
        else:
            raise ValueError("latency_modality_invalid")
    return attempts


def _latency_attempt_id(cell_id: str, attempt_index: int, retry_index: int) -> str:
    return f"g3l-{cell_id}-a{attempt_index:02d}-r{retry_index}"


def load_latency_attempts(
    store: AttemptArtifactStore,
    attempts: Sequence[Mapping[str, Any]],
    config_hash: str,
    formal_hash: str,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    allowed = {(str(item["unit_id"]), int(item["attempt_index"])) for item in attempts}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for path in sorted(store.attempts.glob("g3l-*.json")):
        record = _json(path)
        key = (str(record.get("unit_id", "")), int(record.get("attempt_index", -1)))
        if key not in allowed:
            raise ValueError("latency_attempt_unknown_frozen_identity")
        if (
            record.get("config_file_sha256") != config_hash
            or record.get("formal_manifest_file_sha256") != formal_hash
        ):
            raise ValueError("latency_attempt_frozen_identity_mismatch")
        _record_state(record)
        grouped.setdefault(key, []).append(record)
    for key, records in grouped.items():
        records.sort(key=lambda item: int(item.get("retry_index", -1)))
        indexes = [int(item.get("retry_index", -1)) for item in records]
        if indexes != list(range(len(indexes))):
            raise ValueError(f"latency_retry_sequence_invalid:{key[0]}:{key[1]}")
    return grouped


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
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_summary(
    attempts: Sequence[Mapping[str, Any]],
    grouped: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    terminal_counts: Counter[str] = Counter()
    cell_values: dict[str, list[float]] = {}
    cell_expected_steady: Counter[str] = Counter()
    for attempt in attempts:
        key = (str(attempt["unit_id"]), int(attempt["attempt_index"]))
        if attempt["latency_kind"] == "steady":
            cell_expected_steady[attempt["cell_id"]] += 1
        records = grouped.get(key, ())
        if not records:
            continue
        last = records[-1]
        state = _record_state(last)
        terminal_counts[state.value] += 1
        if state is TerminalState.SUCCESS and attempt["latency_kind"] == "steady":
            result = last.get("result")
            if isinstance(result, Mapping) and isinstance(
                result.get("model_path_seconds"), (int, float)
            ):
                cell_values.setdefault(attempt["cell_id"], []).append(
                    float(result["model_path_seconds"])
                )
    cells = []
    for cell_id in sorted(cell_expected_steady):
        values = cell_values.get(cell_id, [])
        cells.append(
            {
                "cell_id": cell_id,
                "steady_successes": len(values),
                "steady_attempts_expected": cell_expected_steady[cell_id],
                "median_seconds": statistics.median(values) if values else None,
                "p95_seconds": _quantile(values, 0.95),
            }
        )
    coverage = sum(terminal_counts.values())
    complete_cells = sum(
        item["steady_successes"] == item["steady_attempts_expected"] for item in cells
    )
    return {
        "latency_attempts_expected": len(attempts),
        "latency_terminal_coverage": coverage,
        "latency_terminal_counts": dict(sorted(terminal_counts.items())),
        "latency_cells_expected": len(cells),
        "latency_cells_with_all_steady_successes": complete_cells,
        "cell_metrics": cells,
        "terminal_verdict": "PASS" if coverage == len(attempts) else "FAIL",
        "metrics_completeness": "COMPLETE"
        if complete_cells == len(cells)
        else "COMPLETE_WITH_FAILED_STEADY_ATTEMPTS",
    }


def _quality_state(
    output_root: Path, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], int, int]:
    path = output_root / "g3-quality-summary.json"
    summary = _json(path)
    if (
        summary.get("schema_version") != "evidencegate.g3.quality-summary.v1"
        or summary.get("verdict") != "PASS"
        or summary.get("config_file_sha256") != contract["config_file_sha256"]
        or summary.get("formal_manifest_file_sha256") != contract["formal_file_sha256"]
    ):
        raise ValueError("g3_quality_pass_required_before_latency")
    return (
        summary,
        int(summary["quality_attempts_started"]),
        int(summary["quality_retries_used"]),
    )


def run_latency(
    contract: dict[str, Any],
    attempts: list[dict[str, Any]],
    gpu_receipt: dict[str, Any],
    gpu_hash: str,
    output_root: Path,
) -> dict[str, Any]:
    config = contract["config"]
    quality, quality_attempts, quality_retries = _quality_state(output_root, contract)
    store = AttemptArtifactStore(output_root)
    recovery = store.recover()
    if recovery["fail_closed"]:
        raise RuntimeError("g3_attempt_store_recovery_failed")
    grouped = load_latency_attempts(
        store, attempts, contract["config_file_sha256"], contract["formal_file_sha256"]
    )
    latency_retries = sum(max(0, len(records) - 1) for records in grouped.values())
    retry_cap = int(config["limits"]["retry_attempt_budget"])
    if quality_retries + latency_retries > retry_cap:
        raise ValueError("shared_retry_budget_exceeded")
    total_latency_calls = sum(len(records) for records in grouped.values())
    total_call_cap = int(config["limits"]["total_call_hard_cap"])
    g2_calls = 36
    cap_seconds = float(config["limits"]["gpu_hour_hard_cap"]) * 3600.0
    inference_timeout = float(config["runtime"]["latency_inference_timeout_seconds"])
    load_timeout = float(config["runtime"]["model_load_timeout_seconds"])
    loader = FormalAssetLoader(
        contract["video_samples"],
        contract["sequence_receipts"],
        contract["codec_entries"],
        sources_already_validated=True,
    )
    scheduler = SingleScheduler(output_root)
    scheduler.acquire("evidencegate:g3-formal-latency")
    base_ids = {
        "config_file_sha256": contract["config_file_sha256"],
        "formal_manifest_file_sha256": contract["formal_file_sha256"],
        "formal_manifest_sha256": contract["formal"]["manifest_sha256"],
        "gpu_admission_receipt_sha256": gpu_hash,
        "quality_summary_file_sha256": _sha(output_root / "g3-quality-summary.json"),
    }
    hard_stop = False
    try:
        for model_key in sorted(contract["layers"]):
            model_attempts = [item for item in attempts if item["model"] == model_key]
            pending = []
            for item in model_attempts:
                existing = grouped.get(
                    (item["unit_id"], int(item["attempt_index"])), []
                )
                if not existing:
                    pending.append(item)
                elif (
                    _record_state(existing[-1]) in RETRYABLE_STATES
                    and quality_retries + latency_retries < retry_cap
                ):
                    pending.append(item)
            if not pending:
                continue
            if g2_calls + quality_attempts + total_latency_calls >= total_call_cap:
                hard_stop = True
                break
            load_started = time.perf_counter()
            try:
                scheduler.reserve_gpu(
                    load_timeout, cap_seconds, f"g3-latency-model-load-{model_key}"
                )
            except RuntimeError as exc:
                if str(exc) != "gpu_hour_cap_would_be_exceeded":
                    raise
                scheduler.hard_stop()
                hard_stop = True
                break
            try:
                model, processor, load_receipt = _load_layer(
                    contract["layers"][model_key],
                    contract["model_index"],
                    output_root,
                    phase="g3-latency",
                    snapshot_already_validated=True,
                )
                scheduler.settle_gpu(time.perf_counter() - load_started)
            except Exception:
                scheduler.settle_gpu(time.perf_counter() - load_started)
                raise
            try:
                by_cell: dict[str, list[dict[str, Any]]] = {}
                for item in pending:
                    by_cell.setdefault(item["cell_id"], []).append(item)
                ordered_cells = sorted(
                    by_cell.values(),
                    key=lambda block: (
                        block[0]["modality"],
                        block[0]["sample_id"],
                        block[0]["resolved_condition"],
                        block[0]["repeat_index"],
                    ),
                )
                for block in ordered_cells:
                    template = block[0]
                    plugin, question, expected, io_receipt = _prepare_unit(
                        template, contract, loader
                    )
                    for item in sorted(
                        block, key=lambda value: int(value["attempt_index"])
                    ):
                        if (
                            g2_calls + quality_attempts + total_latency_calls
                            >= total_call_cap
                        ):
                            hard_stop = True
                            break
                        attempt_index = int(item["attempt_index"])
                        key = (item["unit_id"], attempt_index)
                        existing = grouped.get(key, [])
                        retry_index = len(existing)
                        if (
                            retry_index > 0
                            and quality_retries + latency_retries >= retry_cap
                        ):
                            continue
                        attempt_id = _latency_attempt_id(
                            item["cell_id"], attempt_index, retry_index
                        )
                        try:
                            scheduler.reserve_gpu(
                                inference_timeout, cap_seconds, attempt_id
                            )
                        except RuntimeError as exc:
                            if str(exc) != "gpu_hour_cap_would_be_exceeded":
                                raise
                            scheduler.hard_stop()
                            hard_stop = True
                            break
                        started = time.time()
                        monotonic = time.perf_counter()
                        result: dict[str, Any] | None = None
                        try:
                            result = _generate(
                                model,
                                processor,
                                plugin,
                                config["prompts"],
                                item["modality"],
                                question,
                                inference_timeout,
                            )
                            result.update(
                                {
                                    "plugin": plugin.to_dict(),
                                    "io_receipt": {**io_receipt, "cache_state": "warm"},
                                    "expected": expected,
                                    "correct": result["parsed_output"] == expected,
                                    "model_path_seconds": float(
                                        result["preprocess_seconds"]
                                    )
                                    + float(result["transfer_seconds"])
                                    + float(result["generate_seconds"]),
                                }
                            )
                            result["resource_ledger"] = _resource_ledger(result, plugin)
                            state, error = TerminalState.SUCCESS, None
                        except ValueError as exc:
                            state, error = (
                                TerminalState.INVALID,
                                f"{type(exc).__name__}:{exc}",
                            )
                        except TimeoutError as exc:
                            state, error = (
                                TerminalState.TIMEOUT,
                                f"{type(exc).__name__}:{exc}",
                            )
                        except Exception as exc:
                            try:
                                import torch

                                state = (
                                    TerminalState.OOM
                                    if isinstance(exc, torch.cuda.OutOfMemoryError)
                                    else TerminalState.CRASH
                                )
                            except Exception:
                                state = TerminalState.CRASH
                            error = f"{type(exc).__name__}:{exc}"
                        elapsed = time.perf_counter() - monotonic
                        scheduler.settle_gpu(elapsed)
                        record = {
                            **base_ids,
                            **item,
                            "phase": "g3-latency",
                            "attempt_id": attempt_id,
                            "retry_index": retry_index,
                            "prompt_contract_sha256": manifest_sha256(
                                config["prompts"][item["modality"]]
                            ),
                            "model_load_artifact_sha256": load_receipt[
                                "artifact_sha256"
                            ],
                            "started_at_unix": started,
                            "ended_at_unix": time.time(),
                            "gpu_wall_seconds": elapsed,
                            "terminal_state": state.value,
                            "result": result,
                            "error": error,
                        }
                        store.write(record, attempt_id=attempt_id)
                        grouped.setdefault(key, []).append(record)
                        total_latency_calls += 1
                        if retry_index > 0:
                            latency_retries += 1
                        print(
                            json.dumps(
                                {
                                    "latency_attempts": total_latency_calls,
                                    "cell_id": item["cell_id"],
                                    "attempt_index": attempt_index,
                                    "state": state.value,
                                }
                            ),
                            flush=True,
                        )
                    if hard_stop:
                        break
                if hard_stop:
                    break
            finally:
                del model, processor
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
    finally:
        scheduler.release()
    if hard_stop:
        for item in attempts:
            key = (item["unit_id"], int(item["attempt_index"]))
            if key in grouped:
                continue
            attempt_id = _latency_attempt_id(
                item["cell_id"], int(item["attempt_index"]), 0
            )
            record = {
                **base_ids,
                **item,
                "phase": "g3-latency",
                "attempt_id": attempt_id,
                "retry_index": 0,
                "prompt_contract_sha256": manifest_sha256(
                    config["prompts"][item["modality"]]
                ),
                "terminal_state": TerminalState.NOT_RUN_DUE_TO_HARD_STOP.value,
                "result": None,
                "error": "formal_hard_stop_before_latency_attempt",
            }
            store.write(record, attempt_id=attempt_id)
            grouped[key] = [record]
    records = [record for key in sorted(grouped) for record in grouped[key]]
    snapshot = write_content_addressed_jsonl(output_root / "latency", records)
    summary = {
        "schema_version": "evidencegate.g3.latency-summary.v1",
        **base_ids,
        "selected_physical_index": gpu_receipt["selected_physical_index"],
        "selected_gpu_uuid": gpu_receipt["selected_gpu_uuid"],
        "g2_calls": g2_calls,
        "quality_attempts_started": quality_attempts,
        "latency_attempts_started": total_latency_calls,
        "shared_retries_used": quality_retries + latency_retries,
        "total_calls_through_latency": g2_calls
        + quality_attempts
        + total_latency_calls,
        "total_call_hard_cap": total_call_cap,
        "gpu_seconds_consumed_g3_cumulative": scheduler.consumed_seconds,
        "hard_stop": hard_stop,
        "snapshot_path": str(snapshot),
        "snapshot_sha256": snapshot.stem,
        **latency_summary(attempts, grouped),
    }
    write_immutable_manifest(output_root / "g3-latency-summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EvidenceGate G3 formal latency runner"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--image-scope", type=Path, required=True)
    parser.add_argument("--video-scope", type=Path, required=True)
    parser.add_argument("--codec-manifest", type=Path, required=True)
    parser.add_argument("--model-index", type=Path, required=True)
    parser.add_argument("--g2-output-index", type=Path, required=True)
    parser.add_argument("--prompt-adjustment", type=Path, required=True)
    parser.add_argument("--gpu-admission-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    contract = validate_quality_contract(
        config_path=args.config,
        formal_manifest_path=args.formal_manifest,
        scope_path=args.scope,
        image_scope_path=args.image_scope,
        video_scope_path=args.video_scope,
        codec_manifest_path=args.codec_manifest,
        model_index_path=args.model_index,
        g2_output_index_path=args.g2_output_index,
        prompt_adjustment_path=args.prompt_adjustment,
    )
    if str(args.output_root) != contract["config"]["runtime"]["g3_output_root"]:
        raise ValueError("g3_output_root_mismatch")
    attempts = validate_latency_contract(contract)
    gpu_receipt, gpu_hash = _validate_gpu_admission(args.gpu_admission_receipt)
    summary = run_latency(contract, attempts, gpu_receipt, gpu_hash, args.output_root)
    return 0 if summary["terminal_verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
