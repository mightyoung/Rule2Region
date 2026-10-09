from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from evidencegate.codec_manifest import validate_codec_derivative_manifest
from evidencegate.formal_manifest import (
    IMAGE_CONDITIONS,
    VIDEO_CONDITIONS,
    validate_formal_manifest,
)
from evidencegate.ledger import IOReceipt
from evidencegate.manifest import manifest_sha256, write_immutable_manifest
from evidencegate.runner import (
    AttemptArtifactStore,
    SingleScheduler,
    write_content_addressed_jsonl,
)
from evidencegate.smoke_inputs import (
    decode_video_rgb,
    load_candidate_features,
    prepare_image_input,
    prepare_video_input,
)
from evidencegate.terminals import TerminalState
from research_tools.g0_video_freeze import frame_tree_hash
from research_tools.g2_smoke import _generate, _load_layer, _validate_model_index


RETRYABLE_STATES = {TerminalState.TIMEOUT, TerminalState.OOM, TerminalState.CRASH}
TERMINAL_VALUES = {state.value for state in TerminalState}
MODEL_FAMILIES = {"llava_onevision_05b", "qwen25_vl_3b"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path.name}")
    return value


def _validate_execution_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = _json(path)
    material = dict(payload)
    identity = material.pop("execution_config_sha256", None)
    if payload.get(
        "schema_version"
    ) != "evidencegate.g1.execution-config.v8" or identity != manifest_sha256(material):
        raise ValueError("formal_execution_config_identity_or_schema_invalid")
    return payload, _sha(path)


def _validate_gpu_admission(path: Path) -> tuple[dict[str, Any], str]:
    receipt = _json(path)
    if receipt.get("verdict") != "ADMITTED":
        raise ValueError("gpu_admission_not_admitted")
    if time.time() - float(receipt.get("observed_at_unix", 0)) > 300:
        raise ValueError("gpu_admission_missing_or_stale")
    selected = str(receipt.get("selected_physical_index"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != selected:
        raise ValueError("cuda_visible_devices_does_not_match_admission")
    if not str(receipt.get("selected_gpu_uuid", "")).strip():
        raise ValueError("gpu_admission_uuid_missing")
    return receipt, _sha(path)


def enumerate_video_source_paths(
    sample: Mapping[str, Any], sequence_receipt: Mapping[str, Any]
) -> tuple[Path, ...]:
    start = int(sample["start_frame_inclusive"])
    end = int(sample["end_frame_exclusive"])
    if start < 1 or end <= start or end - start != int(sample["frame_count"]):
        raise ValueError("video_window_frame_bounds_invalid")
    gt_path = Path(str(sequence_receipt["gt_path"]))
    image_root = gt_path.parent.parent / "img1"
    paths = tuple(image_root / f"{index:06d}.jpg" for index in range(start, end))
    if paths[0].name != sample["first_frame"] or paths[-1].name != sample["last_frame"]:
        raise ValueError("video_window_boundary_names_mismatch")
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"video_source_frames_missing:{sample['sample_id']}:{missing[0]}"
        )
    return paths


def expected_label(sample: Mapping[str, Any], modality: str) -> bool | int:
    if modality == "image":
        label = str(sample.get("label", "")).lower()
        if label not in {"yes", "no"} or not str(sample.get("text", "")).strip():
            raise ValueError("formal_image_label_or_question_invalid")
        return label == "yes"
    if modality == "video":
        label = sample.get("label")
        if (
            not isinstance(label, Mapping)
            or not isinstance(label.get("count"), int)
            or int(label["count"]) < 0
        ):
            raise ValueError("formal_video_label_invalid")
        if not str(sample.get("label_contract", "")).strip():
            raise ValueError("formal_video_label_contract_missing")
        return int(label["count"])
    raise ValueError("formal_unit_modality_invalid")


def _validate_prompt_adjustment(path: Path, expected_hash: str) -> dict[str, Any]:
    if _sha(path) != expected_hash:
        raise ValueError("prompt_adjustment_receipt_hash_mismatch")
    receipt = _json(path)
    evidence = receipt.get("source_evidence", {})
    if (
        receipt.get("schema_version") != "evidencegate.g2.prompt-adjustment.v1"
        or evidence.get("formal_samples_used") is not False
        or int(receipt.get("additional_model_calls", -1)) != 0
        or int(receipt.get("adjustment_ordinal", -1)) != 1
        or int(receipt.get("adjustment_limit", -1)) != 1
        or receipt.get("further_prompt_or_output_format_adjustment_allowed")
        is not False
    ):
        raise ValueError("prompt_adjustment_policy_invalid")
    return receipt


def _validate_model_index_contract(
    path: Path, expected_hash: str, *, verify_files: bool
) -> dict[str, Any]:
    if verify_files:
        return _validate_model_index(path, expected_hash)
    if _sha(path) != expected_hash:
        raise ValueError("model_cache_index_file_sha256_mismatch")
    index = _json(path)
    material = dict(index)
    identity = material.pop("manifest_sha256", None)
    if index.get(
        "schema_version"
    ) != "evidencegate.g1.model-cache-index.v1" or identity != manifest_sha256(
        material
    ):
        raise ValueError("model_cache_index_identity_or_schema_invalid")
    models = index.get("models")
    if not isinstance(models, Mapping) or set(models) != MODEL_FAMILIES:
        raise ValueError("model_cache_index_family_set_mismatch")
    for entry in models.values():
        if not all(
            str(entry.get(key, "")).strip()
            for key in (
                "path",
                "manifest_path",
                "repo_id",
                "revision",
                "snapshot_manifest_sha256",
                "manifest_file_sha256",
            )
        ):
            raise ValueError("model_cache_index_entry_invalid")
    return index


def validate_quality_contract(
    *,
    config_path: Path,
    formal_manifest_path: Path,
    scope_path: Path,
    image_scope_path: Path,
    video_scope_path: Path,
    codec_manifest_path: Path,
    model_index_path: Path,
    g2_output_index_path: Path,
    prompt_adjustment_path: Path,
    verify_model_files: bool = True,
    verify_source_files: bool = True,
) -> dict[str, Any]:
    config, config_file_sha = _validate_execution_config(config_path)
    formal = _json(formal_manifest_path)
    validate_formal_manifest(formal)
    scope = _json(scope_path)
    image_scope = _json(image_scope_path)
    video_scope = _json(video_scope_path)
    if (
        _sha(scope_path) != config["source_scope_sha256"]
        or _sha(scope_path) != formal["source_scope_file_sha256"]
    ):
        raise ValueError("formal_scope_file_hash_mismatch")
    if _sha(config_path) != formal["execution_config_file_sha256"]:
        raise ValueError("formal_config_file_hash_mismatch")
    if scope.get("scope_verdict") != "BROAD":
        raise ValueError("formal_scope_must_be_broad")
    sources = scope.get("source_manifests", {})
    if _sha(image_scope_path) != sources.get("image", {}).get("file_sha256"):
        raise ValueError("formal_image_scope_hash_mismatch")
    if _sha(video_scope_path) != sources.get("video", {}).get("file_sha256"):
        raise ValueError("formal_video_scope_hash_mismatch")
    if _sha(codec_manifest_path) != config["codec_derivative_manifest_sha256"]:
        raise ValueError("formal_codec_manifest_file_hash_mismatch")
    validate_codec_derivative_manifest(
        codec_manifest_path, video_scope_path, verify_files=verify_source_files
    )
    if _sha(g2_output_index_path) != config["runtime"]["g2_output_index_sha256"]:
        raise ValueError("g2_output_index_file_hash_mismatch")
    g2_index = _json(g2_output_index_path)
    if (
        int(g2_index.get("attempts", -1)) != 36
        or g2_index.get("schema_version") != "evidencegate.g2.output-index.v1"
    ):
        raise ValueError("g2_output_index_contract_invalid")
    _validate_prompt_adjustment(
        prompt_adjustment_path, config["runtime"]["prompt_adjustment_receipt_sha256"]
    )
    model_index = _validate_model_index_contract(
        model_index_path,
        config["runtime"]["model_cache_index_sha256"],
        verify_files=verify_model_files,
    )

    image_samples = {
        item["image"]: item
        for item in image_scope["samples"]
        if item["tier"] == "formal"
    }
    video_samples = {item["sample_id"]: item for item in video_scope["samples"]}
    codec_entries = {
        item["sample_id"]: item for item in _json(codec_manifest_path)["entries"]
    }
    layers = {item["model_key"]: item for item in config["model_layers"]}
    if verify_source_files:
        for sample in image_samples.values():
            source_path = Path(sample["source"]["path"])
            if (
                not source_path.is_file()
                or source_path.stat().st_size != int(sample["source"]["bytes"])
                or _sha(source_path) != sample["source"]["sha256"]
            ):
                raise ValueError(f"formal_image_source_file_mismatch:{sample['image']}")
        for sample in video_samples.values():
            sequence_key = str(sample["sequence_id"]).removeprefix("MOT17-")
            paths = enumerate_video_source_paths(
                sample, video_scope["sequence_receipts"][sequence_key]
            )
            if frame_tree_hash(paths) != sample["frame_tree_sha256"]:
                raise ValueError(
                    f"formal_video_frame_tree_hash_mismatch:{sample['sample_id']}"
                )
    units = formal["units"]
    if units != scope["matrix"]["quality_units"]:
        # The formal compiler adds only resolved_condition and layer identity.
        stripped = [
            {
                key: value
                for key, value in unit.items()
                if key not in {"resolved_condition", "model_layer_identity"}
            }
            for unit in units
        ]
        if stripped != scope["matrix"]["quality_units"]:
            raise ValueError("formal_units_do_not_preserve_frozen_matrix")
    aliases = config["runtime"]["formal_condition_resolution"]
    for unit in units:
        modality = unit["modality"]
        if modality == "image":
            sample = image_samples.get(unit["sample_id"])
            allowed = IMAGE_CONDITIONS
        elif modality == "video":
            sample = video_samples.get(unit["sample_id"])
            allowed = VIDEO_CONDITIONS
            if unit["sample_id"] not in codec_entries:
                raise ValueError("formal_video_codec_entry_missing")
        else:
            raise ValueError("formal_unit_modality_invalid")
        if sample is None:
            raise ValueError(f"formal_sample_not_frozen:{unit['sample_id']}")
        expected_resolved = aliases.get(modality, {}).get(
            unit["condition"], unit["condition"]
        )
        if (
            unit.get("resolved_condition") != expected_resolved
            or expected_resolved not in allowed
        ):
            raise ValueError("formal_condition_resolution_mismatch")
        layer = layers.get(unit["model"])
        if (
            layer is None
            or unit.get("model_layer_identity") != layer["model_layer_identity"]
        ):
            raise ValueError("formal_model_layer_identity_mismatch")
        expected_label(sample, modality)
    if Counter(unit["model"] for unit in units) != Counter(
        {"0.5b-bf16": 590, "0.5b-nf4": 132, "3b-bf16": 132, "3b-nf4": 132}
    ):
        raise ValueError("formal_model_cardinality_mismatch")
    return {
        "config": config,
        "config_file_sha256": config_file_sha,
        "formal": formal,
        "formal_file_sha256": _sha(formal_manifest_path),
        "scope": scope,
        "image_samples": image_samples,
        "video_samples": video_samples,
        "sequence_receipts": video_scope["sequence_receipts"],
        "codec_entries": codec_entries,
        "model_index": model_index,
        "layers": layers,
        "g2_output_index_file_sha256": _sha(g2_output_index_path),
    }


@dataclass
class LoadedVideo:
    frames: tuple[np.ndarray, ...]
    codec_features: tuple[Any, ...]
    bitrate_frames: tuple[np.ndarray, ...]
    original_encoded_bytes: int
    reduced_encoded_bytes: int
    io_receipt: dict[str, Any]


class FormalAssetLoader:
    def __init__(
        self,
        video_samples: Mapping[str, dict[str, Any]],
        sequence_receipts: Mapping[str, dict[str, Any]],
        codec_entries: Mapping[str, dict[str, Any]],
        *,
        sources_already_validated: bool = False,
    ) -> None:
        self.video_samples = video_samples
        self.sequence_receipts = sequence_receipts
        self.codec_entries = codec_entries
        self.sources_already_validated = sources_already_validated
        self._cached_id: str | None = None
        self._cached: LoadedVideo | None = None

    def load_video(self, sample_id: str) -> tuple[LoadedVideo, str]:
        if self._cached_id == sample_id and self._cached is not None:
            return self._cached, "warm"
        io_started = time.perf_counter()
        sample = self.video_samples[sample_id]
        sequence_key = str(sample["sequence_id"]).removeprefix("MOT17-")
        paths = enumerate_video_source_paths(
            sample, self.sequence_receipts[sequence_key]
        )
        if (
            not self.sources_already_validated
            and frame_tree_hash(paths) != sample["frame_tree_sha256"]
        ):
            raise ValueError(f"formal_video_frame_tree_hash_mismatch:{sample_id}")
        frames = tuple(np.asarray(Image.open(path).convert("RGB")) for path in paths)
        entry = self.codec_entries[sample_id]
        guided = entry["derivatives"]["codec_guided_crf18"]
        bitrate = entry["derivatives"]["bitrate_only_crf32"]
        guided_path = Path(guided["feature_path"])
        bitrate_path = Path(bitrate["path"])
        if not self.sources_already_validated and (
            _sha(guided_path) != guided["feature_sha256"]
            or _sha(bitrate_path) != bitrate["sha256"]
        ):
            raise ValueError(f"formal_video_derivative_hash_mismatch:{sample_id}")
        features = load_candidate_features(guided_path)
        bitrate_frames = decode_video_rgb(bitrate_path)
        if len(features) != len(frames) or len(bitrate_frames) != len(frames):
            raise ValueError(
                f"formal_video_derivative_cardinality_mismatch:{sample_id}"
            )
        original_bytes = sum(path.stat().st_size for path in paths)
        loaded = LoadedVideo(
            frames=frames,
            codec_features=features,
            bitrate_frames=bitrate_frames,
            original_encoded_bytes=original_bytes,
            reduced_encoded_bytes=int(bitrate["bytes"]),
            io_receipt={
                **IOReceipt(
                    stage="formal_video_input",
                    source_path=f"{paths[0]}..{paths[-1]}",
                    source_sha256=sample["frame_tree_sha256"],
                    storage_class="nfs",
                    cache_state="cold",
                    bytes_read=original_bytes
                    + int(bitrate["bytes"])
                    + guided_path.stat().st_size,
                    wall_clock_ms=(time.perf_counter() - io_started) * 1000.0,
                    included_in_latency=False,
                ).to_dict(),
                "schema_version": "evidencegate.g3.input-io.v1",
                "sample_id": sample_id,
                "source_frame_count": len(frames),
                "codec_feature_sha256": guided["feature_sha256"],
                "bitrate_video_sha256": bitrate["sha256"],
                "bitrate_video_bytes": int(bitrate["bytes"]),
            },
        )
        self._cached_id, self._cached = sample_id, loaded
        return loaded, "cold"


def _prepare_unit(
    unit: Mapping[str, Any], contract: Mapping[str, Any], loader: FormalAssetLoader
) -> tuple[Any, str | None, bool | int, dict[str, Any]]:
    modality = unit["modality"]
    if modality == "image":
        sample = contract["image_samples"][unit["sample_id"]]
        io_started = time.perf_counter()
        plugin = prepare_image_input(sample, unit["resolved_condition"])
        receipt = {
            **IOReceipt(
                stage="formal_image_input",
                source_path=sample["source"]["path"],
                source_sha256=sample["source"]["sha256"],
                storage_class="nfs",
                cache_state="cold",
                bytes_read=int(sample["source"]["bytes"]),
                wall_clock_ms=(time.perf_counter() - io_started) * 1000.0,
                included_in_latency=False,
            ).to_dict(),
            "schema_version": "evidencegate.g3.input-io.v1",
            "sample_id": unit["sample_id"],
        }
        return plugin, sample["text"], expected_label(sample, modality), receipt
    loaded, cache_state = loader.load_video(unit["sample_id"])
    plugin = prepare_video_input(
        loaded.frames,
        unit["resolved_condition"],
        source_sha256=contract["video_samples"][unit["sample_id"]]["frame_tree_sha256"],
        codec_features=loaded.codec_features,
        bitrate_frames=loaded.bitrate_frames,
        original_encoded_bytes=loaded.original_encoded_bytes,
        reduced_encoded_bytes=loaded.reduced_encoded_bytes,
        window_id=unit["sample_id"],
    )
    receipt = {**loaded.io_receipt, "cache_state": cache_state}
    return (
        plugin,
        None,
        expected_label(contract["video_samples"][unit["sample_id"]], modality),
        receipt,
    )


def _record_state(record: Mapping[str, Any]) -> TerminalState:
    value = record.get("terminal_state")
    if value not in TERMINAL_VALUES:
        raise ValueError("attempt_terminal_state_invalid")
    return TerminalState(value)


def _resource_ledger(result: Mapping[str, Any], plugin: Any) -> dict[str, Any]:
    return {
        "schema_version": "evidencegate.g3.resource-ledger.v1",
        "records": [
            {
                "stage": "plugin",
                "encoded_bytes": int(plugin.encoded_bytes),
                "decoded_pixels": int(plugin.decoded_pixels),
                "model_pixels": int(plugin.model_pixels),
                "visual_tokens": 0,
                "prefill_tokens": 0,
                "vram_bytes": 0,
                "wall_clock_ms": 0.0,
                "combined": False,
            },
            {
                "stage": "processor",
                "encoded_bytes": 0,
                "decoded_pixels": 0,
                "model_pixels": 0,
                "visual_tokens": 0,
                "prefill_tokens": 0,
                "vram_bytes": 0,
                "wall_clock_ms": float(result["preprocess_seconds"]) * 1000.0,
                "combined": False,
            },
            {
                "stage": "transfer",
                "encoded_bytes": 0,
                "decoded_pixels": 0,
                "model_pixels": 0,
                "visual_tokens": 0,
                "prefill_tokens": 0,
                "vram_bytes": 0,
                "wall_clock_ms": float(result["transfer_seconds"]) * 1000.0,
                "combined": False,
            },
            {
                "stage": "vision_projector_prefill_generation_combined",
                "encoded_bytes": 0,
                "decoded_pixels": 0,
                "model_pixels": 0,
                "visual_tokens": 0,
                "prefill_tokens": int(result["prefill_tokens"]),
                "vram_bytes": int(result["peak_vram_bytes"]),
                "wall_clock_ms": float(result["generate_seconds"]) * 1000.0,
                "combined": True,
            },
        ],
        "visual_token_measurement": "unavailable_from_frozen_processor_api",
    }


def load_quality_attempts(
    store: AttemptArtifactStore, unit_ids: set[str], config_hash: str, formal_hash: str
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_attempt_ids: set[str] = set()
    for path in sorted(store.attempts.glob("g3q-*.json")):
        record = _json(path)
        attempt_id = str(record.get("attempt_id", ""))
        unit_id = str(record.get("unit_id", ""))
        if attempt_id in seen_attempt_ids:
            raise ValueError("duplicate_quality_attempt_id")
        seen_attempt_ids.add(attempt_id)
        if unit_id not in unit_ids:
            raise ValueError("quality_attempt_unknown_unit")
        if (
            record.get("config_file_sha256") != config_hash
            or record.get("formal_manifest_file_sha256") != formal_hash
        ):
            raise ValueError("quality_attempt_frozen_identity_mismatch")
        _record_state(record)
        grouped[unit_id].append(record)
    for unit_id, records in grouped.items():
        records.sort(key=lambda item: int(item.get("retry_index", -1)))
        indexes = [int(item.get("retry_index", -1)) for item in records]
        if indexes != list(range(len(indexes))) or len(
            {item["attempt_id"] for item in records}
        ) != len(records):
            raise ValueError(f"quality_retry_sequence_invalid:{unit_id}")
    return dict(grouped)


def quality_gate_summary(
    units: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    success_minimum: int = 937,
) -> dict[str, Any]:
    unit_ids = {str(unit["unit_id"]) for unit in units}
    terminal: dict[str, str] = {}
    correctness: list[bool] = []
    for unit_id in unit_ids:
        records = grouped.get(unit_id, ())
        if not records:
            continue
        last = records[-1]
        terminal[unit_id] = _record_state(last).value
        result = last.get("result")
        if (
            terminal[unit_id] == TerminalState.SUCCESS.value
            and isinstance(result, Mapping)
            and isinstance(result.get("correct"), bool)
        ):
            correctness.append(bool(result["correct"]))
    counts = Counter(terminal.values())
    successes = counts[TerminalState.SUCCESS.value]
    coverage = len(terminal)
    verdict = (
        "PASS" if successes >= success_minimum and coverage == len(unit_ids) else "FAIL"
    )
    return {
        "quality_logic_units": len(unit_ids),
        "quality_success_minimum": success_minimum,
        "quality_terminal_coverage_required": len(unit_ids),
        "quality_successes": successes,
        "quality_terminal_coverage": coverage,
        "terminal_counts": dict(sorted(counts.items())),
        "scored_successes": len(correctness),
        "correct_outputs": sum(correctness),
        "accuracy_on_successful_scored_units": (sum(correctness) / len(correctness))
        if correctness
        else None,
        "verdict": verdict,
    }


def _attempt_id(unit_id: str, retry_index: int) -> str:
    safe = unit_id.replace(":", "-")
    return f"g3q-{safe}-r{retry_index}"


def _write_not_run(
    store: AttemptArtifactStore,
    unit: Mapping[str, Any],
    base_ids: Mapping[str, Any],
    prompt_contract_sha256: str,
) -> dict[str, Any]:
    record = {
        **base_ids,
        **unit,
        "attempt_id": _attempt_id(str(unit["unit_id"]), 0),
        "retry_index": 0,
        "phase": "g3-quality",
        "prompt_contract_sha256": prompt_contract_sha256,
        "terminal_state": TerminalState.NOT_RUN_DUE_TO_HARD_STOP.value,
        "result": None,
        "error": "formal_hard_stop_before_first_attempt",
    }
    store.write(record, attempt_id=record["attempt_id"])
    return record


def run_quality(
    contract: dict[str, Any],
    gpu_receipt: dict[str, Any],
    gpu_hash: str,
    output_root: Path,
) -> dict[str, Any]:
    config = contract["config"]
    units = list(contract["formal"]["units"])
    unit_ids = {unit["unit_id"] for unit in units}
    store = AttemptArtifactStore(output_root)
    recovery = store.recover()
    if recovery["fail_closed"]:
        raise RuntimeError("g3_attempt_store_recovery_failed")
    grouped = load_quality_attempts(
        store, unit_ids, contract["config_file_sha256"], contract["formal_file_sha256"]
    )
    retry_used = sum(max(0, len(records) - 1) for records in grouped.values())
    retry_cap = int(config["limits"]["retry_attempt_budget"])
    if retry_used > retry_cap:
        raise ValueError("quality_retry_budget_already_exceeded")
    total_started = sum(len(records) for records in grouped.values())
    g2_calls = 36
    call_cap = int(config["limits"]["total_call_hard_cap"])
    cap_seconds = float(config["limits"]["gpu_hour_hard_cap"]) * 3600.0
    inference_timeout = float(config["runtime"]["quality_inference_timeout_seconds"])
    load_timeout = float(config["runtime"]["model_load_timeout_seconds"])
    loader = FormalAssetLoader(
        contract["video_samples"],
        contract["sequence_receipts"],
        contract["codec_entries"],
        sources_already_validated=True,
    )
    scheduler = SingleScheduler(output_root)
    scheduler.acquire("evidencegate:g3-formal-quality")
    base_ids = {
        "config_file_sha256": contract["config_file_sha256"],
        "formal_manifest_file_sha256": contract["formal_file_sha256"],
        "formal_manifest_sha256": contract["formal"]["manifest_sha256"],
        "gpu_admission_receipt_sha256": gpu_hash,
    }
    hard_stop = False
    try:
        for model_key in sorted(contract["layers"]):
            model_units = [unit for unit in units if unit["model"] == model_key]
            model_units.sort(
                key=lambda item: (
                    item["modality"],
                    item["sample_id"],
                    item["resolved_condition"],
                    item["unit_id"],
                )
            )
            pending = []
            for unit in model_units:
                records = grouped.get(unit["unit_id"], [])
                if not records:
                    pending.append(unit)
                elif (
                    _record_state(records[-1]) in RETRYABLE_STATES
                    and len(records) - 1 < retry_cap
                ):
                    pending.append(unit)
            if not pending:
                continue
            if g2_calls + total_started >= call_cap:
                hard_stop = True
                break
            load_started = time.perf_counter()
            try:
                scheduler.reserve_gpu(
                    load_timeout, cap_seconds, f"g3-model-load-{model_key}"
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
                    phase="g3",
                    snapshot_already_validated=True,
                )
                scheduler.settle_gpu(time.perf_counter() - load_started)
            except Exception:
                scheduler.settle_gpu(time.perf_counter() - load_started)
                raise
            try:
                for unit in pending:
                    records = grouped.get(unit["unit_id"], [])
                    retry_index = len(records)
                    if retry_index > 0 and retry_used >= retry_cap:
                        continue
                    if g2_calls + total_started >= call_cap:
                        hard_stop = True
                        break
                    attempt_id = _attempt_id(unit["unit_id"], retry_index)
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
                        plugin, question, expected, io_receipt = _prepare_unit(
                            unit, contract, loader
                        )
                        result = _generate(
                            model,
                            processor,
                            plugin,
                            config["prompts"],
                            unit["modality"],
                            question,
                            inference_timeout,
                        )
                        result.update(
                            {
                                "plugin": plugin.to_dict(),
                                "io_receipt": io_receipt,
                                "expected": expected,
                                "correct": result["parsed_output"] == expected,
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
                        **unit,
                        "phase": "g3-quality",
                        "attempt_id": attempt_id,
                        "retry_index": retry_index,
                        "model_load_artifact_sha256": load_receipt["artifact_sha256"],
                        "prompt_contract_sha256": manifest_sha256(
                            config["prompts"][unit["modality"]]
                        ),
                        "started_at_unix": started,
                        "ended_at_unix": time.time(),
                        "gpu_wall_seconds": elapsed,
                        "terminal_state": state.value,
                        "result": result,
                        "error": error,
                    }
                    store.write(record, attempt_id=attempt_id)
                    grouped.setdefault(unit["unit_id"], []).append(record)
                    total_started += 1
                    if retry_index > 0:
                        retry_used += 1
                    print(
                        json.dumps(
                            {
                                "quality_attempts": total_started,
                                "unit_id": unit["unit_id"],
                                "retry_index": retry_index,
                                "state": state.value,
                            }
                        ),
                        flush=True,
                    )
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
        for unit in units:
            if unit["unit_id"] not in grouped:
                record = _write_not_run(
                    store,
                    unit,
                    base_ids,
                    manifest_sha256(config["prompts"][unit["modality"]]),
                )
                grouped[unit["unit_id"]] = [record]
    records = [record for unit_id in sorted(grouped) for record in grouped[unit_id]]
    snapshot = write_content_addressed_jsonl(output_root, records)
    summary = {
        "schema_version": "evidencegate.g3.quality-summary.v1",
        **base_ids,
        "selected_physical_index": gpu_receipt["selected_physical_index"],
        "selected_gpu_uuid": gpu_receipt["selected_gpu_uuid"],
        "quality_attempts_started": total_started,
        "quality_retries_used": retry_used,
        "quality_retry_budget": retry_cap,
        "g2_calls": g2_calls,
        "total_calls_through_quality": g2_calls + total_started,
        "total_call_hard_cap": call_cap,
        "gpu_seconds_consumed_g3": scheduler.consumed_seconds,
        "hard_stop": hard_stop,
        "snapshot_path": str(snapshot),
        "snapshot_sha256": snapshot.stem,
        **quality_gate_summary(
            units, grouped, int(config["limits"]["quality_success_minimum"])
        ),
    }
    write_immutable_manifest(output_root / "g3-quality-summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EvidenceGate G3 formal quality runner"
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
    parser.add_argument("--gpu-admission-receipt", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
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
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "verdict": "PASS",
                    "quality_units": len(contract["formal"]["units"]),
                    "formal_images": len(contract["image_samples"]),
                    "formal_video_windows": len(contract["video_samples"]),
                    "gpu_calls_started": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.gpu_admission_receipt is None:
        raise ValueError("gpu_admission_receipt_required_for_execution")
    gpu_receipt, gpu_hash = _validate_gpu_admission(args.gpu_admission_receipt)
    summary = run_quality(contract, gpu_receipt, gpu_hash, args.output_root)
    return 0 if summary["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
