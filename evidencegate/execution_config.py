from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .codec_manifest import validate_codec_derivative_manifest
from .manifest import canonical_json, manifest_sha256
from .scope_manifest import ScopeManifest

_SEED = 20260810
_FROZEN_G0_SCOPE_SHA256 = "5c246d3b348169130055059456170b00b4300e0c6d1ab451ec244f7fbc58441f"
_FROZEN_IMAGE_MANIFEST_SHA256 = "b1b28483ff57de90eaf9e360de1ebd39f52437f96e94070e1d3e0d7be5ef828a"
_FROZEN_VIDEO_MANIFEST_SHA256 = "889c7b61ed58d3f31d79baf496335411255474b02ec8a0ad5fbaf67cb52dc893"


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{name}_must_be_sha256")
    return value.lower()


def canonical_quantization_config(model_key: str, model: Mapping[str, Any]) -> dict[str, Any]:
    precision = str(model["precision"])
    if precision == "bf16":
        return {"load_in_4bit": False, "precision": "bf16", "torch_dtype": "bfloat16"}
    if precision != "language_weight_nf4":
        raise ValueError("unsupported_precision")
    repository = str(model["model_id"])
    if repository.startswith("llava-hf/"):
        skip_modules = ["vision_tower", "multi_modal_projector", "lm_head"]
    elif repository.startswith("Qwen/"):
        skip_modules = ["visual", "lm_head"]
    else:
        raise ValueError(f"nf4_skip_modules_not_registered:{model_key}")
    return {
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "language_weight_only": True,
        "llm_int8_skip_modules": skip_modules,
        "load_in_4bit": True,
        "precision": "language_weight_nf4",
        "torch_dtype": "bfloat16",
    }


@dataclass(frozen=True)
class ModelLayerIdentity:
    model_key: str
    repository_id: str
    repository_revision: str
    precision: str
    quantization_config: dict[str, Any]
    quantization_config_hash: str
    model_layer_identity: str


def model_layer_identity(model_key: str, model: Mapping[str, Any]) -> ModelLayerIdentity:
    quantization = canonical_quantization_config(model_key, model)
    quantization_hash = hashlib.sha256(canonical_json(quantization)).hexdigest()
    revision = str(model["revision"])
    precision = str(model["precision"])
    identity = _hash_text("|".join((model_key, revision, precision, quantization_hash)))
    return ModelLayerIdentity(
        model_key,
        str(model["model_id"]),
        revision,
        precision,
        quantization,
        quantization_hash,
        identity,
    )


@dataclass(frozen=True)
class LatencyCell:
    cell_id: str
    model_key: str
    model_layer_identity: str
    modality: str
    evidence_state: str
    sample_id: str
    repeat_index: int
    warmup_attempts: int = 4
    steady_attempts: int = 10
    io_policy: str = "staged_warm"
    include_in_formal_latency: bool = True


@dataclass(frozen=True)
class SmokeUnit:
    unit_id: str
    model_key: str
    model_layer_identity: str
    modality: str
    asset_id: str
    condition: str
    fault_mode: str | None = None


def _ranked(ids: tuple[str, ...], prefix: str) -> list[str]:
    return sorted(ids, key=lambda sample_id: _hash_text(f"{prefix}|{_SEED}|{sample_id}"))


def compile_latency_cells(scope: ScopeManifest, layers: Mapping[str, ModelLayerIdentity]) -> tuple[LatencyCell, ...]:
    image_sentinels = _ranked(scope.image_deployment_ids, "latency-image")[:2]
    static = tuple(sample.sample_id for sample in scope.video_samples if sample.stratum == "STATIC_PRIMARY" and sample.sample_id in scope.video_deployment_ids)
    moving = tuple(sample.sample_id for sample in scope.video_samples if sample.stratum == "MOVING_CAMERA_STRESS" and sample.sample_id in scope.video_deployment_ids)
    video_sentinels = (_ranked(static, "latency-video")[0], _ranked(moving, "latency-video")[0])
    values: list[LatencyCell] = []
    for model_key, modality, state, sample_id, repeat_index in itertools.chain(
        itertools.product(sorted(layers), ("image",), ("native", "resize50"), image_sentinels, range(2)),
        itertools.product(sorted(layers), ("video",), ("dense", "codec_guided50"), video_sentinels, range(2)),
    ):
        layer = layers[model_key]
        material = "|".join((layer.model_layer_identity, modality, state, sample_id, str(repeat_index)))
        values.append(LatencyCell(_hash_text(material)[:16], model_key, layer.model_layer_identity, modality, state, sample_id, repeat_index))
    if len(values) != 64 or len({cell.cell_id for cell in values}) != 64:
        raise AssertionError("latency_cells_must_be_64_unique")
    if sum(cell.warmup_attempts + cell.steady_attempts for cell in values) != 896:
        raise AssertionError("latency_attempts_must_be_896")
    return tuple(values)


def _smoke_image(scope: ScopeManifest, layer: ModelLayerIdentity, condition: str) -> str:
    images = tuple(sorted(scope.development_image_ids))
    digest = int(_hash_text(f"smoke-image|{_SEED}|{layer.model_layer_identity}|{condition}"), 16)
    return images[digest % len(images)]


def _smoke_unit(layer: ModelLayerIdentity, modality: str, asset: str, condition: str, fault: str | None = None) -> SmokeUnit:
    material = "|".join((layer.model_layer_identity, modality, asset, condition, fault or "none", str(_SEED)))
    return SmokeUnit(_hash_text(material)[:16], layer.model_key, layer.model_layer_identity, modality, asset, condition, fault)


def compile_smoke_units(scope: ScopeManifest, layers: Mapping[str, ModelLayerIdentity]) -> tuple[SmokeUnit, ...]:
    values: list[SmokeUnit] = []
    primary = layers["0.5b-bf16"]
    for condition in ("native", "jpeg", "resize50", "random_crop50", "center_crop50"):
        values.append(_smoke_unit(primary, "image", _smoke_image(scope, primary, condition), condition))
    for condition in ("dense", "uniform50", "random50", "rgb_difference50", "codec_guided50", "bitrate_only", "resize50"):
        values.append(_smoke_unit(primary, "video", scope.development_video_fixture_sha256, condition))
    for key in ("0.5b-nf4", "3b-bf16", "3b-nf4"):
        layer = layers[key]
        for condition in ("native", "resize50", "jpeg"):
            values.append(_smoke_unit(layer, "image", _smoke_image(scope, layer, condition), condition))
        for condition in ("dense", "codec_guided50", "bitrate_only", "resize50"):
            values.append(_smoke_unit(layer, "video", scope.development_video_fixture_sha256, condition))
    images = tuple(sorted(scope.development_image_ids))
    values.extend((
        _smoke_unit(primary, "image", images[0], "native", "invalid_output"),
        _smoke_unit(primary, "video", scope.development_video_fixture_sha256, "dense", "timeout"),
        _smoke_unit(primary, "image", images[1], "resize50", "crash_or_oom"),
    ))
    if len(values) != 36 or len({unit.unit_id for unit in values}) != 36:
        raise AssertionError("smoke_units_must_be_36_unique")
    return tuple(values)


@dataclass(frozen=True)
class ExecutionConfig:
    schema_version: str
    source_scope_sha256: str
    seed: int
    model_layers: tuple[ModelLayerIdentity, ...]
    latency_cells: tuple[LatencyCell, ...]
    smoke_units: tuple[SmokeUnit, ...]
    query_provider: dict[str, Any]
    codec: dict[str, Any]
    gpu_reservation: dict[str, Any]
    raw_store: dict[str, Any]
    limits: dict[str, Any]
    prompts: dict[str, Any]
    runtime: dict[str, Any]
    codec_derivative_manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return manifest_sha256(self.to_dict())


def compile_execution_config(
    scope_path: str | Path,
    *,
    query_provider_resolution_sha256: str,
    runtime_environment_receipt_sha256: str,
    development_video_manifest_sha256: str,
    model_cache_index_sha256: str,
    g2_output_index_sha256: str,
    prompt_adjustment_receipt_sha256: str,
    codec_derivative_manifest_sha256: str | None = None,
    codec_derivative_manifest_path: str | Path | None = None,
    video_scope_path: str | Path | None = None,
) -> ExecutionConfig:
    path = Path(scope_path)
    payload = path.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    if source_hash != _FROZEN_G0_SCOPE_SHA256:
        raise ValueError("g0_scope_sha256_not_frozen_authority")
    raw = json.loads(payload)
    if raw.get("scope_verdict") != "BROAD":
        raise ValueError("execution_config_requires_broad_scope")
    gates = raw["gates"]
    expected = (986, 64, 896, 36, 189, 2107)
    actual = (gates["quality_logic_units"], gates["latency_cells"], gates["latency_attempts"], gates["smoke_call_cap"], gates["retry_attempt_budget"], gates["total_call_hard_cap"])
    if actual != expected:
        raise ValueError(f"g0_gate_cardinality_mismatch:{actual}")
    scope = ScopeManifest.from_mapping(raw["scope_manifest"])
    if scope.image_source_manifest_sha256 != _FROZEN_IMAGE_MANIFEST_SHA256 or scope.video_source_manifest_sha256 != _FROZEN_VIDEO_MANIFEST_SHA256:
        raise ValueError("g0_upstream_manifest_hash_mismatch")
    layers = {key: model_layer_identity(key, value) for key, value in raw["models"].items()}
    if codec_derivative_manifest_path is not None:
        if codec_derivative_manifest_sha256 is not None or video_scope_path is None:
            raise ValueError("codec_manifest_path_requires_video_scope_and_no_hash_override")
        codec_derivative_manifest_sha256 = validate_codec_derivative_manifest(
            codec_derivative_manifest_path,
            video_scope_path,
        )
    if codec_derivative_manifest_sha256 is None:
        raise ValueError("codec_derivative_manifest_required")
    codec_hash = _require_sha256(codec_derivative_manifest_sha256, "codec_derivative_manifest_sha256")
    query_hash = _require_sha256(query_provider_resolution_sha256, "query_provider_resolution_sha256")
    runtime_hash = _require_sha256(runtime_environment_receipt_sha256, "runtime_environment_receipt_sha256")
    development_video_hash = _require_sha256(development_video_manifest_sha256, "development_video_manifest_sha256")
    model_cache_index_hash = _require_sha256(model_cache_index_sha256, "model_cache_index_sha256")
    g2_output_index_hash = _require_sha256(g2_output_index_sha256, "g2_output_index_sha256")
    prompt_adjustment_hash = _require_sha256(prompt_adjustment_receipt_sha256, "prompt_adjustment_receipt_sha256")
    config = ExecutionConfig(
        "evidencegate.g1.execution-config.v8",
        source_hash,
        _SEED,
        tuple(layers[key] for key in sorted(layers)),
        compile_latency_cells(scope, layers),
        compile_smoke_units(scope, layers),
        {
            "mode": "global_center_crop50",
            "resolution_manifest_sha256": query_hash,
            "reporting_label": "global_center_crop50_fallback_not_query_aware",
            "input_dependence": "image_dimensions_only",
            "extra_lmm_calls": 0,
            "formal_samples_may_select_provider": False,
            "native_attention_verdict": "rejected_for_g2_causal_order_and_budget",
            "replaces_condition": "query_aware_crop50",
            "replacement_condition": "center_crop50",
        },
        {
            "ffmpeg_version": "6.1.4",
            "encoder": "libx264",
            "codec": "h264",
            "profile": "high",
            "pixel_format": "yuv420p",
            "preset": "medium",
            "gop_seconds": 2,
            "min_keyint_equals_keyint": True,
            "scenecut": 0,
            "b_frames": 2,
            "refs": 3,
            "codec_guided_crf": 18,
            "bitrate_only_crf": 32,
            "runtime_encoder_substitution_allowed": False,
        },
        {
            "max_active_attempts_per_gpu": 1,
            "max_schedulers_per_output_root": 1,
            "admission_rule": "consumed_plus_reserved_plus_new_le_cap",
            "unknown_crash_charge": "full_reservation",
        },
        {
            "attempt_commit": "partial_fsync_rename_dirfsync_then_wal_line_fsync",
            "snapshot": "exclusive_content_addressed_jsonl",
            "overwrite_allowed": False,
            "output_root": "/srv/project-assets/evidencegate-v2",
        },
        {
            "quality_logic_units": gates["quality_logic_units"],
            "quality_success_minimum": gates["quality_success_minimum"],
            "quality_terminal_coverage_required": gates["quality_terminal_coverage_required"],
            "latency_cells": gates["latency_cells"],
            "latency_attempts": gates["latency_attempts"],
            "smoke_call_cap": gates["smoke_call_cap"],
            "retry_attempt_budget": gates["retry_attempt_budget"],
            "total_call_hard_cap": gates["total_call_hard_cap"],
            "gpu_hour_recommended": gates["gpu_hour_recommended"],
            "gpu_hour_hard_cap": gates["gpu_hour_hard_cap"],
        },
        {
            "image": {
                "template": "{question}\nAnswer exactly yes or no, with no other text.",
                "parser": "exact_case_insensitive_yes_or_no",
                "max_new_tokens": 4,
                "do_sample": False,
            },
            "video": {
                "template": "What is the maximum number of people simultaneously visible in any single frame of this clip? Return exactly {\"count\": N} and no other text, where N is a non-negative integer.",
                "assistant_prefix": "{\"count\": ",
                "parser": "exact_json_single_count_0_99",
                "max_new_tokens": 8,
                "do_sample": False,
            },
        },
        {
            "python_executable": "/srv/project/.venv-evidencegate-g2/bin/python",
            "runtime_environment_receipt_sha256": runtime_hash,
            "nvidia_library_glob": "/usr/local/lib/python3.10/dist-packages/nvidia/*/lib",
            "model_cache_root": "/srv/project-assets/evidencegate-v2/models",
            "model_cache_index_sha256": model_cache_index_hash,
            "g2_output_index_sha256": g2_output_index_hash,
            "prompt_adjustment_receipt_sha256": prompt_adjustment_hash,
            "g2_output_root": "/srv/project-assets/evidencegate-v2/runs/g2-smoke-v1",
            "development_video_manifest_sha256": development_video_hash,
            "model_load_timeout_seconds": 900,
            "smoke_inference_timeout_seconds": 180,
            "allow_model_download_during_g2": False,
            "cuda_visible_device_count": 1,
            "processor_use_fast": False,
            "g3_output_root": "/srv/project-assets/evidencegate-v2/runs/g3-formal-v1",
            "quality_inference_timeout_seconds": 180,
            "input_transforms": {
                "image_jpeg_quality": 75,
                "image_area_fraction": 0.5,
                "video_frame_fraction": 0.5,
                "video_bitrate_only_crf": 32,
                "video_codec_guided_crf": 18,
            },
            "formal_condition_resolution": {
                "image": {
                    "query_aware_crop50": "center_crop50",
                    "reduced": "center_crop50",
                },
                "video": {
                    "native": "dense",
                    "reduced": "codec_guided50",
                    "jpeg": "bitrate_only",
                },
            },
        },
        codec_hash,
    )
    if len(config.latency_cells) != 64 or len(config.smoke_units) != 36:
        raise AssertionError("execution_config_cardinality_invalid")
    return config
