from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evidencegate.manifest import manifest_sha256, write_immutable_manifest
from evidencegate.model_runtime import parse_model_output, render_prompt
from evidencegate.model_snapshot import (
    validate_model_cache_index,
    validate_model_snapshot,
)
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
from research_tools.model_compatibility.harness import (
    FIXED_MODELS,
    _dtype_inventory,
    _validate_language_weight_nf4,
)


MODEL_FAMILY = {"0.5b": "llava_onevision_05b", "3b": "qwen25_vl_3b"}
FAULT_EXPECTED = {
    "invalid_output": TerminalState.INVALID,
    "timeout": TerminalState.TIMEOUT,
    "crash_or_oom": TerminalState.OOM,
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_execution_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_bytes())
    identity = payload.pop("execution_config_sha256")
    if (
        identity != manifest_sha256(payload)
        or payload.get("schema_version") != "evidencegate.g1.execution-config.v6"
    ):
        raise ValueError("execution_config_identity_or_schema_invalid")
    return payload, _sha(path)


def _validate_model_index(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _sha(path) != expected_sha256:
        raise ValueError("model_cache_index_file_sha256_mismatch")
    index = json.loads(path.read_text(encoding="utf-8"))
    validate_model_cache_index(index)
    if set(index["models"]) != set(MODEL_FAMILY.values()):
        raise ValueError("model_cache_index_family_set_mismatch")
    return index


def _load_layer(
    layer: dict[str, Any],
    model_index: dict[str, Any],
    output_root: Path,
    *,
    phase: str = "g2",
    snapshot_already_validated: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    import transformers
    from transformers import AutoProcessor, BitsAndBytesConfig

    size = "0.5b" if layer["model_key"].startswith("0.5b") else "3b"
    family = MODEL_FAMILY[size]
    spec = FIXED_MODELS[family]
    snapshot = model_index["models"][family]
    snapshot_path = Path(snapshot["path"])
    manifest = json.loads(Path(snapshot["manifest_path"]).read_text(encoding="utf-8"))
    if manifest["repo_id"] != spec.repo_id or manifest["revision"] != spec.revision:
        raise ValueError("model_snapshot_identity_mismatch")
    if not snapshot_already_validated:
        validate_model_snapshot(snapshot_path, manifest)
    processor = AutoProcessor.from_pretrained(
        snapshot_path, local_files_only=True, use_fast=False
    )
    model_class = getattr(transformers, spec.model_class)
    quantization = layer["quantization_config"]
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }
    if quantization["load_in_4bit"]:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quantization["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=quantization["bnb_4bit_use_double_quant"],
            llm_int8_skip_modules=list(quantization["llm_int8_skip_modules"]),
        )
    torch.cuda.reset_peak_memory_stats()
    model = model_class.from_pretrained(snapshot_path, **kwargs)
    model.eval()
    inventory = _dtype_inventory(model, spec)
    problems = (
        _validate_language_weight_nf4(inventory) if quantization["load_in_4bit"] else []
    )
    if problems:
        raise RuntimeError("nf4_inventory_invalid:" + "|".join(problems))
    receipt = {
        "schema_version": f"evidencegate.{phase}.model-load.v1",
        "model_key": layer["model_key"],
        "model_layer_identity": layer["model_layer_identity"],
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "snapshot_manifest_sha256": manifest["manifest_sha256"],
        "processor_class": processor.__class__.__name__,
        "model_class": model.__class__.__name__,
        "quantization_config": quantization,
        "dtype_inventory": inventory,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }
    receipt_identity = manifest_sha256(receipt)
    artifact = write_immutable_manifest(
        output_root / "model-load" / f"{layer['model_key']}.{receipt_identity}.json",
        receipt,
    )
    receipt["artifact_sha256"] = artifact.sha256
    return model, processor, receipt


def _generate(
    model: Any,
    processor: Any,
    plugin: Any,
    prompt_config: dict[str, Any],
    modality: str,
    question: str | None,
    timeout: float,
) -> dict[str, Any]:
    import torch

    prompt_text, assistant_prefix = render_prompt(
        prompt_config, modality, question=question
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": modality}, {"type": "text", "text": prompt_text}],
        }
    ]
    chat = (
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        + assistant_prefix
    )
    if modality == "image":
        media_kwargs = {"images": [Image.fromarray(plugin.model_frames[0])]}
    else:
        media_kwargs = {"videos": [np.stack(plugin.model_frames, axis=0)]}
    preprocess_started = time.perf_counter()
    inputs = processor(text=[chat], padding=True, return_tensors="pt", **media_kwargs)
    preprocess_seconds = time.perf_counter() - preprocess_started
    cuda_device = next(
        parameter.device
        for parameter in model.parameters()
        if parameter.device.type == "cuda"
    )
    transfer_started = time.perf_counter()
    inputs = {
        name: value.to(cuda_device) if hasattr(value, "to") else value
        for name, value in inputs.items()
    }
    transfer_seconds = time.perf_counter() - transfer_started
    torch.cuda.reset_peak_memory_stats(cuda_device)
    prompt_length = int(inputs["input_ids"].shape[-1])
    generate_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(prompt_config[modality]["max_new_tokens"]),
            do_sample=bool(prompt_config[modality]["do_sample"]),
            use_cache=True,
            max_time=timeout,
            return_dict_in_generate=True,
        )
    generate_seconds = time.perf_counter() - generate_started
    continuation = generated.sequences[:, prompt_length:]
    continuation_text = processor.batch_decode(continuation, skip_special_tokens=True)[
        0
    ].strip()
    raw_text = assistant_prefix + continuation_text
    parsed = parse_model_output(raw_text, prompt_config[modality]["parser"])
    return {
        "prompt_sha256": hashlib.sha256(chat.encode("utf-8")).hexdigest(),
        "assistant_prefix": assistant_prefix,
        "processor_input_keys": sorted(inputs),
        "processor_input_shapes": {
            key: list(value.shape)
            for key, value in inputs.items()
            if hasattr(value, "shape")
        },
        "prefill_tokens": prompt_length,
        "raw_output": raw_text,
        "parsed_output": parsed,
        "preprocess_seconds": preprocess_seconds,
        "transfer_seconds": transfer_seconds,
        "generate_seconds": generate_seconds,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(cuda_device)),
    }


def _attempt_record(
    base: dict[str, Any],
    state: TerminalState,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {**base, "terminal_state": state.value, "result": result, "error": error}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-scope", type=Path, required=True)
    parser.add_argument("--development-video-manifest", type=Path, required=True)
    parser.add_argument("--model-index", type=Path, required=True)
    parser.add_argument("--gpu-admission-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config, config_file_hash = _validate_execution_config(args.config)
    if str(args.output_root) != config["runtime"]["g2_output_root"]:
        raise ValueError("g2_output_root_mismatch")
    if (
        _sha(args.development_video_manifest)
        != config["runtime"]["development_video_manifest_sha256"]
    ):
        raise ValueError("development_video_manifest_hash_mismatch")
    gpu_admission = json.loads(args.gpu_admission_receipt.read_text(encoding="utf-8"))
    if (
        gpu_admission.get("verdict") != "ADMITTED"
        or time.time() - float(gpu_admission.get("observed_at_unix", 0)) > 300
    ):
        raise ValueError("gpu_admission_missing_or_stale")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(
        gpu_admission["selected_physical_index"]
    ):
        raise ValueError("cuda_visible_devices_does_not_match_admission")
    gpu_admission_hash = _sha(args.gpu_admission_receipt)
    image_scope = json.loads(args.image_scope.read_text(encoding="utf-8"))
    image_samples = {
        item["image"]: item
        for item in image_scope["samples"]
        if item["tier"] == "development"
    }
    development = json.loads(
        args.development_video_manifest.read_text(encoding="utf-8")
    )
    model_index = _validate_model_index(
        args.model_index, config["runtime"]["model_cache_index_sha256"]
    )
    source_video = Path(development["source_path"])
    source_frames = decode_video_rgb(source_video)
    guided = development["derivatives"]["codec_guided_crf18"]
    bitrate = development["derivatives"]["bitrate_only_crf32"]
    codec_features = load_candidate_features(guided["feature_path"])
    bitrate_frames = decode_video_rgb(bitrate["path"])
    store = AttemptArtifactStore(args.output_root)
    recovery = store.recover()
    if recovery["fail_closed"]:
        raise RuntimeError("attempt_store_recovery_failed")
    existing = {
        json.loads(path.read_text(encoding="utf-8"))["unit_id"]
        for path in store.attempts.glob("*.json")
    }
    smoke_units = config["smoke_units"]
    if len(smoke_units) != 36 or len({unit["unit_id"] for unit in smoke_units}) != 36:
        raise ValueError("smoke_identity_cardinality_invalid")
    layers = {layer["model_key"]: layer for layer in config["model_layers"]}
    scheduler = SingleScheduler(args.output_root)
    scheduler.acquire("evidencegate:g2-smoke")
    cap_seconds = float(config["limits"]["gpu_hour_hard_cap"]) * 3600
    inference_timeout = float(config["runtime"]["smoke_inference_timeout_seconds"])
    load_timeout = float(config["runtime"]["model_load_timeout_seconds"])
    try:
        for model_key in sorted(layers):
            pending = [
                unit
                for unit in smoke_units
                if unit["model_key"] == model_key and unit["unit_id"] not in existing
            ]
            for unit in [item for item in pending if item.get("fault_mode")]:
                fault = unit["fault_mode"]
                base = {
                    "unit_id": unit["unit_id"],
                    "attempt_id": f"g2-{unit['unit_id']}",
                    "retry_index": 0,
                    "config_file_sha256": config_file_hash,
                    "gpu_admission_receipt_sha256": gpu_admission_hash,
                    **unit,
                    "fault_injected": True,
                }
                store.write(
                    _attempt_record(
                        base, FAULT_EXPECTED[fault], error=f"injected:{fault}"
                    ),
                    attempt_id=base["attempt_id"],
                )
                existing.add(unit["unit_id"])
            normal = [unit for unit in pending if not unit.get("fault_mode")]
            if not normal:
                continue
            load_started = time.perf_counter()
            scheduler.reserve_gpu(load_timeout, cap_seconds, f"model-load-{model_key}")
            try:
                model, processor, load_receipt = _load_layer(
                    layers[model_key], model_index, args.output_root
                )
                scheduler.settle_gpu(time.perf_counter() - load_started)
            except Exception as exc:
                scheduler.settle_gpu(time.perf_counter() - load_started)
                for unit in normal:
                    base = {
                        "unit_id": unit["unit_id"],
                        "attempt_id": f"g2-{unit['unit_id']}",
                        "retry_index": 0,
                        "config_file_sha256": config_file_hash,
                        "gpu_admission_receipt_sha256": gpu_admission_hash,
                        **unit,
                        "fault_injected": False,
                    }
                    store.write(
                        _attempt_record(
                            base,
                            TerminalState.CRASH,
                            error=f"model_load_failed:{type(exc).__name__}",
                        ),
                        attempt_id=base["attempt_id"],
                    )
                continue
            for unit in normal:
                started = time.time()
                base = {
                    "unit_id": unit["unit_id"],
                    "attempt_id": f"g2-{unit['unit_id']}",
                    "retry_index": 0,
                    "config_file_sha256": config_file_hash,
                    "gpu_admission_receipt_sha256": gpu_admission_hash,
                    "model_load_artifact_sha256": load_receipt["artifact_sha256"],
                    **unit,
                    "fault_injected": False,
                    "started_at_unix": started,
                }
                scheduler.reserve_gpu(
                    inference_timeout, cap_seconds, base["attempt_id"]
                )
                attempt_started = time.perf_counter()
                try:
                    if unit["modality"] == "image":
                        sample = image_samples[unit["asset_id"]]
                        plugin = prepare_image_input(sample, unit["condition"])
                        question = sample["text"]
                        expected = sample["label"].lower() == "yes"
                    else:
                        plugin = prepare_video_input(
                            source_frames,
                            unit["condition"],
                            source_sha256=development["source_sha256"],
                            codec_features=codec_features,
                            bitrate_frames=bitrate_frames,
                            original_encoded_bytes=source_video.stat().st_size,
                            reduced_encoded_bytes=bitrate["bytes"],
                            window_id=unit["asset_id"],
                        )
                        question = None
                        expected = None
                    result = _generate(
                        model,
                        processor,
                        plugin,
                        config["prompts"],
                        unit["modality"],
                        question,
                        inference_timeout,
                    )
                    result["plugin"] = plugin.to_dict()
                    result["expected"] = expected
                    result["correct"] = (
                        result["parsed_output"] == expected
                        if expected is not None
                        else None
                    )
                    state, error = TerminalState.SUCCESS, None
                except ValueError as exc:
                    result, state, error = (
                        None,
                        TerminalState.INVALID,
                        f"{type(exc).__name__}:{exc}",
                    )
                except TimeoutError as exc:
                    result, state, error = (
                        None,
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
                    result, error = None, f"{type(exc).__name__}:{exc}"
                elapsed = time.perf_counter() - attempt_started
                scheduler.settle_gpu(elapsed)
                base["ended_at_unix"] = time.time()
                base["gpu_wall_seconds"] = elapsed
                store.write(
                    _attempt_record(base, state, result=result, error=error),
                    attempt_id=base["attempt_id"],
                )
                print(
                    json.dumps(
                        {
                            "completed": len(existing) + 1,
                            "unit_id": unit["unit_id"],
                            "state": state.value,
                        }
                    ),
                    flush=True,
                )
                existing.add(unit["unit_id"])
            del model, processor
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
    finally:
        scheduler.release()
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(store.attempts.glob("*.json"))
    ]
    relevant = [
        record
        for record in records
        if record["unit_id"] in {unit["unit_id"] for unit in smoke_units}
    ]
    snapshot = write_content_addressed_jsonl(args.output_root, relevant)
    expected_states = {
        unit["unit_id"]: FAULT_EXPECTED[unit["fault_mode"]].value
        if unit.get("fault_mode")
        else TerminalState.SUCCESS.value
        for unit in smoke_units
    }
    actual_states = {record["unit_id"]: record["terminal_state"] for record in relevant}
    unexpected = sorted(
        unit_id
        for unit_id, expected in expected_states.items()
        if actual_states.get(unit_id) != expected
    )
    summary = {
        "schema_version": "evidencegate.g2.smoke-summary.v1",
        "config_file_sha256": config_file_hash,
        "gpu_admission_receipt_sha256": gpu_admission_hash,
        "selected_physical_index": gpu_admission["selected_physical_index"],
        "selected_gpu_uuid": gpu_admission["selected_gpu_uuid"],
        "attempted_units": len(relevant),
        "terminal_coverage": len(actual_states),
        "call_cap": 36,
        "unexpected_terminal_units": unexpected,
        "snapshot_path": str(snapshot),
        "snapshot_sha256": snapshot.stem,
        "gpu_seconds_consumed": scheduler.consumed_seconds,
        "verdict": "PASS"
        if len(relevant) == 36 and len(actual_states) == 36 and not unexpected
        else "FAIL",
    }
    write_immutable_manifest(args.output_root / "g2-smoke-summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
