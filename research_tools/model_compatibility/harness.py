from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ModelSpec:
    family: str
    repo_id: str
    revision: str
    model_class: str
    vision_prefixes: tuple[str, ...]
    projector_prefixes: tuple[str, ...]
    language_prefixes: tuple[str, ...]
    nf4_skip_modules: tuple[str, ...]


FIXED_MODELS: dict[str, ModelSpec] = {
    "llava_onevision_05b": ModelSpec(
        family="llava_onevision_05b",
        repo_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        revision="74dd0bf867a4cda7950c17663794267c60cf4b40",
        model_class="LlavaOnevisionForConditionalGeneration",
        vision_prefixes=("vision_tower",),
        projector_prefixes=("multi_modal_projector",),
        language_prefixes=("language_model",),
        nf4_skip_modules=("vision_tower", "multi_modal_projector", "lm_head"),
    ),
    "qwen25_vl_3b": ModelSpec(
        family="qwen25_vl_3b",
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
        model_class="Qwen2_5_VLForConditionalGeneration",
        vision_prefixes=("visual", "model.visual"),
        projector_prefixes=("visual.merger", "model.visual.merger"),
        language_prefixes=("model.language_model", "language_model", "model"),
        nf4_skip_modules=("visual", "lm_head"),
    ),
}

REQUIRED_RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "family",
    "repo_id",
    "revision_requested",
    "revision_loaded",
    "precision",
    "input_manifest",
    "environment",
    "weight_manifest",
    "processor",
    "generation",
    "dtype_inventory",
    "peak_vram_bytes",
    "checks",
}

COUNT_RE = re.compile(r"\{\s*\"count\"\s*:\s*(-?\d+)\s*\}")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment(torch: Any) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "accelerate": _package_version("accelerate"),
        "bitsandbytes": _package_version("bitsandbytes"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "pillow": _package_version("pillow"),
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count() if cuda_available else 0)
        ],
    }


def _load_frames(frames_dir: Path, frame_count: int) -> tuple[Any, list[dict[str, Any]]]:
    from PIL import Image
    import numpy as np

    paths = sorted(
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if len(paths) < frame_count:
        raise ValueError(
            f"need at least {frame_count} image frames, found {len(paths)} in {frames_dir}"
        )
    selected = paths[:frame_count]
    arrays = []
    manifest = []
    expected_size: tuple[int, int] | None = None
    for index, path in enumerate(selected):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if expected_size is None:
                expected_size = rgb.size
            if rgb.size != expected_size:
                raise ValueError("all frames must have the same width and height")
            arrays.append(np.asarray(rgb))
            manifest.append(
                {
                    "index": index,
                    "name": path.name,
                    "sha256": _sha256(path),
                    "width": rgb.width,
                    "height": rgb.height,
                }
            )
    return np.stack(arrays, axis=0), manifest


def _snapshot_manifest(snapshot_dir: Path) -> list[dict[str, Any]]:
    allowed = {
        ".json",
        ".safetensors",
        ".model",
        ".txt",
        ".jinja",
        ".py",
    }
    files = sorted(
        path
        for path in snapshot_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    )
    if not any(path.suffix.lower() == ".safetensors" for path in files):
        raise RuntimeError("snapshot contains no safetensors weights")
    return [
        {
            "path": path.relative_to(snapshot_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]


def _module_bucket(name: str, spec: ModelSpec) -> str:
    if any(prefix in name for prefix in spec.projector_prefixes):
        return "projector"
    if any(prefix in name for prefix in spec.vision_prefixes):
        return "vision"
    if any(prefix in name for prefix in spec.language_prefixes):
        return "language"
    return "other"


def _dtype_inventory(model: Any, spec: ModelSpec) -> dict[str, Any]:
    inventory: dict[str, dict[str, int]] = {
        "vision": {},
        "projector": {},
        "language": {},
        "other": {},
    }
    quantized_modules: dict[str, list[str]] = {
        "vision": [],
        "projector": [],
        "language": [],
        "other": [],
    }
    for name, parameter in model.named_parameters():
        bucket = _module_bucket(name, spec)
        dtype = str(parameter.dtype)
        inventory[bucket][dtype] = inventory[bucket].get(dtype, 0) + parameter.numel()
    for name, module in model.named_modules():
        if module.__class__.__name__ in {"Linear4bit", "Params4bit"}:
            quantized_modules[_module_bucket(name, spec)].append(name)
    return {
        "parameter_numel_by_dtype": inventory,
        "quantized_module_names": quantized_modules,
    }


def _tensor_dtypes(value: Any) -> list[str]:
    dtypes: set[str] = set()
    if hasattr(value, "dtype"):
        dtypes.add(str(value.dtype))
    elif isinstance(value, dict):
        for child in value.values():
            dtypes.update(_tensor_dtypes(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            dtypes.update(_tensor_dtypes(child))
    elif hasattr(value, "key_cache") and hasattr(value, "value_cache"):
        dtypes.update(_tensor_dtypes(value.key_cache))
        dtypes.update(_tensor_dtypes(value.value_cache))
    return sorted(dtypes)


def _activation_probe(
    model: Any,
    inputs: dict[str, Any],
    spec: ModelSpec,
    torch: Any,
) -> tuple[dict[str, Any], list[str]]:
    selected: dict[str, tuple[str, Any]] = {}
    for name, module in model.named_modules():
        bucket = _module_bucket(name, spec)
        if bucket not in {"vision", "projector", "language"} or bucket in selected:
            continue
        if any(True for _ in module.parameters(recurse=False)) or name in {
            *spec.vision_prefixes,
            *spec.projector_prefixes,
            *spec.language_prefixes,
        }:
            selected[bucket] = (name, module)

    observed: dict[str, dict[str, Any]] = {}
    handles = []
    for bucket, (name, module) in selected.items():
        def hook(_module: Any, _args: Any, output: Any, *, key: str = bucket, module_name: str = name) -> None:
            dtypes = _tensor_dtypes(output)
            if dtypes:
                observed[key] = {"module": module_name, "dtypes": dtypes}

        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            output = model(**inputs, use_cache=True, return_dict=True)
        kv_dtypes = _tensor_dtypes(getattr(output, "past_key_values", None))
    finally:
        for handle in handles:
            handle.remove()
    del output
    torch.cuda.empty_cache()
    return observed, kv_dtypes


def _parse_count(text: str) -> int:
    match = COUNT_RE.search(text)
    if not match:
        raise ValueError("generation did not contain an exact JSON count object")
    value = int(match.group(1))
    if not 0 <= value <= 99:
        raise ValueError("parsed count is outside the frozen 0..99 contract")
    return value


def _validate_language_weight_nf4(inventory: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    quantized = inventory["quantized_module_names"]
    if not quantized["language"]:
        problems.append("no language module was observed as bitsandbytes Linear4bit")
    if quantized["vision"]:
        problems.append("vision modules were quantized; required language-weight NF4 only")
    if quantized["projector"]:
        problems.append("projector modules were quantized; required language-weight NF4 only")
    return problems


def validate_receipt(receipt: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    missing = sorted(REQUIRED_RECEIPT_FIELDS - receipt.keys())
    if missing:
        problems.append(f"missing receipt fields: {missing}")
    if receipt.get("status") != "passed":
        problems.append("receipt status is not passed")
    if receipt.get("family") not in FIXED_MODELS:
        problems.append("unknown or substituted model family")
    else:
        spec = FIXED_MODELS[receipt["family"]]
        if receipt.get("repo_id") != spec.repo_id:
            problems.append("repo_id does not match frozen contract")
        if receipt.get("revision_requested") != spec.revision:
            problems.append("revision_requested does not match frozen contract")
        if receipt.get("revision_loaded") != spec.revision:
            problems.append("loaded revision does not match requested frozen revision")
    if receipt.get("precision") not in {"bf16", "nf4"}:
        problems.append("precision is neither bf16 nor nf4")
    if not receipt.get("input_manifest"):
        problems.append("real multi-frame input manifest is empty")
    environment = receipt.get("environment") or {}
    if not environment.get("cuda_available"):
        problems.append("CUDA execution was not observed")
    if not receipt.get("weight_manifest"):
        problems.append("weight manifest is empty")
    processor = receipt.get("processor") or {}
    if not any("video" in key for key in processor.get("input_keys", [])):
        problems.append("native video processor tensor was not observed")
    generation = receipt.get("generation") or {}
    if not generation.get("parser_valid"):
        problems.append("generation parser did not pass")
    if not generation.get("kv_cache_dtypes"):
        problems.append("KV-cache dtype was not observed")
    dtype_inventory = receipt.get("dtype_inventory") or {}
    parameter_dtypes = dtype_inventory.get("parameter_numel_by_dtype") or {}
    for bucket in ("vision", "projector", "language"):
        if not parameter_dtypes.get(bucket):
            problems.append(f"{bucket} dtype inventory is empty")
    if receipt.get("peak_vram_bytes", 0) <= 0:
        problems.append("peak VRAM was not observed")
    checks = receipt.get("checks") or {}
    if not checks or not all(checks.values()):
        problems.append("one or more compatibility checks failed")
    return problems


def validate_matrix(receipts: Iterable[dict[str, Any]]) -> list[str]:
    receipt_list = list(receipts)
    expected = {
        (family, precision)
        for family in FIXED_MODELS
        for precision in ("bf16", "nf4")
    }
    observed = {
        (receipt.get("family"), receipt.get("precision")) for receipt in receipt_list
    }
    problems: list[str] = []
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected, key=str)
    if missing:
        problems.append(f"missing fixed compatibility cells: {missing}")
    if unexpected:
        problems.append(f"unexpected or substituted compatibility cells: {unexpected}")
    if len(receipt_list) != len(expected):
        problems.append("compatibility matrix must contain exactly four receipts")
    for receipt in receipt_list:
        cell = f"{receipt.get('family')}:{receipt.get('precision')}"
        problems.extend(f"{cell}: {problem}" for problem in validate_receipt(receipt))
    return problems


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, BitsAndBytesConfig
    import transformers

    spec = FIXED_MODELS[args.family]
    receipt: dict[str, Any] = {
        "schema_version": "codecchange.model-compatibility.v1",
        "status": "failed",
        "family": spec.family,
        "repo_id": spec.repo_id,
        "revision_requested": spec.revision,
        "revision_loaded": None,
        "precision": args.precision,
        "input_manifest": [],
        "environment": _environment(torch),
        "weight_manifest": [],
        "processor": {},
        "generation": {},
        "dtype_inventory": {},
        "peak_vram_bytes": 0,
        "checks": {},
        "started_at_unix": time.time(),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the BF16/NF4 compatibility gate")
    if args.precision == "nf4":
        for package in ("bitsandbytes", "accelerate"):
            if _package_version(package) is None:
                raise RuntimeError(f"{package} is required for NF4 compatibility")

    frames, input_manifest = _load_frames(args.frames_dir, args.frame_count)
    receipt["input_manifest"] = input_manifest
    snapshot_dir = Path(
        snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            local_files_only=not args.allow_download,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "*.txt",
                "*.jinja",
                "*.py",
                "*.tiktoken",
                "*.yaml",
                "*.yml",
            ],
            ignore_patterns=[
                "*.onnx",
                "onnx/**",
                "*.gguf",
                "*.h5",
                "*.msgpack",
                "*.ot",
            ],
        )
    )
    receipt["revision_loaded"] = snapshot_dir.name
    if snapshot_dir.name != spec.revision:
        raise RuntimeError("resolved snapshot does not match the frozen revision")
    receipt["weight_manifest"] = _snapshot_manifest(snapshot_dir)

    processor = AutoProcessor.from_pretrained(
        snapshot_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    receipt["processor"] = {
        "class": processor.__class__.__name__,
        "commit_hash": getattr(processor, "_commit_hash", None),
    }

    model_cls = getattr(transformers, spec.model_class)
    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "device_map": "auto",
    }
    if args.precision == "bf16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=list(spec.nf4_skip_modules),
        )
        load_kwargs["torch_dtype"] = torch.bfloat16

    torch.cuda.reset_peak_memory_stats()
    model = model_cls.from_pretrained(snapshot_dir, **load_kwargs)
    model.eval()
    receipt["dtype_inventory"] = _dtype_inventory(model, spec)
    receipt["dtype_inventory"]["nf4_skip_modules"] = (
        list(spec.nf4_skip_modules) if args.precision == "nf4" else []
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {
                    "type": "text",
                    "text": (
                        'Count visible people across these frames. Return exactly '
                        '{"count": <integer from 0 to 99>}.'
                    ),
                },
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    assistant_prefix = '{"count": '
    prompt += assistant_prefix
    inputs = processor(
        text=[prompt],
        videos=[frames],
        padding=True,
        return_tensors="pt",
    )
    cuda_device = next(
        parameter.device
        for parameter in model.parameters()
        if parameter.device.type == "cuda"
    )
    inputs = {
        name: tensor.to(cuda_device) if hasattr(tensor, "to") else tensor
        for name, tensor in inputs.items()
    }
    receipt["processor"].update(
        {
            "input_keys": sorted(inputs),
            "input_dtypes": {name: _tensor_dtypes(value) for name, value in inputs.items()},
            "video_shape": list(frames.shape),
            "prompt_contract": "json_count_v2_assistant_prefix",
            "assistant_prefix": assistant_prefix,
        }
    )
    activation_dtypes, kv_cache_dtypes = _activation_probe(
        model,
        inputs,
        spec,
        torch,
    )
    receipt["dtype_inventory"]["activation_dtypes"] = activation_dtypes
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )
    prompt_length = inputs["input_ids"].shape[-1]
    continuation = generated.sequences[:, prompt_length:]
    continuation_text = processor.batch_decode(
        continuation,
        skip_special_tokens=True,
    )[0].strip()
    text = assistant_prefix + continuation_text
    parser_error: str | None = None
    try:
        parsed_count = _parse_count(text)
        parser_valid = True
    except ValueError as exc:
        parsed_count = None
        parser_valid = False
        parser_error = str(exc)
    receipt["generation"] = {
        "raw_text": text,
        "parsed_count": parsed_count,
        "parser_valid": parser_valid,
        "parser_error": parser_error,
        "sequence_shape": list(generated.sequences.shape),
        "kv_cache_dtypes": kv_cache_dtypes,
    }
    receipt["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())

    inventory = receipt["dtype_inventory"]
    checks = {
        "fixed_revision": receipt["revision_loaded"] == spec.revision,
        "real_multiframe_processor": (
            frames.shape[0] == args.frame_count
            and any("video" in key for key in receipt["processor"]["input_keys"])
        ),
        "parser_valid": receipt["generation"]["parser_valid"],
        "kv_cache_dtype_observed": bool(receipt["generation"]["kv_cache_dtypes"]),
        "activation_dtypes_observed": all(
            bucket in receipt["dtype_inventory"]["activation_dtypes"]
            for bucket in ("vision", "projector", "language")
        ),
        "peak_vram_observed": receipt["peak_vram_bytes"] > 0,
        "weight_hashes_sealed": all(
            len(item.get("sha256", "")) == 64 for item in receipt["weight_manifest"]
        ),
    }
    if args.precision == "bf16":
        language_dtypes = inventory["parameter_numel_by_dtype"]["language"]
        checks["language_bf16_observed"] = language_dtypes.get("torch.bfloat16", 0) > 0
    else:
        nf4_problems = _validate_language_weight_nf4(inventory)
        receipt["nf4_validation_problems"] = nf4_problems
        checks["language_weight_nf4_only"] = not nf4_problems
    receipt["checks"] = checks
    receipt["status"] = "passed" if all(checks.values()) else "failed"
    receipt["finished_at_unix"] = time.time()
    return receipt


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fail-closed fixed-revision Video-LMM BF16/NF4 smoke."
    )
    parser.add_argument("--family", required=True, choices=sorted(FIXED_MODELS))
    parser.add_argument("--precision", required=True, choices=("bf16", "nf4"))
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--frame-count", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit download of the exact frozen snapshot; default is cache-only.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt: dict[str, Any]
    try:
        receipt = _run(args)
    except Exception as exc:
        spec = FIXED_MODELS[args.family]
        receipt = {
            "schema_version": "codecchange.model-compatibility.v1",
            "status": "failed",
            "family": spec.family,
            "repo_id": spec.repo_id,
            "revision_requested": spec.revision,
            "revision_loaded": None,
            "precision": args.precision,
            "input_manifest": [],
            "environment": {},
            "weight_manifest": [],
            "processor": {},
            "generation": {"parser_valid": False},
            "dtype_inventory": {},
            "peak_vram_bytes": 0,
            "checks": {},
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "finished_at_unix": time.time(),
        }
    _write_receipt(args.receipt, receipt)
    problems = validate_receipt(receipt)
    if problems:
        for problem in problems:
            print(f"BLOCKED: {problem}", file=sys.stderr)
        return 2
    print(f"VERIFIED: {args.family} {args.precision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
