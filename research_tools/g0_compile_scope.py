from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from evidencegate import ScopeManifest, VideoSample, compile_matrix, write_immutable_manifest


MODELS = {
    "0.5b-bf16": {
        "model_id": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "revision": "74dd0bf867a4cda7950c17663794267c60cf4b40",
        "precision": "bf16",
        "role": "primary_full_matrix",
    },
    "0.5b-nf4": {
        "model_id": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "revision": "74dd0bf867a4cda7950c17663794267c60cf4b40",
        "precision": "language_weight_nf4",
        "role": "deployment_subset",
    },
    "3b-bf16": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "precision": "bf16",
        "role": "deployment_subset",
    },
    "3b-nf4": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "precision": "language_weight_nf4",
        "role": "deployment_subset",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sealed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest_not_object:{path}")
    stored = value.get("manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    actual = hashlib.sha256(
        (json.dumps(unsigned, ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()
    if stored != actual:
        raise ValueError(f"manifest_hash_mismatch:{path}")
    return value


def build_scope(video: dict[str, Any], image: dict[str, Any]) -> ScopeManifest:
    video_samples = tuple(
        VideoSample(
            sample["sample_id"],
            sample["sequence_id"],
            sample["camera_motion_stratum"],
        )
        for sample in video["samples"]
    )
    formal_images = tuple(
        sample["image"] for sample in image["samples"] if sample["tier"] == "formal"
    )
    video_deployment = tuple(
        sample["sample_id"]
        for sample in video["samples"]
        if sample["deployment_member"]
    )
    image_deployment = tuple(
        sample["image"]
        for sample in image["samples"]
        if sample["deployment_member"]
    )
    development_images = tuple(
        sample["image"]
        for sample in image["samples"]
        if sample["tier"] == "development"
    )
    return ScopeManifest(
        video_samples,
        formal_images,
        video_deployment,
        image_deployment,
        development_images,
        "/srv/project-assets/evidencegate-v2/development/video/cfr-bframes.mp4",
        "515d5077f6b5d190c846989d987cd2e3e8932c60aa460c1bf3735237e6142c07",
        video["manifest_sha256"],
        image["manifest_sha256"],
    )


def compile_scope_manifest(video_path: Path, image_path: Path) -> dict[str, Any]:
    video = load_sealed(video_path)
    image = load_sealed(image_path)
    scope = build_scope(video, image)
    matrix = compile_matrix("BROAD", scope)
    if matrix.total_calls != 2107:
        raise AssertionError("broad_hard_cap_must_be_2107")
    success_minimum = math.ceil(matrix.quality_count * 0.95)
    if success_minimum != 937:
        raise AssertionError("quality_success_minimum_must_be_937")
    return {
        "schema_version": "evidencegate.g0.execution-scope.v1",
        "scope_verdict": "BROAD",
        "seed": 20260810,
        "source_manifests": {
            "video": {
                "path": str(video_path.resolve()),
                "file_sha256": sha256_file(video_path),
                "manifest_sha256": video["manifest_sha256"],
            },
            "image": {
                "path": str(image_path.resolve()),
                "file_sha256": sha256_file(image_path),
                "manifest_sha256": image["manifest_sha256"],
            },
        },
        "scope_manifest": scope.to_dict(),
        "models": MODELS,
        "prompts": {
            "video_count": (
                "Count the maximum number of visible people present at the same "
                "time across this visual evidence. Respond exactly as JSON: "
                '{"count": <non-negative integer>}.'
            ),
            "pope_yes_no": (
                "{pope_question}\nRespond exactly as JSON using one lowercase "
                'answer: {"answer": "yes"} or {"answer": "no"}.'
            ),
        },
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 32,
            "seed": 20260810,
            "parser_policy": "strict_json_no_default_value",
        },
        "conditions": {
            "video_primary": [
                "dense",
                "uniform50",
                "random50",
                "rgb_difference50",
                "codec_guided50",
                "bitrate_only",
                "resize50",
            ],
            "image_primary": [
                "native",
                "jpeg",
                "resize50",
                "random_crop50",
                "query_aware_crop50",
            ],
            "video_deployment": ["native", "reduced", "jpeg", "resize50"],
            "image_deployment": ["native", "reduced", "jpeg"],
        },
        "matrix": matrix.to_dict(),
        "gates": {
            "quality_logic_units": 986,
            "quality_success_minimum": success_minimum,
            "quality_terminal_coverage_required": 986,
            "latency_cells": 64,
            "latency_attempts": 896,
            "smoke_call_cap": 36,
            "retry_attempt_budget": 189,
            "total_call_hard_cap": 2107,
            "gpu_hour_recommended": 12,
            "gpu_hour_hard_cap": 24,
        },
        "claim_boundaries": {
            "static_primary_windows": 17,
            "moving_camera_stress_windows": 33,
            "pooling_for_primary_claims": False,
            "primary_video_estimand": "sequence_equal_within_STATIC_PRIMARY",
            "stress_video_estimand": "sequence_equal_within_MOVING_CAMERA_STRESS",
            "general_statistical_significance_claim": False,
            "unseen_camera_generalization_claim": False,
            "official_mot17_or_coco_archive_hash_match_claim": False,
        },
        "development_isolation": {
            "video_fixture_sha256": "515d5077f6b5d190c846989d987cd2e3e8932c60aa460c1bf3735237e6142c07",
            "image_samples": 12,
            "formal_samples_may_be_used_for_tuning": False,
        },
        "storage": {
            "remote_root": "/srv/project-assets/evidencegate-v2",
            "shared_nfs_io_must_be_separately_measured": True,
            "latency_input_cache_policy": "freeze and record either local staging or explicit NFS read stage",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile EvidenceGate G0 scope.")
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    value = compile_scope_manifest(arguments.video_manifest, arguments.image_manifest)
    artifact = write_immutable_manifest(arguments.output, value)
    print(
        json.dumps(
            {
                "output": artifact.path,
                "sha256": artifact.sha256,
                "bytes": artifact.bytes_written,
                "scope": value["scope_verdict"],
                "quality": value["gates"]["quality_logic_units"],
                "total_calls": value["gates"]["total_call_hard_cap"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
