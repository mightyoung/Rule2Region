from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evidencegate.execution_config import compile_execution_config
from evidencegate.manifest import manifest_sha256, write_immutable_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate codec receipts and freeze the G1 execution config.")
    parser.add_argument("--execution-scope", type=Path, required=True)
    parser.add_argument("--video-scope", type=Path, required=True)
    parser.add_argument("--codec-manifest", type=Path, required=True)
    parser.add_argument("--query-provider-resolution", type=Path, required=True)
    parser.add_argument("--runtime-environment-receipt", type=Path, required=True)
    parser.add_argument("--development-video-manifest", type=Path, required=True)
    parser.add_argument("--model-cache-index", type=Path, required=True)
    parser.add_argument("--g2-output-index", type=Path, required=True)
    parser.add_argument("--prompt-adjustment-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from evidencegate.query_provider import validate_query_provider_resolution

    query_provider_resolution_sha256 = validate_query_provider_resolution(args.query_provider_resolution)
    config = compile_execution_config(
        args.execution_scope,
        query_provider_resolution_sha256=query_provider_resolution_sha256,
        runtime_environment_receipt_sha256=hashlib.sha256(args.runtime_environment_receipt.read_bytes()).hexdigest(),
        development_video_manifest_sha256=hashlib.sha256(args.development_video_manifest.read_bytes()).hexdigest(),
        model_cache_index_sha256=hashlib.sha256(args.model_cache_index.read_bytes()).hexdigest(),
        g2_output_index_sha256=hashlib.sha256(args.g2_output_index.read_bytes()).hexdigest(),
        prompt_adjustment_receipt_sha256=hashlib.sha256(args.prompt_adjustment_receipt.read_bytes()).hexdigest(),
        codec_derivative_manifest_path=args.codec_manifest,
        video_scope_path=args.video_scope,
    )
    payload = config.to_dict()
    payload["execution_config_sha256"] = manifest_sha256(payload)
    artifact = write_immutable_manifest(args.output, payload)
    print(json.dumps({
        "output": artifact.path,
        "file_sha256": artifact.sha256,
        "bytes": artifact.bytes_written,
        "execution_config_sha256": payload["execution_config_sha256"],
        "latency_cells": len(config.latency_cells),
        "latency_attempts": sum(cell.warmup_attempts + cell.steady_attempts for cell in config.latency_cells),
        "smoke_units": len(config.smoke_units),
        "total_call_hard_cap": config.limits["total_call_hard_cap"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
