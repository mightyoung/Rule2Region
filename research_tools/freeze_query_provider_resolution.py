from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegate.manifest import manifest_sha256, write_immutable_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the predeclared zero-call query-provider fallback.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": "evidencegate.g1.query-provider-resolution.v1",
        "decision": "PREDECLARED_FALLBACK",
        "selected_provider": "global_center_crop50",
        "reporting_label": "global_center_crop50_fallback_not_query_aware",
        "input_dependence": "image_dimensions_only",
        "provider_region_rule": "one_pixel_at_floor_center_then_expand_to_nearest_half_area_crop",
        "target_area_fraction": 0.5,
        "extra_lmm_calls": 0,
        "formal_samples_may_select_provider": False,
        "ground_truth_used": False,
        "native_attention_verdict": "REJECTED_FOR_G2",
        "native_attention_rejection_reasons": [
            "crop selection precedes model input but native attention exists only after the native input forward pass",
            "a second cropped forward pass would violate the frozen zero-extra-LMM-call contract",
            "charging the native provider pass would destroy latency and input-budget comparability"
        ],
        "claim_boundary": "Results must be reported as deterministic center-crop fallback, not query-aware attention selection.",
    }
    payload["resolution_sha256"] = manifest_sha256(payload)
    artifact = write_immutable_manifest(args.output, payload)
    print(json.dumps({
        "output": artifact.path,
        "file_sha256": artifact.sha256,
        "resolution_sha256": payload["resolution_sha256"],
        "selected_provider": payload["selected_provider"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
