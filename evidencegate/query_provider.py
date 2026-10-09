from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .manifest import manifest_sha256


def global_center_provider_region(width: int, height: int) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("image_dimensions_must_be_positive")
    return min(width - 1, width // 2), min(height - 1, height // 2), 1, 1


def validate_query_provider_resolution(path: str | Path) -> str:
    source = Path(path)
    payload_bytes = source.read_bytes()
    payload = json.loads(payload_bytes)
    identity = str(payload.pop("resolution_sha256", ""))
    if len(identity) != 64 or manifest_sha256(payload) != identity:
        raise ValueError("query_provider_resolution_identity_mismatch")
    required = {
        "schema_version": "evidencegate.g1.query-provider-resolution.v1",
        "decision": "PREDECLARED_FALLBACK",
        "selected_provider": "global_center_crop50",
        "input_dependence": "image_dimensions_only",
        "extra_lmm_calls": 0,
        "formal_samples_may_select_provider": False,
        "ground_truth_used": False,
        "native_attention_verdict": "REJECTED_FOR_G2",
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("query_provider_resolution_contract_mismatch")
    if payload.get("reporting_label") != "global_center_crop50_fallback_not_query_aware":
        raise ValueError("query_provider_reporting_label_mismatch")
    return hashlib.sha256(payload_bytes).hexdigest()
