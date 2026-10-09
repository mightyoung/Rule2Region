from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import sha256_file

SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")

REQUIRED_TEXT_FIELDS = (
    "source_video_id",
    "agreement_version",
    "agreement_signed_at",
    "release_id",
    "archive_path",
    "archive_sha256",
    "video_path",
    "video_sha256",
    "annotation_revision",
    "annotation_path",
    "annotation_sha256",
    "split",
    "scene_id",
    "timeline_mapping_version",
    "timeline_mapping_path",
    "timeline_mapping_sha256",
)

REQUIRED_PERMISSION_FIELDS = (
    "cloud_processing_allowed",
    "derivative_processing_allowed",
    "redistribution_prohibited_acknowledged",
)


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    errors: tuple[str, ...]


def validate_admission_receipt(
    receipt: dict[str, Any],
    *,
    verify_files: bool = True,
    base_dir: str | Path | None = None,
) -> AdmissionDecision:
    if not isinstance(receipt, dict):
        return AdmissionDecision(False, ("receipt_not_object",))
    errors: list[str] = []
    for field in REQUIRED_TEXT_FIELDS:
        value = receipt.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"missing_or_invalid:{field}")

    for field in REQUIRED_PERMISSION_FIELDS:
        if receipt.get(field) is not True:
            errors.append(f"permission_not_confirmed:{field}")

    for field in (
        "archive_sha256",
        "video_sha256",
        "annotation_sha256",
        "timeline_mapping_sha256",
    ):
        value = receipt.get(field)
        if isinstance(value, str) and not SHA256_PATTERN.fullmatch(value):
            errors.append(f"invalid_sha256:{field}")

    numeric_contracts = {
        "video_time_base_seconds": lambda value: value > 0,
        "annotation_time_base_seconds": lambda value: value > 0,
        "timeline_offset_seconds": lambda value: True,
        "video_duration_seconds": lambda value: value > 0,
        "annotation_duration_seconds": lambda value: value > 0,
        "source_fps": lambda value: value > 0,
        "timeline_max_error_seconds": lambda value: value >= 0,
    }
    for field, predicate in numeric_contracts.items():
        value = receipt.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not predicate(float(value))
        ):
            errors.append(f"missing_or_invalid:{field}")

    fps = receipt.get("source_fps")
    max_error = receipt.get("timeline_max_error_seconds")
    if (
        isinstance(fps, (int, float))
        and not isinstance(fps, bool)
        and isinstance(max_error, (int, float))
        and not isinstance(max_error, bool)
        and fps > 0
        and max_error > 0.5 / fps + 1e-12
    ):
        errors.append("timeline_error_exceeds_half_frame")

    if verify_files:
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        for path_field, hash_field in (
            ("archive_path", "archive_sha256"),
            ("video_path", "video_sha256"),
            ("annotation_path", "annotation_sha256"),
            ("timeline_mapping_path", "timeline_mapping_sha256"),
        ):
            path_value = receipt.get(path_field)
            expected_hash = receipt.get(hash_field)
            if not isinstance(path_value, str):
                continue
            path = Path(path_value)
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                errors.append(f"file_not_found:{path_field}")
                continue
            if isinstance(expected_hash, str) and SHA256_PATTERN.fullmatch(
                expected_hash
            ):
                try:
                    actual_hash = sha256_file(path).lower()
                except OSError as error:
                    errors.append(
                        f"file_unreadable:{path_field}:{error.__class__.__name__}"
                    )
                else:
                    if actual_hash != expected_hash.lower():
                        errors.append(f"hash_mismatch:{path_field}")

        mapping_value = receipt.get("timeline_mapping_path")
        if isinstance(mapping_value, str):
            mapping_path = Path(mapping_value)
            if not mapping_path.is_absolute():
                mapping_path = root / mapping_path
            if mapping_path.is_file():
                try:
                    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    errors.append(
                        f"timeline_mapping_invalid:{error.__class__.__name__}"
                    )
                else:
                    if not isinstance(mapping, dict):
                        errors.append("timeline_mapping_not_object")
                    else:
                        for field in (
                            "source_video_id",
                            "timeline_mapping_version",
                        ):
                            if mapping.get(field) != receipt.get(field):
                                errors.append(f"timeline_mapping_mismatch:{field}")
                        for field in numeric_contracts:
                            mapped = mapping.get(field)
                            expected = receipt.get(field)
                            if (
                                isinstance(mapped, bool)
                                or not isinstance(mapped, (int, float))
                                or isinstance(expected, bool)
                                or not isinstance(expected, (int, float))
                                or not math.isclose(
                                    float(mapped),
                                    float(expected),
                                    rel_tol=0.0,
                                    abs_tol=1e-12,
                                )
                            ):
                                errors.append(f"timeline_mapping_mismatch:{field}")

    return AdmissionDecision(admitted=not errors, errors=tuple(sorted(set(errors))))
