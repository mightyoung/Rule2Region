from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _number(name: str, value: int | float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name}_must_be_nonnegative_finite")


@dataclass(frozen=True)
class ResourceRecord:
    stage: str
    encoded_bytes: int = 0
    decoded_pixels: int = 0
    model_pixels: int = 0
    visual_tokens: int = 0
    prefill_tokens: int = 0
    vram_bytes: int = 0
    wall_clock_ms: float = 0.0
    combined: bool = False

    def __post_init__(self) -> None:
        if not self.stage.strip():
            raise ValueError("stage_required")
        for field in ("encoded_bytes", "decoded_pixels", "model_pixels", "visual_tokens", "prefill_tokens", "vram_bytes", "wall_clock_ms"):
            _number(field, getattr(self, field))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IOReceipt:
    stage: str
    source_path: str
    source_sha256: str
    storage_class: str
    cache_state: str
    bytes_read: int
    wall_clock_ms: float
    included_in_latency: bool
    staging_path: str | None = None
    staging_sha256: str | None = None
    target_path: str | None = None
    target_sha256: str | None = None
    bytes_written: int = 0
    started_at: str | None = None
    ended_at: str | None = None
    atomic_seal_state: str = "not_applicable"
    error: str | None = None
    attempt_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.stage.strip() or not self.source_path.strip():
            raise ValueError("io_stage_and_source_path_required")
        if not _SHA256.fullmatch(self.source_sha256.lower()):
            raise ValueError("source_sha256_must_be_sha256")
        if self.storage_class not in {"nfs", "local", "memory"}:
            raise ValueError("invalid_storage_class")
        if self.cache_state not in {"cold", "warm", "staged"}:
            raise ValueError("invalid_cache_state")
        _number("bytes_read", self.bytes_read)
        _number("wall_clock_ms", self.wall_clock_ms)
        if (self.staging_path is None) != (self.staging_sha256 is None):
            raise ValueError("staging_path_and_hash_must_be_paired")
        if self.staging_sha256 is not None and not _SHA256.fullmatch(
            self.staging_sha256.lower()
        ):
            raise ValueError("staging_sha256_must_be_sha256")

    def validate_output_receipt(self) -> None:
        if not self.target_path or not self.target_sha256:
            raise ValueError("output_target_receipt_required")
        if not _SHA256.fullmatch(self.target_sha256.lower()):
            raise ValueError("target_sha256_must_be_sha256")
        if self.attempt_artifact_sha256 is not None and not _SHA256.fullmatch(self.attempt_artifact_sha256.lower()):
            raise ValueError("attempt_artifact_sha256_must_be_sha256")
        _number("bytes_written", self.bytes_written)
        if not self.started_at or not self.ended_at:
            raise ValueError("output_timestamps_required")
        if self.atomic_seal_state not in {"sealed", "failed"}:
            raise ValueError("output_atomic_terminal_state_required")
        if self.atomic_seal_state == "sealed" and self.error:
            raise ValueError("sealed_output_cannot_have_error")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StageResourceLedger:
    def __init__(self) -> None:
        self._records: list[ResourceRecord] = []
        self._io_receipts: list[IOReceipt] = []

    def record(self, stage: str, **resources: int | float | bool) -> ResourceRecord:
        record = ResourceRecord(stage=stage, **resources)
        self._records.append(record)
        return record

    def add(self, record: ResourceRecord) -> None:
        if not isinstance(record, ResourceRecord):
            raise TypeError("record_must_be_resource_record")
        self._records.append(record)

    @property
    def records(self) -> tuple[ResourceRecord, ...]:
        return tuple(self._records)

    def record_io(self, receipt: IOReceipt) -> None:
        if not isinstance(receipt, IOReceipt):
            raise TypeError("receipt_must_be_io_receipt")
        self._io_receipts.append(receipt)

    @property
    def io_receipts(self) -> tuple[IOReceipt, ...]:
        return tuple(self._io_receipts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [record.to_dict() for record in self._records],
            "io_receipts": [receipt.to_dict() for receipt in self._io_receipts],
        }

    def totals(self) -> dict[str, int | float]:
        fields = ("encoded_bytes", "decoded_pixels", "model_pixels", "visual_tokens", "prefill_tokens", "vram_bytes", "wall_clock_ms")
        totals = {
            field: sum(getattr(record, field) for record in self._records)
            for field in fields
            if field != "vram_bytes"
        }
        totals["vram_bytes"] = max(
            (record.vram_bytes for record in self._records), default=0
        )
        return totals


ResourceLedger = StageResourceLedger
