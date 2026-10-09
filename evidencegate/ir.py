from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODALITIES = {"image", "video"}


def _nonnegative(name: str, value: int | float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}_must_be_numeric")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name}_must_be_nonnegative_finite")


def _hash(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value.lower()):
        raise ValueError(f"{name}_must_be_sha256")
    return value.lower()


@dataclass(frozen=True)
class VisualEvidenceIR:
    """Validated representation of one visual evidence item.

    Resource fields are intentionally separate: encoded bytes are not decoded
    pixels, and visual tokens are not prefill tokens.  ``stage`` may be a
    combined stage (for example ``vision+prefill``) and is therefore opaque.
    """

    modality: str
    source_id: str
    source_sha256: str
    derived_sha256: str
    derived_kind: str = "native"
    width: int = 0
    height: int = 0
    frame_index: int | None = None
    encoded_bytes: int = 0
    decoded_pixels: int = 0
    model_pixels: int = 0
    visual_tokens: int = 0
    prefill_tokens: int = 0
    vram_bytes: int = 0
    wall_clock_ms: float = 0.0
    stage: str = "input"

    def __post_init__(self) -> None:
        if self.modality not in _MODALITIES:
            raise ValueError("modality_must_be_image_or_video")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id_required")
        _hash(self.source_sha256, "source_sha256")
        _hash(self.derived_sha256, "derived_sha256")
        if not isinstance(self.derived_kind, str) or not self.derived_kind.strip():
            raise ValueError("derived_kind_required")
        for name in (
            "width", "height", "encoded_bytes", "decoded_pixels", "model_pixels",
            "visual_tokens", "prefill_tokens", "vram_bytes", "wall_clock_ms",
        ):
            _nonnegative(name, getattr(self, name))
        if isinstance(self.width, bool) or isinstance(self.height, bool):
            raise ValueError("dimensions_must_be_integers")
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise ValueError("dimensions_must_be_integers")
        if self.frame_index is not None:
            if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
                raise ValueError("frame_index_must_be_nonnegative_integer")
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("stage_required")

    @property
    def source_hash(self) -> str:
        return self.source_sha256

    @property
    def derived_hash(self) -> str:
        return self.derived_sha256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VisualEvidenceIR":
        data = dict(value)
        # Accept the concise hash spellings at ingestion, while serializing one
        # canonical schema.
        data.setdefault("source_sha256", data.pop("source_hash", None))
        data.setdefault("derived_sha256", data.pop("derived_hash", None))
        return cls(**data)


def validate_visual_evidence_ir(value: VisualEvidenceIR | Mapping[str, Any]) -> tuple[str, ...]:
    """Return sorted validation errors without replacing invalid values."""
    try:
        if isinstance(value, VisualEvidenceIR):
            value.__post_init__()
        elif isinstance(value, Mapping):
            VisualEvidenceIR.from_mapping(value)
        else:
            return ("not_visual_evidence_ir_or_mapping",)
    except (TypeError, ValueError) as error:
        return (str(error),)
    return ()


def hash_bytes(data: bytes) -> str:
    """Small helper for adapters constructing source/derived identities."""
    return hashlib.sha256(data).hexdigest()
