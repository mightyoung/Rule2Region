"""Dataset adapters with fail-closed parsing contracts."""

from .industrial_g0 import (
    DatasetAdmission,
    IndustrialFrameRecord,
    PersonBox,
    PilotVideo,
    build_pilot_manifest,
    select_evenly_spaced_frames,
)
from .virat_release2 import parse_object_lines

__all__ = [
    "DatasetAdmission",
    "IndustrialFrameRecord",
    "PersonBox",
    "PilotVideo",
    "build_pilot_manifest",
    "parse_object_lines",
    "select_evenly_spaced_frames",
]
