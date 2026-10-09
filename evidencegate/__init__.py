"""Small, auditable EvidenceGate core contracts.

The package deliberately contains data contracts and deterministic planning
logic only.  Model execution and dataset adapters remain outside this core.
"""

from .ir import VisualEvidenceIR, validate_visual_evidence_ir
from .scope import ScopeVerdict, decide_scope, scope_verdict
from .scope_manifest import ScopeManifest, VideoSample
from .matrix import (
    ExperimentUnit,
    MatrixPlan,
    compile_matrix,
    compile_experiment_matrix,
)
from .terminals import (
    AttemptRecord,
    TerminalRecord,
    TerminalState,
    aggregate_terminals,
)
from .ledger import IOReceipt, ResourceLedger, ResourceRecord, StageResourceLedger
from .manifest import (
    ManifestArtifact,
    ManifestConflictError,
    canonical_json,
    manifest_sha256,
    write_immutable_manifest,
)
from .plugins import (
    PluginResult,
    image_native,
    image_jpeg,
    image_resize50,
    image_random_crop50,
    image_query_aware_crop50,
    image_center_crop50,
    video_select,
    video_resize50,
    video_bitrate_only,
    image_resize_50,
    image_random_crop_50,
    image_query_aware_crop_50,
    image_center_crop_50,
    video_bitrate,
)
from .execution_config import (
    ExecutionConfig,
    LatencyCell,
    ModelLayerIdentity,
    SmokeUnit,
    canonical_quantization_config,
    compile_execution_config,
    compile_latency_cells,
    compile_smoke_units,
    model_layer_identity,
)
from .codec_manifest import validate_codec_derivative_manifest
from .query_provider import global_center_provider_region, validate_query_provider_resolution

__all__ = [
    "VisualEvidenceIR",
    "validate_visual_evidence_ir",
    "ScopeVerdict",
    "decide_scope",
    "scope_verdict",
    "ScopeManifest",
    "VideoSample",
    "ExperimentUnit",
    "MatrixPlan",
    "compile_matrix",
    "compile_experiment_matrix",
    "AttemptRecord",
    "TerminalRecord",
    "TerminalState",
    "aggregate_terminals",
    "ResourceLedger",
    "StageResourceLedger",
    "ResourceRecord",
    "IOReceipt",
    "ManifestArtifact",
    "ManifestConflictError",
    "canonical_json",
    "manifest_sha256",
    "write_immutable_manifest",
    "PluginResult",
    "image_native",
    "image_jpeg",
    "image_resize50",
    "image_random_crop50",
    "image_query_aware_crop50",
    "image_center_crop50",
    "video_select",
    "video_resize50",
    "video_bitrate_only",
    "image_resize_50",
    "image_random_crop_50",
    "image_query_aware_crop_50",
    "image_center_crop_50",
    "video_bitrate",
    "ExecutionConfig",
    "LatencyCell",
    "ModelLayerIdentity",
    "SmokeUnit",
    "canonical_quantization_config",
    "compile_execution_config",
    "compile_latency_cells",
    "compile_smoke_units",
    "model_layer_identity",
    "validate_codec_derivative_manifest",
    "global_center_provider_region",
    "validate_query_provider_resolution",
]
