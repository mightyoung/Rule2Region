from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from .scope import ScopeVerdict
from .scope_manifest import ScopeManifest


@dataclass(frozen=True)
class ExperimentUnit:
    unit_id: str
    family: str
    modality: str
    sample_id: str
    condition: str
    model: str = "0.5b-bf16"
    sequence_id: str | None = None
    camera_motion_stratum: str | None = None
    deployment_member: bool = False
    attempt_index: int = 0
    retry_of: str | None = None


@dataclass(frozen=True)
class MatrixPlan:
    scope: ScopeVerdict
    quality_units: tuple[ExperimentUnit, ...]
    latency_attempts: tuple[ExperimentUnit, ...]
    smoke_units: tuple[ExperimentUnit, ...]
    retry_budget: int

    @property
    def quality_count(self) -> int:
        return len(self.quality_units)

    @property
    def latency_count(self) -> int:
        return len(self.latency_attempts)

    @property
    def smoke_count(self) -> int:
        return len(self.smoke_units)

    @property
    def retry_count(self) -> int:
        return self.retry_budget

    @property
    def total_calls(self) -> int:
        return self.quality_count + self.latency_count + self.smoke_count + self.retry_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.value,
            "quality_count": self.quality_count,
            "latency_count": self.latency_count,
            "smoke_count": self.smoke_count,
            "retry_count": self.retry_count,
            "total_calls": self.total_calls,
            "quality_units": [u.__dict__ for u in self.quality_units],
            "latency_attempts": [u.__dict__ for u in self.latency_attempts],
            "smoke_units": [u.__dict__ for u in self.smoke_units],
            "retry_budget": self.retry_budget,
        }


def _unit(
    family: str,
    modality: str,
    sample: str,
    condition: str,
    model: str = "0.5b-bf16",
    *,
    sequence_id: str | None = None,
    camera_motion_stratum: str | None = None,
    deployment_member: bool = False,
    index: int = 0,
    retry_of: str | None = None,
) -> ExperimentUnit:
    # Attempt and retry identity are execution metadata, not logical unit
    # identity.  In particular, all 14 latency attempts share one cell ID.
    identity_index = None if family == "latency" else index
    identity_condition = "attempts" if family == "latency" else condition
    material = json.dumps([family, modality, sample, identity_condition, model, identity_index], separators=(",", ":"))
    digest = hashlib.sha256(material.encode()).hexdigest()[:16]
    return ExperimentUnit(
        f"{family}:{digest}",
        family,
        modality,
        sample,
        condition,
        model,
        sequence_id,
        camera_motion_stratum,
        deployment_member,
        index,
        retry_of,
    )


def _quality(scope: ScopeVerdict, manifest: ScopeManifest) -> list[ExperimentUnit]:
    values: list[ExperimentUnit] = []
    if scope in (ScopeVerdict.BROAD, ScopeVerdict.VIDEO_ONLY):
        for sample in manifest.video_samples:
            for condition in ("dense", "uniform50", "random50", "rgb_difference50", "codec_guided50", "bitrate_only", "resize50"):
                values.append(
                    _unit(
                        "quality",
                        "video",
                        sample.sample_id,
                        condition,
                        sequence_id=sample.sequence_id,
                        camera_motion_stratum=sample.stratum,
                        deployment_member=sample.sample_id in manifest.video_deployment_ids,
                    )
                )
    if scope in (ScopeVerdict.BROAD, ScopeVerdict.IMAGE_ONLY):
        for sample in manifest.image_ids:
            for condition in ("native", "jpeg", "resize50", "random_crop50", "center_crop50"):
                values.append(
                    _unit(
                        "quality",
                        "image",
                        sample,
                        condition,
                        deployment_member=sample in manifest.image_deployment_ids,
                    )
                )
    if scope is ScopeVerdict.BROAD:
        for model in ("0.5b-nf4", "3b-bf16", "3b-nf4"):
            for sample in manifest.video_deployment_ids:
                video = next(item for item in manifest.video_samples if item.sample_id == sample)
                for condition in ("native", "reduced", "jpeg", "resize50"):
                    values.append(
                        _unit(
                            "deployment",
                            "video",
                            sample,
                            condition,
                            model,
                            sequence_id=video.sequence_id,
                            camera_motion_stratum=video.stratum,
                            deployment_member=True,
                        )
                    )
            for sample in manifest.image_deployment_ids:
                for condition in ("native", "reduced", "jpeg"):
                    values.append(
                        _unit(
                            "deployment",
                            "image",
                            sample,
                            condition,
                            model,
                            deployment_member=True,
                        )
                    )
    elif scope is ScopeVerdict.IMAGE_ONLY:
        # Image-only fallback has the 12-image deployment subset at 3 levels.
        for model in ("0.5b-nf4", "3b-bf16", "3b-nf4"):
            for sample in manifest.image_deployment_ids:
                for condition in ("native", "reduced", "jpeg"):
                    values.append(
                        _unit(
                            "deployment",
                            "image",
                            sample,
                            condition,
                            model,
                            deployment_member=True,
                        )
                    )
    return values


def _latency(scope: ScopeVerdict) -> list[ExperimentUnit]:
    cells = 64 if scope is ScopeVerdict.BROAD else (28 if scope is ScopeVerdict.IMAGE_ONLY else 0)
    return [
        _unit(
            "latency",
            "mixed",
            f"cell-{i:02d}",
            f"attempt-{j:02d}",
            index=j,
        )
        for i in range(cells)
        for j in range(14)
    ]


def _smoke(scope: ScopeVerdict) -> list[ExperimentUnit]:
    count = 36 if scope is ScopeVerdict.BROAD else (12 if scope is ScopeVerdict.IMAGE_ONLY else 0)
    return [_unit("smoke", "mixed", f"smoke-{i:02d}", "development") for i in range(count)]


def compile_matrix(scope: str | ScopeVerdict, manifest: ScopeManifest | None = None) -> MatrixPlan:
    try:
        verdict = scope if isinstance(scope, ScopeVerdict) else ScopeVerdict(str(scope).upper())
    except ValueError as error:
        raise ValueError("scope_must_be_BROAD_IMAGE_ONLY_VIDEO_ONLY_or_STOP") from error
    if verdict is ScopeVerdict.STOP:
        plan = MatrixPlan(verdict, (), (), (), 0)
        return plan
    if verdict is ScopeVerdict.VIDEO_ONLY:
        raise ValueError("video_only_scope_requires_precise_preregistration")
    if manifest is None:
        raise ValueError("scope_manifest_required")
    quality = tuple(_quality(verdict, manifest))
    latency = tuple(_latency(verdict))
    smoke = tuple(_smoke(verdict))
    retry_budget = 189 if verdict is ScopeVerdict.BROAD else (74 if verdict is ScopeVerdict.IMAGE_ONLY else 0)
    expected = {
        ScopeVerdict.BROAD: (986, 896, 36, 189, 2107),
        ScopeVerdict.IMAGE_ONLY: (348, 392, 12, 74, 826),
        ScopeVerdict.VIDEO_ONLY: (504, 0, 0, 0, 504),
    }[verdict]
    actual = (len(quality), len(latency), len(smoke), retry_budget, len(quality) + len(latency) + len(smoke) + retry_budget)
    if actual != expected:
        raise AssertionError(f"matrix_cardinality_mismatch:{verdict.value}:{actual}!={expected}")
    logical_ids = [u.unit_id for u in (*quality, *smoke)]
    if len(logical_ids) != len(set(logical_ids)):
        raise AssertionError("matrix_logical_unit_ids_not_unique")
    latency_ids = {u.unit_id for u in latency}
    if len(latency_ids) != (64 if verdict is ScopeVerdict.BROAD else 28 if verdict is ScopeVerdict.IMAGE_ONLY else 0):
        raise AssertionError("latency_cell_ids_not_exact")
    return MatrixPlan(verdict, quality, latency, smoke, retry_budget)


compile_experiment_matrix = compile_matrix
