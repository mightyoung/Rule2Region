from pathlib import Path

import pytest

from research_tools.g0_freeze import seal_manifest, select_pope_records


def _records(variant: str) -> list[dict[str, object]]:
    records = []
    for label in ("yes", "no"):
        for index in range(20):
            records.append(
                {
                    "image": f"COCO_val2014_{variant}_{label}_{index:03d}.jpg",
                    "label": label,
                    "question_id": f"{variant}-{label}-{index}",
                    "text": f"question {index}",
                }
            )
    return records


def test_selection_is_balanced_deterministic_and_image_disjoint() -> None:
    inputs = {variant: _records(variant) for variant in ("random", "popular", "adversarial")}
    first = select_pope_records(inputs, seed=17)
    second = select_pope_records(inputs, seed=17)
    assert first == second
    assert sum(record["tier"] == "formal" for record in first) == 48
    assert sum(record["tier"] == "development" for record in first) == 12
    assert len({record["image"] for record in first}) == 60
    for tier, expected in (("formal", 8), ("development", 2)):
        for variant in ("random", "popular", "adversarial"):
            for label in ("yes", "no"):
                assert sum(
                    record["tier"] == tier
                    and record["pope_variant"] == variant
                    and record["label"] == label
                    for record in first
                ) == expected


def test_selection_fails_when_unique_image_quota_is_impossible() -> None:
    shared = [
        {
            "image": "same.jpg",
            "label": label,
            "question_id": f"q-{label}",
            "text": "question",
        }
        for label in ("yes", "no")
    ]
    inputs = {variant: shared for variant in ("random", "popular", "adversarial")}
    with pytest.raises(ValueError, match="insufficient unique images"):
        select_pope_records(inputs, seed=17, formal_per_label=1, development_per_label=0)


def test_seal_manifest_refuses_different_content(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    first_hash = seal_manifest(target, {"schema_version": "v1", "value": 1})
    assert seal_manifest(target, {"schema_version": "v1", "value": 1}) == first_hash
    with pytest.raises(FileExistsError, match="immutable manifest differs"):
        seal_manifest(target, {"schema_version": "v1", "value": 2})
