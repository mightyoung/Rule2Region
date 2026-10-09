import json
from pathlib import Path

import pytest
from PIL import Image

from research_tools.qwen_roi_teacher import (
    CATEGORIES,
    MODEL_REVISION,
    MODEL_REPO,
    load_roi_manifest,
    parse_teacher_output,
    run_teacher,
)


@pytest.mark.parametrize("value", CATEGORIES)
def test_parser_accepts_only_categories(value):
    assert parse_teacher_output(value) == value
    assert parse_teacher_output(json.dumps({"category": value})) == value


@pytest.mark.parametrize("value", ["two people", "{\"count\": 2}", "real_person extra", ""])
def test_parser_rejects_counting_or_non_categorical_output(value):
    with pytest.raises(ValueError):
        parse_teacher_output(value)


def test_manifest_resolves_relative_images_and_enforces_bound(tmp_path: Path):
    image = tmp_path / "roi.png"
    Image.new("RGB", (10, 10), "white").save(image)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [{"id": "a", "image": "roi.png", "roi": {"x": 1, "y": 2, "width": 4, "height": 5}}]}), encoding="utf-8")
    loaded = load_roi_manifest(manifest, max_items=1)
    assert loaded[0]["image_path"] == str(image)
    assert loaded[0]["roi"] == [1.0, 2.0, 4.0, 5.0]
    with pytest.raises(ValueError, match="max_items"):
        load_roi_manifest(manifest, max_items=0)


def test_manifest_rejects_missing_image(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"image": "missing.png"}]), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_roi_manifest(manifest)


def test_run_teacher_seals_receipt_and_loads_pinned_model_lazily(tmp_path: Path):
    image = tmp_path / "roi.png"
    Image.new("RGB", (10, 10), "white").save(image)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"image": image.name}]), encoding="utf-8")
    output = tmp_path / "receipt.json"
    calls = []

    class Processor:
        def __call__(self, **kwargs):
            return {}

        def batch_decode(self, *_args, **_kwargs):
            return ["unclear"]

    class Model:
        def eval(self):
            return self

        def generate(self, **_kwargs):
            return [0]

    def loader(repo, revision):
        calls.append((repo, revision))
        return Model(), Processor()

    receipt = run_teacher(manifest, output, model_loader=loader)
    assert calls == [(MODEL_REPO, MODEL_REVISION)]
    assert receipt["items"][0]["category"] == "unclear"
    assert output.exists()
    first = output.read_bytes()
    run_teacher(manifest, output, model_loader=loader)
    assert output.read_bytes() == first
