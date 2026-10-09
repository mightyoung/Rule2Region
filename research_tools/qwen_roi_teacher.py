"""Offline Qwen2.5-VL semantic teacher for a bounded ROI manifest.

The model is deliberately loaded inside :func:`run_teacher`; importing this
module therefore remains cheap and does not require torch/transformers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from evidencegate.manifest import canonical_json, write_immutable_manifest

MODEL_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
CATEGORIES = (
    "real_person",
    "poster_or_screen",
    "reflection",
    "statue_or_mannequin",
    "partial_person",
    "unclear",
)
DEFAULT_MAX_ITEMS = 64
PROMPT = (
    "Classify the visual subject in this image crop. Return exactly one token "
    "from: real_person, poster_or_screen, reflection, statue_or_mannequin, "
    "partial_person, unclear. Do not count people and do not explain."
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_teacher_output(output: str) -> str:
    """Parse a model response, accepting only one of the six categories."""
    text = str(output).strip()
    candidate = text
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("teacher output is not valid JSON") from exc
        if not isinstance(obj, dict):
            raise ValueError("teacher JSON must be an object")
        if set(obj) not in ({"category"}, {"label"}):
            raise ValueError("teacher JSON must contain only category or label")
        candidate = obj.get("category", obj.get("label", ""))
        if not isinstance(candidate, str):
            raise ValueError("teacher JSON category must be a string")
        candidate = candidate.strip()
    if candidate not in CATEGORIES:
        raise ValueError(f"invalid teacher category: {candidate!r}")
    return candidate


def load_roi_manifest(path: str | Path, *, max_items: int = DEFAULT_MAX_ITEMS) -> list[dict[str, Any]]:
    """Load and validate a JSON ROI manifest, without touching model code."""
    if max_items < 1:
        raise ValueError("max_items must be positive")
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("ROI manifest must be a JSON list or an object with items")
    if len(entries) > max_items:
        raise ValueError(f"ROI manifest exceeds max_items={max_items}")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"manifest item {index} must be an object")
        raw_path = item.get("image") or item.get("image_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"manifest item {index} missing image path")
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        roi = item.get("roi")
        if roi is not None:
            if isinstance(roi, dict):
                try:
                    roi = [roi[k] for k in ("x", "y", "width", "height")]
                except KeyError as exc:
                    raise ValueError(f"manifest item {index} has incomplete roi") from exc
            if not isinstance(roi, (list, tuple)) or len(roi) != 4:
                raise ValueError(f"manifest item {index} roi must be [x,y,width,height]")
            if any(not isinstance(v, (int, float)) for v in roi) or roi[2] <= 0 or roi[3] <= 0:
                raise ValueError(f"manifest item {index} roi dimensions must be positive")
            roi = [float(v) for v in roi]
        result.append({"id": str(item.get("id", index)), "image_path": str(image_path), "roi": roi})
    return result


def _load_model(repo_id: str, revision: str) -> tuple[Any, Any]:
    """Lazily import and load the pinned Qwen model and processor."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(repo_id, revision=revision)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo_id, revision=revision)
    model.eval()
    return model, processor


def _predict(model: Any, processor: Any, image: Any) -> str:
    import torch

    inputs = processor(text=PROMPT, images=image, return_tensors="pt")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return parse_teacher_output(text)


def run_teacher(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    max_items: int = DEFAULT_MAX_ITEMS,
    model_loader: Callable[[str, str], tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Run bounded offline inference and atomically seal an immutable receipt."""
    entries = load_roi_manifest(manifest_path, max_items=max_items)
    loader = model_loader or _load_model
    model, processor = loader(MODEL_REPO, MODEL_REVISION)
    from PIL import Image

    predictions: list[dict[str, Any]] = []
    for entry in entries:
        source = Path(entry["image_path"])
        source_bytes = source.read_bytes()
        with Image.open(source) as image:
            image.load()
            if entry["roi"] is not None:
                x, y, width, height = entry["roi"]
                image = image.crop((x, y, x + width, y + height))
            label = _predict(model, processor, image)
            predictions.append({"id": entry["id"], "image_sha256": sha256_bytes(source_bytes), "roi": entry["roi"], "category": label})
    receipt: dict[str, Any] = {
        "schema_version": "human-count.qwen-roi-teacher.v1",
        "status": "passed",
        "model": {"repo_id": MODEL_REPO, "revision": MODEL_REVISION},
        "model_sha256": sha256_bytes(canonical_json({"repo_id": MODEL_REPO, "revision": MODEL_REVISION})),
        "prompt": PROMPT,
        "prompt_sha256": sha256_bytes(PROMPT.encode("utf-8")),
        "manifest_sha256": sha256_bytes(canonical_json(entries)),
        "max_items": max_items,
        "items": predictions,
    }
    write_immutable_manifest(output_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Qwen ROI semantic teacher")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
    args = parser.parse_args()
    receipt = run_teacher(args.manifest, args.output, max_items=args.max_items)
    print(json.dumps({"status": receipt["status"], "items": len(receipt["items"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
