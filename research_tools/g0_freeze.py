from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


POPE_VARIANTS = ("random", "popular", "adversarial")
LABELS = ("yes", "no")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(seed: int, variant: str, record: Mapping[str, Any]) -> str:
    identity = "|".join(
        (
            str(seed),
            variant,
            str(record["label"]),
            Path(str(record["image"])).name,
            str(record["question_id"]),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            records.append(value)
    return records


def select_pope_records(
    records_by_variant: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    seed: int,
    formal_per_label: int = 8,
    development_per_label: int = 2,
) -> list[dict[str, Any]]:
    """Freeze balanced, image-disjoint POPE records without outcome-based selection."""

    required = {"image", "label", "question_id", "text"}
    strata: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for variant in POPE_VARIANTS:
        records = list(records_by_variant.get(variant, ()))
        for record in records:
            if not required.issubset(record):
                raise ValueError(f"{variant}: POPE record missing required fields")
            if record["label"] not in LABELS:
                raise ValueError(f"{variant}: unsupported label {record['label']!r}")
        for label in LABELS:
            candidates = [record for record in records if record["label"] == label]
            strata[(variant, label)] = sorted(
                candidates,
                key=lambda record: (
                    _rank(seed, variant, record),
                    Path(str(record["image"])).name,
                    str(record["question_id"]),
                ),
            )

    used_images: set[str] = set()
    selected: list[dict[str, Any]] = []

    def take(tier: str, quota: int) -> None:
        for variant in POPE_VARIANTS:
            for label in LABELS:
                taken = 0
                for record in strata[(variant, label)]:
                    image_name = Path(str(record["image"])).name
                    if image_name in used_images:
                        continue
                    selected.append(
                        {
                            "tier": tier,
                            "pope_variant": variant,
                            "image": image_name,
                            "label": label,
                            "question_id": record["question_id"],
                            "text": record["text"],
                        }
                    )
                    used_images.add(image_name)
                    taken += 1
                    if taken == quota:
                        break
                if taken != quota:
                    raise ValueError(
                        f"insufficient unique images for {tier}:{variant}:{label}; "
                        f"needed={quota}, found={taken}"
                    )

    take("formal", formal_per_label)
    take("development", development_per_label)
    return selected


def _image_metadata(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "width": width,
        "height": height,
        "mode": mode,
        "decoder": "PIL.verify",
    }


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def seal_manifest(path: Path, manifest: dict[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    manifest_hash = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    sealed = {**unsigned, "manifest_sha256": manifest_hash}
    payload = canonical_bytes(sealed)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable manifest differs: {path}")
        return manifest_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return manifest_hash


def build_image_manifest(
    *,
    pope_root: Path,
    coco_root: Path,
    seed: int,
    pope_commit: str,
) -> dict[str, Any]:
    paths = {
        variant: pope_root / "output" / "coco" / f"coco_pope_{variant}.json"
        for variant in POPE_VARIANTS
    }
    records = {variant: load_jsonl(path) for variant, path in paths.items()}
    selected = select_pope_records(records, seed=seed)
    deployment_counts: dict[tuple[str, str], int] = defaultdict(int)
    for record in selected:
        key = (record["pope_variant"], record["label"])
        is_deployment = record["tier"] == "formal" and deployment_counts[key] < 2
        record["deployment_member"] = is_deployment
        if is_deployment:
            deployment_counts[key] += 1
    enriched: list[dict[str, Any]] = []
    for record in selected:
        image_path = coco_root / record["image"]
        if not image_path.is_file():
            raise FileNotFoundError(f"selected COCO image is missing: {image_path}")
        enriched.append({**record, "source": _image_metadata(image_path)})
    counts: dict[str, int] = defaultdict(int)
    for record in enriched:
        counts[f"{record['tier']}:{record['pope_variant']}:{record['label']}"] += 1
    if sum(bool(record["deployment_member"]) for record in enriched) != 12:
        raise AssertionError("image_deployment_subset_must_have_12_samples")
    if any(
        record["deployment_member"] and record["tier"] != "formal"
        for record in enriched
    ):
        raise AssertionError("development_images_cannot_enter_deployment_subset")
    return {
        "schema_version": "evidencegate.g0.image-scope.v2",
        "seed": seed,
        "selection_rule": "sha256-ranked, variant/label-balanced, image-disjoint",
        "pope_commit": pope_commit,
        "pope_sources": {
            variant: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "records": len(records[variant]),
            }
            for variant, path in paths.items()
        },
        "formal_samples": 48,
        "development_samples": 12,
        "deployment_samples": 12,
        "counts": dict(sorted(counts.items())),
        "samples": enriched,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze EvidenceGate G0 image scope.")
    parser.add_argument("--pope-root", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--pope-commit",
        default="08d957b917e5a378a2f99d35b6293c536a66298b",
    )
    arguments = parser.parse_args()
    manifest = build_image_manifest(
        pope_root=arguments.pope_root,
        coco_root=arguments.coco_root,
        seed=arguments.seed,
        pope_commit=arguments.pope_commit,
    )
    manifest_hash = seal_manifest(arguments.output, manifest)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "manifest_sha256": manifest_hash,
                "formal_samples": manifest["formal_samples"],
                "development_samples": manifest["development_samples"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
