"""Registered skills the visual agent may load. One home per skill body."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SKILLS_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SkillSpec:
    name: str
    dirname: str
    description: str
    required_for_submit: bool = True


REGISTRY: tuple[SkillSpec, ...] = (
    SkillSpec(
        "file-browse",
        "file_browse",
        "How to list and read the context directory, and how tool results come back.",
    ),
    SkillSpec(
        "count-schedule",
        "count_schedule",
        "How to draw count regions and pick yolo11s/yolo26 and conf.",
    ),
)


def spec_by_name(name: str) -> SkillSpec | None:
    key = (name or "").strip()
    for spec in REGISTRY:
        if spec.name == key:
            return spec
    return None


def source_skill_md(spec: SkillSpec) -> Path:
    return SKILLS_ROOT / spec.dirname / "SKILL.md"


def install_skills(dest: Path) -> list[dict[str, Any]]:
    dest.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in REGISTRY:
        src = source_skill_md(spec)
        if not src.is_file():
            raise FileNotFoundError(f"skill_missing:{spec.name}:{src}")
        target = dest / spec.name
        target.mkdir(exist_ok=True)
        text = src.read_text(encoding="utf-8")
        (target / "SKILL.md").write_text(text, encoding="utf-8")
        rows.append(
            {
                "name": spec.name,
                "description": spec.description,
                "path": f"skills/{spec.name}/SKILL.md",
                "required_for_submit": spec.required_for_submit,
            }
        )
    (dest / "INDEX.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def required_for_submit() -> tuple[str, ...]:
    return tuple(spec.name for spec in REGISTRY if spec.required_for_submit)
