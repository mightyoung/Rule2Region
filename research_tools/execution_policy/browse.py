"""Fail-closed file tools. Paths must stay inside one context directory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_tools.execution_policy.skills.registry import required_for_submit


MAX_READ_CHARS = 80_000
TOOLS = frozenset(
    {
        "list_skills",
        "load_skill",
        "list_dir",
        "read_file",
        "list_tools",
        "query_tool",
        "verify_last_tool",
    }
)
TOOL_CATALOG = {
    "list_skills": {
        "args": [],
        "purpose": "List registered skills and whether each is loaded.",
    },
    "load_skill": {
        "args": ["path"],
        "purpose": "path is a skill name. Load that SKILL.md into this turn.",
    },
    "list_tools": {
        "args": [],
        "purpose": "Return the catalog of callable tools so you can verify names.",
    },
    "query_tool": {
        "args": ["path"],
        "purpose": "path is a tool name. Return that tool's arguments, purpose, and last result.",
    },
    "verify_last_tool": {
        "args": [],
        "purpose": "Verify the previous tool actually ran and whether its path is still readable.",
    },
    "list_dir": {
        "args": ["path"],
        "purpose": "List files and directories under a path inside the context root.",
    },
    "read_file": {
        "args": ["path"],
        "purpose": "Read a UTF-8 text file inside the context root.",
    },
    "submit_card": {
        "args": ["regions"],
        "purpose": "Submit the H1 schedule card after required skills are loaded and a file was read.",
    },
}


class PathEscapesContext(ValueError):
    pass


class FileBrowser:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"context_missing:{self.root}")
        self.history: list[dict[str, Any]] = []
        self.loaded: set[str] = set()
        self.required_for_submit = frozenset(required_for_submit())
        self.listed_files: set[str] = set()
        self.listed_dirs: set[str] = set()
        self.loaded_skill_text: dict[str, str] = {}
        self.read_boxes: dict[str, str] = {}

    def _norm(self, rel: str) -> str:
        return (rel or ".").replace("\\", "/").strip().lstrip("./")

    def _gated_context_path(self, rel: str) -> bool:
        norm = self._norm(rel)
        if norm in {"", ".", "skills"} or norm.startswith("skills/"):
            return False
        return True

    def _join(self, parent: str, name: str) -> str:
        prefix = self._norm(parent)
        if prefix in {"", "."}:
            return name
        return f"{prefix}/{name}"

    def missing_for_submit(self) -> list[str]:
        return sorted(self.required_for_submit - self.loaded)

    def memory(self) -> dict[str, Any]:
        return {
            "loaded_skills": dict(self.loaded_skill_text),
            "listed_files": sorted(self.listed_files),
            "read_boxes": dict(self.read_boxes),
        }

    def list_skills(self) -> dict[str, Any]:
        index_path = self.root / "skills" / "INDEX.json"
        if not index_path.is_file():
            return {"ok": False, "error": "skills_index_missing"}
        rows = json.loads(index_path.read_text(encoding="utf-8"))
        for row in rows:
            row["loaded"] = row.get("name") in self.loaded
        return {
            "ok": True,
            "skills": rows,
            "loaded": sorted(self.loaded),
            "required_for_submit": sorted(self.required_for_submit),
        }

    def _skill_name(self, name: str) -> str:
        key = (name or "").strip().replace("\\", "/").strip("/")
        if key.startswith("skills/"):
            key = key[len("skills/") :]
        if key.endswith("/SKILL.md"):
            key = key[: -len("/SKILL.md")]
        return key.strip("/")

    def load_skill(self, name: str) -> dict[str, Any]:
        key = self._skill_name(name)
        target = self.root / "skills" / key / "SKILL.md"
        try:
            resolved = self.resolve(f"skills/{key}/SKILL.md")
        except PathEscapesContext as exc:
            return {"ok": False, "error": str(exc)}
        if key in {"", ".", ".."} or "/" in key or resolved != target.resolve() or not resolved.is_file():
            return {"ok": False, "error": "unknown_skill", "path": name}
        text = resolved.read_text(encoding="utf-8")
        self.loaded.add(key)
        self.loaded_skill_text[key] = text
        return {"ok": True, "skill": key, "loaded": sorted(self.loaded), "text": text}

    def resolve(self, rel: str) -> Path:
        text = (rel or ".").strip() or "."
        raw = Path(text)
        target = raw.resolve() if raw.is_absolute() else (self.root / text).resolve()
        if target != self.root and self.root not in target.parents:
            raise PathEscapesContext(f"path_escapes_context:{rel}")
        return target

    def list_dir(self, rel: str = ".") -> dict[str, Any]:
        if self._gated_context_path(rel) and "file-browse" not in self.loaded:
            return {"ok": False, "error": "skill_not_loaded", "skill": "file-browse", "path": rel}
        target = self.resolve(rel)
        if not target.is_dir():
            return {"ok": False, "error": "not_a_directory", "path": rel}
        entries = []
        omitted = 0
        summary_only = self._norm(rel) == "history_boxes"
        for child in sorted(target.iterdir(), key=lambda item: item.name):
            kind = "dir" if child.is_dir() else "file"
            if summary_only and kind == "file" and not child.name.startswith("_"):
                omitted += 1
                continue
            child_rel = self._join(rel, child.name)
            entries.append({"name": child.name, "kind": kind})
            if kind == "dir":
                self.listed_dirs.add(child_rel)
            else:
                self.listed_files.add(child_rel)
        result = {"ok": True, "path": rel or ".", "entries": entries}
        if omitted:
            result["omitted_frame_files"] = omitted
            result["hint"] = "read history_boxes/_focus.json; per-frame extras are not listed"
        return result

    def read_file(self, rel: str) -> dict[str, Any]:
        if self._gated_context_path(rel) and "file-browse" not in self.loaded:
            return {"ok": False, "error": "skill_not_loaded", "skill": "file-browse", "path": rel}
        norm = self._norm(rel)
        if norm not in self.listed_files:
            return {
                "ok": False,
                "error": "path_not_in_listing",
                "path": rel,
                "hint": "list_dir the parent first; observer-frame files are not listed",
                "listed_files": sorted(self.listed_files),
            }
        if norm in self.read_boxes:
            return {
                "ok": False,
                "error": "already_read",
                "path": rel,
                "hint": "submit_card now; do not reread the same file",
            }
        if norm.startswith("history_boxes/") and any(path.startswith("history_boxes/") for path in self.read_boxes):
            return {
                "ok": False,
                "error": "history_boxes_already_read",
                "path": rel,
                "hint": "submit_card from history_boxes/_focus.json; do not read another history_boxes file",
            }
        target = self.resolve(rel)
        if not target.is_file():
            return {"ok": False, "error": "not_a_file", "path": rel}
        text = target.read_text(encoding="utf-8")
        if len(text) > MAX_READ_CHARS:
            return {"ok": False, "error": "too_large", "path": rel, "chars": len(text)}
        if norm.startswith("history_boxes/"):
            self.read_boxes[norm] = text
        return {"ok": True, "path": rel, "text": text}

    def query_tool(self, name: str) -> dict[str, Any]:
        key = (name or "").strip()
        spec = TOOL_CATALOG.get(key)
        if spec is None:
            return {"ok": False, "error": "unknown_tool", "path": name, "tools": sorted(TOOL_CATALOG)}
        last = next((row for row in reversed(self.history) if row.get("action") == key), None)
        return {"ok": True, "tool": key, **spec, "last": last}

    def verify_last_tool(self) -> dict[str, Any]:
        if not self.history:
            return {"ok": False, "error": "no_tool_called_yet"}
        last = dict(self.history[-1])
        path = str(last.get("path") or "")
        readable = None
        if last.get("action") in {"list_dir", "read_file"} and path:
            try:
                target = self.resolve(path)
                readable = target.exists()
            except PathEscapesContext as exc:
                last["escape"] = str(exc)
                readable = False
        last["path_still_readable"] = readable
        last["verified"] = bool(last.get("ok")) and readable is not False
        return {"ok": True, "last": last}

    def call(self, action: str, path: str) -> dict[str, Any]:
        if action == "list_skills":
            result: dict[str, Any] = self.list_skills()
        elif action == "load_skill":
            result = self.load_skill(path)
        elif action == "list_tools":
            result = {"ok": True, "tools": TOOL_CATALOG}
        elif action == "query_tool":
            result = self.query_tool(path)
        elif action == "verify_last_tool":
            result = self.verify_last_tool()
        elif action == "list_dir":
            result = self.list_dir(path)
        elif action == "read_file":
            result = self.read_file(path)
        else:
            result = {"ok": False, "error": "unknown_tool", "action": action}
        self.history.append({"action": action, "path": path, "ok": result.get("ok")})
        return result
