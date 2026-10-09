"""Read-only Gradio dashboard for precomputed supervision assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TABLE_HEADERS = ["Condition", "MAE", "AP50", "Recall", "Small recall", "Budget", "p50 ms", "p95 ms", "FPS"]


def load_demo_index(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "fieldpack.supervision-demo-index.v1":
        raise ValueError("unsupported_demo_index")
    if payload.get("evidence_status") != "ENGINEERING_PILOT_NOT_FORMAL_STATISTICS":
        raise ValueError("demo_evidence_status_missing")
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("demo_assets_missing")
    return payload


def asset_choices(index: dict[str, Any]) -> list[str]:
    return [f"{item['sequence_id']} | {item['condition']}" for item in index["assets"]]


def select_asset(index: dict[str, Any], choice: str) -> tuple[str, list[list[Any]], str]:
    for item, label in zip(index["assets"], asset_choices(index)):
        if label == choice:
            status = f"工程Pilot；{item['frames']}帧；视频SHA-256 `{item['video_sha256']}`"
            return item["video"], item["condition_table"], status
    raise KeyError("demo_asset_choice_missing")


def build_app(index_path: Path) -> Any:
    try:
        import gradio as gr  # type: ignore
    except ImportError as exc:
        raise RuntimeError("gradio_required_for_dashboard") from exc
    index = load_demo_index(index_path)
    choices = asset_choices(index)
    initial_video, initial_table, initial_status = select_asset(index, choices[0])
    with gr.Blocks(title="FieldPack G0 Cascade Pilot") as app:
        gr.Markdown("# FieldPack G0 人员计数级联\n只读工程Pilot展示；不是正式统计结果，也不触发模型推理。")
        status = gr.Markdown(initial_status)
        selector = gr.Dropdown(choices=choices, value=choices[0], label="视频与条件")
        video = gr.Video(value=initial_video, label="Supervision标注视频", interactive=False)
        table = gr.Dataframe(headers=TABLE_HEADERS, value=initial_table, interactive=False, label="同一序列条件对比")
        selector.change(lambda choice: select_asset(index, choice), inputs=selector, outputs=(video, table, status))
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve read-only FieldPack dashboard")
    parser.add_argument("--index", type=Path, required=True); parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app(args.index).launch(server_name=args.host, server_port=args.port, show_error=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
