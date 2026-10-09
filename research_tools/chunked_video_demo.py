from __future__ import annotations

import argparse
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evidencegate.demo import active_segment, aggregate_segment_counts, plan_overlapping_segments
from evidencegate.model_runtime import parse_model_output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(path: Path) -> tuple[list[np.ndarray], float]:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = float(stream.average_rate or stream.base_rate or 0)
        if rate <= 0:
            raise ValueError("video_fps_unavailable")
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24").copy())
    if not frames:
        raise ValueError("video_frames_unavailable")
    return frames, rate


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _fit(frame: np.ndarray, width: int, height: int) -> Image.Image:
    image = Image.fromarray(frame)
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "#11151c")
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def render_demo(
    reference_path: Path,
    compressed_path: Path,
    receipt: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    reference, fps = _decode(reference_path)
    compressed, compressed_fps = _decode(compressed_path)
    if abs(fps - compressed_fps) > 0.01:
        raise ValueError("comparison_fps_mismatch")
    frame_count = min(len(reference), len(compressed))
    byte_reduction = 1.0 - compressed_path.stat().st_size / reference_path.stat().st_size
    width, height = 1280, 720
    panel_width, panel_height = 620, 420
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=round(fps))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}
        for frame_index in range(frame_count):
            now = frame_index / fps
            current = active_segment(receipt["segments"], now)
            canvas = Image.new("RGB", (width, height), "#0b0f14")
            canvas.paste(_fit(reference[frame_index], panel_width, panel_height), (20, 80))
            canvas.paste(_fit(compressed[frame_index], panel_width, panel_height), (660, 80))
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 20), "EvidenceGate memory-bounded video inference", font=_font(30), fill="#f5f7fa")
            draw.text((20, 52), "Reference / CRF18", font=_font(20), fill="#8bd5ff")
            draw.text((660, 52), "Low bitrate / CRF32", font=_font(20), fill="#ffd580")
            if current:
                count = current.get("count")
                status = current.get("state", "unknown")
                detail = (
                    f"Chunk {int(current['index']) + 1}/{len(receipt['segments'])}  "
                    f"{float(current['start_seconds']):.1f}-{float(current['end_seconds']):.1f}s  "
                    f"count={count if count is not None else 'N/A'}  state={status}"
                )
                metrics = (
                    f"input_tokens={current.get('input_tokens', 'N/A')}  "
                    f"peak_VRAM={float(current.get('peak_vram_bytes', 0)) / 1024**3:.2f} GiB  "
                    f"latency={float(current.get('latency_seconds', 0)):.2f}s"
                )
                draw.text((24, 530), detail, font=_font(24), fill="#ffffff")
                draw.text((24, 570), metrics, font=_font(22), fill="#b9c2cc")
            aggregate = receipt["aggregate"]
            draw.text(
                (24, 620),
                "Aggregation (no cross-chunk summation): "
                f"max-visible={aggregate['max_visible_count']}  "
                f"median={aggregate['median_segment_count']}  "
                f"ground-truth={receipt.get('expected_count', 'not supplied')}",
                font=_font(22),
                fill="#9ee6a8",
            )
            draw.text(
                (660, 508),
                f"Encoded-byte reduction: {byte_reduction:.1%} (decoded dimensions unchanged)",
                font=_font(18),
                fill="#ffd580",
            )
            timeline_left, timeline_right, timeline_y = 24, 1256, 682
            draw.line((timeline_left, timeline_y, timeline_right, timeline_y), fill="#4a5563", width=6)
            progress = timeline_left + int((timeline_right - timeline_left) * frame_index / max(1, frame_count - 1))
            draw.line((timeline_left, timeline_y, progress, timeline_y), fill="#55b7ff", width=6)
            video_frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return {
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "frames": frame_count,
        "fps": fps,
    }


def run_inference(
    input_path: Path,
    model_path: Path,
    *,
    segment_seconds: float,
    overlap_seconds: float,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    frames, fps = _decode(input_path)
    segments = plan_overlapping_segments(
        len(frames), fps, segment_seconds=segment_seconds, overlap_seconds=overlap_seconds
    )
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    ).eval()
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {
                        "type": "text",
                        "text": (
                            "Count the maximum number of visible people at one time in this short clip. "
                            'Respond exactly as JSON: {"count": <integer from 0 to 99>}.'
                        ),
                    },
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    ) + '{"count": '
    results: list[dict[str, Any]] = []
    for segment in segments:
        clip = np.stack(frames[segment.start_frame : segment.end_frame], axis=0)
        started = time.perf_counter()
        state = "success"
        count: int | None = None
        raw_text = ""
        input_tokens: int | None = None
        peak_vram = 0
        try:
            inputs = processor(text=[prompt], videos=[clip], padding=True, return_tensors="pt")
            inputs = {key: value.to("cuda:0") if hasattr(value, "to") else value for key, value in inputs.items()}
            input_tokens = int(inputs["input_ids"].shape[-1])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                )
            continuation = generated.sequences[:, inputs["input_ids"].shape[-1] :]
            raw_text = processor.batch_decode(continuation, skip_special_tokens=True)[0]
            full_text = '{"count": ' + raw_text
            count = int(parse_model_output(full_text, "exact_json_single_count_0_99"))
            peak_vram = int(torch.cuda.max_memory_allocated(0))
        except torch.OutOfMemoryError as exc:
            state = "OOM"
            raw_text = str(exc)
            peak_vram = int(torch.cuda.max_memory_allocated(0))
        except (ValueError, KeyError, RuntimeError) as exc:
            state = "error"
            raw_text = f"{type(exc).__name__}:{exc}"
            peak_vram = int(torch.cuda.max_memory_allocated(0))
        finally:
            inputs = None
            generated = None
            continuation = None
            clip = None
            torch.cuda.empty_cache()
        results.append(
            {
                **segment.to_dict(),
                "state": state,
                "count": count,
                "input_tokens": input_tokens,
                "peak_vram_bytes": peak_vram,
                "latency_seconds": time.perf_counter() - started,
                "raw_text": raw_text,
            }
        )
    return {
        "schema_version": "evidencegate.chunked-demo.v1",
        "input": {"path": str(input_path), "sha256": _sha256(input_path), "fps": fps, "frames": len(frames)},
        "model": {"path": str(model_path), "precision": "bf16"},
        "policy": {
            "segment_seconds": segment_seconds,
            "overlap_seconds": overlap_seconds,
            "aggregation": "max_visible_and_median; never sum across chunks",
            "formal_evidence": False,
        },
        "segments": results,
        "aggregate": aggregate_segment_counts(results),
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description="Build a memory-bounded chunked VLM demo.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compressed", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.2)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    else:
        if args.model is None:
            parser.error("--model is required unless --render-only is used")
        receipt = run_inference(
            args.reference,
            args.model,
            segment_seconds=args.segment_seconds,
            overlap_seconds=args.overlap_seconds,
        )
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.expected_count is not None:
        receipt["expected_count"] = args.expected_count
    if args.output:
        receipt["render"] = render_demo(args.reference, args.compressed, receipt, args.output)
        args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(args.receipt), "aggregate": receipt["aggregate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
