from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import imageio_ffmpeg
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.model_selector import run_colorizer
from tools.TemporalSmoothing import apply_temporal_smoothing

REPORT_ROOT = REPO_ROOT / "reports" / "benchmark_suite_20260622_fresh_sources"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
BENCHMARK_FPS = 4
SOURCE_ROOT = REPO_ROOT / "benchmark_sources_new"


@dataclass
class SceneSpec:
    key: str
    label: str
    category: str
    source: str
    start: float
    duration: float
    notes: str


SCENES: list[SceneSpec] = [
    SceneSpec(
        key="nature_short",
        label="Nature Short",
        category="nature",
        source=str(SOURCE_ROOT / "nature_short.mp4"),
        start=0.0,
        duration=4.0,
        notes="water, sky, landscape, simple outdoor palette",
    ),
    SceneSpec(
        key="urban_short",
        label="Urban Short",
        category="urban",
        source=str(SOURCE_ROOT / "urban_short.mp4"),
        start=0.0,
        duration=4.0,
        notes="city, roads, buildings, wider structural scene",
    ),
    SceneSpec(
        key="people_short",
        label="People Short",
        category="people",
        source=str(SOURCE_ROOT / "people_short.mp4"),
        start=0.0,
        duration=4.0,
        notes="single face, skin tone, hair, indoor interview framing",
    ),
    SceneSpec(
        key="animation_short",
        label="Animation Short",
        category="animation",
        source=str(SOURCE_ROOT / "animation_short.mp4"),
        start=0.0,
        duration=4.0,
        notes="stylized animated scene, non-photoreal colors, hard benchmark for realism bias",
    ),
]


MODELS: list[tuple[str, dict[str, Any]]] = [
    (
        "chromanet",
        {
            "model_name": "colorize_chromanet_v3",
            "kwargs": {
                "use_gpu": True,
                "input_size": 256,
                "batch_size": 12,
                "prefetch_workers": 4,
                "save_workers": 4,
                "max_prefetch_batches": 2,
                "confidence_threshold": 0.30,
                "saturation_gain": 1.00,
                "grain_amount": 0.00,
                "style_preset": "realistic",
            },
        },
    ),
    (
        "instcolorization",
        {
            "model_name": "instcolorization2025",
            "kwargs": {
                "use_gpu": True,
                "style": "trained",
                "input_size": 256,
            },
        },
    ),
    (
        "zhang_siggraph17",
        {
            "model_name": "colorize_zhang",
            "kwargs": {
                "use_gpu": True,
                "variant": "siggraph17",
                "input_size": 224,
                "batch_size": 12,
                "num_threads": 8,
                "prefetch_workers": 4,
                "save_workers": 2,
                "progress": True,
            },
        },
    ),
    (
        "enhanced_zhang",
        {
            "model_name": "Enhanced Zhang (Bebo's Experiment)",
            "kwargs": {
                "use_gpu": True,
                "variant": "siggraph17",
                "input_size": 256,
                "batch_size": 12,
                "num_threads": 8,
                "prefetch_workers": 4,
                "save_workers": 2,
                "saturation_gain": 1.18,
                "contrast_gain": 1.06,
                "neutralize_ab_bias": 0.18,
                "progress": True,
            },
        },
    ),
]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def cut_scene_clip(scene: SceneSpec, scene_dir: Path) -> Path:
    clip_path = scene_dir / f"{scene.key}_color_clip.mp4"
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-ss",
        str(scene.start),
        "-t",
        str(scene.duration),
        "-i",
        scene.source,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(clip_path),
    ]
    _run(cmd)
    return clip_path


def make_grayscale_clip(color_clip_path: Path, scene_dir: Path, scene_key: str) -> Path:
    gray_clip_path = scene_dir / f"{scene_key}_gray_clip.mp4"
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-i",
        str(color_clip_path),
        "-vf",
        "format=gray",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(gray_clip_path),
    ]
    _run(cmd)
    return gray_clip_path


def extract_frames(clip_path: Path, frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-i",
        str(clip_path),
        "-vf",
        f"fps={BENCHMARK_FPS}",
        "-q:v",
        "2",
        str(frames_dir / "frame_%05d.jpg"),
    ]
    _run(cmd)


def sorted_frames(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}],
        key=lambda p: p.name,
    )


def make_montage(
    color_frame: Path,
    gray_frame: Path,
    raw_frame: Path,
    smooth_frame: Path,
    out_path: Path,
    title: str,
) -> None:
    color = Image.open(color_frame).convert("RGB")
    gray = Image.open(gray_frame).convert("RGB")
    raw = Image.open(raw_frame).convert("RGB")
    smooth = Image.open(smooth_frame).convert("RGB")
    width, height = gray.size
    montage = Image.new("RGB", (width * 4, height + 34), color=(0, 0, 0))
    montage.paste(color, (0, 34))
    montage.paste(gray, (width, 34))
    montage.paste(raw, (width * 2, 34))
    montage.paste(smooth, (width * 3, 34))
    draw = ImageDraw.Draw(montage)
    draw.text(
        (8, 8),
        f"{title} | color ref | input gray | raw | flow_chroma",
        fill=(255, 255, 255),
    )
    montage.save(out_path)


def pick_mid_frame(folder: Path) -> Path:
    frames = sorted_frames(folder)
    return frames[len(frames) // 2]


def find_matching_frame(folder: Path, stem: str) -> Path | None:
    for frame in sorted_frames(folder):
        if frame.stem == stem:
            return frame
    return None


def benchmark_scene(scene: SceneSpec) -> dict[str, Any]:
    scene_dir = REPORT_ROOT / scene.key
    _ensure_clean_dir(scene_dir)
    color_clip_path = cut_scene_clip(scene, scene_dir)
    gray_clip_path = make_grayscale_clip(color_clip_path, scene_dir, scene.key)
    frames_color_dir = scene_dir / "frames_color"
    frames_dir = scene_dir / "frames_gray"
    extract_frames(color_clip_path, frames_color_dir)
    extract_frames(gray_clip_path, frames_dir)
    color_frames = sorted_frames(frames_color_dir)
    gray_frames = sorted_frames(frames_dir)
    scene_report: dict[str, Any] = {
        "scene": asdict(scene),
        "color_clip_path": str(color_clip_path),
        "gray_clip_path": str(gray_clip_path),
        "frames_color_dir": str(frames_color_dir),
        "frames_dir": str(frames_dir),
        "frame_count": len(gray_frames),
        "models": [],
    }

    for model_key, spec in MODELS:
        model_dir = scene_dir / model_key
        raw_dir = model_dir / "raw"
        smooth_dir = model_dir / "flow_chroma"
        screenshot_dir = model_dir / "screenshots"
        raw_dir.mkdir(parents=True, exist_ok=True)
        smooth_dir.mkdir(parents=True, exist_ok=True)
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        raw_seconds = None
        smooth_seconds = None
        error_text = None
        try:
            t0 = time.perf_counter()
            run_colorizer(
                model_name=spec["model_name"],
                frames_dir=frames_dir,
                color_dir=raw_dir,
                models_dir=REPO_ROOT / "models",
                preview=False,
                **spec["kwargs"],
            )
            raw_seconds = time.perf_counter() - t0

            t1 = time.perf_counter()
            apply_temporal_smoothing(
                input_folder=str(raw_dir),
                output_folder=str(smooth_dir),
                gray_input_folder=str(frames_dir),
                mode="flow_chroma",
                flow_mix=0.9,
                motion_strength=1.2,
            )
            smooth_seconds = time.perf_counter() - t1
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"

        gray_mid = pick_mid_frame(frames_dir)
        color_mid = find_matching_frame(frames_color_dir, gray_mid.stem) or color_frames[len(color_frames) // 2]
        raw_mid = find_matching_frame(raw_dir, gray_mid.stem)
        smooth_mid = find_matching_frame(smooth_dir, gray_mid.stem)
        montage_path = screenshot_dir / f"{scene.key}_{model_key}_mid_montage.jpg"
        if error_text is None and raw_mid is not None and smooth_mid is not None:
            make_montage(color_mid, gray_mid, raw_mid, smooth_mid, montage_path, f"{scene.label} | {model_key}")

        scene_report["models"].append(
            {
                "model_key": model_key,
                "raw_dir": str(raw_dir),
                "smooth_dir": str(smooth_dir),
                "raw_seconds": round(raw_seconds, 3) if raw_seconds is not None else None,
                "smooth_seconds": round(smooth_seconds, 3) if smooth_seconds is not None else None,
                "raw_frame_count": len(sorted_frames(raw_dir)),
                "smooth_frame_count": len(sorted_frames(smooth_dir)),
                "mid_montage": str(montage_path) if montage_path.exists() else None,
                "error": error_text,
            }
        )
    return scene_report


def write_report(scene_reports: list[dict[str, Any]]) -> None:
    report_path = REPORT_ROOT / "README_BENCHMARK.md"
    lines = [
        "# Benchmark Suite Report",
        "",
        "This folder contains short grayscale benchmark scenes run through each available colorization backend, then through `flow_chroma` using `flow_mix=0.9` and `motion_strength=1.2`.",
        "",
        f"Frames are extracted at `{BENCHMARK_FPS} fps` for analysis speed.",
        "",
        "## Scenes",
        "",
    ]
    for scene_report in scene_reports:
        scene = scene_report["scene"]
        lines.extend(
            [
                f"### {scene['label']}",
                f"- Category: `{scene['category']}`",
                f"- Source: `{scene['source']}`",
                f"- Start / duration: `{scene['start']}s / {scene['duration']}s`",
                f"- Notes: {scene['notes']}",
                f"- Frame count: `{scene_report['frame_count']}`",
                "",
                "#### Runs",
                "",
            ]
        )
        for model in scene_report["models"]:
            lines.extend(
                [
                    f"- `{model['model_key']}`:",
                    f"  - raw: `{model['raw_dir']}`",
                    f"  - flow_chroma: `{model['smooth_dir']}`",
                    f"  - raw time: `{model['raw_seconds']}s`",
                    f"  - smoothing time: `{model['smooth_seconds']}s`",
                    f"  - screenshot: `{model['mid_montage']}`",
                    f"  - error: `{model['error']}`",
                ]
            )
        lines.extend(
            [
                "",
                "#### Review Notes",
                "",
                "- Fill in visual strengths here after reviewing the montage screenshots.",
                "- Fill in failures here after reviewing the montage screenshots.",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    reports = []
    for scene in SCENES:
        print(f"[benchmark] scene={scene.key}")
        reports.append(benchmark_scene(scene))
        (REPORT_ROOT / "benchmark_results.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    write_report(reports)
    print(f"[done] benchmark outputs -> {REPORT_ROOT}")


if __name__ == "__main__":
    main()
