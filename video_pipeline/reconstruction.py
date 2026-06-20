"""Video reconstruction utilities."""

from __future__ import annotations

from pathlib import Path

from tools.console import status
from tools.FFmpeg.rebuild_video import build_video_from_frames


def rebuild_video_output(
    color_dir: Path,
    smooth_dir: Path | None,
    source_video: Path,
    fps: int = 24,
) -> None:
    final_frames = smooth_dir if smooth_dir and smooth_dir.exists() else color_dir
    label = "smoothed" if smooth_dir and final_frames == smooth_dir else "colorized"

    default_output = source_video.with_name(f"{source_video.stem}_{label}.mp4")
    try:
        build_video_from_frames(
            frames_dir=str(final_frames),
            output_path=str(default_output),
            codec="h264",
            fps=fps,
            prefer_gpu=True,
            source_audio=str(source_video),
        )
        status(f"[ok] video rebuild saved: {default_output}")
        return default_output
    except Exception as e:
        status(f"[warn] video rebuild failed: {e}")
        raise e


__all__ = ["rebuild_video_output"]
