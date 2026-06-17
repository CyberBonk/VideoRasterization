"""Video frame extraction utilities."""

from importlib import import_module
from pathlib import Path

import imageio_ffmpeg
from tools.console import status

ffmpeg_tools = import_module("tools.FFmpeg.FFmpeg_utilization")


def extract_frames(video_path: Path, temp_root: Path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    frames_dir = ffmpeg_tools.extract_frames(ffmpeg_path, video_path, temp_root)
    if not frames_dir or not Path(frames_dir).exists():
        status("[error] failed to extract frames.")
        return None
    status(f"[ok] extracted frames dir: {frames_dir}")
    return Path(frames_dir)


__all__ = ["extract_frames"]
