"""User input helpers."""

from importlib import import_module
from pathlib import Path
from tools.console import status

input_selector = import_module("tools.input_selector")


def select_input_video():
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    video_path = input_selector.get_input_video_path(allowed_exts=VIDEO_EXTS)
    if not video_path:
        status("[error] no video selected. exiting.")
        return None
    status(f"[ok] selected video: {video_path}")
    return Path(video_path)


__all__ = ["select_input_video"]
