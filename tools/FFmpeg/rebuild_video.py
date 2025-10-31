"""Rebuild a video from a folder of numbered frame images using ffmpeg."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Sequence

import imageio_ffmpeg


_CODEC_MAP: dict[str, tuple[str, str]] = {
    "h264": ("h264_nvenc", "libx264"),
    "h265": ("hevc_nvenc", "libx265"),
    "av1": ("av1_nvenc", "libaom-av1"),
}


def _resolve_frames_pattern(frames_path: Path) -> tuple[str, str]:
    candidates = sorted(frames_path.glob("*.jpg"))
    ext = ".jpg"
    if not candidates:
        candidates = sorted(frames_path.glob("*.png"))
        ext = ".png"
    if not candidates:
        raise FileNotFoundError(f"No .jpg or .png frames found in {frames_path}")

    first = candidates[0].name
    match = re.match(r"(.*?)(\d+)(\.\w+)$", first)
    if not match:
        raise ValueError(f"Cannot infer frame numbering pattern from '{first}'")
    prefix, digits, suffix = match.groups()
    pattern = f"{prefix}%0{len(digits)}d{suffix}"
    return pattern, ext


def _available_encoders(ffmpeg_path: str) -> set[str]:
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.SubprocessError:
        return set()

    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("V"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            encoders.add(parts[1])
    return encoders


def _select_encoder(codec: str, prefer_gpu: bool, encoders: set[str]) -> str:
    gpu_encoder, cpu_encoder = _CODEC_MAP[codec]
    if prefer_gpu and (not encoders or gpu_encoder in encoders):
        return gpu_encoder
    return cpu_encoder


def _print_command(cmd: Sequence[str]) -> None:
    printable = " ".join(shlex.quote(arg) for arg in cmd)
    print(f"[info] ffmpeg command: {printable}")


def build_video_from_frames(
    frames_dir: str,
    output_path: str,
    codec: str = "h264",
    fps: int = 24,
    prefer_gpu: bool = True,
) -> None:
    codec = codec.lower()
    if codec not in _CODEC_MAP:
        raise ValueError(f"Unsupported codec '{codec}'. Expected one of {tuple(_CODEC_MAP)}.")

    frames_path = Path(frames_dir).expanduser().resolve()
    if not frames_path.is_dir():
        raise NotADirectoryError(f"Frames directory not found: {frames_path}")

    pattern, ext = _resolve_frames_pattern(frames_path)
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    encoders = _available_encoders(ffmpeg_path) if prefer_gpu else set()
    encoder = _select_encoder(codec, prefer_gpu, encoders)

    output_file = Path(output_path).expanduser()
    if codec == "av1":
        output_file = output_file.with_suffix(".mkv")

    preset = "p4" if encoder.endswith("nvenc") else "medium"
    input_pattern = str((frames_path / pattern).as_posix())

    cmd = [
        ffmpeg_path,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        input_pattern,
        "-c:v",
        encoder,
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        str(output_file),
    ]

    _print_command(cmd)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"ffmpeg failed with return code {err.returncode} when encoding {frames_path}"
        ) from err

    print(f"[ok] video written to {output_file}")
