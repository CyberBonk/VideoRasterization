from pathlib import Path
import subprocess

def _run_ffmpeg(args: list[str]) -> None:
    # uses system ffmpeg; ensure it's on PATH
    # raises CalledProcessError on failure
    subprocess.run(["ffmpeg", "-y", *args], check=True)

def extract_frames(input_video: Path, frames_dir: Path, qscale: int = 2, hwaccel: str|None = "auto") -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = []
    if hwaccel == "auto":
        cmd += ["-hwaccel","auto"]
    elif hwaccel and hwaccel.lower() != "none":
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", str(input_video), "-qscale:v", str(qscale), str(frames_dir / "%06d.png")]
    _run_ffmpeg(cmd)

def extract_audio(input_video: Path, out_audio: Path, hwaccel: str|None = "auto") -> None:
    out_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = []
    if hwaccel == "auto":
        cmd += ["-hwaccel","auto"]
    elif hwaccel and hwaccel.lower() != "none":
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", str(input_video), "-vn", "-acodec", "copy", str(out_audio)]
    _run_ffmpeg(cmd)
