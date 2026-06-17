from pathlib import Path
from tools.console import status
# import imageio_ffmpeg

def _clean_path_text(text: str) -> str:
    t = text.strip().strip("'\"")
    if t.lower().startswith("file://"):
        t = t.split("://", 1)[1]
    return t

def get_input_video_path(allowed_exts: set[str]) -> Path:
    while True:
        print("Enter path to input video:")
        ptxt = _clean_path_text(input("> "))
        if not ptxt:
            status("[warn] empty input. try again.")
            continue

        p = Path(ptxt).expanduser().resolve()
        if not p.exists():
            status(f"[warn] not found: {p}")
            continue
        if p.is_dir():
            status("[warn] that is a folder. please provide a file.")
            continue
        if p.suffix.lower() not in allowed_exts:
            exts = ", ".join(sorted(allowed_exts))
            status(f"[warn] unsupported extension '{p.suffix}'. allowed: {exts}")
            continue

        # # confirm that imageio-ffmpeg can find its ffmpeg binary
        # ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        # print(f"[ok] Found bundled FFmpeg at: {ffmpeg_path}")
        return p
