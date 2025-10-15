from importlib import import_module
from pathlib import Path
import sys
import imageio_ffmpeg

# ensure the repo root is importable on any machine (no absolute paths)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    input_selector = import_module("tools.input_selector")
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    video_path = input_selector.get_input_video_path(allowed_exts=VIDEO_EXTS)
    print(f"[ok] selected video: {video_path}")

    temp_root = Path("temp")
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    extract = import_module("tools.FFmpeg.FFmpeg_utilization")
    frames_dir = extract.extract_frames(ffmpeg_path, video_path, temp_root)
    if frames_dir is not None:
        print(f"[ok] extracted frames dir: {frames_dir}")




if __name__ == "__main__":
    main()
