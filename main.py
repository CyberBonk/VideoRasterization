# main.py
# role:
#   orchestrates the full video colorization pipeline
# flow:
#   1) select input video
#   2) extract frames via FFmpeg
#   3) choose AI model (Zhang)
#   4) run colorization (optimized CPU or GPU)
#   5) temporal smoothing (ONNX / NumPy)
#   6) generate report
#   7) rebuild final video output

from pathlib import Path
import sys

import torch

# --------------------------------------------------------------------
# --- 1) Imports -----------------------------------------------------
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video_pipeline.env import HAS_IPEX, LOGICAL  # noqa: F401
from video_pipeline.frame_extraction import extract_frames
from video_pipeline.input_handling import select_input_video
from video_pipeline.colorization import run_colorization
from video_pipeline.smoothing import apply_temporal_smoothing_step
from video_pipeline.window_prompt import ask_temporal_window
from video_pipeline.reporting import generate_report
from video_pipeline.reconstruction import rebuild_video_output

# --------------------------------------------------------------------
# --- 2) Main pipeline -----------------------------------------------
# --------------------------------------------------------------------
def main():
    print("=== VideoRasterization start ===")
    temp_root = ROOT / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)


    video_path = select_input_video()
    if not video_path:
        return

    frames_dir = extract_frames(video_path, temp_root)
    if not frames_dir:
        return


    model_name = "colorize_zhang"
    use_gpu = torch.cuda.is_available()
    print(f"[info] GPU available: {use_gpu}")
    window_size = ask_temporal_window()
    color_dir = run_colorization(frames_dir, model_name, use_gpu)
    smooth_dir = apply_temporal_smoothing_step(color_dir, window_size)
    generate_report(frames_gray_dir=frames_dir, frames_color_dir=color_dir)
    rebuild_video_output(color_dir=color_dir, smooth_dir=smooth_dir, source_video=video_path, fps=24)

    print("[done] pipeline finished.")
    print("=== End ===")


if __name__ == "__main__":
    main()
