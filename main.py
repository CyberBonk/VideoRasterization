# main.py


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
from tools import model_selector, inference_options
from tools.console import status

# --------------------------------------------------------------------
# --- 2) Main pipeline -----------------------------------------------
# --------------------------------------------------------------------
def main():
    status("=== VideoRasterization start ===")
    temp_root = ROOT / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)


    video_path = select_input_video()
    if not video_path:
        return

    frames_dir = extract_frames(video_path, temp_root)
    if not frames_dir:
        return

    available_models = model_selector.scan_available_models(ROOT / "tools" / "AImodels")
    model_name = inference_options.choose_colorization_model(available_models)
    model_options = inference_options.choose_chromanet_options(model_name)


    use_gpu = torch.cuda.is_available()
    status(f"[info] GPU available: {use_gpu}")
    window_size = ask_temporal_window()
    color_dir = run_colorization(frames_dir, model_name, use_gpu, **model_options)
    smooth_dir = apply_temporal_smoothing_step(color_dir, window_size)
    generate_report(frames_gray_dir=frames_dir, frames_color_dir=color_dir)
    rebuild_video_output(color_dir=color_dir, smooth_dir=smooth_dir, source_video=video_path, fps=24)

    status("[done] pipeline finished.")
    status("=== End ===")


if __name__ == "__main__":
    main()
