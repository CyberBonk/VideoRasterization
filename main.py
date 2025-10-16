# main.py
# role:
# - central runner that calls the other modules
# - all user choices happen here (so a future GUI can reuse the same logic)
#
# flow:
#   1) ask user for input video (tools.input_selector)
#   2) extract frames + audio to temp/ (tools.FFmpeg.FFmpeg_utilization)
#   3) pick AI model (tools.model_selector)  ← scans tools/AImodels
#      - if model is Zhang, also ask for variant + preview (tools.inference_options)
#   4) run colorizer (tools.model_selector.run_colorizer)
#   5) (optional next steps: smoothing, rebuild video) – add later

from importlib import import_module
from pathlib import Path
import sys
import imageio_ffmpeg

# ---- force full CPU usage for BLAS backends (set before torch import) ----
import os
import multiprocessing as mp
LOGICAL = mp.cpu_count() or 8
os.environ.setdefault("OMP_NUM_THREADS", str(LOGICAL))
os.environ.setdefault("MKL_NUM_THREADS", str(LOGICAL))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(LOGICAL))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(LOGICAL))


# make repo root importable (portable on any machine)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    # -----------------------------
    # 1) INPUT: pick the video file
    #    who: tools.input_selector.get_input_video_path
    # -----------------------------
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    input_selector = import_module("tools.input_selector")
    video_path = input_selector.get_input_video_path(allowed_exts=VIDEO_EXTS)
    if not video_path:
        print("[error] no video selected. exiting.")
        return
    print(f"[ok] selected video: {video_path}")

    # -----------------------------
    # 2) EXTRACT: frames (and audio if you do that inside your FFmpeg module)
    #    who: tools.FFmpeg.FFmpeg_utilization.extract_frames
    #    note: we pass a known ffmpeg path (bundled or system)
    # -----------------------------
    temp_root = ROOT / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()  # or use your bundled one
    ffmpeg_tools = import_module("tools.FFmpeg.FFmpeg_utilization")

    # expected return: Path to frames folder (e.g., temp/frames_YYYYmmdd/000001.png ...)
    frames_dir = ffmpeg_tools.extract_frames(ffmpeg_path, video_path, temp_root)
    if not frames_dir or not Path(frames_dir).exists():
        print("[error] failed to extract frames. exiting.")
        return
    print(f"[ok] extracted frames dir: {frames_dir}")

    # -----------------------------
    # 3) MODEL CHOICE: pick available AI model by scanning tools/AImodels
    #    who: tools.model_selector.select_model
    #    rule: if only one model present → auto-pick; if none → error
    # -----------------------------
    model_selector = import_module("tools.model_selector")
    models_root = ROOT / "tools" / "AImodels"
    model_name = model_selector.select_model(models_root)
    if not model_name:
        print("[error] no AI models found. exiting.")
        return
    print(f"[ok] selected model: {model_name}")

    # -----------------------------
    # 3.1) MODEL-SPECIFIC OPTIONS (still chosen in main.py)
    #      example: Zhang has two variants (eccv16 / siggraph17) and optional preview
    #      who: tools.inference_options
    # -----------------------------
    zhang_variant = None
    preview = False
    use_gpu = True # set True if you want to try CUDA later


    # -----------------------------
    # 4) COLORIZE: run the chosen model on the frames
    #    who: tools.model_selector.run_colorizer
    #    params:
    #      - frames_dir: where PNGs live
    #      - color_dir: where colored PNGs go
    #      - models_dir: keep for API consistency (unused by PyTorch Zhang)
    #      - zhang_variant / preview / use_gpu: optional args handled by model if supported
    # -----------------------------
    frames_path = Path(frames_dir)
    color_dir = frames_path.parent / f"{frames_path.name}_colorized"
    color_dir.mkdir(parents=True, exist_ok=True)

    model_selector.run_colorizer(
        model_name=model_name,
        frames_dir=frames_path,
        color_dir=color_dir,
        models_dir=ROOT / "models",
        zhang_variant=zhang_variant,  # selector maps to 'variant'
        preview=False,
        use_gpu=False,  # CPU path
        batch_size=None,  # auto from threads
        num_threads=None,  # auto = logical cores
        input_size=256,  # 224 speeds up a bit if you want
        progress=True,
        prefetch_workers=None,  # auto (≈ threads/2, capped)
        save_workers=4,  # try 0, 2, or 4 depending on disk
    )

    print(f"[ok] colorization complete: {color_dir}")

    # -----------------------------
    # 5) NEXT STEPS (placeholders)
    #    - temporal smoothing module
    #    - rebuild video with FFmpeg (merge color_dir + audio → output.mp4)
    #    we will add them later so main.py stays the single caller.
    # -----------------------------
    # TODO: call tools.temporal.smooth_sequence(color_dir, alpha=?)
    # TODO: call tools.FFmpeg.FFmpeg_utilization.rebuild_video(...)

    print("[done] pipeline finished.")

    # 5) one-shot report (PNG + JSON)
    report = import_module("tools.preview_report")
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_png = reports_dir / "report.png"
    report_json = reports_dir / "report.json"

    try:
        report.generate_report(
            frames_gray_dir=frames_path,
            frames_color_dir=color_dir,
            out_png=report_png,
            out_json=report_json,
        )
        print(f"[ok] report saved:\n - {report_png}\n - {report_json}")
    except Exception as e:
        print(f"[warn] report failed: {e}")


if __name__ == "__main__":
    main()
