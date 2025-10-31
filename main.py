# main.py
# role:
#   orchestrates the full video colorization pipeline
# flow:
#   1) select input video
#   2) extract frames via FFmpeg
#   3) choose AI model (Zhang)
#   4) run colorization (optimized CPU or GPU)
#   5) generate report

from importlib import import_module
from pathlib import Path
import sys, os, multiprocessing as mp
import imageio_ffmpeg
import torch

# --------------------------------------------------------------------
# --- 0) Environment setup for max CPU performance -------------------
# --------------------------------------------------------------------
LOGICAL = mp.cpu_count() or 8
os.environ["OMP_NUM_THREADS"] = str(LOGICAL)
os.environ["MKL_NUM_THREADS"] = str(LOGICAL)
os.environ["OPENBLAS_NUM_THREADS"] = str(LOGICAL)
os.environ["NUMEXPR_NUM_THREADS"] = str(LOGICAL)

torch.set_num_threads(LOGICAL // 2)
torch.set_num_interop_threads(LOGICAL // 2)

# Optional Intel oneDNN/IPEX detection
try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False

# --------------------------------------------------------------------
# --- 1) Import helpers ----------------------------------------------
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

input_selector = import_module("tools.input_selector")
ffmpeg_tools = import_module("tools.FFmpeg.FFmpeg_utilization")
model_selector = import_module("tools.model_selector")
report = import_module("tools.preview_report")

# --------------------------------------------------------------------
# --- 2) main pipeline -----------------------------------------------
# --------------------------------------------------------------------
def main():

    # ----- INPUT -----
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    video_path = input_selector.get_input_video_path(allowed_exts=VIDEO_EXTS)
    if not video_path:
        print("[error] no video selected. exiting.")
        return
    print(f"[ok] selected video: {video_path}")

    # ----- EXTRACT -----
    temp_root = ROOT / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    frames_dir = ffmpeg_tools.extract_frames(ffmpeg_path, video_path, temp_root)
    if not frames_dir or not Path(frames_dir).exists():
        print("[error] failed to extract frames. exiting.")
        return
    print(f"[ok] extracted frames dir: {frames_dir}")

    # ----- MODEL -----
    models_root = ROOT / "tools" / "AImodels"
    model_name = model_selector.select_model(models_root)
    if not model_name:
        print("[error] no AI models found. exiting.")
        return
    print(f"[ok] selected model: {model_name}")

    zhang_variant = None
    use_gpu = torch.cuda.is_available()
    print(f"[info] GPU available: {use_gpu}")

    # ----- SETTINGS -----
    try:
        window_size = int(input("Enter temporal smoothing window (odd number, default=9): ") or 9)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size < 3:
            window_size = 3
        print(f"[info] using temporal window size: {window_size}")
    except Exception:
        window_size = 9
        print("[warn] invalid input, using default window size 9.")

    # ----- COLORIZE -----
    frames_path = Path(frames_dir)
    color_dir = frames_path.parent / f"{frames_path.name}_colorized"
    color_dir.mkdir(parents=True, exist_ok=True)

    # if IPEX is installed, prepare optimization
    if HAS_IPEX and not use_gpu:
        print("[info] optimizing model via Intel IPEX oneDNN backend...")
        ipex.enable_onednn_fusion(True)
        ipex.set_fp32_math_mode(mode="BF16")

    model_selector.run_colorizer(
        model_name=model_name,
        frames_dir=frames_path,
        color_dir=color_dir,
        models_dir=ROOT / "models",
        zhang_variant=zhang_variant,
        preview=False,
        use_gpu=use_gpu,
        batch_size=12,
        num_threads=LOGICAL,
        input_size=224,
        progress=True,
        prefetch_workers=LOGICAL // 4,
        save_workers=2,
    )

    print(f"[ok] colorization complete: {color_dir}")

    # ----- REPORT -----
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

    print("[done] pipeline finished.")


if __name__ == "__main__":
    main()
