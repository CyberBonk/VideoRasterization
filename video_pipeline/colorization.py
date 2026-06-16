"""Frame colorization helpers."""

from importlib import import_module
from pathlib import Path

from .env import HAS_IPEX, LOGICAL, ipex

model_selector = import_module("tools.model_selector")

ROOT = Path(__file__).resolve().parent.parent
INST_MODEL_NAMES = {
    "instcolorization2025",
    "inst_colorization",
    "colorize_instcolorization2025",
}


def run_colorization(
    frames_path: Path, model_name: str, use_gpu: bool, **model_options
) -> Path:
    color_dir = frames_path.parent / f"{frames_path.name}_colorized"
    color_dir.mkdir(parents=True, exist_ok=True)

    if HAS_IPEX and not use_gpu and ipex:
        print("[info] optimizing with Intel oneDNN (IPEX)...")
        ipex.enable_onednn_fusion(True)
        try:
            ipex.set_fp32_math_mode(mode="BF16")
        except Exception:
            pass

    if model_name in INST_MODEL_NAMES:
        from tools.AImodels.instcolorization2025.inst_backend import colorize_frames_inst

        frame_paths = [
            p
            for p in sorted(frames_path.glob("*"))
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ]
        colorize_frames_inst(
            frame_paths=frame_paths,
            output_dir=color_dir,
            style="eccv16",
            device="cuda" if use_gpu else None,
            image_size=256,
        )
    else:
        model_selector.run_colorizer(
            model_name=model_name,
            frames_dir=frames_path,
            color_dir=color_dir,
            models_dir=ROOT / "models",
            zhang_variant=None,
            preview=False,
            use_gpu=use_gpu,
            batch_size=12,
            num_threads=LOGICAL,
            input_size=224,
            progress=True,
            prefetch_workers=LOGICAL // 4,
            save_workers=2,
            **model_options,
        )
    print(f"[ok] colorization complete: {color_dir}")
    return color_dir


__all__ = ["run_colorization"]
