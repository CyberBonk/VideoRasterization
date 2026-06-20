"""Frame colorization helpers."""

from importlib import import_module
from pathlib import Path

from .env import HAS_IPEX, LOGICAL, ipex
from .frame_deduplication import deduplicate_consecutive_frames, expand_duplicate_outputs
from tools.console import status

model_selector = import_module("tools.model_selector")

ROOT = Path(__file__).resolve().parent.parent
CHROMANET_MODEL_NAMES = {"colorize_chromanet_v3", "chromanet_v3", "chromanet"}
INST_MODEL_NAMES = {
    "instcolorization2025",
    "inst_colorization",
    "colorize_instcolorization2025",
}
ENHANCED_ZHANG_MODEL_NAMES = {"Enhanced Zhang (Bebo's Experiment)"}


def run_colorization(
    frames_path: Path, model_name: str, use_gpu: bool, **model_options
) -> Path:
    color_dir = frames_path.parent / f"{frames_path.name}_colorized"
    color_dir.mkdir(parents=True, exist_ok=True)
    model_frames_path, duplicate_map = deduplicate_consecutive_frames(frames_path)

    if HAS_IPEX and not use_gpu and ipex:
        status("[info] optimizing with Intel oneDNN (IPEX)...")
        ipex.enable_onednn_fusion(True)
        try:
            ipex.set_fp32_math_mode(mode="BF16")
        except Exception:
            pass

    if model_name in INST_MODEL_NAMES:
        from tools.AImodels.instcolorization2025.inst_backend import colorize_frames_inst

        frame_paths = [
            p
            for p in sorted(model_frames_path.glob("*"))
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ]
        colorize_frames_inst(
            frame_paths=frame_paths,
            output_dir=color_dir,
            style=model_options.get("style", "trained"),
            device="cuda" if use_gpu else None,
            image_size=256,
        )
    else:
        input_size = model_options.pop(
            "input_size",
            256 if model_name in ENHANCED_ZHANG_MODEL_NAMES else 224,
        )
        model_selector.run_colorizer(
            model_name=model_name,
            frames_dir=model_frames_path,
            color_dir=color_dir,
            models_dir=ROOT / "models",
            zhang_variant=None,
            preview=False,
            use_gpu=use_gpu,
            batch_size=12,
            num_threads=LOGICAL,
            input_size=input_size,
            progress=True,
            prefetch_workers=4 if model_name in CHROMANET_MODEL_NAMES else max(2, LOGICAL // 4),
            save_workers=4 if model_name in CHROMANET_MODEL_NAMES else 2,
            max_prefetch_batches=2,
            **model_options,
        )
    expand_duplicate_outputs(color_dir, duplicate_map)
    status(f"[ok] colorization complete: {color_dir}")
    return color_dir


__all__ = ["run_colorization"]
