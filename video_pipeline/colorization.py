"""Frame colorization helpers."""

from importlib import import_module
from pathlib import Path

from .env import HAS_IPEX, LOGICAL, ipex

model_selector = import_module("tools.model_selector")

ROOT = Path(__file__).resolve().parent.parent


def run_colorization(frames_path: Path, model_name: str, use_gpu: bool) -> Path:
    color_dir = frames_path.parent / f"{frames_path.name}_colorized"
    color_dir.mkdir(parents=True, exist_ok=True)

    if HAS_IPEX and not use_gpu and ipex:
        print("[info] optimizing with Intel oneDNN (IPEX)...")
        ipex.enable_onednn_fusion(True)
        try:
            ipex.set_fp32_math_mode(mode="BF16")
        except Exception:
            pass

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
    )
    print(f"[ok] colorization complete: {color_dir}")
    return color_dir


__all__ = ["run_colorization"]
