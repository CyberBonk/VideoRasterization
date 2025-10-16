from pathlib import Path
import importlib
from typing import Optional

def scan_available_models(models_root: Path) -> list[str]:
    if not models_root.exists():
        return []
    # a model is any file named colorize_*.py
    return [p.stem for p in models_root.glob("colorize_*.py")]

def select_model(models_root: Path) -> Optional[str]:
    available = scan_available_models(models_root)
    if not available:
        print("[error] no AI models found in tools/AImodels/")
        return None
    if len(available) == 1:
        print(f"[ok] only one model found → {available[0]}")
        return available[0]
    print("Choose model:")
    for i, name in enumerate(available, 1):
        print(f"{i}) {name}")
    ans = input("number: ").strip()
    try:
        idx = int(ans) - 1
        return available[idx]
    except Exception:
        print("[warn] invalid choice; using first.")
        return available[0]

def run_colorizer(model_name: str, frames_dir, color_dir, models_dir, *, zhang_variant=None, preview=False, use_gpu=False):
    """Dynamically import tools.AImodels.<model_name> and call colorize_dir."""
    if model_name is None:
        print("[error] no valid model selected.")
        return

    mod = importlib.import_module(f"tools.AImodels.{model_name}")
    if not hasattr(mod, "colorize_dir"):
        print(f"[error] {model_name}.py has no function 'colorize_dir'")
        return

    # Pass optional args only if the function supports them
    try:
        mod.colorize_dir(
            frames_dir=Path(frames_dir),
            out_dir=Path(color_dir),
            models_dir=Path(models_dir),
            variant=zhang_variant,    # Zhang wrapper accepts this
            preview=preview,
            use_gpu=use_gpu,
        )
    except TypeError:
        # fallback for simpler models that only take (frames_dir, out_dir, models_dir)
        mod.colorize_dir(Path(frames_dir), Path(color_dir), Path(models_dir))
