from pathlib import Path
import importlib
from typing import Optional

def scan_available_models(models_root: Path) -> list[str]:
    if not models_root.exists():
        return []
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

def run_colorizer(model_name, frames_dir, color_dir, models_dir, **kwargs):
    mod = importlib.import_module(f"tools.AImodels.{model_name}")
    if not hasattr(mod, "colorize_dir"):
        print(f"[error] {model_name}.py has no function 'colorize_dir'")
        return
    if "zhang_variant" in kwargs and "variant" not in kwargs:
        kwargs["variant"] = kwargs.pop("zhang_variant")
    mod.colorize_dir(
        frames_dir=Path(frames_dir),
        out_dir=Path(color_dir),
        models_dir=Path(models_dir),
        **kwargs,
    )


