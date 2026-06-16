from typing import Literal, Sequence


CHROMANET_MODEL_NAMES = {"colorize_chromanet_v3", "chromanet_v3", "chromanet"}


def _ask_float(prompt: str, default: float, min_value: float, max_value: float) -> float:
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except Exception:
        print(f"[warn] invalid number; using {default:g}.")
        return default
    if value < min_value or value > max_value:
        print(f"[warn] out of range; using {default:g}.")
        return default
    return value


def choose_zhang_variant() -> Literal["eccv16", "siggraph17"]:
    print("Choose the Zhang model format:")
    print("1) eccv16  (2016)")
    print("2) siggraph17  (2017, recommended, default)")
    ans = input("choose the model format (default [2]): ").strip()

    if ans == "1":
        return "eccv16"
    return "siggraph17"


def choose_colorization_model(available: Sequence[str]) -> str:
    if not available:
        return "colorize_zhang"
    print("Choose AI model backend:")
    for i, name in enumerate(available, 1):
        print(f"{i}) {name}")
    ans = input(f"number (default [1]): ").strip()
    if not ans:
        return available[0]
    try:
        idx = int(ans) - 1
        return available[idx]
    except Exception:
        print("[warn] invalid choice; using first.")
        return available[0]


def choose_chromanet_options(model_name: str) -> dict:
    if model_name not in CHROMANET_MODEL_NAMES:
        return {}

    print("ChromaNet color strength:")
    print("0) default / realistic")
    print("1) mild")
    print("2) vivid")
    print("3) max / experimental")
    print("4) custom")
    raw = input("color strength (default [0]): ").strip()
    try:
        level = max(0, min(4, int(raw or "0")))
    except Exception:
        print("[warn] invalid strength; using default.")
        level = 0

    presets = {
        0: {"confidence_threshold": 0.30, "saturation_gain": 1.00, "grain_amount": 0.0},
        1: {"confidence_threshold": 0.20, "saturation_gain": 1.15, "grain_amount": 0.0},
        2: {"confidence_threshold": 0.10, "saturation_gain": 1.35, "grain_amount": 0.0},
        3: {"confidence_threshold": 0.00, "saturation_gain": 1.60, "grain_amount": 0.0},
    }
    if level == 4:
        print("Custom ChromaNet settings:")
        confidence_filter = _ask_float(
            "confidence filter 0.00-1.00 (default 1.00, lower = more color): ",
            default=1.00,
            min_value=0.0,
            max_value=1.0,
        )
        color_amount = _ask_float(
            "color amount 0.00-1.00 (default 0.50): ",
            default=0.50,
            min_value=0.0,
            max_value=1.0,
        )
        grain_amount = _ask_float(
            "film grain 0.00-1.00 (default 0.00): ",
            default=0.00,
            min_value=0.0,
            max_value=1.0,
        )
        opts = {
            "confidence_threshold": 0.30 * confidence_filter,
            "saturation_gain": 0.50 + 2.50 * color_amount,
            "grain_amount": grain_amount,
        }
    else:
        opts = presets[level]
    print(
        f"[info] ChromaNet strength={level} "
        f"confidence_threshold={opts['confidence_threshold']:.2f} "
        f"saturation_gain={opts['saturation_gain']:.2f} "
        f"grain_amount={opts['grain_amount']:.2f}"
    )
    return opts
