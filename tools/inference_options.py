from typing import Literal, Sequence


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
