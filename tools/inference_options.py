from typing import Literal

def choose_zhang_variant() -> Literal["eccv16", "siggraph17"]:
    print("Choose the Zhang model format:")
    print("1) eccv16  (2016)")
    print("2) siggraph17  (2017, recommended, default)")
    ans = input("choose the model format (default [2]): ").strip()

    if ans == "1":
        return "eccv16"
    return "siggraph17"

def ask_preview() -> bool:
    ans = input("Show preview windows (y/N)? ").strip().lower()
    return ans in {"y", "yes"}
