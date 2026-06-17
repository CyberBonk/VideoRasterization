"""Window selection prompt for temporal smoothing."""

from tools.console import status


def ask_temporal_window() -> int | None:
    """Ask for temporal smoothing window size (default: off, recommended: 9)."""
    raw = input("Temporal smoothing window (odd number, blank = off, recommended = 9): ").strip()
    if raw == "":
        status("[info] temporal smoothing disabled by default.")
        return None
    try:
        window_size = int(raw)
    except Exception:
        status("[warn] invalid input, disabling temporal smoothing.")
        return None

    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        window_size = 3
    status(f"[info] using temporal window size: {window_size}")
    return window_size


__all__ = ["ask_temporal_window"]
