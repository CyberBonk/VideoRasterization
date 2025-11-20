"""Window selection prompt for temporal smoothing."""


def ask_temporal_window() -> int:
    """Ask for temporal smoothing window size."""
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
    return window_size


__all__ = ["ask_temporal_window"]
