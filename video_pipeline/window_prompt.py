"""Prompt helpers for temporal smoothing."""

from tools.console import status


def ask_temporal_smoothing_options() -> dict:
    print("Temporal smoothing mode:")
    print("0) off")
    print("1) flow_chroma  (recommended, motion-compensated color stabilization)")
    print("2) legacy_average  (older sliding-window blend, can blur motion)")
    raw = input("mode (default [1]): ").strip()
    if raw == "0":
        status("[info] temporal smoothing disabled.")
        return {"mode": "off"}
    if raw == "2":
        window_raw = input("legacy window size (odd number, default [9]): ").strip()
        try:
            window_size = int(window_raw or "9")
        except Exception:
            status("[warn] invalid legacy window; using 9.")
            window_size = 9
        if window_size % 2 == 0:
            window_size -= 1
        if window_size < 3:
            window_size = 3
        status(f"[info] temporal smoothing mode=legacy_average window={window_size}")
        return {"mode": "legacy_average", "window_size": window_size}

    mix_raw = input("flow memory 0.00-1.00 (default [0.75]): ").strip()
    strength_raw = input("motion confidence strength 0.00-1.50 (default [1.00]): ").strip()
    try:
        flow_mix = float(mix_raw or "0.75")
    except Exception:
        flow_mix = 0.75
    try:
        motion_strength = float(strength_raw or "1.00")
    except Exception:
        motion_strength = 1.0
    flow_mix = max(0.0, min(1.0, flow_mix))
    motion_strength = max(0.0, min(1.5, motion_strength))
    status(
        f"[info] temporal smoothing mode=flow_chroma flow_mix={flow_mix:.2f} "
        f"motion_strength={motion_strength:.2f}"
    )
    return {
        "mode": "flow_chroma",
        "flow_mix": flow_mix,
        "motion_strength": motion_strength,
    }


__all__ = ["ask_temporal_smoothing_options"]
