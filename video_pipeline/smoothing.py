"""Temporal smoothing helpers."""

from pathlib import Path

from tools.console import status
from tools.TemporalSmoothing import apply_temporal_smoothing


def apply_temporal_smoothing_step(
    gray_dir: Path,
    color_dir: Path,
    smoothing_options: dict | None,
):
    mode = (smoothing_options or {}).get("mode", "off")
    if mode == "off":
        status("[info] temporal smoothing skipped; using raw colorized frames.")
        return None

    suffix = "TemporalSmoothed" if mode == "legacy_average" else "FlowChromaStabilized"
    smooth_dir = color_dir.parent / f"{color_dir.name}_{suffix}"
    smooth_dir.mkdir(parents=True, exist_ok=True)

    apply_temporal_smoothing(
        input_folder=color_dir,
        output_folder=smooth_dir,
        gray_input_folder=gray_dir,
        mode=mode,
        use_onnx=mode == "legacy_average",
        window_size=int((smoothing_options or {}).get("window_size", 9)),
        flow_mix=float((smoothing_options or {}).get("flow_mix", 0.75)),
        motion_strength=float((smoothing_options or {}).get("motion_strength", 1.0)),
    )
    status(f"[ok] temporal smoothing complete ({mode}): {smooth_dir}")
    return smooth_dir


__all__ = ["apply_temporal_smoothing_step"]
