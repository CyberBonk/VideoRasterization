"""Temporal smoothing helpers."""

from pathlib import Path

from tools.TemporalSmoothing import apply_temporal_smoothing


def apply_temporal_smoothing_step(color_dir: Path, window_size: int):
    smooth_dir = color_dir.parent / f"{color_dir.name}_TemporalSmoothed"
    smooth_dir.mkdir(parents=True, exist_ok=True)

    apply_temporal_smoothing(
        input_folder=color_dir,
        output_folder=smooth_dir,
        use_onnx=True,
        window_size=window_size,
    )
    print(f"[ok] temporal smoothing complete: {smooth_dir}")
    return smooth_dir


__all__ = ["apply_temporal_smoothing_step"]
