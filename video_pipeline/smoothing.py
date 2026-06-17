"""Temporal smoothing helpers."""

from pathlib import Path

from tools.console import status
from tools.TemporalSmoothing import apply_temporal_smoothing


def apply_temporal_smoothing_step(color_dir: Path, window_size: int | None):
    """
    Run temporal smoothing if requested.

    When window_size is falsy (None/0), this is a no-op so the pipeline can
    reuse the existing colorized frames without duplicating them.
    """
    if not window_size or window_size < 3:
        status("[info] temporal smoothing skipped; using raw colorized frames.")
        return None

    smooth_dir = color_dir.parent / f"{color_dir.name}_TemporalSmoothed"
    smooth_dir.mkdir(parents=True, exist_ok=True)

    apply_temporal_smoothing(
        input_folder=color_dir,
        output_folder=smooth_dir,
        use_onnx=True,
        window_size=window_size,
    )
    status(f"[ok] temporal smoothing complete: {smooth_dir}")
    return smooth_dir


__all__ = ["apply_temporal_smoothing_step"]
