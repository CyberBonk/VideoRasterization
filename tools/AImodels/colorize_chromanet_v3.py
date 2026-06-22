"""VideoRasterization adapter for locally trained ChromaNet v3."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from tools.console import status

ROOT = Path(__file__).resolve().parents[2]
CHROMANET_ROOT = ROOT / "ChromaNet_v3_complete" / "chromanet_v3"
DEFAULT_CHECKPOINT = ROOT / "ChromaNet_v3_complete" / "chromanet_v3" / "checkpoints" / "checkpoint_latest.pth"


def colorize_dir(
    frames_dir: Path,
    out_dir: Path,
    models_dir: Optional[Path] = None,
    use_gpu: bool = True,
    input_size: int = 256,
    batch_size: Optional[int] = 12,

    prefetch_workers: int = 4,
    save_workers: int = 4,
    max_prefetch_batches: int = 2,
    confidence_threshold: float = 0.3,
    saturation_gain: float = 1.0,
    grain_amount: float = 0.0,
    style_preset: str = "realistic",
    checkpoint: Optional[Path] = None,
    cancel_event=None,
    pause_event=None,
    **_: object,
) -> None:
    """Colorize extracted frames using trained ChromaNet v3 checkpoint."""
    _ = models_dir
    checkpoint_path = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"ChromaNet checkpoint not found: {checkpoint_path}. Train model first."
        )

    chromanet_path = str(CHROMANET_ROOT)
    if chromanet_path not in sys.path:
        sys.path.insert(0, chromanet_path)

    from inference.colorizer import ChromaColorizer

    device = "cuda" if use_gpu else "cpu"
    colorizer = ChromaColorizer(
        checkpoint_path=checkpoint_path,
        device=device,
        image_size=input_size,
        save_confidence=False,
        confidence_threshold=confidence_threshold,
        saturation_gain=saturation_gain,
        grain_amount=grain_amount,
    )
    status(
        f"[start] ChromaNet v3 | checkpoint={checkpoint_path.name} | device={device} "
        f"| style={style_preset} "
        f"| confidence_threshold={confidence_threshold:.2f} "
        f"| saturation_gain={saturation_gain:.2f} "
        f"| grain_amount={grain_amount:.2f}"
    )
    colorizer.colorize_folder(
        frames_dir,
        out_dir,
        batch_size=batch_size or 1,
        prefetch_workers=prefetch_workers,
        save_workers=save_workers,
        max_prefetch_batches=max_prefetch_batches,
        cancel_event=cancel_event,
        pause_event=pause_event,
    )


__all__ = ["colorize_dir"]
