from pathlib import Path
from typing import Optional

from tools.AImodels.instcolorization2025.inst_backend import colorize_frames_inst


def colorize_dir(
    frames_dir: Path,
    out_dir: Path,
    models_dir: Optional[Path] = None,
    style: str = "siggraph17",
    preview: bool = False,
    use_gpu: bool = False,
    batch_size: Optional[int] = None,
    num_threads: Optional[int] = None,
    input_size: int = 256,
    **_: dict,
) -> None:
    """
    Thin adapter to match the existing model_selector contract.
    """
    _ = (models_dir, preview, batch_size, num_threads)  # unused but kept for parity
    frames_dir = Path(frames_dir)
    frame_paths = [
        p
        for p in sorted(frames_dir.glob("*"))
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ]
    colorize_frames_inst(
        frame_paths=frame_paths,
        output_dir=out_dir,
        style=style,
        device="cuda" if use_gpu else None,
        image_size=input_size,
    )
