from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from . import networks
from . import util as inst_util
from .colorize_inst import (
    load_weights,
    select_device,
    make_opt,
    prepare_transforms,
    run_siggraph,
)
from .siggraph_loader import load_eccv16, load_siggraph17

DEFAULT_IMAGE_SIZE = 256


def _list_frames(frame_paths: Iterable[Path | str]) -> list[Path]:
    paths = [Path(p) for p in frame_paths]
    return sorted([p for p in paths if p.is_file()], key=lambda p: p.name)


def colorize_frames_inst(
    frame_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    style: str = "siggraph17",
    device: Optional[str] = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    dtype: str = "float32",
) -> None:
    """
    Colorize a set of grayscale frames using InstColorization.

    Args:
        frame_paths: Iterable of image file paths (png/jpg/bmp/tiff).
        output_dir: Directory where colorized frames are written.
        style: "siggraph17" (default) or "eccv16".
        device: torch device string; auto-select if None.
        image_size: resize target for inference.
        dtype: "float32" (default) or "float16".
    """
    target_device = select_device(device)
    target_dtype = torch.float16 if dtype == "float16" else torch.float32
    if target_dtype == torch.float16 and target_device.type == "cpu":
        target_dtype = torch.float32

    transform = prepare_transforms(image_size)
    opt = make_opt(image_size)

    net = networks.SIGGRAPHGenerator(
        opt.input_nc + opt.output_nc + 1,
        opt.output_nc,
        norm_layer=networks.get_norm_layer(opt.norm),
        use_tanh=True,
        classification=opt.classification,
    )
    net.to(device=target_device, dtype=target_dtype)
    net.eval()

    weight_path = load_siggraph17() if style == "siggraph17" else load_eccv16()
    load_weights(net, str(weight_path), target_device)

    paths = _list_frames(frame_paths)
    if not paths:
        print("[warn] InstColorization: no frames to colorize")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for frame_path in paths:
        img = Image.open(frame_path).convert("RGB")
        full_tensor = transform(img)
        full_lab = inst_util.get_colorization_data([full_tensor.unsqueeze(0)], opt, ab_thresh=0, p=1.0)
        if full_lab is None:
            continue

        with torch.inference_mode():
            out_reg = run_siggraph(net, full_lab, opt, target_device, target_dtype)
            out_rgb = inst_util.lab2rgb(
                torch.cat((full_lab["A"].to(target_device, target_dtype), out_reg), dim=1), opt
            )

        out_np = torch.clamp(out_rgb, 0.0, 1.0).detach().cpu().numpy()[0].transpose(1, 2, 0)
        out_img = Image.fromarray((out_np * 255).astype(np.uint8))
        out_img.save(out_dir / frame_path.name)


__all__ = ["colorize_frames_inst"]
