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
from .siggraph_loader import load_eccv16

DEFAULT_IMAGE_SIZE = 256
INST_STYLE_ECCV16 = "inst_eccv16"


def _list_frames(frame_paths: Iterable[Path | str]) -> list[Path]:
    paths = [Path(p) for p in frame_paths]
    return sorted([p for p in paths if p.is_file()], key=lambda p: p.name)


def _resolve_style(style: str) -> str:
    """
    InstColorization currently only supports the ECCV16 checkpoint.
    TODO: add inst_siggraph17 once a compatible checkpoint is trained for networks.py.
    """
    normalized = style.lower()
    if normalized in {"eccv16", INST_STYLE_ECCV16}:
        return INST_STYLE_ECCV16
    print(f"[info] InstColorization supports ECCV16 only; using ECCV16 instead of '{style}'.")
    return INST_STYLE_ECCV16


def save_colorized_fullres(orig_path: Path, color_small: np.ndarray, out_path: Path) -> None:
    """
    Save a colorized frame resized back to the original resolution.
    """
    with Image.open(orig_path) as im:
        orig_w, orig_h = im.size

    color_img = Image.fromarray(color_small.astype("uint8"))
    color_img = color_img.resize((orig_w, orig_h), Image.BICUBIC)
    color_img.save(out_path, quality=95)


def colorize_frames_inst(
    frame_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    style: str = INST_STYLE_ECCV16,
    device: Optional[str] = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    dtype: str = "float32",
) -> None:
    """
    Colorize a set of grayscale frames using InstColorization.

    Args:
        frame_paths: Iterable of image file paths (png/jpg/bmp/tiff).
        output_dir: Directory where colorized frames are written.
        style: InstColorization backend label (ECCV16-only for now).
        device: torch device string; auto-select if None.
        image_size: resize target for inference.
        dtype: "float32" (default) or "float16".
    """
    resolved_style = _resolve_style(style)
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

    # ECCV16-only backend; keep label explicit for future variants.
    _ = resolved_style  # explicit for readability
    weight_path = load_eccv16()
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
        out_u8 = (out_np * 255.0).round().astype(np.uint8)
        save_colorized_fullres(frame_path, out_u8, out_dir / frame_path.name)


__all__ = ["colorize_frames_inst"]
