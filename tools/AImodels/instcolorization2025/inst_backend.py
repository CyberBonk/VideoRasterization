from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color as skcolor

from . import networks
from . import util as inst_util
from .colorize_inst import (
    load_weights,
    select_device,
    make_opt,
    prepare_transforms,
    run_siggraph,
)
from .siggraph_loader import load_siggraph17

DEFAULT_IMAGE_SIZE = 256
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_ROOT = BASE_DIR.parents[2] / "checkpoints"
TRAINED_FULL_CHECKPOINT = CHECKPOINT_ROOT / "instcolorization" / "coco_full_256_train2017" / "latest_net_G.pth"
INST_STYLE_TRAINED = "inst_trained_full"
INST_STYLE_SIGGRAPH17 = "inst_siggraph17"


def _list_frames(frame_paths: Iterable[Path | str]) -> list[Path]:
    paths = [Path(p) for p in frame_paths]
    return sorted([p for p in paths if p.is_file()], key=lambda p: p.name)


def _resolve_style(style: str) -> str:
    """
    Full-branch InstColorization uses the SIGGRAPH generator architecture.
    Prefer the local train2017-tuned checkpoint when it exists.
    """
    normalized = style.lower()
    if normalized in {"trained", "train2017", INST_STYLE_TRAINED}:
        return INST_STYLE_TRAINED
    if normalized in {"siggraph17", INST_STYLE_SIGGRAPH17}:
        return INST_STYLE_SIGGRAPH17
    print(f"[info] InstColorization unknown style '{style}'; using trained checkpoint when available.")
    return INST_STYLE_TRAINED


def _resolve_weights(resolved_style: str) -> Path:
    if resolved_style == INST_STYLE_TRAINED and TRAINED_FULL_CHECKPOINT.is_file():
        return TRAINED_FULL_CHECKPOINT
    if resolved_style == INST_STYLE_TRAINED:
        print(f"[warn] trained InstColorization checkpoint not found: {TRAINED_FULL_CHECKPOINT}")
        print("[info] falling back to SIGGRAPH17 base weights.")
    return load_siggraph17()


def save_colorized_fullres(orig_path: Path, ab_small: torch.Tensor, opt, out_path: Path) -> None:
    with Image.open(orig_path) as im:
        rgb_full = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0

    height, width = rgb_full.shape[:2]
    l_full = skcolor.rgb2lab(rgb_full)[:, :, 0]
    ab_full = F.interpolate(
        ab_small.float(), size=(height, width), mode="bicubic", align_corners=False
    )[0].detach().cpu().numpy() * opt.ab_norm
    lab = np.stack([l_full, ab_full[0], ab_full[1]], axis=2).astype(np.float32)
    rgb = np.clip(skcolor.lab2rgb(lab), 0.0, 1.0)
    Image.fromarray((rgb * 255.0).round().astype(np.uint8)).save(out_path, quality=95)


def colorize_frames_inst(
    frame_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    style: str = INST_STYLE_TRAINED,
    device: Optional[str] = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    dtype: str = "float32",
) -> None:
    """
    Colorize a set of grayscale frames using InstColorization.

    Args:
        frame_paths: Iterable of image file paths (png/jpg/bmp/tiff).
        output_dir: Directory where colorized frames are written.
        style: InstColorization backend label.
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

    weight_path = _resolve_weights(resolved_style)
    print(f"[start] InstColorization | checkpoint={weight_path.name} | device={target_device}")
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
        save_colorized_fullres(frame_path, out_reg, opt, out_dir / frame_path.name)


__all__ = ["colorize_frames_inst"]
