from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from pathlib import Path
from typing import List, Literal, Optional
import os
import time

import numpy as np
from PIL import Image, ImageEnhance
from skimage import color

from tools.AImodels.zhang_model import (
    eccv16,
    siggraph17,
    load_img,
    postprocess_tens,
    preprocess_img,
)
from tools.console import status

try:
    import torch
except Exception:
    torch = None  # type: ignore

LOGICAL = os.cpu_count() or 8
T_THREADS = int(os.getenv("VC_THREADS", LOGICAL))
if torch is not None:
    torch.set_grad_enabled(False)
    try:
        torch.set_num_threads(T_THREADS)
        torch.set_num_interop_threads(T_THREADS)
    except RuntimeError:
        pass
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True


def _list_frames(frames_dir: Path) -> List[Path]:
    d = Path(frames_dir)
    paths = {
        p.resolve(): p
        for p in chain(
            d.glob("*.png"),
            d.glob("*.PNG"),
            d.glob("*.jpg"),
            d.glob("*.JPG"),
            d.glob("*.jpeg"),
            d.glob("*.JPEG"),
        )
    }
    return sorted(paths.values(), key=lambda p: p.name)


def _auto_batch_size(threads: int) -> int:
    return max(6, min(24, (threads * 3) // 4))


def _pick_model(variant: Literal["eccv16", "siggraph17"], device: str):
    model = eccv16(pretrained=True).eval() if variant == "eccv16" else siggraph17(pretrained=True).eval()
    if torch is not None:
        model = model.to(device)
    return model


def _save_rgb(img_np: np.ndarray, out_path: Path) -> None:
    arr = (np.clip(img_np, 0.0, 1.0) * 255.0).astype("uint8")
    Image.fromarray(arr).save(out_path)


def _prep_one(path: Path, input_size: int):
    img = load_img(str(path))
    l_orig, l_rs = preprocess_img(img, HW=(input_size, input_size))
    return (path, l_orig, l_rs)


def _prefetch_batch(paths: List[Path], input_size: int, workers: int):
    with ThreadPoolExecutor(max_workers=max(2, workers)) as ex:
        futs = [ex.submit(_prep_one, p, input_size) for p in paths]
        results = [f.result() for f in as_completed(futs)]
    results.sort(key=lambda t: t[0].name)
    l_orig_list = [t[1] for t in results]
    l_rs_list = [t[2] for t in results]
    return l_orig_list, l_rs_list


def _enhance_rgb(
    img_np: np.ndarray,
    saturation_gain: float,
    contrast_gain: float,
    neutralize_ab_bias: float,
) -> np.ndarray:
    rgb = np.clip(img_np, 0.0, 1.0).astype(np.float32)
    lab = color.rgb2lab(rgb)
    ab = lab[:, :, 1:3]
    ab_mean = ab.mean(axis=(0, 1), keepdims=True)
    lab[:, :, 1:3] = np.clip((ab - (ab_mean * neutralize_ab_bias)) * saturation_gain, -128.0, 127.0)
    corrected = np.clip(color.lab2rgb(lab), 0.0, 1.0)

    if abs(contrast_gain - 1.0) > 1e-3:
        pil = Image.fromarray((corrected * 255.0).astype("uint8"))
        pil = ImageEnhance.Contrast(pil).enhance(contrast_gain)
        corrected = np.asarray(pil, dtype=np.float32) / 255.0
    return corrected


def colorize_dir(
    frames_dir: Path,
    out_dir: Path,
    models_dir: Optional[Path] = None,
    variant: Literal["eccv16", "siggraph17"] = "siggraph17",
    preview: bool = False,
    use_gpu: bool = False,
    batch_size: Optional[int] = None,
    num_threads: Optional[int] = None,
    input_size: int = 256,
    saturation_gain: float = 1.18,
    contrast_gain: float = 1.06,
    neutralize_ab_bias: float = 0.18,
    progress: bool = True,
    prefetch_workers: Optional[int] = None,
    save_workers: int = 2,
    **_: object,
) -> None:
    _ = (models_dir, preview)
    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch is None:
        raise RuntimeError("PyTorch is required for Enhanced Zhang backend.")

    threads_info = T_THREADS
    if num_threads is not None:
        torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(num_threads)
        except RuntimeError:
            pass
        threads_info = num_threads

    frame_list = _list_frames(frames_dir)
    if not frame_list:
        status(f"[error] no frames found (png/jpg) in: {frames_dir}")
        return
    total = len(frame_list)

    if batch_size is None:
        batch_size = _auto_batch_size(threads_info)
    if prefetch_workers is None:
        prefetch_workers = min(12, max(4, threads_info // 2))

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    status(
        f"[start] Enhanced Zhang (Bebo's Experiment) | base={variant} | device={device} "
        f"| frames={total} | batch={batch_size} | saturation_gain={saturation_gain:.2f} "
        f"| contrast_gain={contrast_gain:.2f} | neutralize_ab_bias={neutralize_ab_bias:.2f}"
    )

    t0_load = time.time()
    model = _pick_model(variant, device)
    status(f"[ok] enhanced Zhang model ready in {time.time() - t0_load:.1f}s")

    saver = ThreadPoolExecutor(max_workers=save_workers) if save_workers > 0 else None
    pending = []
    done = 0
    t0 = time.time()

    while done < total:
        batch_paths = frame_list[done : done + batch_size]
        l_orig_list, l_rs_list = _prefetch_batch(batch_paths, input_size, prefetch_workers)
        l_rs_b = torch.cat(l_rs_list, dim=0).to(device)

        with torch.inference_mode():
            out_ab_b = model(l_rs_b).cpu()

        for j, p in enumerate(batch_paths):
            out_img = postprocess_tens(l_orig_list[j], out_ab_b[j : j + 1])
            out_img = _enhance_rgb(
                out_img,
                saturation_gain=saturation_gain,
                contrast_gain=contrast_gain,
                neutralize_ab_bias=neutralize_ab_bias,
            )
            if saver:
                pending.append(saver.submit(_save_rgb, out_img, out_dir / p.name))
            else:
                _save_rgb(out_img, out_dir / p.name)

        done += len(batch_paths)
        if progress:
            elapsed = time.time() - t0
            fps = done / max(elapsed, 1e-6)
            status(f"[progress] {done}/{total} frames ({(done / total) * 100:.1f}%) | {fps:.2f} fps")

    if saver:
        for future in as_completed(pending):
            future.result()
        saver.shutdown(wait=True)

    elapsed = time.time() - t0
    status(f"[ok] enhanced Zhang colorization complete: {total}/{total} in {elapsed:.1f}s")


__all__ = ["colorize_dir"]
