from pathlib import Path
from typing import Literal, Optional, List
from itertools import chain
import os, time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tools.AImodels.zhang_model import (
    eccv16, siggraph17, load_img, preprocess_img, postprocess_tens
)

# ---------------- CPU threads + torch setup ----------------
try:
    import torch
    LOGICAL = os.cpu_count() or 8                      # logical cores
    T_THREADS = int(os.getenv("VC_THREADS", LOGICAL))  # allow override
    torch.set_grad_enabled(False)
    torch.set_num_threads(T_THREADS)
    torch.set_num_interop_threads(T_THREADS)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
except Exception:
    torch = None  # type: ignore


# ---------------- helpers ----------------
def _list_frames(frames_dir: Path) -> List[Path]:
    d = Path(frames_dir)
    paths = {
        p.resolve(): p
        for p in chain(
        d.glob("*.png"), d.glob("*.PNG"),
        d.glob("*.jpg"), d.glob("*.JPG"),
        d.glob("*.jpeg"), d.glob("*.JPEG"),
        )
    }
    return sorted(paths.values(), key=lambda p: p.name)

def _auto_batch_size(threads: int) -> int:
    # bigger cap to keep modern CPUs busy
    return max(6, min(24, (threads * 3) // 4))

def _pick_model(variant: Literal["eccv16", "siggraph17"], use_gpu: bool):
    m = eccv16(pretrained=True).eval() if variant == "eccv16" else siggraph17(pretrained=True).eval()
    # CPU path only here
    return m

def _save_rgb(img_np, out_path: Path):
    import numpy as np
    try:
        import cv2
        arr = (np.clip(img_np, 0.0, 1.0) * 255.0).astype("uint8")
        bgr = arr[:, :, ::-1]
        cv2.imwrite(str(out_path), bgr)
    except Exception:
        from PIL import Image
        arr = (np.clip(img_np, 0.0, 1.0) * 255.0).astype("uint8")
        Image.fromarray(arr).save(str(out_path))

def _prep_one(path: Path, input_size: int):
    """Load image + preprocess to tensors (runs in thread)."""
    img = load_img(str(path))
    l_orig, l_rs = preprocess_img(img, HW=(input_size, input_size))
    return (path, l_orig, l_rs)

def _prefetch_batch(paths: List[Path], input_size: int, workers: int):
    """Prepare a batch in parallel (I/O + resize) to smooth CPU spikes."""
    with ThreadPoolExecutor(max_workers=max(2, workers)) as ex:
        futs = [ex.submit(_prep_one, p, input_size) for p in paths]
        results = []
        for f in as_completed(futs):
            results.append(f.result())
    # keep original order
    results.sort(key=lambda t: t[0].name)
    l_orig_list = [t[1] for t in results]
    l_rs_list   = [t[2] for t in results]
    return l_orig_list, l_rs_list


# ---------------- main API ----------------
def colorize_dir(
    frames_dir: Path,
    out_dir: Path,
    models_dir: Optional[Path] = None,         # unused
    variant: Literal["eccv16", "siggraph17"] = "siggraph17",
    preview: bool = False,                      # ignored
    use_gpu: bool = False,                      # ignored (CPU-only here)
    batch_size: Optional[int] = None,          # auto if None
    num_threads: Optional[int] = None,         # override torch threads
    input_size: int = 256,                     # try 224 for speed
    sat_gain: float = 1.20,                    # reserved for later post
    contrast_gain: float = 1.05,               # reserved for later post
    preview_every: int = 10,                   # unused
    progress: bool = True,                     # print batch progress
    prefetch_workers: Optional[int] = None,    # auto if None (I/O threads)
    save_workers: int = 0,                     # 0=save inline, >0=parallel save
    **_: object,
) -> None:
    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # torch threads override
    threads_info = T_THREADS if torch is not None else (os.cpu_count() or 8)
    if (torch is not None) and (num_threads is not None):
        torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(num_threads)
        except RuntimeError:
            pass
        threads_info = num_threads

    # list frames
    frame_list = _list_frames(frames_dir)
    if not frame_list:
        print(f"[error] no frames found (png/jpg) in: {frames_dir}")
        return
    total = len(frame_list)

    # batch size
    if batch_size is None:
        batch_size = _auto_batch_size(threads_info)

    # prefetch workers (I/O-bound; don’t overdo)
    if prefetch_workers is None:
        prefetch_workers = min(12, max(4, threads_info // 2))

    print(f"[start] zhang={variant} | frames={total} | batch={batch_size} | threads={threads_info} | prefetch={prefetch_workers} | save_workers={save_workers}")

    # build model (may incur one-time weight load)
    t0_load = time.time()
    mdl = _pick_model(variant, use_gpu=False)
    print(f"[ok] model ready in {time.time() - t0_load:.1f}s")

    # optional executor for parallel saves
    saver = ThreadPoolExecutor(max_workers=save_workers) if save_workers > 0 else None
    pending = []  # list of futures for save ops

    # main loop
    done = 0
    t0 = time.time()

    while done < total:
        batch_paths = frame_list[done : done + batch_size]

        # prefetch batch
        l_orig_list, l_rs_list = _prefetch_batch(batch_paths, input_size, prefetch_workers)

        # stack + forward
        if torch is not None:
            l_rs_b = torch.cat(l_rs_list, dim=0)
            with torch.inference_mode():
                out_ab_b = mdl(l_rs_b).cpu()
        else:
            out_ab_b = [mdl(x).cpu() for x in l_rs_list]

        # postprocess + save
        if torch is not None:
            for j, p in enumerate(batch_paths):
                out_img = postprocess_tens(l_orig_list[j], out_ab_b[j:j+1])
                if saver:
                    pending.append(saver.submit(_save_rgb, out_img, out_dir / p.name))
                else:
                    _save_rgb(out_img, out_dir / p.name)
        else:
            for j, p in enumerate(batch_paths):
                out_img = postprocess_tens(l_orig_list[j], out_ab_b[j])
                if saver:
                    pending.append(saver.submit(_save_rgb, out_img, out_dir / p.name))
                else:
                    _save_rgb(out_img, out_dir / p.name)

        done += len(batch_paths)

        if progress:
            pct = (done / total)
            elapsed = time.time() - t0
            speed = done / max(elapsed, 1e-6)
            eta_s = (total - done) / max(speed, 1e-6)
            print(f"[colorize] {done}/{total}  ({int(pct*100)}%)  elapsed {elapsed:.1f}s  ETA {eta_s:.1f}s", end="\r", flush=True)

    # finish pending saves
    if saver:
        for f in as_completed(pending):
            _ = f.result()
        saver.shutdown(wait=True)

    if progress:
        elapsed = time.time() - t0
        print(f"\n[ok] colorization complete: {total}/{total} in {elapsed:.1f}s")
