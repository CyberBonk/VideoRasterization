# tools/preview_report.py
from pathlib import Path
import json
from typing import List, Tuple
import numpy as np

try:
    import cv2  # faster for IO + HSV
    _USE_CV2 = True
except Exception:
    from PIL import Image
    _USE_CV2 = False

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def _read_rgb(p: Path) -> np.ndarray:
    if _USE_CV2:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read {p}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        img = Image.open(p).convert("RGB")
        return np.array(img)


def _to_hsv(rgb: np.ndarray) -> np.ndarray:
    if _USE_CV2:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    else:
        # simple RGB->HSV without cv2
        import colorsys
        h, w, _ = rgb.shape
        flat = rgb.reshape(-1, 3) / 255.0
        hsv = np.array([colorsys.rgb_to_hsv(*px) for px in flat], dtype=np.float32)
        hsv[..., 1:] *= 255.0  # S,V to 0..255 for parity with cv2
        return hsv.reshape(h, w, 3)


def _pick_three(paths: List[Path]) -> Tuple[Path, Path, Path]:
    n = len(paths)
    return paths[0], paths[n // 2], paths[-1]


def _match_name(gray_dir: Path, color_dir: Path, name: str) -> Tuple[Path, Path]:
    # try same filename with {png,jpg,jpeg} in both dirs
    exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    for e in exts:
        g = gray_dir / (Path(name).stem + e)
        c = color_dir / (Path(name).stem + e)
        if g.exists() and c.exists():
            return g, c
    # fallback: exact name in both
    g = gray_dir / name
    c = color_dir / name
    return g, c


def _glob_images(d: Path) -> List[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    out: List[Path] = []
    for pat in exts:
        out.extend(d.glob(pat))
    return sorted(out, key=lambda p: p.name)


def _mean_rgb(img: np.ndarray) -> Tuple[float, float, float]:
    m = img.reshape(-1, 3).mean(0)
    return float(m[0]), float(m[1]), float(m[2])


def generate_report(frames_gray_dir: Path, frames_color_dir: Path, out_png: Path, out_json: Path) -> None:
    frames_gray_dir = Path(frames_gray_dir)
    frames_color_dir = Path(frames_color_dir)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    gray_list = _glob_images(frames_gray_dir)
    color_list = _glob_images(frames_color_dir)
    if not gray_list or not color_list:
        raise RuntimeError("no images found for report")

    # pick first/middle/last by colorized list (safer)
    p1, p2, p3 = _pick_three(color_list)
    pairs = []
    for p in (p1, p2, p3):
        g, c = _match_name(frames_gray_dir, frames_color_dir, p.name)
        if not g.exists() or not c.exists():
            continue
        pairs.append((g, c))
    if not pairs:
        raise RuntimeError("could not pair images between gray and colorized")

    # compute stats over ALL colorized frames
    # mean RGB and saturation histogram
    all_means = []
    all_sats = []
    for p in color_list:
        rgb = _read_rgb(p)
        all_means.append(_mean_rgb(rgb))
        hsv = _to_hsv(rgb)
        all_sats.append(hsv[..., 1].reshape(-1))  # S channel 0..255

    mean_rgb = tuple(float(np.array(all_means).mean(0)[i]) for i in range(3))
    sats = np.concatenate(all_sats, axis=0)

    # ---- draw report (1 png) ----
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 3, height_ratios=[1,1,1], width_ratios=[1,1,1], hspace=0.35, wspace=0.2)

    # row 1: first frame (gray vs color)
    for col, (g, c) in enumerate([pairs[0]]):
        rgb_g = _read_rgb(g)
        rgb_c = _read_rgb(c)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(rgb_g); ax1.set_title(f"First (gray): {g.name}"); ax1.axis("off")
        ax2 = fig.add_subplot(gs[0, 1:])
        ax2.imshow(rgb_c); ax2.set_title(f"First (color): {c.name}"); ax2.axis("off")

    # row 2: middle frame
    if len(pairs) >= 2:
        g, c = pairs[1]
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(_read_rgb(g)); ax3.set_title(f"Middle (gray): {g.name}"); ax3.axis("off")
        ax4 = fig.add_subplot(gs[1, 1:])
        ax4.imshow(_read_rgb(c)); ax4.set_title(f"Middle (color): {c.name}"); ax4.axis("off")

    # row 3: hist + last color frame
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(sats, bins=40)
    ax5.set_title("Saturation histogram (HSV.S, 0..255)")
    ax5.set_xlabel("S"); ax5.set_ylabel("count")

    g_last, c_last = pairs[-1]
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.imshow(_read_rgb(c_last)); ax6.set_title(f"Last (color): {c_last.name}"); ax6.axis("off")

    fig.suptitle(f"Colorization report\nMean RGB: {mean_rgb}", fontsize=12)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- write JSON stats ----
    stats = {
        "frames_gray_dir": str(frames_gray_dir),
        "frames_color_dir": str(frames_color_dir),
        "num_frames": len(color_list),
        "mean_rgb": {"r": mean_rgb[0], "g": mean_rgb[1], "b": mean_rgb[2]},
        "saturation_histogram": {"bins": 40, "range": [0, 255]},
    }
    out_json.write_text(json.dumps(stats, indent=2))
