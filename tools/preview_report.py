# tools/preview_report.py
from pathlib import Path
from typing import List, Tuple
import json
import numpy as np

# ------------------------------------------------------------------
# Optional accelerated image IO (cv2), fallback to PIL
# ------------------------------------------------------------------
try:
    import cv2
    _USE_CV2 = True
except Exception:
    from PIL import Image
    _USE_CV2 = False

# ------------------------------------------------------------------
# Headless matplotlib
# ------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# ---- Image IO helpers --------------------------------------------
# ------------------------------------------------------------------
def _read_rgb(path: Path) -> np.ndarray:
    """Read RGB image using cv2 or PIL."""
    if _USE_CV2:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        img = Image.open(path).convert("RGB")
        return np.array(img)


def _to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB → HSV conversion using cv2 or a fallback Python implementation."""
    if _USE_CV2:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    import colorsys
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*px) for px in flat], dtype=np.float32)
    hsv[..., 1:] *= 255.0  # match cv2 range (S,V 0..255)
    return hsv.reshape(h, w, 3)


# ------------------------------------------------------------------
# ---- Frame selection + matching ----------------------------------
# ------------------------------------------------------------------
def _glob_images(folder: Path) -> List[Path]:
    """Return all images in folder sorted by name."""
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    files = []
    for pat in patterns:
        files.extend(folder.glob(pat))
    return sorted(files, key=lambda p: p.name)


def _pick_three(paths: List[Path]) -> Tuple[Path, Path, Path]:
    """Pick first, middle, last frames."""
    return paths[0], paths[len(paths) // 2], paths[-1]


def _match_name(gray_dir: Path, color_dir: Path, name: str) -> Tuple[Path, Path]:
    """Find matching gray/color frame with flexible extensions."""
    exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    stem = Path(name).stem

    for e in exts:
        g = gray_dir / f"{stem}{e}"
        c = color_dir / f"{stem}{e}"
        if g.exists() and c.exists():
            return g, c

    # fallback: literal name
    return gray_dir / name, color_dir / name


# ------------------------------------------------------------------
# ---- Pixel statistics ---------------------------------------------
# ------------------------------------------------------------------
def _mean_rgb(img: np.ndarray) -> Tuple[float, float, float]:
    """Compute mean R,G,B over image."""
    m = img.reshape(-1, 3).mean(0)
    return float(m[0]), float(m[1]), float(m[2])


# ------------------------------------------------------------------
# ---- Main reporting function --------------------------------------
# ------------------------------------------------------------------
def generate_report(frames_gray_dir: Path, frames_color_dir: Path,
                    out_png: Path, out_json: Path) -> None:

    frames_gray_dir = Path(frames_gray_dir)
    frames_color_dir = Path(frames_color_dir)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    gray_frames = _glob_images(frames_gray_dir)
    color_frames = _glob_images(frames_color_dir)

    if not gray_frames or not color_frames:
        raise RuntimeError("No images found for report generation")

    # pick three key frames from the colorized sequence
    first_f, middle_f, last_f = _pick_three(color_frames)

    # pair them with corresponding gray frames
    pairs = []
    for p in (first_f, middle_f, last_f):
        g, c = _match_name(frames_gray_dir, frames_color_dir, p.name)
        if g.exists() and c.exists():
            pairs.append((g, c))

    if not pairs:
        raise RuntimeError("Could not pair gray and colorized frames")

    # aggregate stats (RGB means + saturation hist)
    rgb_means = []
    sat_values = []

    for p in color_frames:
        rgb = _read_rgb(p)
        rgb_means.append(_mean_rgb(rgb))

        hsv = _to_hsv(rgb)
        sat_values.append(hsv[..., 1].reshape(-1))

    mean_rgb = tuple(np.array(rgb_means).mean(0))
    sats_flat = np.concatenate(sat_values, axis=0)

    # ------------------------------------------------------------------
    # Draw composite report
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(
        3, 3,
        height_ratios=[1, 1, 1],
        width_ratios=[1, 1, 1],
        hspace=0.35,
        wspace=0.2
    )

    # --- Row 1: first frame (gray + color)
    g, c = pairs[0]
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(_read_rgb(g))
    ax.set_title(f"First (gray): {g.name}")
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 1:])
    ax.imshow(_read_rgb(c))
    ax.set_title(f"First (color): {c.name}")
    ax.axis("off")

    # --- Row 2: middle frame
    if len(pairs) >= 2:
        g, c = pairs[1]

        ax = fig.add_subplot(grid[1, 0])
        ax.imshow(_read_rgb(g))
        ax.set_title(f"Middle (gray): {g.name}")
        ax.axis("off")

        ax = fig.add_subplot(grid[1, 1:])
        ax.imshow(_read_rgb(c))
        ax.set_title(f"Middle (color): {c.name}")
        ax.axis("off")

    # --- Row 3: histogram + last color frame
    ax = fig.add_subplot(grid[2, 0])
    ax.hist(sats_flat, bins=40)
    ax.set_title("Saturation histogram (HSV.S, 0..255)")
    ax.set_xlabel("S")
    ax.set_ylabel("Count")

    g_last, c_last = pairs[-1]
    ax = fig.add_subplot(grid[2, 1:])
    ax.imshow(_read_rgb(c_last))
    ax.set_title(f"Last (color): {c_last.name}")
    ax.axis("off")

    fig.suptitle(f"Colorization Report\nMean RGB: {mean_rgb}", fontsize=12)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    stats = {
        "frames_gray_dir": str(frames_gray_dir),
        "frames_color_dir": str(frames_color_dir),
        "num_frames": len(color_frames),
        "mean_rgb": {"r": mean_rgb[0], "g": mean_rgb[1], "b": mean_rgb[2]},
        "saturation_histogram": {"bins": 40, "range": [0, 255]},
    }

    out_json.write_text(json.dumps(stats, indent=2))
