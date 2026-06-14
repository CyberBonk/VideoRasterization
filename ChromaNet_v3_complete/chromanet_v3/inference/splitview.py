"""
splitview.py — Before/After Split View  ★ NEW in v3 ★

Generates a side-by-side comparison video:
  Left half  = original grayscale
  Right half = colorized
  A moving white divider line separates them

This is purely a presentation tool — no model changes needed.
It makes the demo video extremely impressive.

Usage:
    python inference/splitview.py \\
        --gray    path/to/gray_frames/ \\
        --color   path/to/color_frames/ \\
        --output  splitview.mp4 \\
        --fps     24 \\
        --divider-speed 0.3
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir()
                   if p.suffix.lower() in _IMG_EXTS])


def _load_rgb(path: Path, size: tuple[int,int] | None = None) -> np.ndarray:
    """Load image as RGB uint8 numpy array, optionally resize."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.BICUBIC)
    return np.array(img, dtype=np.uint8)


def _to_gray_rgb(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale-RGB (3-channel gray)."""
    gray = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=2)


def build_split_frame(
    gray_img: np.ndarray,
    color_img: np.ndarray,
    divider_x: int,
    divider_width: int = 4,
) -> np.ndarray:
    """
    Compose one split-view frame.

    Args:
        gray_img:     [H, W, 3] grayscale image (RGB format)
        color_img:    [H, W, 3] colorized image
        divider_x:    X position of divider line
        divider_width: Width of white divider line in pixels

    Returns:
        Composite frame [H, W, 3]
    """
    H, W = gray_img.shape[:2]
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    # Left of divider = gray, right = color
    split = max(0, min(divider_x, W))
    frame[:, :split]  = gray_img[:, :split]
    frame[:, split:]  = color_img[:, split:]

    # White divider line
    d_left  = max(0, split - divider_width // 2)
    d_right = min(W, split + divider_width // 2)
    frame[:, d_left:d_right] = 255

    # Small triangle marker at top of divider
    arrow_h = 20
    for i in range(arrow_h):
        w_half = arrow_h - i
        l = max(0, split - w_half)
        r = min(W, split + w_half)
        frame[i, l:r] = 255

    return frame


def generate_splitview(
    gray_dir:       str | Path,
    color_dir:      str | Path,
    output_path:    str | Path,
    fps:            int   = 24,
    divider_speed:  float = 0.3,    # divider moves this fraction of width per second
    static_seconds: float = 1.0,    # seconds the divider stays at center before moving
) -> None:
    """
    Generate a split-view comparison video.

    The divider starts at the center and sweeps left→right→left.

    Args:
        gray_dir:       Folder of grayscale frames.
        color_dir:      Folder of colorized frames.
        output_path:    Output video file path (.mp4).
        fps:            Frames per second.
        divider_speed:  How fast the divider moves (fraction of width per second).
        static_seconds: How long divider stays still at start.
    """
    try:
        import imageio_ffmpeg
        import subprocess, shlex, tempfile
    except ImportError:
        print("[error] imageio_ffmpeg not installed. Run: pip install imageio-ffmpeg")
        return

    gray_dir   = Path(gray_dir)
    color_dir  = Path(color_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gray_frames  = _collect(gray_dir)
    color_frames = _collect(color_dir)

    if not gray_frames or not color_frames:
        print("[error] no frames found")
        return

    # Match frame count
    n_frames = min(len(gray_frames), len(color_frames))
    print(f"[splitview] {n_frames} frames | fps={fps}")

    # Load first frame to get dimensions
    sample    = _load_rgb(color_frames[0])
    H, W      = sample.shape[:2]

    # Pre-compute divider positions for each frame
    # Pattern: center → sweep right → sweep left (ping-pong)
    center_x   = W // 2
    static_frm = int(static_seconds * fps)
    speed_px   = divider_speed * W  # pixels per second

    divider_positions = []
    for i in range(n_frames):
        if i < static_frm:
            divider_positions.append(center_x)
        else:
            t = (i - static_frm) / fps
            # Oscillate: center + amplitude * sin(t * speed)
            amplitude = W * 0.4
            x = center_x + amplitude * np.sin(t * speed_px / W * np.pi)
            divider_positions.append(int(np.clip(x, W*0.05, W*0.95)))

    # Write frames to temp dir and encode with ffmpeg
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print("[splitview] composing frames...")

        for i, (gp, cp) in enumerate(zip(gray_frames[:n_frames],
                                          color_frames[:n_frames])):
            gray_img  = _to_gray_rgb(_load_rgb(gp,  (W, H)))
            color_img = _load_rgb(cp, (W, H))
            frame     = build_split_frame(gray_img, color_img, divider_positions[i])

            from PIL import Image
            Image.fromarray(frame).save(tmp_dir / f"frame_{i:06d}.png")

            if (i+1) % 50 == 0 or (i+1) == n_frames:
                print(f"  {i+1}/{n_frames}", end="\r", flush=True)

        print(f"\n[splitview] encoding video → {output_path}")
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", str(tmp_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)

    print(f"[ok] split-view video saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate before/after split-view video")
    parser.add_argument("--gray",    required=True, help="Grayscale frames folder")
    parser.add_argument("--color",   required=True, help="Colorized frames folder")
    parser.add_argument("--output",  default="splitview.mp4")
    parser.add_argument("--fps",     type=int,   default=24)
    parser.add_argument("--divider-speed", type=float, default=0.3)
    parser.add_argument("--static-seconds", type=float, default=1.0)
    args = parser.parse_args()

    generate_splitview(
        gray_dir       = args.gray,
        color_dir      = args.color,
        output_path    = args.output,
        fps            = args.fps,
        divider_speed  = args.divider_speed,
        static_seconds = args.static_seconds,
    )


if __name__ == "__main__":
    main()
