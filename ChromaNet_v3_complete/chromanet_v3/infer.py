"""
infer.py — ChromaNet v3 Inference Entry Point
==============================================
Colorize images, folders of frames, or full videos.

Usage:
    # Single image
    python infer.py --mode image --input photo.jpg --output colorized.png

    # Folder of grayscale frames
    python infer.py --mode frames --input ./frames_gray --output ./frames_color

    # Full video (grayscale in → colorized RGB out)
    python infer.py --mode video --input gray_video.mp4 --output colorized.mp4

    # Video with confidence maps saved
    python infer.py --mode video --input gray.mp4 --output color.mp4 --save-confidence
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import color as skcolor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "chromanet"
sys.path.insert(0, str(ROOT))

from inference.colorizer import ChromaColorizer
from model.confidence import apply_confidence, save_confidence_heatmap


def lab_to_rgb_np(L_norm: np.ndarray, AB_norm: np.ndarray) -> np.ndarray:
    """L in [-1,1], AB in [-1,1] → RGB uint8"""
    L  = (L_norm + 1.0) * 50.0
    AB = AB_norm * 110.0
    lab = np.stack([L, AB[0], AB[1]], axis=2)
    rgb = np.clip(skcolor.lab2rgb(lab), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def colorize_frame_tensor(model, frame_rgb_np, device, img_size=256):
    """
    Colorize one frame.
    frame_rgb_np: numpy (H,W,3) RGB uint8
    Returns     : numpy (H,W,3) RGB uint8 colorized at original resolution
    """
    orig_h, orig_w = frame_rgb_np.shape[:2]
    img = Image.fromarray(frame_rgb_np).resize((img_size, img_size), Image.BICUBIC)
    rgb_f = np.array(img, dtype=np.float32) / 255.0
    lab   = skcolor.rgb2lab(rgb_f).astype(np.float32)
    L_np  = (lab[:, :, 0] / 50.0) - 1.0
    L_t   = torch.from_numpy(L_np).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out  = model(L_t)
        AB   = out["ab"]
        conf = out.get("confidence")
        if conf is not None:
            AB = apply_confidence(AB, conf)

    AB_np     = AB[0].cpu().numpy()          # (2, H, W)
    colorized = lab_to_rgb_np(L_np, AB_np)
    colorized = cv2.resize(colorized, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
    return colorized, (conf[0].cpu() if conf is not None else None)


def load_model(checkpoint_path: str, device):
    import yaml
    from model.chromaNet import build_model
    ck    = torch.load(checkpoint_path, map_location=device)
    cfg   = ck.get("cfg", {})
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"[ChromaNet v3] loaded from {checkpoint_path}")
    print(f"[ChromaNet v3] device = {device}")
    return model


def infer_image(args, model, device):
    colorizer = ChromaColorizer.__new__(ChromaColorizer)
    colorizer.model         = model
    colorizer.device        = device
    colorizer.image_size    = args.img_size
    colorizer.save_confidence = args.save_confidence
    colorizer.colorize_image(args.input, args.output,
                              conf_path=args.output.replace(".png","_conf.png").replace(".jpg","_conf.png")
                              if args.save_confidence else None)
    print(f"Saved: {args.output}")


def infer_frames(args, model, device):
    from inference.colorizer import ChromaColorizer
    colorizer = ChromaColorizer.__new__(ChromaColorizer)
    colorizer.model           = model
    colorizer.device          = device
    colorizer.image_size      = args.img_size
    colorizer.save_confidence = args.save_confidence
    colorizer.colorize_folder(args.input, args.output)


def infer_video(args, model, device):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.input}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {width}x{height} @ {fps:.1f}fps — {total} frames")

    out_path = args.output or "colorized.mp4"
    writer   = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    conf_dir = None
    if args.save_confidence:
        conf_dir = Path(out_path).stem + "_confidence_maps"
        Path(conf_dir).mkdir(exist_ok=True)
        print(f"[confidence] maps → {conf_dir}/")

    # Temporal smoothing state
    prev_ab = None
    alpha   = args.temporal_alpha

    with tqdm(total=total, desc="Colorizing") as pbar:
        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            colorized, conf = colorize_frame_tensor(model, frame_rgb, device, args.img_size)

            # Temporal smoothing
            if prev_ab is not None and alpha > 0:
                lab_c    = skcolor.rgb2lab(colorized.astype(np.float32) / 255.0)
                ab_cur   = lab_c[:, :, 1:].astype(np.float32)
                ab_blend = (1 - alpha) * ab_cur + alpha * prev_ab
                L_cur    = lab_c[:, :, 0]
                lab_out  = np.stack([L_cur, ab_blend[:,:,0], ab_blend[:,:,1]], axis=2)
                colorized = (np.clip(skcolor.lab2rgb(lab_out), 0, 1) * 255).astype(np.uint8)
                prev_ab  = ab_blend
            else:
                lab_c   = skcolor.rgb2lab(colorized.astype(np.float32) / 255.0)
                prev_ab = lab_c[:, :, 1:].astype(np.float32)

            # Save confidence map
            if args.save_confidence and conf is not None and conf_dir:
                save_confidence_heatmap(conf, f"{conf_dir}/frame_{frame_idx:06d}.png")

            writer.write(cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"\n[done] Colorized video → {out_path}")
    if conf_dir:
        print(f"[done] Confidence maps → {conf_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="ChromaNet v3 Inference")
    p.add_argument("--mode",   choices=["image","frames","video"], default="video")
    p.add_argument("--input",  required=True,  help="Input file or folder")
    p.add_argument("--output", default=None,   help="Output file or folder")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_DIR / "checkpoint_latest.pth"),
                   help="Path to trained checkpoint")
    p.add_argument("--img-size",        type=int,   default=256, dest="img_size")
    p.add_argument("--temporal-alpha",  type=float, default=0.3, dest="temporal_alpha",
                   help="Temporal smoothing 0=off 0.5=strong")
    p.add_argument("--save-confidence", action="store_true", dest="save_confidence",
                   help="Save confidence heatmaps alongside output")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find checkpoint automatically if not specified
    ckpt = args.checkpoint
    if not Path(ckpt).exists():
        ckpts = sorted(DEFAULT_CHECKPOINT_DIR.glob("*.pth"))
        if not ckpts:
            print("ERROR: No checkpoint found. Run train.py first.")
            sys.exit(1)
        # prefer _best, then _final, then latest
        best = [c for c in ckpts if "_best" in c.name]
        ckpt = str(best[-1] if best else ckpts[-1])
        print(f"[auto] using checkpoint: {ckpt}")

    model = load_model(ckpt, device)

    if   args.mode == "image":  infer_image(args, model, device)
    elif args.mode == "frames": infer_frames(args, model, device)
    elif args.mode == "video":  infer_video(args, model, device)
