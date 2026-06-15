"""colorizer.py — ChromaNet v3 Inference"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color as skcolor
from model.chromaNet import build_model
from model.confidence import apply_confidence, save_confidence_heatmap

_EXTS = {".jpg",".jpeg",".png",".bmp",".tiff",".tif"}


class ChromaColorizer:
    def __init__(self, checkpoint_path, device=None,
                 image_size=256, save_confidence=False) -> None:
        self.image_size       = image_size
        self.save_confidence  = save_confidence
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        ck  = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = build_model(ck.get("cfg", {}))
        self.model.load_state_dict(ck["model_state"])
        self.model.to(self.device).eval()
        print(f"[ChromaColorizer] v3 | device={self.device}")

    def _pre(self, img: Image.Image):
        rgb_image = img.convert("RGB")
        orig = rgb_image.size

        # Keep source luminance at full resolution. Only model input is resized.
        rgb_full = np.asarray(rgb_image, dtype=np.float32) / 255.0
        L_full = skcolor.rgb2lab(rgb_full).astype(np.float32)[:, :, 0]

        model_image = rgb_image.resize(
            (self.image_size, self.image_size), Image.Resampling.BICUBIC
        )
        rgb_model = np.asarray(model_image, dtype=np.float32) / 255.0
        lab_model = skcolor.rgb2lab(rgb_model).astype(np.float32)
        L_model = torch.from_numpy((lab_model[:, :, 0] / 50.0) - 1.0)
        L_model = L_model.unsqueeze(0).unsqueeze(0)
        return L_model.to(self.device), orig, L_full

    def _post(self, L_full, AB, orig):
        width, height = orig
        Ld = L_full
        AB_full = F.interpolate(
            AB.float(), size=(height, width), mode="bicubic", align_corners=False
        )
        ABd = AB_full[0].cpu().numpy() * 110.0
        lab = np.stack([Ld, ABd[0], ABd[1]], axis=2)
        rgb = np.clip(skcolor.lab2rgb(lab), 0.0, 1.0)
        return Image.fromarray((rgb*255).astype(np.uint8))

    @torch.no_grad()
    def colorize_image(self, inp, out_path=None, conf_path=None):
        img      = Image.open(inp)
        L, orig, L_full = self._pre(img)
        out      = self.model(L)
        AB       = out["ab"]
        conf     = out.get("confidence")
        if conf is not None: AB = apply_confidence(AB, conf)
        result   = self._post(L_full, AB, orig)
        if out_path: result.save(out_path)
        if self.save_confidence and conf is not None:
            cp = conf_path or str(out_path).replace(".png","_conf.png").replace(".jpg","_conf.png")
            save_confidence_heatmap(conf, cp)
        return result

    @torch.no_grad()
    def colorize_folder(self, inp_dir, out_dir, ext=".png", batch_size=1):
        inp_dir = Path(inp_dir); out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.save_confidence: (out_dir/"confidence_maps").mkdir(exist_ok=True)
        frames = sorted([p for p in inp_dir.iterdir() if p.suffix.lower() in _EXTS])
        if not frames: print(f"[warn] no images in {inp_dir}"); return
        batch_size = max(1, int(batch_size))
        print(f"[ChromaColorizer] {len(frames)} frames | batch={batch_size}...")
        done = 0
        for start in range(0, len(frames), batch_size):
            batch_paths = frames[start:start + batch_size]
            prepared = [self._pre(Image.open(fp)) for fp in batch_paths]
            L = torch.cat([item[0] for item in prepared], dim=0)
            originals = [item[1] for item in prepared]
            full_luminance = [item[2] for item in prepared]

            with torch.amp.autocast(
                "cuda", enabled=self.device.type == "cuda", dtype=torch.bfloat16
            ):
                output = self.model(L)
                AB = output["ab"]
                confidence = output.get("confidence")
                if confidence is not None:
                    AB = apply_confidence(AB, confidence)

            for index, fp in enumerate(batch_paths):
                result = self._post(
                    full_luminance[index], AB[index:index + 1], originals[index]
                )
                result.save(out_dir/(fp.stem+ext))
                if self.save_confidence and confidence is not None:
                    cp = out_dir/"confidence_maps"/(fp.stem+"_conf.png")
                    save_confidence_heatmap(confidence[index:index + 1], cp)

            done += len(batch_paths)
            if done % 48 == 0 or done == len(frames):
                print(f"  {done}/{len(frames)}", end="\r", flush=True)
        print(f"\n[done] output: {out_dir}")

__all__ = ["ChromaColorizer"]
