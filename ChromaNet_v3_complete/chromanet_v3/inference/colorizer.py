"""colorizer.py — ChromaNet v3 Inference"""
from __future__ import annotations
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color as skcolor
from model.chromaNet import build_model
from model.confidence import save_confidence_heatmap

_EXTS = {".jpg",".jpeg",".png",".bmp",".tiff",".tif"}

try:
    from tools.console import status
except Exception:
    status = print

try:
    import cv2
except Exception:
    cv2 = None


class ChromaColorizer:
    def __init__(self, checkpoint_path, device=None,
                 image_size=256, save_confidence=False,
                 confidence_threshold=0.3, saturation_gain=1.0,
                 grain_amount=0.0) -> None:
        self.image_size       = image_size
        self.save_confidence  = save_confidence
        self.confidence_threshold = float(confidence_threshold)
        self.saturation_gain = float(saturation_gain)
        self.grain_amount = max(0.0, min(float(grain_amount), 1.0))
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        ck  = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = build_model(ck.get("cfg", {}))
        self.model.load_state_dict(ck["model_state"])
        self.model.to(self.device).eval()
        status(f"[ChromaColorizer] v3 | device={self.device}")

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
        ) * self.saturation_gain
        ABd = AB_full[0].cpu().numpy() * 110.0
        lab = np.stack([Ld, ABd[0], ABd[1]], axis=2).astype(np.float32)
        if cv2 is not None:
            rgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0.0, 1.0)
        else:
            rgb = np.clip(skcolor.lab2rgb(lab), 0.0, 1.0)
        if self.grain_amount > 0:
            noise = np.random.normal(0.0, 0.035 * self.grain_amount, rgb.shape[:2])
            rgb = np.clip(rgb + noise[:, :, None], 0.0, 1.0)
        return Image.fromarray((rgb*255).astype(np.uint8))

    @torch.no_grad()
    def colorize_image(self, inp, out_path=None, conf_path=None):
        img      = Image.open(inp)
        L, orig, L_full = self._pre(img)
        out      = self.model(L)
        AB       = out["ab"]
        conf     = out.get("confidence")
        if conf is not None:
            AB = self._apply_confidence(AB, conf)
        result   = self._post(L_full, AB, orig)
        if out_path: result.save(out_path)
        if self.save_confidence and conf is not None:
            cp = conf_path or str(out_path).replace(".png","_conf.png").replace(".jpg","_conf.png")
            save_confidence_heatmap(conf, cp)
        return result

    @torch.no_grad()
    def colorize_folder(
        self, inp_dir, out_dir, ext=".png", batch_size=1,
        prefetch_workers=4, save_workers=4, max_prefetch_batches=2,
        cancel_event=None, pause_event=None,
    ):
        inp_dir = Path(inp_dir); out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.save_confidence: (out_dir/"confidence_maps").mkdir(exist_ok=True)
        frames = sorted([p for p in inp_dir.iterdir() if p.suffix.lower() in _EXTS])
        if not frames:
            status(f"[warn] no images in {inp_dir}")
            return
        batch_size = max(1, int(batch_size))
        prefetch_workers = max(1, int(prefetch_workers))
        save_workers = max(1, int(save_workers))
        max_inflight = max(1, int(max_prefetch_batches))
        status(
            f"[ChromaColorizer] {len(frames)} frames | batch={batch_size} "
            f"| prep={prefetch_workers} save={save_workers}..."
        )

        def prep_frame(fp: Path):
            with Image.open(fp) as img:
                return fp, self._pre(img)

        def save_frame(fp: Path, result: Image.Image) -> None:
            result.save(out_dir/(fp.stem+ext))

        def submit_batch(executor, batch_paths):
            return [executor.submit(prep_frame, fp) for fp in batch_paths]

        frame_batches = [
            frames[start:start + batch_size]
            for start in range(0, len(frames), batch_size)
        ]
        done = 0
        started_at = time.perf_counter()
        last_report = 0

        with ThreadPoolExecutor(max_workers=prefetch_workers) as prep_pool, \
                ThreadPoolExecutor(max_workers=save_workers) as save_pool:
            next_batch = 0
            inflight = []
            while next_batch < len(frame_batches) and len(inflight) < max_inflight:
                batch_paths = frame_batches[next_batch]
                inflight.append((next_batch, submit_batch(prep_pool, batch_paths)))
                next_batch += 1

            pending_saves = set()
            while inflight:
                if pause_event:
                    pause_event.wait()
                    
                if cancel_event and cancel_event.is_set():
                    status("\n[warn] Colorization cancelled by user.")
                    break
                    
                batch_index, prep_futures = inflight.pop(0)
                prepared_results = [future.result() for future in prep_futures]
                prepared_results.sort(key=lambda item: item[0].name)

                batch_paths = [item[0] for item in prepared_results]
                prepared = [item[1] for item in prepared_results]
                L = torch.cat([item[0] for item in prepared], dim=0)
                originals = [item[1] for item in prepared]
                full_luminance = [item[2] for item in prepared]

                while next_batch < len(frame_batches) and len(inflight) < max_inflight:
                    paths = frame_batches[next_batch]
                    inflight.append((next_batch, submit_batch(prep_pool, paths)))
                    next_batch += 1

                with torch.amp.autocast(
                    "cuda", enabled=self.device.type == "cuda", dtype=torch.bfloat16
                ):
                    output = self.model(L)
                    AB = output["ab"]
                    confidence = output.get("confidence")
                    if confidence is not None:
                        AB = self._apply_confidence(AB, confidence)

                for index, fp in enumerate(batch_paths):
                    result = self._post(
                        full_luminance[index], AB[index:index + 1], originals[index]
                    )
                    pending_saves.add(save_pool.submit(save_frame, fp, result))
                    if self.save_confidence and confidence is not None:
                        cp = out_dir/"confidence_maps"/(fp.stem+"_conf.png")
                        save_confidence_heatmap(confidence[index:index + 1], cp)

                done += len(batch_paths)
                if done - last_report >= max(batch_size * 8, 48) or done == len(frames):
                    elapsed = max(time.perf_counter() - started_at, 1e-6)
                    fps = done / elapsed
                    status(
                        f"[progress] {done}/{len(frames)} frames "
                        f"({done / len(frames) * 100:.1f}%) | {fps:.2f} fps",
                    )
                    last_report = done

                if len(pending_saves) >= save_workers * 3:
                    completed, pending_saves = wait(
                        pending_saves, return_when=FIRST_COMPLETED
                    )
                    for future in completed:
                        future.result()

            for future in pending_saves:
                future.result()
        status(f"[done] output: {out_dir}")

    def _apply_confidence(self, ab: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        threshold = max(0.0, min(self.confidence_threshold, 0.95))
        scale = ((conf - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
        return ab * scale

__all__ = ["ChromaColorizer"]
