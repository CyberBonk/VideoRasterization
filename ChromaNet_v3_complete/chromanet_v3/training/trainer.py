"""trainer.py — ChromaNet v3 Training Loop"""
from __future__ import annotations
import json, os, shutil, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from .metrics import compute_metrics
from model.confidence import apply_confidence
from model.losses import lab_to_rgb


class Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, cfg, device) -> None:
        self.model      = model.to(device)
        self.loss_fn    = loss_fn.to(device)
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.cfg        = cfg
        self.device     = device
        tc = cfg.get("training", {})
        self.epochs          = tc.get("epochs", 40)
        self.grad_clip       = tc.get("grad_clip", 1.0)
        self.log_every       = tc.get("log_every", 50)
        self.save_every      = tc.get("save_every", 5)
        self.preview_every   = tc.get("preview_every", 1)
        self.max_hours       = tc.get("max_hours")
        self.checkpoint_dir  = Path(tc.get("checkpoint_dir","./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir = self.checkpoint_dir.parent / "previews"
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp    = tc.get("mixed_precision", True) and device.type == "cuda"
        amp_name = str(tc.get("amp_dtype", "bfloat16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        ec = cfg.get("evaluation", {})
        self.metric_names = ec.get("metrics", ["psnr","ssim","colorfulness"])
        self.history: list[dict] = []
        self.best_psnr   = 0.0
        self.start_epoch = 0

    def _checkpoint_payload(self, epoch: int) -> dict:
        return {"epoch": epoch,
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state":    self.scaler.state_dict(),
                "best_psnr":       self.best_psnr,
                "history":         self.history,
                "cfg":             self.cfg}

    def _assert_finite_model(self) -> None:
        for name, value in self.model.state_dict().items():
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                raise FloatingPointError(
                    f"Refusing to save non-finite model tensor: {name}")

    def save_checkpoint(self, epoch: int, tag: str = "") -> Path:
        self._assert_finite_model()
        p = self.checkpoint_dir / f"checkpoint_epoch{epoch:03d}{tag}.pth"
        tmp = p.with_suffix(p.suffix + ".tmp")
        torch.save(self._checkpoint_payload(epoch), tmp)
        os.replace(tmp, p)
        print(f"  [ckpt] {p}")
        return p

    def update_latest(self, source: Path) -> None:
        latest = self.checkpoint_dir / "checkpoint_latest.pth"
        tmp = latest.with_suffix(latest.suffix + ".tmp")
        shutil.copyfile(source, tmp)
        os.replace(tmp, latest)

    def load_checkpoint(self, path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model_state"])
        self.optimizer.load_state_dict(ck["optimizer_state"])
        self.scheduler.load_state_dict(ck["scheduler_state"])
        if self.scaler.is_enabled() and ck.get("scaler_state"):
            self.scaler.load_state_dict(ck["scaler_state"])
        self.best_psnr   = ck.get("best_psnr", 0.0)
        self.history     = ck.get("history", [])
        self.start_epoch = int(ck["epoch"])
        print(f"[resume] completed epoch {ck['epoch']}; continuing at epoch {self.start_epoch + 1}")

    def _step(self, batch: dict) -> dict[str, float]:
        L   = batch["L"].to(self.device)
        AB  = batch["AB"].to(self.device)
        RGB = batch["RGB"].to(self.device)
        has_temp = "L_next" in batch
        if has_temp:
            L_next = batch["L_next"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            out = self.model(L)
            pred_AB  = out["ab"]
            conf     = out.get("confidence")
            ab_s1    = out.get("ab_s1")
            ab_s2    = out.get("ab_s2")
            disp_AB  = apply_confidence(pred_AB, conf) if conf is not None else pred_AB
            pred_RGB = lab_to_rgb(L, disp_AB)

            pred_AB_next = None
            if has_temp:
                out_n = self.model(L_next)
                pred_AB_next = out_n["ab"]

            losses = self.loss_fn(
                pred_ab=pred_AB, target_ab=AB,
                pred_rgb=pred_RGB, target_rgb=RGB,
                ab_s1=ab_s1, ab_s2=ab_s2,
                confidence=conf,
                pred_ab_next=pred_AB_next,
                L_curr=L if has_temp else None,
                L_next=L_next if has_temp else None,
            )

        if not torch.isfinite(losses["total"]):
            details = ", ".join(
                f"{name}={float(value.detach().float().cpu())}"
                for name, value in losses.items()
            )
            raise FloatingPointError(f"Non-finite training loss detected: {details}")

        self.scaler.scale(losses["total"]).backward()
        self.scaler.unscale_(self.optimizer)
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip, error_if_nonfinite=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def _save_preview(self, epoch: int, L: torch.Tensor,
                      pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> None:
        count = min(4, L.size(0))
        gray = ((L[:count] + 1.0) / 2.0).clamp(0, 1).repeat(1, 3, 1, 1)
        rows = []
        for idx in range(count):
            row = torch.cat([gray[idx], pred_rgb[idx], target_rgb[idx]], dim=2)
            rows.append(row)
        grid = torch.cat(rows, dim=1).detach().cpu()
        grid = torch.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        array = (grid.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(array).save(self.preview_dir / f"epoch_{epoch:03d}.jpg", quality=92)

    def _validate(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.eval()
        tl: dict[str,float] = {}
        tm: dict[str,float] = {}
        n = 0
        for batch in loader:
            L   = batch["L"].to(self.device)
            AB  = batch["AB"].to(self.device)
            RGB = batch["RGB"].to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                out      = self.model(L)
                pred_AB  = out["ab"]
                conf     = out.get("confidence")
                ab_s1    = out.get("ab_s1")
                ab_s2    = out.get("ab_s2")
                disp_AB  = apply_confidence(pred_AB, conf) if conf is not None else pred_AB
                pred_RGB = lab_to_rgb(L, disp_AB)
                losses   = self.loss_fn(pred_ab=pred_AB, target_ab=AB,
                                        pred_rgb=pred_RGB, target_rgb=RGB,
                                        ab_s1=ab_s1, ab_s2=ab_s2, confidence=conf)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Non-finite validation loss detected")
            bs = L.size(0)
            for k,v in losses.items(): tl[k] = tl.get(k,0.0) + v.item()*bs
            for k,v in compute_metrics(pred_RGB, RGB, self.metric_names).items():
                tm[k] = tm.get(k,0.0) + v*bs
            if n == 0 and epoch % self.preview_every == 0:
                self._save_preview(epoch, L, pred_RGB, RGB)
            n += bs
        self.model.train()
        return {**{f"val_{k}":v/n for k,v in tl.items()},
                **{f"val_{k}":v/n for k,v in tm.items()}}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict]:
        print(f"\n{'='*64}")
        amp_label = str(self.amp_dtype).replace("torch.", "") if self.use_amp else "off"
        print(f"  ChromaNet v3  |  RTX 4070  |  AMP={amp_label}")
        print("  Memory + Scene + MultiScale + Confidence + Temporal + FreqLoss")
        print(f"{'='*64}\n")

        run_started = time.monotonic()
        max_seconds = self.max_hours * 3600.0 if self.max_hours else None

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            el: dict[str,float] = {}
            ns = 0
            t0 = time.time()

            for step, batch in enumerate(train_loader):
                sl = self._step(batch)
                for k,v in sl.items(): el[k] = el.get(k,0.0)+v
                ns += 1
                if (step+1) % self.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"  Ep {epoch+1:03d} | Step {step+1:05d} "
                          f"| total={el['total']/ns:.4f} "
                          f"freq={el.get('frequency',0)/ns:.4f} "
                          f"ms={el.get('multiscale',0)/ns:.4f} "
                          f"temp={el.get('temporal',0)/ns:.4f} "
                          f"| lr={lr:.6f}")

                if max_seconds and time.monotonic() - run_started >= max_seconds:
                    path = self.save_checkpoint(epoch, "_time_limit")
                    self.update_latest(path)
                    print(f"\n[stop] Reached {self.max_hours:g} hour limit during epoch {epoch + 1}.")
                    print("[stop] Resume with: --resume latest")
                    return self.history

            avg   = {k:v/ns for k,v in el.items()}
            vstat = self._validate(val_loader, epoch + 1)
            self.scheduler.step()
            psnr  = vstat.get("val_psnr", 0.0)
            print(f"\nEpoch {epoch+1:03d}/{self.epochs} "
                  f"train={avg['total']:.4f} val={vstat['val_total']:.4f} "
                  f"PSNR={psnr:.2f}dB  {time.time()-t0:.0f}s\n")

            rec = {"epoch": epoch+1,
                   **{f"train_{k}":v for k,v in avg.items()},
                   **vstat, "lr": self.optimizer.param_groups[0]["lr"]}
            self.history.append(rec)

            if psnr > self.best_psnr:
                self.best_psnr = psnr
                self.save_checkpoint(epoch+1, "_best")
            if (epoch+1) % self.save_every == 0:
                epoch_path = self.save_checkpoint(epoch+1)
                self.update_latest(epoch_path)

            (self.checkpoint_dir/"history.json").write_text(
                json.dumps(self.history, indent=2), encoding="utf-8")

        final_path = self.save_checkpoint(self.epochs, "_final")
        self.update_latest(final_path)
        print(f"[done] Best PSNR: {self.best_psnr:.2f} dB")
        return self.history

__all__ = ["Trainer"]
