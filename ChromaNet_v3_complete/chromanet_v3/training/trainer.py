"""trainer.py — ChromaNet v3 Training Loop"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from skimage import color as skcolor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from .metrics import compute_metrics
from model.confidence import apply_confidence


def _lab_to_rgb(L: torch.Tensor, AB: torch.Tensor) -> torch.Tensor:
    L_d  = (L + 1.0) * 50.0
    AB_d = AB * 110.0
    LAB  = torch.cat([L_d, AB_d], dim=1).detach().cpu().numpy()
    imgs = []
    for b in range(LAB.shape[0]):
        rgb = skcolor.lab2rgb(LAB[b].transpose(1,2,0)).astype(np.float32)
        imgs.append(torch.from_numpy(rgb.transpose(2,0,1)))
    return torch.stack(imgs).to(L.device)


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
        self.checkpoint_dir  = Path(tc.get("checkpoint_dir","./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp    = tc.get("mixed_precision", True) and device.type == "cuda"
        self.scaler     = GradScaler(enabled=self.use_amp)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        ec = cfg.get("evaluation", {})
        self.metric_names = ec.get("metrics", ["psnr","ssim","colorfulness"])
        self.history: list[dict] = []
        self.best_psnr   = 0.0
        self.start_epoch = 0

    def save_checkpoint(self, epoch: int, tag: str = "") -> None:
        p = self.checkpoint_dir / f"checkpoint_epoch{epoch:03d}{tag}.pth"
        torch.save({"epoch": epoch,
                    "model_state":     self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "scaler_state":    self.scaler.state_dict(),
                    "best_psnr":       self.best_psnr,
                    "history":         self.history,
                    "cfg":             self.cfg}, p)
        print(f"  [ckpt] → {p}")

    def load_checkpoint(self, path) -> None:
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model_state"])
        self.optimizer.load_state_dict(ck["optimizer_state"])
        self.scheduler.load_state_dict(ck["scheduler_state"])
        self.scaler.load_state_dict(ck["scaler_state"])
        self.best_psnr   = ck.get("best_psnr", 0.0)
        self.history     = ck.get("history", [])
        self.start_epoch = ck["epoch"] + 1
        print(f"[resume] epoch {ck['epoch']}")

    def _step(self, batch: dict) -> dict[str, float]:
        L   = batch["L"].to(self.device)
        AB  = batch["AB"].to(self.device)
        RGB = batch["RGB"].to(self.device)
        has_temp = "L_next" in batch
        if has_temp:
            L_next = batch["L_next"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=self.use_amp):
            out = self.model(L)
            pred_AB  = out["ab"]
            conf     = out.get("confidence")
            ab_s1    = out.get("ab_s1")
            ab_s2    = out.get("ab_s2")
            disp_AB  = apply_confidence(pred_AB, conf) if conf is not None else pred_AB
            pred_RGB = _lab_to_rgb(L, disp_AB)

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

        self.scaler.scale(losses["total"]).backward()
        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        tl: dict[str,float] = {}
        tm: dict[str,float] = {}
        n = 0
        for batch in loader:
            L   = batch["L"].to(self.device)
            AB  = batch["AB"].to(self.device)
            RGB = batch["RGB"].to(self.device)
            with autocast(enabled=self.use_amp):
                out      = self.model(L)
                pred_AB  = out["ab"]
                conf     = out.get("confidence")
                ab_s1    = out.get("ab_s1")
                ab_s2    = out.get("ab_s2")
                disp_AB  = apply_confidence(pred_AB, conf) if conf is not None else pred_AB
                pred_RGB = _lab_to_rgb(L, disp_AB)
                losses   = self.loss_fn(pred_ab=pred_AB, target_ab=AB,
                                        pred_rgb=pred_RGB, target_rgb=RGB,
                                        ab_s1=ab_s1, ab_s2=ab_s2, confidence=conf)
            bs = L.size(0)
            for k,v in losses.items(): tl[k] = tl.get(k,0.0) + v.item()*bs
            for k,v in compute_metrics(pred_RGB, RGB, self.metric_names).items():
                tm[k] = tm.get(k,0.0) + v*bs
            n += bs
        self.model.train()
        return {**{f"val_{k}":v/n for k,v in tl.items()},
                **{f"val_{k}":v/n for k,v in tm.items()}}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict]:
        print(f"\n{'='*64}")
        print(f"  ChromaNet v3  |  RTX 4070  |  AMP={self.use_amp}")
        print(f"  Memory ✓  Scene ✓  MultiScale ✓  Confidence ✓  Temporal ✓  FreqLoss ✓")
        print(f"{'='*64}\n")

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

            avg   = {k:v/ns for k,v in el.items()}
            vstat = self._validate(val_loader)
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
                self.save_checkpoint(epoch+1)

        self.save_checkpoint(self.epochs, "_final")
        (self.checkpoint_dir/"history.json").write_text(json.dumps(self.history, indent=2))
        print(f"[done] Best PSNR: {self.best_psnr:.2f} dB")
        return self.history

__all__ = ["Trainer"]
