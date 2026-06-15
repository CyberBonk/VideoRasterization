"""scheduler.py — Warmup + Cosine Decay LR Scheduler"""
from __future__ import annotations
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer: Optimizer, cfg: dict) -> LambdaLR:
    tc         = cfg.get("training", {})
    sc         = cfg.get("scheduler", {})
    total      = tc.get("epochs", 40)
    warmup     = sc.get("warmup_epochs", 3)
    lr         = tc.get("lr", 0.001)
    lr_min     = sc.get("min_lr", 0.00001)
    ratio      = lr_min / lr

    def fn(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / max(1, warmup)
        prog   = (epoch - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
        return ratio + (1.0 - ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=fn)

__all__ = ["build_scheduler"]
