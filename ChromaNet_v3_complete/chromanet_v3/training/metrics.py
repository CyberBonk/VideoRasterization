"""metrics.py — PSNR, SSIM, Colorfulness"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1,2,3])
    return float((10.0 * torch.log10(1.0 / (mse + 1e-8))).mean().item())


def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11, sigma: float = 1.5) -> float:
    C1, C2 = 0.01**2, 0.03**2
    coords = torch.arange(window_size, dtype=torch.float32) - window_size//2
    g      = torch.exp(-(coords**2) / (2*sigma**2)); g = g/g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0).repeat(3,1,1,1).to(pred.device)
    pad    = window_size//2
    mu1    = F.conv2d(pred,   kernel, padding=pad, groups=3)
    mu2    = F.conv2d(target, kernel, padding=pad, groups=3)
    s1_sq  = F.conv2d(pred*pred,     kernel, padding=pad, groups=3) - mu1**2
    s2_sq  = F.conv2d(target*target, kernel, padding=pad, groups=3) - mu2**2
    s12    = F.conv2d(pred*target,   kernel, padding=pad, groups=3) - mu1*mu2
    m      = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1_sq+s2_sq+C2))
    return float(m.mean().item())


def colorfulness_score(rgb: torch.Tensor) -> float:
    R,G,B  = rgb[:,0], rgb[:,1], rgb[:,2]
    rg, yb = R-G, 0.5*(R+G)-B
    cf     = (torch.sqrt(rg.flatten(1).std(1)**2 + yb.flatten(1).std(1)**2)
              + 0.3*torch.sqrt(rg.flatten(1).mean(1)**2+yb.flatten(1).mean(1)**2))
    return float(cf.mean().item())


def compute_metrics(pred: torch.Tensor, target: torch.Tensor,
                    names: list[str] | None = None) -> dict[str, float]:
    if names is None: names = ["psnr","ssim","colorfulness"]
    out: dict[str,float] = {}
    if "psnr"         in names: out["psnr"]         = psnr(pred, target)
    if "ssim"         in names: out["ssim"]         = ssim(pred, target)
    if "colorfulness" in names: out["colorfulness"] = colorfulness_score(pred)
    return out

__all__ = ["psnr","ssim","colorfulness_score","compute_metrics"]
