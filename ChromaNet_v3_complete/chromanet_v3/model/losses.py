"""
losses.py — ChromaNet v3 Loss Functions

All 6 losses:
  1. ChrominanceL1         — weighted per-pixel AB error
  2. PerceptualLoss        — VGG feature matching
  3. ColorfulnessLoss      — fights desaturation
  4. TemporalConsistency   — no video flickering
  5. ConfidenceLoss        — calibrated uncertainty
  6. FrequencyAwareLoss    ★ NEW v3 — penalizes edge color bleeding
  7. MultiScaleLoss        ★ NEW v3 — supervises all 3 decoder scales
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from .temporal import TemporalConsistencyLoss


def lab_to_rgb(L: torch.Tensor, AB: torch.Tensor) -> torch.Tensor:
    """Convert normalized Lab tensors to differentiable sRGB tensors."""
    l = (L + 1.0) * 50.0
    a = AB[:, 0:1] * 110.0
    b = AB[:, 1:2] * 110.0

    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    delta = 6.0 / 29.0

    def inv_f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > delta, t.pow(3), 3.0 * delta**2 * (t - 4.0 / 29.0))

    x = 0.95047 * inv_f(fx)
    y = inv_f(fy)
    z = 1.08883 * inv_f(fz)

    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    rgb_linear = torch.cat([r, g, bl], dim=1)
    rgb = torch.where(
        rgb_linear <= 0.0031308,
        12.92 * rgb_linear,
        # Keep the unselected power branch away from zero: x**(1/2.4)
        # has an infinite derivative at zero and can poison backward through
        # torch.where even when the linear branch is selected.
        1.055 * torch.clamp_min(rgb_linear, 0.0031308).pow(1.0 / 2.4) - 0.055,
    )
    return rgb.clamp(0.0, 1.0)


# ── 1. Chrominance-Weighted L1 ────────────────────────────────────────────

class ChrominanceL1Loss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sat    = torch.sqrt(target[:,0:1]**2 + target[:,1:2]**2 + self.eps)
        weight = (sat / (sat.mean(dim=[2,3], keepdim=True) + self.eps)).clamp(0.5, 4.0)
        return (weight * torch.abs(pred - target)).mean()


# ── 2. Perceptual Loss (VGG-16) ───────────────────────────────────────────

_VGG = {"relu1_2": 4, "relu2_2": 9, "relu3_3": 16, "relu4_3": 23}

class PerceptualLoss(nn.Module):
    def __init__(self, layers: list[str] | None = None) -> None:
        super().__init__()
        if layers is None:
            layers = ["relu2_2", "relu3_3"]
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        prev = 0
        for name in layers:
            idx = _VGG[name]
            self.slices.append(vgg[prev: idx + 1])
            prev = idx + 1
        for p in self.parameters():
            p.requires_grad = False
        w = 1.0 / len(layers)
        self.weights = [w] * len(layers)
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = (pred   - self.mean) / self.std
        t = (target - self.mean) / self.std
        loss = torch.tensor(0.0, device=pred.device)
        for sl, w in zip(self.slices, self.weights):
            p = sl(p); t = sl(t)
            loss = loss + w * F.l1_loss(p, t.detach())
        return loss


# ── 3. Colorfulness Loss ──────────────────────────────────────────────────

class ColorfulnessLoss(nn.Module):
    def __init__(self, target: float = 0.15, eps: float = 1e-8) -> None:
        super().__init__()
        self.target = target
        self.eps = eps

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        R, G, B = rgb[:,0], rgb[:,1], rgb[:,2]
        rg = R - G
        yb = 0.5*(R+G) - B
        rg_flat = rg.flatten(1)
        yb_flat = yb.flatten(1)
        spread = torch.sqrt(
            rg_flat.var(1, unbiased=False) + yb_flat.var(1, unbiased=False) + self.eps
        )
        center = torch.sqrt(
            rg_flat.mean(1).square() + yb_flat.mean(1).square() + self.eps
        )
        cf = spread + 0.3 * center
        return F.relu(self.target - cf).mean()


# ── 4. Confidence Calibration Loss ───────────────────────────────────────

class ConfidenceLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                conf: torch.Tensor) -> torch.Tensor:
        err        = torch.abs(pred - target).mean(dim=1, keepdim=True)
        calibration = (conf * err).mean()
        # uncertain regions → smooth AB predictions
        grad_h = (pred[:,:,1:,:] - pred[:,:,:-1,:]).abs().mean(1, keepdim=True)[:,:,:,1:]
        grad_w = (pred[:,:,:,1:] - pred[:,:,:,:-1]).abs().mean(1, keepdim=True)[:,:,1:,:]
        uncertainty = (1.0 - conf[:,:,1:,:][:,:,:,1:])
        smoothness  = (uncertainty * (grad_h + grad_w) / 2.0).mean()
        return calibration + 0.1 * smoothness


# ── 5. Frequency-Aware Loss  ★ NEW v3 ★ ──────────────────────────────────

class FrequencyAwareLoss(nn.Module):
    """
    Splits the AB prediction into low-frequency (smooth regions)
    and high-frequency (edges) components, then weights them differently.

    High-frequency (edge) regions get a higher penalty weight
    because color bleeding across edges is the most visible artifact.

    How the split works:
        low_freq  = gaussian_blur(AB)
        high_freq = AB - low_freq          (residual = edges + details)

    Args:
        edge_weight:  How much harder to penalize edge regions.
                      1.0 = same as flat regions. 3.0 = 3× harder.
        blur_kernel:  Gaussian blur kernel size for low/high split.
    """
    def __init__(self, edge_weight: float = 3.0, blur_kernel: int = 5) -> None:
        super().__init__()
        self.edge_weight = edge_weight
        self.blur_kernel = blur_kernel
        # Build fixed Gaussian kernel
        self._build_kernel(blur_kernel)

    def _build_kernel(self, k: int) -> None:
        sigma  = k / 3.0
        coords = torch.arange(k, dtype=torch.float32) - k // 2
        g      = torch.exp(-(coords**2) / (2 * sigma**2))
        g      = g / g.sum()
        kernel = g.outer(g)                          # [k, k]
        # Apply to 2 channels (A and B) independently
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1)  # [2,1,k,k]
        self.register_buffer("kernel", kernel)

    def _low_freq(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.blur_kernel // 2
        return F.conv2d(x, self.kernel, padding=pad, groups=2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, 2, H, W] predicted AB
            target: [B, 2, H, W] ground-truth AB
        Returns:
            Scalar frequency-aware loss
        """
        # Low frequency components
        pred_low   = self._low_freq(pred)
        target_low = self._low_freq(target)

        # High frequency (edge) components
        pred_high   = pred   - pred_low
        target_high = target - target_low

        # Standard L1 on smooth regions
        loss_low  = F.l1_loss(pred_low, target_low)

        # Weighted L1 on edge regions — penalize bleeding harder
        loss_high = F.l1_loss(pred_high, target_high) * self.edge_weight

        return loss_low + loss_high


# ── 6. Multi-Scale Supervision Loss  ★ NEW v3 ★ ──────────────────────────

class MultiScaleLoss(nn.Module):
    """
    Supervises all 3 decoder output scales independently.

    Each scale gets its own L1 loss against a downsampled version
    of the ground truth. Coarser scales are weighted less.

    Scale weights (default): s1=0.2, s2=0.4, s3(main)=handled by ChrominanceL1
    """
    def __init__(self, weight_s1: float = 0.2, weight_s2: float = 0.4) -> None:
        super().__init__()
        self.weight_s1 = weight_s1
        self.weight_s2 = weight_s2

    def forward(self, ab_s1: torch.Tensor, ab_s2: torch.Tensor,
                target_ab: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ab_s1:     [B, 2, H/4, W/4] coarse scale prediction
            ab_s2:     [B, 2, H/2, W/2] medium scale prediction
            target_ab: [B, 2, H, W]     full resolution ground truth
        """
        # Downsample target to match each scale
        target_s1 = F.interpolate(target_ab, size=ab_s1.shape[2:],
                                  mode="bilinear", align_corners=False)
        target_s2 = F.interpolate(target_ab, size=ab_s2.shape[2:],
                                  mode="bilinear", align_corners=False)

        loss_s1 = F.l1_loss(ab_s1, target_s1) * self.weight_s1
        loss_s2 = F.l1_loss(ab_s2, target_s2) * self.weight_s2
        return loss_s1 + loss_s2


# ── Combined ChromaLoss v3 ────────────────────────────────────────────────

class ChromaLoss(nn.Module):
    """
    Full ChromaNet v3 loss:
        λ_l1    * ChrominanceL1
      + λ_perc  * Perceptual
      + λ_cf    * Colorfulness
      + λ_temp  * Temporal
      + λ_conf  * Confidence
      + λ_freq  * FrequencyAware     ★ new
      + λ_ms    * MultiScale         ★ new
    """
    def __init__(self, lambda_l1: float = 10.0, lambda_perceptual: float = 1.0,
                 lambda_colorfulness: float = 0.5, lambda_temporal: float = 2.0,
                 lambda_confidence: float = 0.5, lambda_freq: float = 2.0,
                 lambda_multiscale: float = 1.0,
                 perceptual_layers: list[str] | None = None) -> None:
        super().__init__()
        self.lw = dict(l1=lambda_l1, perc=lambda_perceptual,
                       cf=lambda_colorfulness, temp=lambda_temporal,
                       conf=lambda_confidence, freq=lambda_freq,
                       ms=lambda_multiscale)
        self.chroma_l1   = ChrominanceL1Loss()
        self.perceptual = (
            PerceptualLoss(layers=perceptual_layers)
            if lambda_perceptual > 0 else None
        )
        self.colorfulness= ColorfulnessLoss()
        self.temporal    = TemporalConsistencyLoss(lambda_temp=lambda_temporal)
        self.conf_loss   = ConfidenceLoss()
        self.freq_loss   = FrequencyAwareLoss()
        self.ms_loss     = MultiScaleLoss()

    def forward(self, pred_ab, target_ab, pred_rgb, target_rgb,
                ab_s1=None, ab_s2=None, confidence=None,
                pred_ab_next=None, L_curr=None, L_next=None,
                ) -> dict[str, torch.Tensor]:

        l1   = self.chroma_l1(pred_ab, target_ab)
        perc = (self.perceptual(pred_rgb, target_rgb) if self.perceptual is not None
                else torch.tensor(0.0, device=pred_ab.device))
        cf   = self.colorfulness(pred_rgb)
        freq = self.freq_loss(pred_ab, target_ab)

        total = (self.lw["l1"]   * l1
               + self.lw["perc"] * perc
               + self.lw["cf"]   * cf
               + self.lw["freq"] * freq)

        losses = {"l1": l1, "perceptual": perc,
                  "colorfulness": cf, "frequency": freq}

        # Multi-scale loss
        if ab_s1 is not None and ab_s2 is not None:
            ms = self.ms_loss(ab_s1, ab_s2, target_ab)
            total = total + self.lw["ms"] * ms
            losses["multiscale"] = ms
        else:
            losses["multiscale"] = torch.tensor(0.0, device=pred_ab.device)

        # Temporal loss
        if pred_ab_next is not None and L_curr is not None and L_next is not None:
            temp = self.temporal(pred_ab, pred_ab_next, L_curr, L_next)
            total = total + temp
            losses["temporal"] = temp
        else:
            losses["temporal"] = torch.tensor(0.0, device=pred_ab.device)

        # Confidence loss
        if confidence is not None:
            conf_l = self.conf_loss(pred_ab, target_ab, confidence)
            total  = total + self.lw["conf"] * conf_l
            losses["confidence"] = conf_l
        else:
            losses["confidence"] = torch.tensor(0.0, device=pred_ab.device)

        losses["total"] = total
        return losses


def build_loss(cfg: dict) -> ChromaLoss:
    lc = cfg.get("loss", {})
    return ChromaLoss(
        lambda_l1          = lc.get("lambda_l1", 10.0),
        lambda_perceptual  = lc.get("lambda_perceptual", 1.0),
        lambda_colorfulness= lc.get("lambda_colorfulness", 0.5),
        lambda_temporal    = lc.get("lambda_temporal", 2.0),
        lambda_confidence  = lc.get("lambda_confidence", 0.5),
        lambda_freq        = lc.get("lambda_freq", 2.0),
        lambda_multiscale  = lc.get("lambda_multiscale", 1.0),
        perceptual_layers  = lc.get("perceptual_layers", ["relu2_2", "relu3_3"]),
    )

__all__ = ["ChromaLoss", "build_loss", "FrequencyAwareLoss", "MultiScaleLoss", "lab_to_rgb"]
