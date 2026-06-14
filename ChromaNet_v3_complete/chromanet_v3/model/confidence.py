"""
confidence.py — Per-Pixel Confidence Map (from v2, unchanged)
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn


class ConfidenceHead(nn.Module):
    def __init__(self, in_channels: int = 32) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.head[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def apply_confidence(ab: torch.Tensor, conf: torch.Tensor,
                     threshold: float = 0.3) -> torch.Tensor:
    scale = ((conf - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
    return ab * scale


def save_confidence_heatmap(confidence: torch.Tensor,
                             output_path: str | Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        conf_np = confidence.detach().cpu().squeeze().numpy()
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(conf_np, cmap="hot", vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Colorization Confidence")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[warn] confidence heatmap failed: {e}")


__all__ = ["ConfidenceHead", "apply_confidence", "save_confidence_heatmap"]
