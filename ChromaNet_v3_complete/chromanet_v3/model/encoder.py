"""
encoder.py — ChromaNet Encoder
VGG-style encoder with residual blocks and progressive downsampling.
Input: L channel [B, 1, H, W]
Output: list of 5 feature maps [s0(fine) ... s4(bottleneck)]
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [ConvBnRelu(in_ch, out_ch), ResBlock(out_ch, out_ch)]
        if downsample:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ChromaEncoder(nn.Module):
    """
    5-stage encoder. Returns features from fine→coarse.
    Args:
        in_channels:    1 for L channel
        stage_channels: output channels per stage
    """
    def __init__(self, in_channels: int = 1,
                 stage_channels: list[int] | None = None) -> None:
        super().__init__()
        if stage_channels is None:
            stage_channels = [64, 128, 256, 512, 512]
        ch = [in_channels] + stage_channels
        self.stages = nn.ModuleList([
            EncoderStage(ch[i], ch[i + 1], downsample=(i < 4))
            for i in range(len(stage_channels))
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


__all__ = ["ChromaEncoder", "ConvBnRelu", "ResBlock"]
