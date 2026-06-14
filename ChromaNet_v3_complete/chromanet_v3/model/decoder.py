"""
decoder.py — ChromaNet Multi-Scale Decoder  ★ UPGRADED in v3 ★

NEW in v3: Multi-Scale Prediction
  Predicts AB at 3 scales simultaneously:
    - Scale 1 (64×64)   → large color regions (sky, ground)
    - Scale 2 (128×128) → medium objects
    - Scale 3 (256×256) → fine details

  All three are fused via a learned weighted sum.
  This fixes both large-region consistency AND fine detail coloring.

Each decoder stage:
    1. Bilinear upsample ×2
    2. GatedSkip fusion with encoder skip
    3. ConvBnRelu ×2
    4. CBAM attention
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CBAM, GatedSkip
from .encoder import ConvBnRelu


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 attention: bool = True) -> None:
        super().__init__()
        self.skip_proj  = nn.Sequential(
            nn.Conv2d(skip_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
        )
        self.gated_skip = GatedSkip(in_ch)
        self.conv1      = ConvBnRelu(in_ch, out_ch)
        self.conv2      = ConvBnRelu(out_ch, out_ch)
        self.attention  = CBAM(out_ch) if attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        skip = self.skip_proj(skip)
        x    = self.gated_skip(skip, x)
        x    = self.conv2(self.conv1(x))
        return self.attention(x)


class ABHead(nn.Module):
    """Predict AB channels from feature map. Output: tanh → [-1,1]"""
    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MultiScaleFusion(nn.Module):
    """
    Fuse AB predictions from 3 scales into one final prediction.
    Learns a per-scale weight (softmax-normalized so they sum to 1).
    """
    def __init__(self) -> None:
        super().__init__()
        # raw logits — softmax gives weights summing to 1
        self.scale_weights = nn.Parameter(torch.ones(3))

    def forward(self, ab_s1: torch.Tensor, ab_s2: torch.Tensor,
                ab_s3: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """
        Args:
            ab_s1: [B,2,H/4,W/4]  coarse scale
            ab_s2: [B,2,H/2,W/2]  medium scale
            ab_s3: [B,2,H,W]      fine scale
            target_size: (H, W) final output size
        Returns:
            Fused AB [B, 2, H, W]
        """
        w = F.softmax(self.scale_weights, dim=0)  # [3] summing to 1

        # Upsample all to target size
        s1 = F.interpolate(ab_s1, size=target_size, mode="bilinear", align_corners=False)
        s2 = F.interpolate(ab_s2, size=target_size, mode="bilinear", align_corners=False)
        s3 = ab_s3  # already at target size

        return w[0] * s1 + w[1] * s2 + w[2] * s3


class ChromaDecoder(nn.Module):
    """
    ChromaNet v3 Decoder with Multi-Scale AB Prediction.

    Returns: (ab_pred, ab_s1, ab_s2, final_features)
      - ab_pred:       [B, 2, H, W]  final fused AB
      - ab_s1:         [B, 2, H/4, W/4] coarse scale (for multi-scale loss)
      - ab_s2:         [B, 2, H/2, W/2] medium scale (for multi-scale loss)
      - final_features:[B, dec[-1], H, W] for confidence head
    """
    def __init__(self, encoder_channels: list[int] | None = None,
                 decoder_channels: list[int] | None = None,
                 attention: bool = True, dropout: float = 0.1) -> None:
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512, 512]
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64, 32]

        enc, dec = encoder_channels, decoder_channels

        self.bottleneck_drop = nn.Dropout2d(dropout)

        self.stages = nn.ModuleList([
            DecoderStage(enc[4], enc[3], dec[0], attention),  # → 1/8
            DecoderStage(dec[0], enc[2], dec[1], attention),  # → 1/4
            DecoderStage(dec[1], enc[1], dec[2], attention),  # → 1/2
            DecoderStage(dec[2], enc[0], dec[3], attention),  # → 1/1
        ])

        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnRelu(dec[3], dec[4]),
            ConvBnRelu(dec[4], dec[4]),
        )

        # ── Multi-scale AB heads ──────────────────────────────────────
        # s1 head branches after stage 1 (1/4 resolution)
        self.ab_head_s1 = ABHead(dec[1])
        # s2 head branches after stage 2 (1/2 resolution)
        self.ab_head_s2 = ABHead(dec[2])
        # s3 head is the main final head
        self.ab_head_s3 = ABHead(dec[4])

        self.fusion = MultiScaleFusion()

    def forward(self, features: list[torch.Tensor]) -> tuple:
        s0, s1, s2, s3, bottleneck = features

        x = self.bottleneck_drop(bottleneck)

        x  = self.stages[0](x, s3)
        x  = self.stages[1](x, s2)

        # ── Scale 1 branch (coarsest) ─────────────────────────────────
        ab_s1 = self.ab_head_s1(x)                   # [B,2,H/4,W/4]

        x  = self.stages[2](x, s1)

        # ── Scale 2 branch (medium) ───────────────────────────────────
        ab_s2 = self.ab_head_s2(x)                   # [B,2,H/2,W/2]

        x  = self.stages[3](x, s0)
        final_features = self.final_conv(x)

        # ── Scale 3 branch (finest) ───────────────────────────────────
        ab_s3 = self.ab_head_s3(final_features)      # [B,2,H,W]

        # ── Fuse all three scales ─────────────────────────────────────
        H, W  = final_features.shape[2:]
        ab_pred = self.fusion(ab_s1, ab_s2, ab_s3, (H, W))

        return ab_pred, ab_s1, ab_s2, final_features


__all__ = ["ChromaDecoder", "MultiScaleFusion"]
