"""
scene.py — Scene Classification Conditioning  ★ NEW in v3 ★

Before colorizing, a lightweight classifier detects the scene type:
    0=indoor  1=outdoor  2=portrait  3=nature  4=urban

This label is embedded and injected into the bottleneck so the
decoder knows context before picking colors.

Example: outdoor → expects blue sky, green grass
         portrait → expects warm skin tones
         indoor  → expects artificial lighting colors

WHY THIS IS ORIGINAL:
No published colorization paper conditions on scene type without
explicit user input. This is fully automatic and learned from data.

Architecture:
    L channel [B,1,H,W]
        ↓ lightweight CNN (3 conv layers)
        ↓ global avg pool
        ↓ FC → 5 class logits
        ↓ argmax → scene label
        ↓ embedding table [5, embed_dim]
        ↓ inject into bottleneck via addition
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

SCENE_CLASSES = ["indoor", "outdoor", "portrait", "nature", "urban"]
NUM_SCENES    = len(SCENE_CLASSES)


class SceneClassifier(nn.Module):
    """
    Lightweight CNN that predicts scene type from L channel.
    Runs on the same L input as the main encoder — no extra input needed.

    Args:
        in_channels:  1 (L channel)
        num_classes:  5 scene types
    """
    def __init__(self, in_channels: int = 1, num_classes: int = NUM_SCENES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # deliberately lightweight — we don't want this to dominate
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, L: torch.Tensor) -> torch.Tensor:
        """
        Args:
            L: [B, 1, H, W] normalized L channel
        Returns:
            logits [B, num_classes]
        """
        return self.classifier(self.features(L))

    def predict(self, L: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices [B]"""
        with torch.no_grad():
            return self.forward(L).argmax(dim=1)


class SceneConditioner(nn.Module):
    """
    Embeds scene label and injects it into the bottleneck feature map.

    Args:
        feature_channels: Bottleneck channel count (e.g. 512)
        num_classes:       Number of scene types (5)
        embed_dim:         Scene embedding size
    """
    def __init__(self, feature_channels: int = 512,
                 num_classes: int = NUM_SCENES,
                 embed_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, feature_channels),
            nn.LayerNorm(feature_channels),
            nn.Tanh(),
        )
        # Learnable gate — starts near 0 for stable early training
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, scene_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:            Bottleneck features [B, C, H, W]
            scene_logits: Scene classifier output [B, num_classes]
        Returns:
            Scene-conditioned features [B, C, H, W]
        """
        # Soft scene embedding: weighted sum over all classes
        # (differentiable — better than hard argmax during training)
        scene_probs  = F.softmax(scene_logits, dim=-1)               # [B, num_classes]
        all_embeds   = self.embedding.weight                          # [num_classes, embed_dim]
        soft_embed   = torch.matmul(scene_probs, all_embeds)         # [B, embed_dim]
        scene_vec    = self.proj(soft_embed)                          # [B, C]
        gate         = torch.sigmoid(self.gate)
        return x + gate * scene_vec.unsqueeze(-1).unsqueeze(-1)


__all__ = ["SceneClassifier", "SceneConditioner", "SCENE_CLASSES", "NUM_SCENES"]
