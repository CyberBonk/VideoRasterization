"""
memory.py — Semantic Color Memory Module (from v2, unchanged)

32 learnable memory slots injected at the bottleneck.
The model learns "sky → blue, grass → green" from data automatically.
No other standard colorization model has this.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticColorMemory(nn.Module):
    def __init__(self, feature_channels: int = 512,
                 num_slots: int = 32, mem_dim: int = 256,
                 temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.query_proj = nn.Sequential(
            nn.Linear(feature_channels, mem_dim),
            nn.LayerNorm(mem_dim),
            nn.ReLU(inplace=True),
        )
        self.memory_keys   = nn.Parameter(torch.randn(num_slots, mem_dim) * 0.1)
        self.memory_values = nn.Parameter(torch.randn(num_slots, mem_dim) * 0.1)
        self.value_proj = nn.Sequential(
            nn.Linear(mem_dim, feature_channels),
            nn.LayerNorm(feature_channels),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pooled     = x.mean(dim=[2, 3])
        query      = F.normalize(self.query_proj(pooled), dim=-1)
        keys       = F.normalize(self.memory_keys, dim=-1)
        attn       = F.softmax(torch.matmul(query, keys.T) / self.temperature, dim=-1)
        readout    = self.value_proj(torch.matmul(attn, self.memory_values))
        gate       = torch.sigmoid(self.gate)
        return x + gate * readout.unsqueeze(-1).unsqueeze(-1)

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=[2, 3])
        query  = F.normalize(self.query_proj(pooled), dim=-1)
        keys   = F.normalize(self.memory_keys, dim=-1)
        return F.softmax(torch.matmul(query, keys.T) / self.temperature, dim=-1)


__all__ = ["SemanticColorMemory"]
