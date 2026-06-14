"""
temporal.py — Temporal Consistency Loss (from v2, unchanged)
Penalizes color flickering between consecutive video frames.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyLoss(nn.Module):
    def __init__(self, motion_threshold: float = 0.08,
                 lambda_temp: float = 2.0) -> None:
        super().__init__()
        self.motion_threshold = motion_threshold
        self.lambda_temp      = lambda_temp

    def _motion_mask(self, L_curr: torch.Tensor,
                     L_next: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(L_curr - L_next)
        return torch.sigmoid((self.motion_threshold - diff) * 20.0)

    def forward(self, AB_curr: torch.Tensor, AB_next: torch.Tensor,
                L_curr: torch.Tensor, L_next: torch.Tensor) -> torch.Tensor:
        mask = self._motion_mask(L_curr, L_next)
        return self.lambda_temp * (torch.abs(AB_curr - AB_next) * mask).mean()


class TemporalVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset) -> None:
        self.dataset = base_dataset

    def __len__(self) -> int:
        return max(0, len(self.dataset) - 1)

    def __getitem__(self, idx: int) -> dict:
        curr  = self.dataset[idx]
        next_ = self.dataset[idx + 1]
        return {
            "L": curr["L"], "AB": curr["AB"], "RGB": curr["RGB"],
            "L_next": next_["L"], "AB_next": next_["AB"], "RGB_next": next_["RGB"],
        }


__all__ = ["TemporalConsistencyLoss", "TemporalVideoDataset"]
