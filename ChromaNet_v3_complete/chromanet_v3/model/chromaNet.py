"""
chromaNet.py — ChromaNet v3  (Full Model)

Pipeline:
    L [B,1,H,W]
        ↓
    ChromaEncoder          → [s0, s1, s2, s3, bottleneck]
        ↓
    SemanticColorMemory    → augments bottleneck with color priors
        ↓
    SceneClassifier        → detects scene type (indoor/outdoor/etc)
    SceneConditioner       → injects scene context into bottleneck
        ↓
    ChromaDecoder          → (ab_pred, ab_s1, ab_s2, final_features)
        ↓                     ↑ multi-scale outputs
    ConfidenceHead         → confidence [B,1,H,W]
        ↓
    Output: ab_pred, ab_s1, ab_s2, confidence

What's new vs v2:
    ★ Scene Classification Conditioning
    ★ Multi-Scale AB Prediction (3 scales fused)
    ★ Frequency-Aware Loss (in losses.py)
    ★ Before/After Split View (in inference/splitview.py)
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .encoder     import ChromaEncoder
from .decoder     import ChromaDecoder
from .memory      import SemanticColorMemory
from .scene       import SceneClassifier, SceneConditioner
from .confidence  import ConfidenceHead


class ChromaNet(nn.Module):
    def __init__(
        self,
        encoder_channels : list[int] | None = None,
        decoder_channels : list[int] | None = None,
        attention        : bool  = True,
        dropout          : float = 0.1,
        memory_slots     : int   = 32,
        mem_dim          : int   = 256,
        use_memory       : bool  = True,
        use_scene        : bool  = True,
        use_confidence   : bool  = True,
    ) -> None:
        super().__init__()

        if encoder_channels is None: encoder_channels = [64, 128, 256, 512, 512]
        if decoder_channels is None: decoder_channels = [512, 256, 128, 64, 32]

        self.use_memory     = use_memory
        self.use_scene      = use_scene
        self.use_confidence = use_confidence

        self.encoder = ChromaEncoder(1, encoder_channels)

        if use_memory:
            self.memory = SemanticColorMemory(
                feature_channels=encoder_channels[-1],
                num_slots=memory_slots, mem_dim=mem_dim,
            )

        if use_scene:
            self.scene_classifier = SceneClassifier(in_channels=1)
            self.scene_conditioner = SceneConditioner(
                feature_channels=encoder_channels[-1]
            )

        self.decoder = ChromaDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            attention=attention, dropout=dropout,
        )

        if use_confidence:
            self.confidence_head = ConfidenceHead(decoder_channels[-1])

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, L: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            L: [B, 1, H, W] normalized L channel [-1, 1]
        Returns:
            dict with keys:
                ab         — [B, 2, H, W] final fused AB prediction
                ab_s1      — [B, 2, H/4, W/4] coarse scale
                ab_s2      — [B, 2, H/2, W/2] medium scale
                confidence — [B, 1, H, W] (if use_confidence)
                scene_logits — [B, 5] (if use_scene)
        """
        features = self.encoder(L)

        # ── Scene classification ──────────────────────────────────────
        scene_logits = None
        if self.use_scene:
            scene_logits = self.scene_classifier(L)
            features[-1] = self.scene_conditioner(features[-1], scene_logits)

        # ── Semantic color memory ─────────────────────────────────────
        if self.use_memory:
            features[-1] = self.memory(features[-1])

        # ── Decode (multi-scale) ──────────────────────────────────────
        ab_pred, ab_s1, ab_s2, final_features = self.decoder(features)

        out = {"ab": ab_pred, "ab_s1": ab_s1, "ab_s2": ab_s2}

        if scene_logits is not None:
            out["scene_logits"] = scene_logits

        # ── Confidence head ───────────────────────────────────────────
        if self.use_confidence:
            out["confidence"] = self.confidence_head(final_features)

        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict) -> ChromaNet:
    mc = cfg.get("model", {})
    return ChromaNet(
        encoder_channels = mc.get("encoder_channels", [64, 128, 256, 512, 512]),
        decoder_channels = mc.get("decoder_channels", [512, 256, 128, 64, 32]),
        attention        = mc.get("attention", True),
        dropout          = mc.get("dropout", 0.1),
        memory_slots     = mc.get("memory_slots", 32),
        mem_dim          = mc.get("mem_dim", 256),
        use_memory       = mc.get("use_memory", True),
        use_scene        = mc.get("use_scene", True),
        use_confidence   = mc.get("use_confidence", True),
    )

__all__ = ["ChromaNet", "build_model"]
