from __future__ import annotations

"""
Model definitions for the hybrid PlantViT + Swin architecture.

This file contains:
- PlantViT-style branch inspired by PMVT / plant-based MobileViT
- Swin Transformer branch
- Fusion classifier head
- HybridPlantViTSwinNet main model
- Configuration dataclasses
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torchvision.models import swin_t, Swin_T_Weights
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This file requires torchvision with Swin Transformer support."
    ) from exc


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class PlantViTConfig:
    """Configuration for the PlantViT-style branch."""

    in_channels: int = 3
    num_classes: int = 4
    dims: Tuple[int, int, int] = (96, 128, 160)
    channels: Tuple[int, int, int, int, int] = (16, 24, 48, 64, 80)
    transformer_depths: Tuple[int, int, int] = (2, 4, 3)
    patch_size: Tuple[int, int] = (2, 2)
    mlp_ratio: float = 2.0
    num_heads: Tuple[int, int, int] = (4, 4, 4)
    dropout: float = 0.0
    attention_dropout: float = 0.0
    cbam_reduction: int = 16
    classifier_dim: int = 512


@dataclass
class SwinBranchConfig:
    """Configuration for the Swin branch."""

    pretrained: bool = True
    freeze_backbone: bool = False
    out_dim: int = 768


@dataclass
class FusionHeadConfig:
    """Configuration for the fusion head."""

    hidden_dim: int = 512
    dropout: float = 0.30
    num_classes: int = 4


# ============================================================
# BASIC BUILDING BLOCKS
# ============================================================


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + activation helper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation if activation is not None else nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SqueezeExciteLikeMLP(nn.Module):
    """Shared MLP used by CBAM channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.pre = ConvBNAct(channels, channels, kernel_size=3)
        self.channel_mlp = SqueezeExciteLikeMLP(channels, reduction)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pre(x)

        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        channel_attn = torch.sigmoid(
            self.channel_mlp(avg_pool) + self.channel_mlp(max_pool)
        )
        x = x * channel_attn

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.sigmoid(self.spatial(torch.cat([avg_map, max_map], dim=1)))
        x = x * spatial_attn
        return x


class InvertedResidual7x7(nn.Module):
    """Inverted residual block with 7x7 depthwise convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 2,
    ) -> None:
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        self.block = nn.Sequential(
            ConvBNAct(in_channels, hidden_dim, kernel_size=1),
            ConvBNAct(
                hidden_dim,
                hidden_dim,
                kernel_size=7,
                stride=stride,
                groups=hidden_dim,
            ),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class MLP(nn.Module):
    """Transformer feed-forward network."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        bsz, n_tokens, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(bsz, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(bsz, n_tokens, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# PLANTVIT-STYLE BRANCH
# ============================================================


class PatchUnfoldFoldMixin:
    """Utility mixin for patch unfold/fold operations."""

    @staticmethod
    def unfold_patches(
        x: Tensor,
        patch_size: Tuple[int, int]
    ) -> Tuple[Tensor, Tuple[int, int, int, int]]:
        b, c, h, w = x.shape
        ph, pw = patch_size

        new_h = math.ceil(h / ph) * ph
        new_w = math.ceil(w / pw) * pw
        if new_h != h or new_w != w:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        x = x.reshape(b, c, new_h // ph, ph, new_w // pw, pw)
        x = x.permute(0, 3, 5, 2, 4, 1).contiguous()
        tokens = x.reshape(b * ph * pw, (new_h // ph) * (new_w // pw), c)
        meta = (b, c, new_h, new_w)
        return tokens, meta

    @staticmethod
    def fold_patches(
        tokens: Tensor,
        meta: Tuple[int, int, int, int],
        patch_size: Tuple[int, int]
    ) -> Tensor:
        b, c, h, w = meta
        ph, pw = patch_size
        x = tokens.reshape(b, ph, pw, h // ph, w // pw, c)
        x = x.permute(0, 5, 3, 1, 4, 2).contiguous()
        x = x.reshape(b, c, h, w)
        return x


class PlantViTBlock(nn.Module, PatchUnfoldFoldMixin):
    """MobileViT-like local-global block with CBAM."""

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        depth: int,
        num_heads: int,
        patch_size: Tuple[int, int],
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        cbam_reduction: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.local_rep = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            ConvBNAct(in_channels, transformer_dim, kernel_size=1),
        )

        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    dim=transformer_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

        self.project_back = ConvBNAct(transformer_dim, in_channels, kernel_size=1)
        self.cbam = CBAM(in_channels, reduction=cbam_reduction)
        self.fusion = ConvBNAct(in_channels * 2, in_channels, kernel_size=3)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        y = self.local_rep(x)

        tokens, meta = self.unfold_patches(y, self.patch_size)
        tokens = self.transformer(tokens)
        y = self.fold_patches(tokens, meta, self.patch_size)

        y = self.project_back(y)
        y = self.cbam(y)

        if y.shape[-2:] != residual.shape[-2:]:
            y = F.interpolate(y, size=residual.shape[-2:], mode="bilinear", align_corners=False)

        y = torch.cat([residual, y], dim=1)
        y = self.fusion(y)
        return y


class PlantViTBackbone(nn.Module):
    """PlantViT-inspired backbone."""

    def __init__(self, cfg: PlantViTConfig) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = cfg.channels
        d0, d1, d2 = cfg.dims
        t0, t1, t2 = cfg.transformer_depths
        h0, h1, h2 = cfg.num_heads

        self.stem = ConvBNAct(cfg.in_channels, c0, kernel_size=3, stride=2)

        self.stage1 = InvertedResidual7x7(c0, c1, stride=1)
        self.stage2 = nn.Sequential(
            InvertedResidual7x7(c1, c2, stride=2),
            PlantViTBlock(
                in_channels=c2,
                transformer_dim=d0,
                depth=t0,
                num_heads=h0,
                patch_size=cfg.patch_size,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                attention_dropout=cfg.attention_dropout,
                cbam_reduction=cfg.cbam_reduction,
            ),
        )
        self.stage3 = nn.Sequential(
            InvertedResidual7x7(c2, c3, stride=2),
            PlantViTBlock(
                in_channels=c3,
                transformer_dim=d1,
                depth=t1,
                num_heads=h1,
                patch_size=cfg.patch_size,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                attention_dropout=cfg.attention_dropout,
                cbam_reduction=cfg.cbam_reduction,
            ),
        )
        self.stage4 = nn.Sequential(
            InvertedResidual7x7(c3, c4, stride=2),
            PlantViTBlock(
                in_channels=c4,
                transformer_dim=d2,
                depth=t2,
                num_heads=h2,
                patch_size=cfg.patch_size,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                attention_dropout=cfg.attention_dropout,
                cbam_reduction=cfg.cbam_reduction,
            ),
        )
        self.stage5 = ConvBNAct(c4, c4 * 4, kernel_size=1)
        self.out_dim = c4 * 4

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x


# ============================================================
# SWIN BRANCH
# ============================================================


class SwinBackbone(nn.Module):
    """Wrapper around torchvision Swin-T."""

    def __init__(self, cfg: SwinBranchConfig) -> None:
        super().__init__()
        weights = Swin_T_Weights.DEFAULT if cfg.pretrained else None
        backbone = swin_t(weights=weights)

        if cfg.freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.features = backbone.features
        self.norm = backbone.norm
        self.permute = backbone.permute
        self.avgpool = backbone.avgpool
        self.flatten = nn.Flatten(1)
        self.out_dim = cfg.out_dim

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x


# ============================================================
# FUSION MODEL
# ============================================================


class FusionHead(nn.Module):
    """Final fusion classifier."""

    def __init__(self, in_dim: int, cfg: FusionHeadConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HybridPlantViTSwinNet(nn.Module):
    """Hybrid network with PlantViT and Swin in parallel."""

    def __init__(
        self,
        plantvit_cfg: PlantViTConfig,
        swin_cfg: SwinBranchConfig,
        fusion_cfg: FusionHeadConfig,
    ) -> None:
        super().__init__()
        self.plantvit = PlantViTBackbone(plantvit_cfg)
        self.swin = SwinBackbone(swin_cfg)

        fusion_in_dim = self.plantvit.out_dim + self.swin.out_dim
        self.classifier = FusionHead(fusion_in_dim, fusion_cfg)

    def forward_features(self, x: Tensor) -> Dict[str, Tensor]:
        plantvit_feat = self.plantvit(x)
        swin_feat = self.swin(x)
        fused = torch.cat([plantvit_feat, swin_feat], dim=1)
        return {
            "plantvit": plantvit_feat,
            "swin": swin_feat,
            "fused": fused,
        }

    def forward(self, x: Tensor) -> Tensor:
        feats = self.forward_features(x)
        logits = self.classifier(feats["fused"])
        return logits


def build_hybrid_model(
    num_classes: int = 4,
    plantvit_cfg: Optional[PlantViTConfig] = None,
    swin_cfg: Optional[SwinBranchConfig] = None,
    fusion_cfg: Optional[FusionHeadConfig] = None,
) -> HybridPlantViTSwinNet:
    """Convenience factory for the hybrid model."""
    plantvit_cfg = plantvit_cfg or PlantViTConfig(num_classes=num_classes)
    swin_cfg = swin_cfg or SwinBranchConfig()
    fusion_cfg = fusion_cfg or FusionHeadConfig(num_classes=num_classes)

    if plantvit_cfg.num_classes != num_classes:
        warnings.warn(
            "PlantViTConfig.num_classes is not directly used by the backbone. "
            "The final number of classes is controlled by FusionHeadConfig.",
            stacklevel=2,
        )

    return HybridPlantViTSwinNet(
        plantvit_cfg=plantvit_cfg,
        swin_cfg=swin_cfg,
        fusion_cfg=fusion_cfg,
    )
