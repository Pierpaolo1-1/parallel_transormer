
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# CONFIGS
# ============================================================

@dataclass
class PlantViTConfig:
    in_channels: int = 3
    num_classes: int = 10
    dims: Tuple[int, int, int] = (48, 64, 96)
    channels: Tuple[int, int, int, int, int] = (16, 24, 32, 48, 64)
    transformer_depths: Tuple[int, int, int] = (1, 1, 2)
    patch_size: Tuple[int, int] = (2, 2)
    mlp_ratio: float = 2.0
    num_heads: Tuple[int, int, int] = (4, 4, 4)
    dropout: float = 0.0
    attention_dropout: float = 0.0
    cbam_reduction: int = 16


@dataclass
class SwinLikeConfig:
    in_channels: int = 3
    embed_dim: int = 32
    depths: Tuple[int, int, int] = (1, 1, 2)
    num_heads: Tuple[int, int, int] = (4, 4, 8)
    window_size: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    attention_dropout: float = 0.0


@dataclass
class FusionHeadConfig:
    hidden_dim: int = 128
    dropout: float = 0.30
    num_classes: int = 10


# ============================================================
# COMMON BLOCKS
# ============================================================

class ConvBNAct(nn.Module):
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


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
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
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerEncoderBlock(nn.Module):
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
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout, attention_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SqueezeExciteLikeMLP(nn.Module):
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
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.pre = ConvBNAct(channels, channels, kernel_size=3)
        self.channel_mlp = SqueezeExciteLikeMLP(channels, reduction)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pre(x)
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        channel_attn = torch.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
        x = x * channel_attn

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.sigmoid(self.spatial(torch.cat([avg_map, max_map], dim=1)))
        x = x * spatial_attn
        return x


class InvertedResidual7x7(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: int = 2) -> None:
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            ConvBNAct(in_channels, hidden_dim, kernel_size=1),
            ConvBNAct(hidden_dim, hidden_dim, kernel_size=7, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


# ============================================================
# PLANTVIT BRANCH
# ============================================================

class PatchUnfoldFoldMixin:
    @staticmethod
    def unfold_patches(x: Tensor, patch_size: Tuple[int, int]):
        b, c, h, w = x.shape
        ph, pw = patch_size
        new_h = math.ceil(h / ph) * ph
        new_w = math.ceil(w / pw) * pw
        if new_h != h or new_w != w:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        x = x.reshape(b, c, new_h // ph, ph, new_w // pw, pw)
        x = x.permute(0, 3, 5, 2, 4, 1).contiguous()
        tokens = x.reshape(b * ph * pw, (new_h // ph) * (new_w // pw), c)
        return tokens, (b, c, new_h, new_w)

    @staticmethod
    def fold_patches(tokens: Tensor, meta, patch_size: Tuple[int, int]):
        b, c, h, w = meta
        ph, pw = patch_size
        x = tokens.reshape(b, ph, pw, h // ph, w // pw, c)
        x = x.permute(0, 5, 3, 1, 4, 2).contiguous()
        x = x.reshape(b, c, h, w)
        return x


class PlantViTBlock(nn.Module, PatchUnfoldFoldMixin):
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
        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(transformer_dim, num_heads, mlp_ratio, dropout, attention_dropout)
            for _ in range(depth)
        ])
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
        return self.fusion(y)


class PlantViTBackbone(nn.Module):
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
            PlantViTBlock(c2, d0, t0, h0, cfg.patch_size, cfg.mlp_ratio, cfg.dropout, cfg.attention_dropout, cfg.cbam_reduction),
        )
        self.stage3 = nn.Sequential(
            InvertedResidual7x7(c2, c3, stride=2),
            PlantViTBlock(c3, d1, t1, h1, cfg.patch_size, cfg.mlp_ratio, cfg.dropout, cfg.attention_dropout, cfg.cbam_reduction),
        )
        self.stage4 = nn.Sequential(
            InvertedResidual7x7(c3, c4, stride=2),
            PlantViTBlock(c4, d2, t2, h2, cfg.patch_size, cfg.mlp_ratio, cfg.dropout, cfg.attention_dropout, cfg.cbam_reduction),
        )
        self.stage5 = ConvBNAct(c4, c4 * 2, kernel_size=1)
        self.out_dim = c4 * 2

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
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


# ============================================================
# SWIN-LIKE BRANCH
# ============================================================

def window_partition(x: Tensor, window_size: int):
    b, h, w, c = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h))
        x = x.permute(0, 2, 3, 1)
    hp, wp = x.shape[1], x.shape[2]
    x = x.view(b, hp // window_size, window_size, wp // window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = x.view(-1, window_size * window_size, c)
    return windows, (hp, wp)


def window_reverse(windows: Tensor, window_size: int, b: int, hp: int, wp: int, c: int):
    x = windows.view(b, hp // window_size, wp // window_size, window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(b, hp, wp, c)
    return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return self.norm(x)


class SwinLikeBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int, mlp_ratio: float, dropout: float, attention_dropout: float) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout, attention_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, h, w, c = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows, (hp, wp) = window_partition(x, self.window_size)
        windows = self.attn(windows)
        x = window_reverse(windows, self.window_size, b, hp, wp, c)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x[:, :h, :w, :]
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, h, w, c = x.shape
        if h % 2 == 1:
            x = F.pad(x.permute(0, 3, 1, 2), (0, 0, 0, 1)).permute(0, 2, 3, 1)
            h += 1
        if w % 2 == 1:
            x = F.pad(x.permute(0, 3, 1, 2), (0, 1, 0, 0)).permute(0, 2, 3, 1)
            w += 1

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        return self.reduction(x)


class SwinLikeStage(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int, mlp_ratio: float, dropout: float, attention_dropout: float, downsample: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinLikeBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinLikeBackbone(nn.Module):
    def __init__(self, cfg: SwinLikeConfig) -> None:
        super().__init__()
        d0 = cfg.embed_dim
        d1 = d0 * 2
        d2 = d1 * 2
        self.patch_embed = PatchEmbed(cfg.in_channels, d0)
        self.stage1 = SwinLikeStage(d0, cfg.depths[0], cfg.num_heads[0], cfg.window_size, cfg.mlp_ratio, cfg.dropout, cfg.attention_dropout, downsample=True)
        self.stage2 = SwinLikeStage(d1, cfg.depths[1], cfg.num_heads[1], cfg.window_size, cfg.mlp_ratio, cfg.dropout, cfg.attention_dropout, downsample=True)
        self.stage3 = SwinLikeStage(d2, cfg.depths[2], cfg.num_heads[2], cfg.window_size, cfg.mlp_ratio, cfg.dropout, cfg.attention_dropout, downsample=False)
        self.norm = nn.LayerNorm(d2)
        self.out_dim = d2

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


# ============================================================
# FUSION
# ============================================================

class FusionHead(nn.Module):
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
    def __init__(self, plantvit_cfg: PlantViTConfig, swin_cfg: SwinLikeConfig, fusion_cfg: FusionHeadConfig) -> None:
        super().__init__()
        self.plantvit = PlantViTBackbone(plantvit_cfg)
        self.swin = SwinLikeBackbone(swin_cfg)
        self.classifier = FusionHead(self.plantvit.out_dim + self.swin.out_dim, fusion_cfg)

    def forward_features(self, x: Tensor) -> Dict[str, Tensor]:
        plantvit_feat = self.plantvit(x)
        swin_feat = self.swin(x)
        fused = torch.cat([plantvit_feat, swin_feat], dim=1)
        return {"plantvit": plantvit_feat, "swin": swin_feat, "fused": fused}

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.forward_features(x)["fused"])


def build_hybrid_model(
    num_classes: int = 10,
    plantvit_cfg: Optional[PlantViTConfig] = None,
    swin_cfg: Optional[SwinLikeConfig] = None,
    fusion_cfg: Optional[FusionHeadConfig] = None,
) -> HybridPlantViTSwinNet:
    plantvit_cfg = plantvit_cfg or PlantViTConfig(num_classes=num_classes)
    swin_cfg = swin_cfg or SwinLikeConfig()
    fusion_cfg = fusion_cfg or FusionHeadConfig(num_classes=num_classes)

    if plantvit_cfg.num_classes != num_classes:
        warnings.warn(
            "PlantViTConfig.num_classes is not directly used by the backbone. "
            "The final number of classes is controlled by FusionHeadConfig.",
            stacklevel=2,
        )

    return HybridPlantViTSwinNet(plantvit_cfg, swin_cfg, fusion_cfg)
