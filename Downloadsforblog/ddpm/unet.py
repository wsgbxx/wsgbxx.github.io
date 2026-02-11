from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default(val, d):
    return d if val is None else val


def timestep_embedding(timesteps: torch.Tensor, dim: int, *, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding（给扩散步数 t 一个“可学习前的特征表示”）。

    timesteps: (B,) 的 int/float tensor。

    为什么要这么做：
    - 直接把 t 当作一个标量喂给网络，信息量太少；embedding 能提供丰富的周期特征。
    - 这个实现是“无参数”的，便于教学/复现实验。

    Returns: (B, dim)
    """

    if timesteps.ndim != 1:
        raise ValueError("timesteps must be a 1D tensor")

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        *,
        dropout: float = 0.0,
        groups: int = 32,
    ) -> None:
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_ch), num_channels=in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_ch)

        self.norm2 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """2D feature map 上的简化 self-attention。

    注意事项：
    - 这里是教学版实现：把 (H,W) 展平成序列长度 L=H*W。
    - L 会随分辨率平方增长，所以我们只在低分辨率层（例如 16x16）开 attention。
    """

    def __init__(self, channels: int, *, num_heads: int = 4, groups: int = 32) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)
        self.num_heads = num_heads
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x).reshape(b, c, h * w)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        scale = head_dim ** -0.5
        attn = torch.softmax(torch.einsum("bnhm,bnhk->bhmk", q * scale, k), dim=-1)
        out = torch.einsum("bhmk,bnhk->bnhm", attn, v)
        out = out.reshape(b, c, h * w)
        out = self.proj(out).reshape(b, c, h, w)
        return x_in + out


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)


@dataclass
class UNetConfig:
    image_size: int = 32
    in_channels: int = 3
    model_channels: int = 128
    out_channels: int = 3
    num_res_blocks: int = 2
    channel_mult: tuple[int, ...] = (1, 2, 2, 2)
    attention_resolutions: tuple[int, ...] = (16,)
    dropout: float = 0.1
    num_heads: int = 4


class DiffusionUNet(nn.Module):
    """A teaching-friendly diffusion UNet predicting noise eps.

    For CIFAR-10 (32x32), a reasonable default is UNetConfig().
    """

    def __init__(self, cfg: UNetConfig, *, time_emb_dim: Optional[int] = None) -> None:
        super().__init__()
        self.cfg = cfg

        time_emb_dim = _default(time_emb_dim, cfg.model_channels * 4)

        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.in_conv = nn.Conv2d(cfg.in_channels, cfg.model_channels, 3, padding=1)

        # Down
        input_ch = cfg.model_channels
        ds = cfg.image_size
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels: list[int] = []

        for level, mult in enumerate(cfg.channel_mult):
            out_ch = cfg.model_channels * mult
            for _ in range(cfg.num_res_blocks):
                self.down_blocks.append(ResBlock(input_ch, out_ch, time_emb_dim, dropout=cfg.dropout))
                input_ch = out_ch
                if ds in cfg.attention_resolutions:
                    self.down_blocks.append(AttentionBlock(input_ch, num_heads=cfg.num_heads))
                self.skip_channels.append(input_ch)
            if level != len(cfg.channel_mult) - 1:
                self.downsamples.append(Downsample(input_ch))
                ds //= 2
            else:
                self.downsamples.append(nn.Identity())

        # Middle
        self.mid1 = ResBlock(input_ch, input_ch, time_emb_dim, dropout=cfg.dropout)
        self.mid_attn = AttentionBlock(input_ch, num_heads=cfg.num_heads) if ds in cfg.attention_resolutions else nn.Identity()
        self.mid2 = ResBlock(input_ch, input_ch, time_emb_dim, dropout=cfg.dropout)

        # Up
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self._skip_channels_up = list(reversed(self.skip_channels))
        for level, mult in reversed(list(enumerate(cfg.channel_mult))):
            out_ch = cfg.model_channels * mult
            for _ in range(cfg.num_res_blocks):
                skip_ch = self._skip_channels_up.pop(0)
                self.up_blocks.append(ResBlock(input_ch + skip_ch, out_ch, time_emb_dim, dropout=cfg.dropout))
                input_ch = out_ch
                if ds in cfg.attention_resolutions:
                    self.up_blocks.append(AttentionBlock(input_ch, num_heads=cfg.num_heads))
            if level != 0:
                self.upsamples.append(Upsample(input_ch))
                ds *= 2
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(num_groups=min(32, input_ch), num_channels=input_ch)
        self.out_conv = nn.Conv2d(input_ch, cfg.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) int64 or float
        t_emb = timestep_embedding(t, self.cfg.model_channels)
        t_emb = self.time_mlp(t_emb)

        hs: list[torch.Tensor] = []
        h = self.in_conv(x)

        # Down path
        di = 0
        ds = self.cfg.image_size
        for level, _mult in enumerate(self.cfg.channel_mult):
            for _ in range(self.cfg.num_res_blocks):
                h = self.down_blocks[di](h, t_emb)
                di += 1
                if di < len(self.down_blocks) and isinstance(self.down_blocks[di], AttentionBlock):
                    h = self.down_blocks[di](h)
                    di += 1
                hs.append(h)
            if level != len(self.cfg.channel_mult) - 1:
                h = self.downsamples[level](h)
                ds //= 2

        # Middle
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # Up path
        ui = 0
        for level, _mult in reversed(list(enumerate(self.cfg.channel_mult))):
            for _ in range(self.cfg.num_res_blocks):
                h = torch.cat([h, hs.pop()], dim=1)
                h = self.up_blocks[ui](h, t_emb)
                ui += 1
                if ui < len(self.up_blocks) and isinstance(self.up_blocks[ui], AttentionBlock):
                    h = self.up_blocks[ui](h)
                    ui += 1
            h = self.upsamples[len(self.cfg.channel_mult) - 1 - level](h)

        return self.out_conv(F.silu(self.out_norm(h)))
