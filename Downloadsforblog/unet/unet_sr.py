from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .blocks import Downsample, ResBlock, Upsample


@dataclass(frozen=True)
class UNetSRConfig:
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 64
    dropout: float = 0.0


class SimpleUNetSR(nn.Module):
    """A small UNet for supervised image-to-image tasks (e.g., super-resolution).

    This is adapted from your original SR UNet, but extracted as a reusable module.

    Forward returns a residual (delta). Typical usage:
        delta = net(x)
        y = clamp(x + delta)
    """

    def __init__(self, cfg: UNetSRConfig = UNetSRConfig()) -> None:
        super().__init__()

        ch1 = cfg.base_channels
        ch2 = cfg.base_channels * 2
        ch3 = cfg.base_channels * 4
        ch4 = cfg.base_channels * 6

        self.in_conv = nn.Conv2d(cfg.in_channels, ch1, 3, padding=1)

        self.d1 = nn.Sequential(ResBlock(ch1, ch1, dropout=cfg.dropout), ResBlock(ch1, ch1, dropout=cfg.dropout))
        self.down1 = Downsample(ch1)

        self.d2 = nn.Sequential(ResBlock(ch1, ch2, dropout=cfg.dropout), ResBlock(ch2, ch2, dropout=cfg.dropout))
        self.down2 = Downsample(ch2)

        self.d3 = nn.Sequential(ResBlock(ch2, ch3, dropout=cfg.dropout), ResBlock(ch3, ch3, dropout=cfg.dropout))
        self.down3 = Downsample(ch3)

        self.mid = nn.Sequential(ResBlock(ch3, ch4, dropout=cfg.dropout), ResBlock(ch4, ch4, dropout=cfg.dropout))

        self.up3 = Upsample(ch4)
        self.u3 = nn.Sequential(ResBlock(ch4 + ch3, ch3, dropout=cfg.dropout), ResBlock(ch3, ch3, dropout=cfg.dropout))

        self.up2 = Upsample(ch3)
        self.u2 = nn.Sequential(ResBlock(ch3 + ch2, ch2, dropout=cfg.dropout), ResBlock(ch2, ch2, dropout=cfg.dropout))

        self.up1 = Upsample(ch2)
        self.u1 = nn.Sequential(ResBlock(ch2 + ch1, ch1, dropout=cfg.dropout), ResBlock(ch1, ch1, dropout=cfg.dropout))

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=min(32, ch1), num_channels=ch1),
            nn.SiLU(),
            nn.Conv2d(ch1, cfg.out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(x)
        s1 = self.d1(x0)
        h = self.down1(s1)

        s2 = self.d2(h)
        h = self.down2(s2)

        s3 = self.d3(h)
        h = self.down3(s3)

        h = self.mid(h)

        h = self.up3(h)
        h = torch.cat([h, s3], dim=1)
        h = self.u3(h)

        h = self.up2(h)
        h = torch.cat([h, s2], dim=1)
        h = self.u2(h)

        h = self.up1(h)
        h = torch.cat([h, s1], dim=1)
        h = self.u1(h)

        return self.out(h)
