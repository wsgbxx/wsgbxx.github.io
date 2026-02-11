from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels: int, *, max_groups: int = 32) -> nn.GroupNorm:
    """A safe GroupNorm chooser that always divides channels."""

    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


class ResBlock(nn.Module):
    """A simple residual block used by teaching UNet variants."""

    def __init__(self, in_ch: int, out_ch: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            group_norm(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            group_norm(out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return h + self.skip(x)


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
