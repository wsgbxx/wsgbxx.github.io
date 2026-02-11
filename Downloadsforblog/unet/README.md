# UNet（监督学习版 / 给博客用）

> 目标：把一个“小而全”的 UNet 模型做成可复用模块，用于**配对的图像到图像任务**，比如：
> - 去噪（x -> y）
> - 超分（LR -> HR）
> - 去模糊、去雾等

---

## 0. 你会看到哪些文件

- `blocks.py`：`ResBlock / Downsample / Upsample` 等小组件
- `unet_sr.py`：`SimpleUNetSR`（一个小型 SR 风格 UNet，forward 输出 residual）

> 注意：扩散模型用的 UNet 在 `ddpm/unet.py`，那份多了 timestep embedding 与 attention，训练目标也不同。

---

## 1. 约定（这个版本的 forward 输出什么？）

`SimpleUNetSR` 的 forward **返回 residual（delta）**：

- `delta = net(x)`
- `y = x + delta`

这样做的好处：
- 残差学习通常更稳定（网络只需学“改动量”）
- 训练初期输出不会乱飞

---

## 2. 核心代码：Blocks（`unet/blocks.py`）

```python
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
```

---

## 3. 核心代码：UNet 主体（`unet/unet_sr.py`）

读这段你只需要抓住：
- down path 保存 skip（`s1/s2/s3`）
- up path `torch.cat([h, skip], dim=1)` 拼回去
- 输出是 residual：`return self.out(h)`

```python
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
```

---

## 4. 怎么用（最小示例）

```python
import torch

from unet.unet_sr import SimpleUNetSR

x = torch.randn(4, 3, 64, 64)
net = SimpleUNetSR()

delta = net(x)
y = (x + delta).clamp(-1, 1)  # 具体 clamp 范围取决于你数据的归一化
```

---

## 5. 常见坑

- **skip 维度对不上**：上采样后 H/W 要和 skip 的 H/W 对齐。
- **输出范围**：如果你训练目标是 [0,1]，最后别 clamp 到 [-1,1]。
- **显存**：UNet 的中间特征图比较大，`base_channels` 不要一上来就开太大。
