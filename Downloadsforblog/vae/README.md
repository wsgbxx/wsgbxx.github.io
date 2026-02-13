# VAE（教学版 / 给博客用）

> 目标：用**尽量少的代码**讲清楚 VAE 的核心数据流：
> `x -> encoder -> (mu, logvar) -> reparameterize -> z -> decoder -> recon`。

---

## 0. 你会看到哪些文件

- `model.py`：`ConvVAE`（encode / reparameterize / decode / forward）
- `losses.py`：`vae_loss`（reconstruction + KL）
- `../scripts/train_vae_cifar10.py`：最小可运行训练脚本（数据、AMP、训练循环、保存 ckpt）

---

## 1. 约定（非常重要）

- 如果 `out_activation="sigmoid"`：
  - **输入 x 期望在 `[0, 1]`**
  - decoder 输出 `recon` 也在 `[0, 1]`
- 如果你把数据预处理成 `[-1, 1]`，那就应该用 `out_activation="tanh"`（并对应修改 loss）。

> 这里的实现刻意保持“模块纯净”：`vae/` 下只管张量计算，不做任何 dataset / 文件 I/O。

---

## 2. 核心代码：模型（`vae/model.py`）

下面这段就是 VAE 的“主干代码”。读的时候只抓 4 个点：
1) `encode` 输出 `(mu, logvar)`
2) `reparameterize` 采样 `z`
3) `decode` 从 `z` 还原
4) `forward` 把它们串起来并返回结构体

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class VAEOutput:
    recon: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor


class ConvVAE(nn.Module):
    """A small convolutional VAE for images.

    Teaching goals:
    - Keep the module *pure* (no dataset / file I/O here)
    - Expose encode / reparameterize / decode

    Input range expectation:
    - If out_activation="sigmoid", input x is expected in [0, 1].
    """

    def __init__(
        self,
        *,
        latent_dim: int = 128,
        base_channels: int = 64,
        image_size: int = 64,
        in_channels: int = 3,
        out_activation: str = "sigmoid",
    ) -> None:
        super().__init__()

        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16 for this architecture")

        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_activation = out_activation

        ch = base_channels

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, ch, 4, 2, 1),
            nn.ReLU(True),

            nn.Conv2d(ch, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),

            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),

            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(True),
        )

        # 只用来推断 flatten 后的维度（避免你手算）；不参与训练。
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            enc_out = self.enc(dummy)
            feat_dim = enc_out.reshape(1, -1).size(1)
            feat_shape = enc_out.shape[1:]

        self._feat_shape = feat_shape

        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, feat_dim)

        act: nn.Module
        if out_activation == "sigmoid":
            act = nn.Sigmoid()
        elif out_activation == "tanh":
            act = nn.Tanh()
        elif out_activation == "none":
            act = nn.Identity()
        else:
            raise ValueError(f"Unknown out_activation={out_activation!r}")

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch * 8, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),

            nn.ConvTranspose2d(ch, in_channels, 4, 2, 1),
            act,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        h = h.reshape(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        # 关键：std = exp(0.5*logvar)，然后 z = mu + eps*std。
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.shape, device=std.device, dtype=std.dtype, generator=generator)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(h.size(0), *self._feat_shape)
        return self.dec(h)

    def forward(
        self,
        x: torch.Tensor,
        *,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> VAEOutput:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, generator=generator) if sample_posterior else mu
        recon = self.decode(z)
        return VAEOutput(recon=recon, mu=mu, logvar=logvar, z=z)
```

---

## 3. 核心代码：loss（`vae/losses.py`）

就两项：
- recon（重建误差）
- KL（把 `q(z|x)` 拉向标准高斯 `N(0,I)`）

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0,
    recon_type: str = "mse",
    reduction: str = "mean",
) -> dict[str, torch.Tensor]:

    if recon_type == "mse":
        recon_loss = F.mse_loss(recon, x, reduction=reduction)

    # KL(q(z|x) || N(0, I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kld_per = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    if reduction == "mean":
        kld = kld_per.mean()
    else: reduction == "sum":
        kld = kld_per.sum()

    loss = recon_loss + beta * kld
    return {"loss": loss, "recon": recon_loss, "kld": kld}
```

---

## 4. 跑起来（CIFAR-10）

> 请提前把 CIFAR-10 放到本地目录。

```bash
# 推荐：用环境变量写短命令
export DATA_ROOT=/path/to/cifar10
python ../scripts/train_vae_cifar10.py --out-dir ./out_vae --epochs 1
```

---

## 5. 常见坑（只列最常见的）

- **输入范围不对**：`sigmoid` 输出只能表达 [0,1]，你却喂了 [-1,1]。
- **loss 与 activation 不匹配**：`BCE` 需要 recon 在 (0,1)；`MSE` 更稳。
- **beta 太大**：KL 压得太狠会造成 recon 很糊，先从 `beta=1` 开始。

