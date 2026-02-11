---
title: VAE
date: 2026-02-11 00:15:46
categories:
  - 项目
  - 生成模型之旅
tags:
  - 生成模型
---
---
title: VAE
date: 2026-02-11 00:15:46
categories:
  - 项目
  - 生成模型之旅
tags:
  - 生成模型
---

# VAE (变分自编码器) 探索

## 简介

VAE从数学角度出发，利用神经网络训练的encoder和decoder，将高维空间图片投射到低维空间中，并用KL散度进行正态分布先验的限制，再投影回高维。有趣的是似乎每种类型的图像都可以在低维空间里找到自己的地盘，因此实现在低维空间里随机取点，却不会产生过于混合模糊的图片（其实还是会的，图二就很糊），可以直接生成全新虚拟图片。虽然数学原理尚没有办法很清晰的理解，但是这个生成图片的思路确实妙不可言！

## 实现细节

### 数据集与参数
- 使用CIFAR-10数据集，输入图片为32x32x3，输出图片也为32x32x3
- 使用MSE损失函数
- 在L4上即可训练

### 模型架构

#### 1. 初始化方法 (`__init__`)
- 设置模型的基本参数：潜在空间维度、通道数、图像尺寸、输入通道数
- 定义编码器（Encoder）：包含4层卷积层，逐步将输入图像缩小并提取特征
- 定义潜在空间映射：将编码后的特征映射到潜在空间的均值和方差
- 定义解码器（Decoder）：包含4层转置卷积层，将潜在向量重建为图像

#### 2. 编码器 (`encode`)
- 将输入图像通过卷积网络进行特征提取
- 将提取的特征展平并通过全连接层映射到潜在空间的均值（mu）和对数方差（logvar）

#### 3. 重参数化技巧 (`reparameterize`)
- 实现VAE的关键技巧，允许模型进行端到端训练
- 通过添加噪声样本，从潜在分布中采样

#### 4. 解码器 (`decode`)
- 将潜在向量通过全连接层扩展，然后重塑为卷积特征图
- 通过转置卷积层逐步重建图像

#### 5. 前向传播 (`forward`)
- 完整的VAE流程：编码 → 重参数化 → 解码
- 返回重建图像、均值、对数方差和潜在向量

### 核心代码

```python
import torch
import torch.nn as nn

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128, channels=64, img_size=32, in_ch=3):
        super().__init__()
        if img_size % 16 != 0:
            raise ValueError("img_size must be divisible by 16 (because 4 downsamples).")

        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch,        channels,   4, 2, 1), nn.ReLU(),
            nn.Conv2d(channels,     channels*2, 4, 2, 1), nn.BatchNorm2d(channels*2), nn.ReLU(),
            nn.Conv2d(channels*2,   channels*4, 4, 2, 1), nn.BatchNorm2d(channels*4), nn.ReLU(),
            nn.Conv2d(channels*4,   channels*8, 4, 2, 1), nn.BatchNorm2d(channels*8), nn.ReLU(),
        )

        was_training = self.encoder.training
        self.encoder.eval()
        with torch.inference_mode():
            x = torch.zeros(1, in_ch, img_size, img_size)
            h = self.encoder(x)
            self._enc_shape = h.shape[1:]          # (C, H, W)
            self._flat_dim = h.flatten(1).size(1)  # C*H*W
        self.encoder.train(was_training)

        # Latent mappings
        self.fc_mu     = nn.Linear(self._flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, latent_dim) 
        self.fc_decode = nn.Linear(latent_dim, self._flat_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels*8, channels*4, 4, 2, 1), nn.BatchNorm2d(channels*4), nn.ReLU(),
            nn.ConvTranspose2d(channels*4, channels*2, 4, 2, 1), nn.BatchNorm2d(channels*2), nn.ReLU(),
            nn.ConvTranspose2d(channels*2, channels,   4, 2, 1), nn.BatchNorm2d(channels),   nn.ReLU(),
            nn.ConvTranspose2d(channels,   in_ch,      4, 2, 1),
            nn.Sigmoid(),  
        )

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z).view(z.size(0), *self._enc_shape)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
```

## 损失函数

损失函数由两部分组成：
- **recon**（重建误差）
- **KL**（把 `q(z|x)` 拉向标准高斯 `N(0,I)`）

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
    reduction: str = "mean",
) -> dict[str, torch.Tensor]:
    red = "sum" if reduction == "sum" else "mean"

    recon_loss = F.mse_loss(recon, x, reduction=red)

    # KL(q(z|x) || N(0, I)) per-sample: [B]
    kld_per = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1)
    kld = kld_per.sum() if red == "sum" else kld_per.mean()

    loss = recon_loss + beta * kld
    return {"loss": loss, "recon": recon_loss, "kld": kld}

```

## 实验结果

### 重建效果
{% asset_img "U-net ，VAE，DDPM初体验_1_老师好我叫苏同学_来自小红书网页版.jpg" %}

### 随机采样生成效果
{% asset_img "U-net ，VAE，DDPM初体验_2_老师好我叫苏同学_来自小红书网页版.jpg" %}
