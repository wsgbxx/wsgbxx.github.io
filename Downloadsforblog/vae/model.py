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
