from __future__ import annotations

import torch

from ddpm import DiffusionConfig, GaussianDiffusion, UNetConfig, DiffusionUNet
from unet import SimpleUNetSR
from vae.model import ConvVAE


def test_vae_forward_shapes() -> None:
    m = ConvVAE(image_size=64, in_channels=3, latent_dim=128, out_activation="sigmoid")
    x = torch.randn(2, 3, 64, 64)
    out = m(x)
    assert out.recon.shape == x.shape
    assert out.mu.shape == (2, 128)
    assert out.logvar.shape == (2, 128)


def test_unet_sr_shapes() -> None:
    m = SimpleUNetSR()
    x = torch.randn(2, 3, 96, 96)
    y = m(x)
    assert y.shape == x.shape


def test_ddpm_forward_and_sample_shapes() -> None:
    device = torch.device("cpu")
    unet = DiffusionUNet(UNetConfig(image_size=32)).to(device)
    diffusion = GaussianDiffusion(DiffusionConfig(timesteps=10))

    x0 = torch.randn(2, 3, 32, 32, device=device)
    t = torch.randint(0, 10, (2,), device=device)

    losses = diffusion.training_losses(unet, x0, t)
    assert "loss" in losses
    assert losses["loss"].ndim == 0

    samples = diffusion.ddim_sample_loop(unet, (2, 3, 32, 32), device=device, steps=5, eta=0.0)
    assert samples.shape == (2, 3, 32, 32)
