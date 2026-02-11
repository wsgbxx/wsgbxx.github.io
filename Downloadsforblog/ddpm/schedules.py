from __future__ import annotations

import math

import torch


def cosine_beta_schedule(timesteps: int, *, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from https://arxiv.org/abs/2102.09672.

    Returns betas in float64 for numerical stability; cast to model dtype later.
    """

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-8, 0.999)


def linear_beta_schedule(
    timesteps: int, *, beta_start: float = 1e-4, beta_end: float = 2e-2
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
