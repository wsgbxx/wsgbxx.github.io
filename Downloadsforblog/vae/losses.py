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
    """VAE loss = recon + beta * KL.

    recon_type:
    - "mse": good default for teaching
    - "bce": if you interpret pixel as Bernoulli; requires recon in (0,1)

    reduction:
    - "mean": average per batch
    - "sum": sum over all dims
    """

    if recon_type == "mse":
        recon_loss = F.mse_loss(recon, x, reduction=reduction)
    elif recon_type == "bce":
        recon_loss = F.binary_cross_entropy(recon, x, reduction=reduction)
    else:
        raise ValueError(f"Unknown recon_type={recon_type!r}")

    # KL(q(z|x) || N(0, I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # Reduce over latent dim; then reduce over batch.
    kld_per = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    if reduction == "mean":
        kld = kld_per.mean()
    elif reduction == "sum":
        kld = kld_per.sum()
    else:
        raise ValueError(f"Unknown reduction={reduction!r}")

    loss = recon_loss + beta * kld
    return {"loss": loss, "recon": recon_loss, "kld": kld}
