from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedules import cosine_beta_schedule


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    """从 1-D buffer `a` 里按 `t` 取值，并 reshape 成可 broadcast 的形状。

    注意事项：
    - 这里假设 `a.shape == (timesteps,)` 且 `t` 在同一 device 上。
    - reshape 成 (B, 1, 1, 1, ...) 可以避免手写很多 expand / view 代码。
    """

    b = t.shape[0]
    out = a.gather(0, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 1000
    beta_schedule: str = "cosine"  # "cosine" or "linear"


class GaussianDiffusion(nn.Module):
    """DDPM training + sampling utilities.

    This module assumes the model predicts epsilon (noise) by default.
    """

    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.beta_schedule != "cosine":
            raise ValueError("Only cosine schedule is implemented in teaching version")

        betas = cosine_beta_schedule(cfg.timesteps).float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0)
        )

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))

        self.register_buffer(
            "posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        ) * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        ) * eps

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        clip_denoised: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = model(x_t, t)
        x0_pred = self.predict_x0_from_eps(x_t, t, eps)
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        mean = _extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred + _extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        var = _extract(self.posterior_variance, t, x_t.shape)
        log_var = _extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def training_losses(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)
        eps_pred = model(x_t, t)
        loss = F.mse_loss(eps_pred, noise, reduction="mean")
        return {"loss": loss}

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        mean, _var, log_var = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised)
        noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
        nonzero_mask = (t != 0).float().reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        *,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        b = shape[0]
        img = torch.randn(shape, device=device, generator=generator)
        for i in reversed(range(self.cfg.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, generator=generator, clip_denoised=clip_denoised)
        return img

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        *,
        device: torch.device,
        steps: int = 50,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """DDIM sampler (eta=0 -> deterministic).

        steps: number of sampling steps (<= timesteps)
        """

        if steps < 1 or steps > self.cfg.timesteps:
            raise ValueError("steps must be in [1, timesteps]")

        b = shape[0]
        img = torch.randn(shape, device=device, generator=generator)

        # Uniform timestep stride for simplicity.
        t_seq = torch.linspace(self.cfg.timesteps - 1, 0, steps, device=device).long()

        for idx in range(steps):
            t = torch.full((b,), int(t_seq[idx].item()), device=device, dtype=torch.long)
            eps = model(img, t)
            x0 = self.predict_x0_from_eps(img, t, eps)
            if clip_denoised:
                x0 = x0.clamp(-1.0, 1.0)

            if idx == steps - 1:
                img = x0
                break

            t_next = torch.full((b,), int(t_seq[idx + 1].item()), device=device, dtype=torch.long)
            a_t = _extract(self.alphas_cumprod, t, img.shape)
            a_next = _extract(self.alphas_cumprod, t_next, img.shape)

            # DDIM update
            sigma = (
                eta
                * torch.sqrt((1 - a_next) / (1 - a_t))
                * torch.sqrt(1 - a_t / a_next)
            )
            noise = torch.randn(img.shape, device=device, dtype=img.dtype, generator=generator)

            dir_xt = torch.sqrt(1.0 - a_next - sigma ** 2) * eps
            img = torch.sqrt(a_next) * x0 + dir_xt + sigma * noise

        return img
