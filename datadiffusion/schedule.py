"""DDPM noise schedule (Ho et al., 2020) with linear and cosine betas (Nichol & Dhariwal, 2021).

Notation: x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, t = 0 .. T-1 (t = 0 is the least noisy).
"""

from __future__ import annotations

import math

import numpy as np
import torch

LINEAR_DEFAULTS = (1e-4, 0.02)


class NoiseSchedule:
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule: str = "cosine",
        beta_start: float | None = None,
        beta_end: float | None = None,
    ):
        if num_timesteps < 1:
            raise ValueError("num_timesteps must be >= 1")
        self.num_timesteps = num_timesteps
        self.schedule = schedule
        if schedule == "linear":
            b0, b1 = (beta_start or LINEAR_DEFAULTS[0], beta_end or LINEAR_DEFAULTS[1])
            if not 0 < b0 < b1 < 1:
                raise ValueError(f"linear schedule needs 0 < beta_start < beta_end < 1, got {b0}, {b1}")
            betas = torch.linspace(b0, b1, num_timesteps, dtype=torch.float64)
        elif schedule == "cosine":
            if beta_start is not None or beta_end is not None:
                raise ValueError("the cosine schedule has no beta_start/beta_end; drop them or use 'linear'")
            s = 0.008
            t = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float64)
            abar = torch.cos((t / num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
            abar = abar / abar[0]
            betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
        else:
            raise ValueError(f"unknown schedule {schedule!r}; use 'linear' or 'cosine'")
        self.beta_start, self.beta_end = beta_start, beta_end

        alphas = 1 - betas
        abar = torch.cumprod(alphas, 0)
        abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])
        f32 = torch.float32
        self.betas = betas.to(f32)
        self.alphas_cumprod = abar.to(f32)
        self.alphas_cumprod_prev = abar_prev.to(f32)
        self.sqrt_abar = abar.sqrt().to(f32)
        self.sqrt_one_minus_abar = (1 - abar).sqrt().to(f32)
        self.posterior_coef_x0 = (betas * abar_prev.sqrt() / (1 - abar)).to(f32)
        self.posterior_coef_xt = ((1 - abar_prev) * alphas.sqrt() / (1 - abar)).to(f32)
        self.posterior_var = (betas * (1 - abar_prev) / (1 - abar)).clamp(min=1e-20).to(f32)

    def __len__(self) -> int:
        return self.num_timesteps

    def _at(self, table: torch.Tensor, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return table.to(like.device)[t].view(-1, *([1] * (like.dim() - 1)))

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._at(self.sqrt_abar, t, x0) * x0 + self._at(self.sqrt_one_minus_abar, t, x0) * noise

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (xt - self._at(self.sqrt_one_minus_abar, t, xt) * eps) / self._at(self.sqrt_abar, t, xt)

    def ddpm_step(
        self,
        eps: torch.Tensor,
        t: int,
        xt: torch.Tensor,
        generator: torch.Generator | None = None,
        clip: float | None = None,
    ) -> torch.Tensor:
        """Ancestral sampling step x_t -> x_{t-1}; no noise is added at t = 0."""
        tt = torch.full((xt.shape[0],), t, dtype=torch.long, device=xt.device)
        x0 = self.predict_x0(xt, tt, eps)
        if clip is not None:
            x0 = x0.clamp(-clip, clip)
        mean = self._at(self.posterior_coef_x0, tt, xt) * x0 + self._at(self.posterior_coef_xt, tt, xt) * xt
        if t == 0:
            return mean
        noise = torch.randn(xt.shape, generator=generator, device=xt.device)
        return mean + self._at(self.posterior_var, tt, xt).sqrt() * noise

    def ddim_timesteps(self, steps: int) -> np.ndarray:
        """`steps` distinct timesteps from T-1 down to 0 (inclusive), evenly spaced."""
        if not 1 <= steps <= self.num_timesteps:
            raise ValueError(f"DDIM steps must be in [1, {self.num_timesteps}], got {steps}")
        return np.unique(np.linspace(0, self.num_timesteps - 1, steps).round().astype(np.int64))[::-1].copy()
