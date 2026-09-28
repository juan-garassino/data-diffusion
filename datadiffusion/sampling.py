"""Reverse diffusion: ancestral DDPM and DDIM (Song et al., 2021), seeded and batched."""

from __future__ import annotations

import logging

import torch

from datadiffusion.schedule import NoiseSchedule

logger = logging.getLogger(__name__)


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    schedule: NoiseSchedule,
    n: int,
    dim: int,
    method: str = "ddim",
    steps: int = 50,
    eta: float = 0.0,
    clip: float | None = None,
    seed: int = 0,
    batch_size: int = 8192,
) -> torch.Tensor:
    """Draw n rows in the model's (normalized) space.

    method="ddpm" runs all T ancestral steps; method="ddim" runs `steps` steps, deterministic
    for eta=0 and equivalent to DDPM's variance for eta=1. `clip` bounds the predicted x0.
    """
    if method not in ("ddpm", "ddim"):
        raise ValueError(f"unknown sampler {method!r}; use 'ddpm' or 'ddim'")
    model.eval()
    device = next(model.parameters()).device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for start in range(0, n, batch_size):
        m = min(batch_size, n - start)
        x = torch.randn(m, dim, generator=gen).to(device)
        if method == "ddpm":
            for t in range(schedule.num_timesteps - 1, -1, -1):
                eps = model(x, torch.full((m,), t, device=device))
                x = schedule.ddpm_step(eps, t, x, generator=gen, clip=clip)
        else:
            x = _ddim(model, schedule, x, steps, eta, clip, gen)
        out.append(x.cpu())
    return torch.cat(out)


def _ddim(model, schedule, x, steps, eta, clip, gen):
    abar = schedule.alphas_cumprod
    ts = schedule.ddim_timesteps(steps)
    for i, t in enumerate(ts):
        a_t = abar[t]
        a_prev = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        eps = model(x, torch.full((x.shape[0],), int(t), device=x.device))
        x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        if clip is not None:
            x0 = x0.clamp(-clip, clip)
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()
        sigma = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).clamp(min=0).sqrt()
        x = a_prev.sqrt() * x0 + (1 - a_prev - sigma**2).clamp(min=0).sqrt() * eps
        if sigma > 0:
            x = x + sigma * torch.randn(x.shape, generator=gen).to(x.device)
    return x
