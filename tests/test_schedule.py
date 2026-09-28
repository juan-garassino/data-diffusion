from __future__ import annotations

import pytest
import torch

from datadiffusion.schedule import NoiseSchedule


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_abar_is_monotone_in_unit_interval(kind):
    s = NoiseSchedule(200, kind)
    abar = s.alphas_cumprod
    assert (abar[1:] < abar[:-1]).all() and abar.max() < 1 and abar.min() > 0


def test_cosine_endpoints():
    s = NoiseSchedule(1000, "cosine")
    assert s.alphas_cumprod[0] > 0.999 and s.alphas_cumprod[-1] < 1e-3


def test_linear_betas_endpoints():
    s = NoiseSchedule(100, "linear", beta_start=1e-4, beta_end=0.02)
    assert s.betas[0].item() == pytest.approx(1e-4) and s.betas[-1].item() == pytest.approx(0.02)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(schedule="cosine", beta_start=1e-4),  # silently ignored before; now an error
        dict(schedule="linear", beta_start=0.02, beta_end=0.01),
        dict(schedule="quadratic"),
        dict(num_timesteps=0),
    ],
)
def test_invalid_schedules_raise(kwargs):
    with pytest.raises(ValueError):
        NoiseSchedule(**kwargs)


def test_add_noise_statistics_and_inverse():
    torch.manual_seed(0)
    s = NoiseSchedule(100, "linear")
    t = torch.full((20000,), 60)
    x0 = torch.zeros(20000, 1)
    noise = torch.randn(20000, 1)
    xt = s.add_noise(x0, noise, t)
    assert xt.std().item() == pytest.approx(s.sqrt_one_minus_abar[60].item(), rel=0.03)
    x0 = torch.randn(20000, 1)
    xt = s.add_noise(x0, noise, t)
    assert torch.allclose(s.predict_x0(xt, t, noise), x0, atol=1e-4)


def test_ddpm_step_at_t0_is_deterministic():
    s = NoiseSchedule(10, "linear")
    x, eps = torch.randn(4, 3), torch.randn(4, 3)
    assert torch.equal(s.ddpm_step(eps, 0, x), s.ddpm_step(eps, 0, x))


def test_ddim_timesteps_span_the_whole_range():
    s = NoiseSchedule(1000, "cosine")
    ts = s.ddim_timesteps(50)
    assert ts[0] == 999 and ts[-1] == 0 and len(ts) == 50 and (ts[:-1] > ts[1:]).all()
    assert list(s.ddim_timesteps(1000)) == list(range(999, -1, -1))
    with pytest.raises(ValueError):
        s.ddim_timesteps(1001)
