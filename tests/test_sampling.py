from __future__ import annotations

import pytest
import torch

from datadiffusion.model import ModelConfig, TabularMLP
from datadiffusion.sampling import sample
from datadiffusion.schedule import NoiseSchedule

MU = torch.tensor([1.5, -0.5, 2.0])


class _Oracle(torch.nn.Module):
    """Exact eps for data = the single point MU: eps = (x_t - sqrt(abar) MU) / sqrt(1 - abar)."""

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, t):
        t = t.long()
        a = self.schedule.sqrt_abar[t][:, None]
        b = self.schedule.sqrt_one_minus_abar[t][:, None]
        return (x - a * MU) / b


@pytest.mark.parametrize("method,steps,eta", [("ddim", 25, 0.0), ("ddim", 25, 1.0), ("ddpm", None, 0.0)])
def test_samplers_recover_a_point_mass(method, steps, eta):
    s = NoiseSchedule(200, "cosine")
    x = sample(_Oracle(s), s, n=256, dim=3, method=method, steps=steps or 50, eta=eta, clip=None, seed=0)
    assert torch.allclose(x.mean(0), MU, atol=0.05) and x.std(0).max() < 0.1


def test_ddim_eta0_is_deterministic_given_seed():
    s = NoiseSchedule(50, "cosine")
    model = TabularMLP(3, ModelConfig(hidden_size=16, num_layers=1, time_embed_dim=8))
    a = sample(model, s, 10, 3, steps=10, seed=1)
    b = sample(model, s, 10, 3, steps=10, seed=1)
    c = sample(model, s, 10, 3, steps=10, seed=2)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_batching_does_not_change_count():
    s = NoiseSchedule(10, "cosine")
    model = TabularMLP(2, ModelConfig(hidden_size=8, num_layers=1, time_embed_dim=8))
    assert sample(model, s, 25, 2, steps=5, batch_size=7).shape == (25, 2)


def test_unknown_sampler():
    s = NoiseSchedule(10)
    with pytest.raises(ValueError):
        sample(_Oracle(s), s, 1, 3, method="euler")


class _Wild(torch.nn.Module):
    """A badly wrong noise predictor (constant -1000), standing in for an extrapolating net."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, t):
        return torch.full_like(x, -1000.0)


def test_x0_clipping_bounds_samples():
    s = NoiseSchedule(100, "cosine")
    wild = sample(_Wild(), s, 64, 3, steps=20, clip=None, seed=0)
    tame = sample(_Wild(), s, 64, 3, steps=20, clip=5.0, seed=0)
    assert wild.abs().max() > 5 and tame.abs().max() <= 5.0 + 1e-5
