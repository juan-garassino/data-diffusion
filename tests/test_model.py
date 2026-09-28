from __future__ import annotations

import torch

from datadiffusion.model import ModelConfig, TabularMLP, sinusoidal_embedding


def test_embedding_shape_and_distinct_timesteps():
    e = sinusoidal_embedding(torch.tensor([0, 1, 500]), 16)
    assert e.shape == (3, 16) and not torch.allclose(e[0], e[1])


def test_mlp_shape_and_time_dependence():
    torch.manual_seed(0)
    net = TabularMLP(9, ModelConfig(hidden_size=32, num_layers=2, time_embed_dim=16)).eval()
    x = torch.randn(4, 9)
    a, b = net(x, torch.zeros(4)), net(x, torch.full((4,), 900.0))
    assert a.shape == (4, 9) and not torch.allclose(a, b)
