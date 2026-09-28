"""Residual MLP that predicts the noise eps(x_t, t) for tabular rows."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 256
    num_layers: int = 4
    dropout: float = 0.0
    time_embed_dim: int = 128

    def to_dict(self) -> dict:
        return asdict(self)


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style embedding of (float) timesteps, built on t's device."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(1, half - 1)
    )
    angles = t.float()[:, None] * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(size), nn.Linear(size, size), nn.GELU(), nn.Dropout(dropout), nn.Linear(size, size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularMLP(nn.Module):
    def __init__(self, input_size: int, config: ModelConfig | None = None):
        super().__init__()
        self.input_size = input_size
        self.config = cfg = config or ModelConfig()
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.hidden_size),
            nn.GELU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )
        self.input_proj = nn.Linear(input_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(
            ResidualBlock(cfg.hidden_size, cfg.dropout) for _ in range(cfg.num_layers)
        )
        self.out = nn.Sequential(nn.LayerNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, input_size))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.time_mlp(sinusoidal_embedding(t, self.config.time_embed_dim))
        for block in self.blocks:
            h = block(h)
        return self.out(h)
