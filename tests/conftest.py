"""Shared fixtures. Tests use tiny tensors, so one torch thread avoids OpenMP spin-wait overhead."""

from __future__ import annotations

import torch

torch.set_num_threads(1)
