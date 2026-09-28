"""Run directories (config, metrics, checkpoint, samples, reports) — local files instead of MLflow."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from datadiffusion.config import ExperimentConfig
from datadiffusion.model import ModelConfig, TabularMLP
from datadiffusion.schedule import NoiseSchedule


def new_run_dir(root: str | Path, name: str) -> Path:
    run = Path(root) / name / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    return run


class MetricsWriter:
    """Append-only JSONL, one object per line, flushed per write."""

    def __init__(self, path: Path):
        self._file = path.open("a")

    def log(self, **fields) -> None:
        self._file.write(json.dumps({"ts": round(time.time(), 3), **fields}) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def save_checkpoint(run: Path, cfg: ExperimentConfig, model: TabularMLP, **extra) -> Path:
    path = run / "model.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(
        {"config": cfg.to_dict(), "input_size": model.input_size, "state_dict": model.state_dict(), **extra},
        tmp,
    )
    tmp.replace(path)
    return path


def load_run(run: str | Path) -> tuple[ExperimentConfig, TabularMLP, NoiseSchedule]:
    payload = torch.load(Path(run) / "model.pt", map_location="cpu", weights_only=True)
    cfg = ExperimentConfig.from_dict(payload["config"])
    model = TabularMLP(payload["input_size"], ModelConfig(**payload["config"]["model"]))
    model.load_state_dict(payload["state_dict"])
    d = cfg.diffusion
    return cfg, model.eval(), NoiseSchedule(d.num_timesteps, d.schedule, d.beta_start, d.beta_end)


def save_table(path: Path, rows: np.ndarray, columns: list[str]) -> Path:
    np.savetxt(path, rows, delimiter=",", header=",".join(columns), comments="", fmt="%.6g")
    return path


def load_table(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
