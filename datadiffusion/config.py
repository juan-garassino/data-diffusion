"""Experiment configuration: nested dataclasses, loadable from TOML (stdlib tomllib)."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from datadiffusion.model import ModelConfig


@dataclass
class DataConfig:
    dataset: str = "california"
    seed: int = 0
    holdout_fraction: float = 0.15
    test_fraction: float = 0.15
    n_samples: int = 5000  # toy datasets only
    noise: float = 0.05  # toy datasets only


@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000
    schedule: str = "cosine"
    beta_start: float | None = None  # linear schedule only
    beta_end: float | None = None


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-6
    warmup_steps: int = 500
    ema_decay: float = 0.999
    patience: int = 25
    val_fraction: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        sections = {
            "data": DataConfig,
            "model": ModelConfig,
            "diffusion": DiffusionConfig,
            "train": TrainConfig,
        }
        unknown = set(d) - {"name", *sections}
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        kwargs = {"name": d.get("name", "experiment")}
        for key, section_cls in sections.items():
            values = d.get(key, {})
            allowed = {f.name for f in fields(section_cls)}
            bad = set(values) - allowed
            if bad:
                raise ValueError(f"unknown keys in [{key}]: {sorted(bad)}")
            kwargs[key] = section_cls(**values)
        return cls(**kwargs)

    @classmethod
    def from_toml(cls, path: str | Path) -> ExperimentConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        with path.open("rb") as f:
            return cls.from_dict(tomllib.load(f))
