from __future__ import annotations

import json

import pytest

from datadiffusion.config import DataConfig, DiffusionConfig, ExperimentConfig, TrainConfig
from datadiffusion.data import load_dataset
from datadiffusion.model import ModelConfig
from datadiffusion.runs import load_run, new_run_dir
from datadiffusion.train import train, warmup_cosine


def _cfg(epochs=3, seed=0):
    return ExperimentConfig(
        name="t",
        data=DataConfig(dataset="moons", n_samples=400),
        model=ModelConfig(hidden_size=32, num_layers=2, time_embed_dim=16),
        diffusion=DiffusionConfig(num_timesteps=50),
        train=TrainConfig(epochs=epochs, batch_size=64, warmup_steps=5, seed=seed, patience=100),
    )


def test_warmup_then_cosine_to_zero():
    assert warmup_cosine(0, 10, 100) == pytest.approx(0.1)
    assert warmup_cosine(9, 10, 100) == pytest.approx(1.0)
    assert warmup_cosine(99, 10, 100) == pytest.approx(0.0, abs=1e-3)


def test_train_writes_run_and_lr_anneals(tmp_path):
    cfg = _cfg()
    run = new_run_dir(tmp_path, cfg.name)
    summary = train(cfg, load_dataset("moons", n_samples=400), run)
    lines = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [line["epoch"] for line in lines] == [1, 2, 3]
    assert lines[-1]["lr"] < 0.01 * cfg.train.lr  # the schedule actually reaches the end (v2's never did)
    assert 1 <= summary["best_epoch"] <= 3
    loaded_cfg, model, schedule = load_run(run)
    assert loaded_cfg == cfg and model.input_size == 2 and len(schedule) == 50


def test_same_seed_same_losses(tmp_path):
    data = load_dataset("moons", n_samples=400)
    a = train(_cfg(epochs=2), data, new_run_dir(tmp_path / "a", "t"))
    b = train(_cfg(epochs=2), data, new_run_dir(tmp_path / "b", "t"))
    assert a == b
