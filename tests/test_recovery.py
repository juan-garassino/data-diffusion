"""End to end: a small model trained for seconds must learn the moons' dependence structure."""

from __future__ import annotations

import time

from datadiffusion import evaluate as ev
from datadiffusion.config import DataConfig, DiffusionConfig, ExperimentConfig, TrainConfig
from datadiffusion.data import Preprocessor, load_dataset
from datadiffusion.model import ModelConfig
from datadiffusion.runs import new_run_dir
from datadiffusion.sampling import sample
from datadiffusion.train import train


def test_moons_are_recovered(tmp_path):
    cfg = ExperimentConfig(
        name="moons",
        data=DataConfig(dataset="moons", n_samples=3000),
        model=ModelConfig(hidden_size=64, num_layers=2, time_embed_dim=32),
        diffusion=DiffusionConfig(num_timesteps=100),
        train=TrainConfig(epochs=40, batch_size=128, lr=3e-3, warmup_steps=20, ema_decay=0.99, patience=100),
    )
    data = load_dataset("moons", n_samples=3000)
    t0 = time.time()
    run = new_run_dir(tmp_path, "moons")
    train(cfg, data, run)
    from datadiffusion.runs import load_run

    _, model, schedule = load_run(run)
    pre = Preprocessor().fit(data.train)
    synth = pre.inverse_transform(
        sample(model, schedule, len(data.holdout), 2, method="ddpm", seed=0).numpy()
    )
    auc_diffusion = ev.c2st_auc(data.holdout, synth)
    auc_marginals = ev.c2st_auc(data.holdout, ev.independent_marginals(data.train, len(data.holdout)))
    elapsed = time.time() - t0
    print(f"moons: C2ST diffusion {auc_diffusion:.3f} vs marginals {auc_marginals:.3f} ({elapsed:.0f}s)")
    assert auc_diffusion < 0.8
    assert auc_diffusion < auc_marginals - 0.05
