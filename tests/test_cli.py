from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from datadiffusion.cli import build_parser, main
from datadiffusion.runs import load_table

CONFIG = """
name = "moons-test"
[data]
dataset = "moons"
n_samples = 600
[model]
hidden_size = 32
num_layers = 2
time_embed_dim = 16
[diffusion]
num_timesteps = 50
[train]
epochs = 2
batch_size = 64
warmup_steps = 5
"""


def test_parser():
    args = build_parser().parse_args(["sample", "runs/x", "--method", "ddpm", "-n", "10"])
    assert args.method == "ddpm" and args.n == 10 and args.steps == 50


def test_train_sample_evaluate_roundtrip(tmp_path, capsys):
    cfg = tmp_path / "c.toml"
    cfg.write_text(CONFIG)
    assert main(["train", "--config", str(cfg), "--out", str(tmp_path / "runs"), "--threads", "1"]) == 0
    run = capsys.readouterr().out.strip().splitlines()[-1]
    assert main(["sample", run, "-n", "40", "--steps", "10"]) == 0
    rows = load_table(Path(run) / "samples.csv")
    assert rows.shape == (40, 2) and np.isfinite(rows).all()
    assert main(["evaluate", run, "--steps", "10"]) == 0
    rep = json.loads((Path(run) / "report.json").read_text())
    assert {"diffusion", "independent_marginals", "gaussian_copula"} <= set(rep["methods"])
    assert (Path(run) / "scatter.png").exists()
