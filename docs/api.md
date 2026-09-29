# Python API

```python
from pathlib import Path

from datadiffusion.config import ExperimentConfig
from datadiffusion.data import Preprocessor, load_dataset
from datadiffusion.evaluate import report, to_markdown
from datadiffusion.runs import load_run, new_run_dir
from datadiffusion.sampling import sample
from datadiffusion.train import train

cfg = ExperimentConfig.from_toml("configs/moons.toml")
data = load_dataset("moons", n_samples=cfg.data.n_samples, seed=cfg.data.seed)

run = new_run_dir("runs", cfg.name)
train(cfg, data, run)                     # writes run/model.pt and run/metrics.jsonl

cfg, model, schedule = load_run(run)
pre = Preprocessor(cfg.data.seed).fit(data.train)
z = sample(model, schedule, n=len(data.holdout), dim=data.dim, method="ddpm", seed=0)
synthetic = pre.inverse_transform(z.numpy())  # original units, same columns as data.columns

rep = report({"diffusion": synthetic}, data)  # adds both baselines
print(to_markdown(rep))
```

## Reference

- `load_dataset(name, seed=0, holdout_fraction=0.15, test_fraction=0.15, n_samples=5000, noise=0.05)
  -> TabularData` with `.train`, `.holdout`, `.test` (raw units), `.columns`, `.target`, `.dim`.
- `Preprocessor(seed).fit(X)`, `.transform(X) -> float32`, `.inverse_transform(Z)`.
- `NoiseSchedule(num_timesteps=1000, schedule="cosine", beta_start=None, beta_end=None)`.
- `TabularMLP(input_size, ModelConfig(hidden_size, num_layers, dropout, time_embed_dim))`.
- `sample(model, schedule, n, dim, method="ddim", steps=50, eta=0.0, clip=5.0, seed=0) -> Tensor`
  (normalized space).
- `train(cfg, data, run_dir) -> {"best_epoch", "best_val_loss", "epochs_run"}`.
- `evaluate.report({name: rows}, data, seed=0) -> dict`; `evaluate.to_markdown(report) -> str`;
  individual metrics `ks_statistic`, `wasserstein_1d`, `correlation_gap`, `c2st_auc`, `tstr`, `dcr`;
  baselines `independent_marginals`, `gaussian_copula`.
