# Architecture

```
load_dataset ──► split train / holdout / test ──► Preprocessor.fit(train)   (quantile → N(0,1))
                                                        │
                        train: ε-prediction on train ◄──┘
                                   │
                        sample: DDPM / DDIM  ──► inverse transform ──► synthetic rows
                                                                          │
                     evaluate: vs holdout / test, next to baselines ◄─────┘
```

## Forward process

`x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε`, with `ε ~ N(0, I)` and `t = 0 … T−1`
(`ᾱ_t = ∏ (1 − β_s)`).

- **Cosine schedule** (default, Nichol & Dhariwal 2021): `ᾱ(t) ∝ cos²((t/T + s)/(1 + s) · π/2)`,
  `s = 0.008`, betas clamped at 0.999. It has no `beta_start`/`beta_end`; passing them is an error.
- **Linear schedule**: `β` from `beta_start` (1e-4) to `beta_end` (0.02).

## Training

Uniform `t ~ U{0 … T−1}`; loss `‖ε_θ(x_t, t) − ε‖²`. AdamW with linear warmup then cosine decay,
sized to the real number of optimizer steps (one per batch); gradient clipping; EMA of the weights.
Validation uses one fixed noise draw so epoch losses are comparable; the best EMA weights are kept
(early stopping on patience).

The model is a residual MLP: `h = W_in x + MLP(sinusoidal(t))`, then `num_layers` pre-norm residual
blocks, then a projection back to the row size.

## Sampling

- **DDPM**: `x_{t−1} = μ̃(x_t, x̂₀) + σ_t z` with the posterior mean and variance of Ho et al. (2020); no
  noise at `t = 0`.
- **DDIM** (Song et al., 2021) on `steps` timesteps evenly spaced from `T−1` to `0`:
  `x_prev = √ᾱ_prev · x̂₀ + √(1 − ᾱ_prev − σ²) · ε_θ + σ z`, with
  `σ = η √((1 − ᾱ_prev)/(1 − ᾱ_t)) √(1 − ᾱ_t/ᾱ_prev)`. `η = 0` is deterministic.
- **x₀ clipping** (default ±5): `x̂₀ = (x_t − √(1−ᾱ_t) ε_θ)/√ᾱ_t` divides by `√ᾱ_{T−1} ≈ 5·10⁻⁴`
  under the cosine schedule, so early prediction errors are amplified thousands of times without it.
  Normalized data never leaves ±5.2 (QuantileTransformer's cap), so clipping at 5 is safe.

## Modules

| Module | Role |
|---|---|
| `data.py` | datasets (California Housing, moons, circles), seeded splits, `Preprocessor` |
| `schedule.py` | `NoiseSchedule`: betas, `add_noise`, `predict_x0`, `ddpm_step`, `ddim_timesteps` |
| `model.py` | `TabularMLP`, `ModelConfig`, sinusoidal time embedding |
| `sampling.py` | `sample(...)` — DDPM or DDIM, seeded, batched, x₀ clipping |
| `train.py` | training loop, EMA, warmup-cosine schedule, early stopping |
| `evaluate.py` | metrics, baselines, `report`, `to_markdown` |
| `runs.py` | run directories, JSONL metrics, checkpoints, CSV tables |
| `config.py` | `ExperimentConfig` (TOML) |
| `plots.py` | report figures (optional `[plots]` extra) |
| `cli.py` | `train`, `sample`, `evaluate`, `baselines` |
