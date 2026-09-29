# dataDiffusion

Synthetic tabular data with a denoising diffusion model (DDPM / DDIM), evaluated **honestly**:
every score is measured on real rows the model never trained on and printed next to two simple
baselines, so you can see what diffusion actually buys you.

- Residual-MLP ε-predictor, cosine or linear noise schedule, EMA weights, early stopping.
- DDPM (ancestral) and DDIM (fast, η-controlled) samplers with x₀ clipping.
- Features and the target are generated **jointly**, so synthetic rows carry their own labels.
- Metrics: per-column KS and Wasserstein, correlation gap, C2ST (can a classifier tell real from
  synthetic?), TSTR (train on synthetic, test on real), DCR (is it copying training rows?).
- Baselines: independent marginals and a Gaussian copula.
- CPU-friendly, small dependency set (torch, numpy, scikit-learn); runs are local directories.

## Install

```bash
uv sync --extra dev        # or: pip install -e ".[plots]"
```

Python 3.11–3.12, torch 2.2.2 (also runs on Intel Macs).

## Use

```bash
# fast sanity check on the 2-D moons (figures show real vs synthetic point clouds)
uv run datadiffusion train --config configs/moons.toml          # prints runs/moons/<timestamp>
uv run datadiffusion evaluate runs/moons/<timestamp> --method ddpm

# California Housing (8 features + target)
uv run datadiffusion train --config configs/california.toml
uv run datadiffusion evaluate runs/california/<timestamp> --method ddpm
uv run datadiffusion sample runs/california/<timestamp> -n 10000 -o synthetic.csv

uv run datadiffusion baselines --dataset california             # baselines alone
```

`evaluate` writes `report.json`, `report.md`, `marginals.png` and `correlations.png` (plus
`scatter.png` for 2-D data) into the run directory. Every run directory also holds `config.json`,
`metrics.jsonl` (per-epoch losses and learning rate) and `model.pt` (config + weights; loads with
`torch.load(weights_only=True)`).

## How it is evaluated

The data is split once (seeded) into **train 70% / holdout 15% / test 15%**. The model and the
quantile preprocessor only ever see *train*.

| Metric | Compared against | Reads as |
|---|---|---|
| KS statistic, Wasserstein-1 / IQR (per column, averaged) | holdout | marginal fidelity; 0 is identical |
| Correlation gap: mean \|Δ corr\| over column pairs | holdout | dependence structure; 0 is identical |
| C2ST AUC: gradient boosting telling real from synthetic | holdout | 0.5 = indistinguishable, 1 = trivially detectable |
| TSTR R²: model trained on synthetic (X, y), scored on real test | test | utility; compare with TRTR (trained on real train) |
| DCR ratio: median distance to nearest *training* row, synthetic ÷ holdout | train | ≈1 healthy; ≪1 means it is copying training rows |

KS p-values are deliberately not reported: with thousands of rows every difference is "significant".
The two baselines bracket the problem: **independent marginals** get every column right and all
dependence wrong; the **Gaussian copula** adds linear dependence. A generator is only interesting
where it beats the copula.

## Limitations

- Numeric columns only; categorical support is not implemented.
- The quantile preprocessor maps back through the training quantiles, so synthetic values never
  leave each column's training range.
- DCR is a heuristic privacy check, not a differential-privacy guarantee.

## Project history

v1 and v2 (tags `v1.0`, `v2.0`) reported a composite score of ~0.92 that turned out to be inflated
(a mis-normalized correlation term) and a "TSTR" that labelled synthetic rows with a model trained on
real data. v3 replaces both with the protocol above; see [CHANGELOG.md](CHANGELOG.md).

## Docs

[docs/architecture.md](docs/architecture.md) — the math and the module map ·
[docs/api.md](docs/api.md) — Python API.

## License

MIT.
