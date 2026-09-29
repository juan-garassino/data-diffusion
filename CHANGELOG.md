# Changelog

## [3.0.0] — unreleased

A rewrite focused on correctness and honest evaluation. Results on California Housing: TSTR R² 0.798
(TRTR 0.854, Gaussian copula 0.560), C2ST AUC 0.581, DCR ratio 1.01 — see README.

### Breaking changes
- New package layout and CLI: `datadiffusion train | sample | evaluate | baselines` over local run
  directories. `main.py`, the self-improvement loop, MLflow tracking, the score-based model/scheduler,
  the transformer model, the animation module and the quadratic schedule are removed.
- Configs are TOML (`[data] [model] [diffusion] [train]`); a missing config file is an error.
- The "composite score" is gone.

### Fixed
- **Composite score was inflated**: the correlation term was divided by √2·n, so a failing
  correlation gap still contributed ~0.92.
- **"TSTR" was not TSTR**: synthetic rows were labelled by a model trained on real data and scored
  on rows the generator had trained on. The target is now generated jointly and TSTR is scored on a
  held-out real test split.
- **LR schedule never annealed**: OneCycleLR was sized for 4× the real optimizer steps.
- **Biased objective**: timesteps were drawn ∝ 1/(t+1) without loss reweighting.
- **DDIM grid** started at (N−1)·ratio instead of T−1 and collapsed to t = 0 when steps > T; η was
  hard-coded to 0.
- **Samples could explode**: predicted x₀ is now clipped (default ±5 in normalized space).
- No seeds anywhere; `--num_samples` ignored; `beta_start`/`beta_end` silently ignored under the
  cosine schedule; `moons`/`line`/`circle` listed but broken; scheduler tensors never moved to the
  model's device.
- The lock pinned torch 2.14, which has no Intel-mac wheel.

### Added
- Seeded train / holdout / test splits; preprocessor fit on train only.
- Evaluation against held-out data: KS, Wasserstein/IQR, correlation gap, C2ST AUC, TSTR vs TRTR,
  DCR — always next to independent-marginals and Gaussian-copula baselines.
- Working `moons` and `circles` toy datasets and an end-to-end recovery test.
- Figures: marginal histograms, correlation heatmaps, 2-D scatter.

## [2.0] and [1.0]

Earlier versions; see the git tags `v2.0` and `v1.0`.
