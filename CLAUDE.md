# dataDiffusion

Tabular DDPM/DDIM with honest evaluation. v3 is a rewrite (tags v1.0/v2.0 are the old code).
Flat package `datadiffusion/` (legacy layout, kept). uv + hatchling; torch pinned to 2.2.2 and
numpy < 2 (Intel-mac wheels), CPU torch on linux via the pytorch-cpu index.

## Commands

- `make install` · `make test` (`make test-ci` skips `slow`) · `make lint`
- `make train-moons` / `make train-california` · `make evaluate RUN=runs/<name>/<ts>` · `make baselines`

## Module map

See `docs/architecture.md` (table). Key invariants:
- Only `data.train` is used for fitting (model and `Preprocessor`); fidelity/C2ST/DCR use `holdout`,
  TSTR uses `test`. The target is the last column and is generated jointly.
- Sampling clips predicted x0 to ±5 by default (cosine schedule makes √ᾱ_T ≈ 5e-4).
- Every score is reported next to the independent-marginals and Gaussian-copula baselines; there is
  no composite score. Report numbers exactly as `report.json` says.
- Checkpoints hold only plain config dicts + tensors (`weights_only=True` loads).

## Tests

`tests/test_sampling.py` has an oracle model (exact ε for a point mass) that both samplers must
recover; `tests/test_recovery.py` trains on moons and must beat the marginals baseline on C2ST.
