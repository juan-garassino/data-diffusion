"""Evaluation of synthetic tables against held-out real data, always next to baselines.

Every score compares against rows the generator never trained on:
- fidelity: per-column KS statistic and Wasserstein-1 (scaled by the column's IQR), mean absolute
  difference of the correlation matrices (off-diagonal);
- detection (C2ST): AUC of a classifier telling real from synthetic — 0.5 is indistinguishable;
- utility (TSTR): R^2 on the real test split of a model trained on synthetic rows, vs the same
  model trained on real rows (TRTR);
- privacy (DCR): distance from synthetic rows to their nearest training row, relative to the same
  distance for real holdout rows — a ratio well below 1 means the generator copies training data.

KS p-values are not reported: at n in the thousands every difference is "significant".
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import QuantileTransformer

# -- one-dimensional distances (scipy-free) -------------------------------------------------


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: max |F_a - F_b|."""
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Wasserstein-1 between two empirical distributions: integral of |F_a - F_b|."""
    a, b = np.sort(a), np.sort(b)
    grid = np.sort(np.concatenate([a, b]))
    deltas = np.diff(grid)
    fa = np.searchsorted(a, grid[:-1], side="right") / len(a)
    fb = np.searchsorted(b, grid[:-1], side="right") / len(b)
    return float(np.sum(np.abs(fa - fb) * deltas))


def marginals(real: np.ndarray, synth: np.ndarray, columns: list[str]) -> dict:
    per_column = {}
    for j, name in enumerate(columns):
        q75, q25 = np.percentile(real[:, j], [75, 25])
        scale = (q75 - q25) or (np.std(real[:, j]) or 1.0)
        per_column[name] = {
            "ks": round(ks_statistic(real[:, j], synth[:, j]), 4),
            "wasserstein_iqr": round(wasserstein_1d(real[:, j], synth[:, j]) / scale, 4),
        }
    return {
        "ks_mean": round(float(np.mean([v["ks"] for v in per_column.values()])), 4),
        "wasserstein_iqr_mean": round(float(np.mean([v["wasserstein_iqr"] for v in per_column.values()])), 4),
        "per_column": per_column,
    }


def correlation_gap(real: np.ndarray, synth: np.ndarray) -> float:
    """Mean |corr_real - corr_synth| over off-diagonal pairs (0 = identical, max 2)."""
    cr, cs = np.corrcoef(real, rowvar=False), np.corrcoef(synth, rowvar=False)
    off = ~np.eye(cr.shape[0], dtype=bool)
    return round(float(np.mean(np.abs(cr - cs)[off])), 4)


# -- learned scores --------------------------------------------------------------------------


def _balanced(real: np.ndarray, synth: np.ndarray, rng: np.random.Generator, max_rows: int):
    n = min(len(real), len(synth), max_rows)
    return real[rng.choice(len(real), n, replace=False)], synth[rng.choice(len(synth), n, replace=False)]


def c2st_auc(real: np.ndarray, synth: np.ndarray, seed: int = 0, max_rows: int = 5000) -> float:
    """Classifier two-sample test: held-out AUC of real-vs-synthetic (0.5 = indistinguishable)."""
    rng = np.random.default_rng(seed)
    r, s = _balanced(real, synth, rng, max_rows)
    X = np.vstack([r, s])
    y = np.r_[np.zeros(len(r)), np.ones(len(s))]
    idx = rng.permutation(len(X))
    cut = int(0.7 * len(X))
    clf = HistGradientBoostingClassifier(random_state=seed).fit(X[idx[:cut]], y[idx[:cut]])
    return round(float(roc_auc_score(y[idx[cut:]], clf.predict_proba(X[idx[cut:]])[:, 1])), 4)


def tstr(train_rows: np.ndarray, test_rows: np.ndarray, seed: int = 0) -> float:
    """R^2 on real test rows of a regressor fit on `train_rows`; target = last column."""
    model = HistGradientBoostingRegressor(random_state=seed).fit(train_rows[:, :-1], train_rows[:, -1])
    return round(float(r2_score(test_rows[:, -1], model.predict(test_rows[:, :-1]))), 4)


def _nearest_distances(queries: np.ndarray, reference: np.ndarray, chunk: int = 1024) -> np.ndarray:
    out = []
    ref_sq = (reference**2).sum(1)
    for i in range(0, len(queries), chunk):
        q = queries[i : i + chunk]
        d2 = (q**2).sum(1)[:, None] + ref_sq[None] - 2 * q @ reference.T
        out.append(np.sqrt(np.maximum(d2.min(1), 0)))
    return np.concatenate(out)


def dcr(
    train: np.ndarray, holdout: np.ndarray, synth: np.ndarray, seed: int = 0, max_rows: int = 2000
) -> dict:
    """Distance to closest training record, in a standardized (quantile-normal) space."""
    rng = np.random.default_rng(seed)
    qt = QuantileTransformer(
        output_distribution="normal", n_quantiles=min(1000, len(train)), random_state=seed
    )
    ref = qt.fit_transform(train)
    pick = lambda a: a[rng.choice(len(a), min(len(a), max_rows), replace=False)]  # noqa: E731
    d_synth = float(np.median(_nearest_distances(qt.transform(pick(synth)), ref)))
    d_hold = float(np.median(_nearest_distances(qt.transform(pick(holdout)), ref)))
    return {
        "dcr_synthetic": round(d_synth, 4),
        "dcr_holdout": round(d_hold, 4),
        "dcr_ratio": round(d_synth / d_hold, 4) if d_hold else float("nan"),
    }


# -- baselines ------------------------------------------------------------------------------


def independent_marginals(train: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Resample each column independently: perfect marginals, no dependence at all."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.choice(train[:, j], n) for j in range(train.shape[1])])


def gaussian_copula(train: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Quantile-normalize, fit one multivariate Gaussian, map back: marginals + linear dependence."""
    qt = QuantileTransformer(
        output_distribution="normal", n_quantiles=min(1000, len(train)), random_state=seed
    )
    z = qt.fit_transform(train)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(z.mean(0), np.cov(z, rowvar=False), size=n)
    return qt.inverse_transform(draws)


BASELINES = {"independent_marginals": independent_marginals, "gaussian_copula": gaussian_copula}


# -- report ---------------------------------------------------------------------------------


def evaluate(synth: np.ndarray, data, seed: int = 0) -> dict:
    """All metrics for one synthetic table (raw space) against data.holdout/test/train."""
    result = {
        **{k: v for k, v in marginals(data.holdout, synth, data.columns).items()},
        "correlation_gap": correlation_gap(data.holdout, synth),
        "c2st_auc": c2st_auc(data.holdout, synth, seed),
        **dcr(data.train, data.holdout, synth, seed),
    }
    if data.target is not None:
        result["tstr_r2"] = tstr(synth, data.test, seed)
    return result


def report(generated: dict[str, np.ndarray], data, seed: int = 0) -> dict:
    """{method: synthetic rows} (baselines added automatically) -> {method: metrics}."""
    n = len(data.holdout)
    methods = dict(generated)
    for name, fn in BASELINES.items():
        methods.setdefault(name, fn(data.train, n, seed))
    out = {
        "dataset": data.name,
        "seed": seed,
        "methods": {k: evaluate(v, data, seed) for k, v in methods.items()},
    }
    if data.target is not None:
        out["trtr_r2"] = tstr(data.train, data.test, seed)
        out["methods"]["real_holdout"] = evaluate(data.holdout, data, seed)
    return out


ROWS = [
    ("ks_mean", "KS (mean, ↓)"),
    ("wasserstein_iqr_mean", "Wasserstein/IQR (mean, ↓)"),
    ("correlation_gap", "Correlation gap (↓)"),
    ("c2st_auc", "C2ST AUC (0.5 best)"),
    ("tstr_r2", "TSTR R² (↑)"),
    ("dcr_ratio", "DCR ratio (≈1; ≪1 = copying)"),
]


def to_markdown(rep: dict) -> str:
    methods = list(rep["methods"])
    lines = ["| Metric | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for key, label in ROWS:
        if all(key not in rep["methods"][m] for m in methods):
            continue
        cells = [f"{rep['methods'][m][key]:.3f}" if key in rep["methods"][m] else "—" for m in methods]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if "trtr_r2" in rep:
        lines.append(f"\nTRTR R² (real train → real test): {rep['trtr_r2']:.3f}")
    return "\n".join(lines) + "\n"
