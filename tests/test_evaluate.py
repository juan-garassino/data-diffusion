from __future__ import annotations

import numpy as np
import pytest

from datadiffusion import evaluate as ev
from datadiffusion.data import TabularData

RNG = np.random.default_rng(0)


def test_ks_and_wasserstein_known_values():
    a = RNG.normal(size=2000)
    assert ev.ks_statistic(a, a) == 0.0
    assert ev.ks_statistic(np.zeros(10), np.ones(10)) == 1.0
    assert ev.wasserstein_1d(a, a + 3.0) == pytest.approx(3.0, rel=1e-6)
    assert ev.wasserstein_1d(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.0


def test_correlation_gap():
    x = RNG.normal(size=(3000, 3))
    x[:, 1] += x[:, 0]
    assert ev.correlation_gap(x, x) == 0.0
    assert ev.correlation_gap(x, RNG.permuted(x, axis=0)) > 0.2


def test_c2st_same_vs_different_distribution():
    a, b = RNG.normal(size=(2000, 2)), RNG.normal(size=(2000, 2))
    assert abs(ev.c2st_auc(a, b) - 0.5) < 0.1
    assert ev.c2st_auc(a, b + 3) > 0.95


def test_tstr_real_equals_real():
    x = RNG.normal(size=(3000, 3))
    rows = np.column_stack([x, x @ [1.0, -2.0, 0.5] + 0.1 * RNG.normal(size=3000)])
    assert ev.tstr(rows[:2000], rows[2000:]) > 0.9


def test_dcr_flags_copies():
    train, holdout = RNG.normal(size=(1000, 3)), RNG.normal(size=(300, 3))
    assert ev.dcr(train, holdout, train[:300])["dcr_ratio"] < 0.1  # memorized
    assert 0.7 < ev.dcr(train, holdout, RNG.normal(size=(300, 3)))["dcr_ratio"] < 1.4


def test_baselines_preserve_what_they_should():
    x = RNG.normal(size=(4000, 2))
    x[:, 1] = x[:, 0] + 0.3 * x[:, 1]
    marg = ev.independent_marginals(x, 4000)
    cop = ev.gaussian_copula(x, 4000)
    assert marg.shape == cop.shape == (4000, 2)
    assert ev.correlation_gap(x, cop) < 0.05 < ev.correlation_gap(x, marg)


def test_report_and_markdown():
    rows = RNG.normal(size=(1500, 3))
    rows[:, 2] = rows[:, 0] - rows[:, 1]
    data = TabularData("toy", ["a", "b", "y"], "y", rows[:1000], rows[1000:1250], rows[1250:])
    rep = ev.report({"diffusion": rows[:250] + 0.01}, data)
    assert set(rep["methods"]) == {"diffusion", "independent_marginals", "gaussian_copula", "real_holdout"}
    assert "tstr_r2" in rep["methods"]["gaussian_copula"] and "trtr_r2" in rep
    md = ev.to_markdown(rep)
    assert md.startswith("| Metric | diffusion |") and "C2ST AUC" in md
