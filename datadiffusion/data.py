"""Datasets, seeded train/holdout/test splits, and the normalizing preprocessor.

The diffusion model is trained only on `train`. `holdout` (real rows it never saw) is what
fidelity and privacy are measured against; `test` is reserved for the TSTR utility score.
The target column, when a dataset has one, is modelled jointly as the last column so
synthetic rows carry their own labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import QuantileTransformer

logger = logging.getLogger(__name__)

DATASETS = ("california", "moons", "circles")


@dataclass
class TabularData:
    name: str
    columns: list[str]
    target: str | None  # name of the last column when it is a supervised target
    train: np.ndarray
    holdout: np.ndarray
    test: np.ndarray

    @property
    def dim(self) -> int:
        return self.train.shape[1]


def _raw(name: str, n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, list[str], str | None]:
    if name == "california":
        from sklearn.datasets import fetch_california_housing

        ds = fetch_california_housing()
        X = np.column_stack([ds.data, ds.target])
        return X.astype(np.float64), list(ds.feature_names) + ["MedHouseVal"], "MedHouseVal"
    if name == "moons":
        from sklearn.datasets import make_moons

        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        return X, ["x", "y"], None
    if name == "circles":
        from sklearn.datasets import make_circles

        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
        return X, ["x", "y"], None
    raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")


def load_dataset(
    name: str,
    seed: int = 0,
    holdout_fraction: float = 0.15,
    test_fraction: float = 0.15,
    n_samples: int = 5000,
    noise: float = 0.05,
) -> TabularData:
    X, columns, target = _raw(name, n_samples, noise, seed)
    perm = np.random.default_rng(seed).permutation(len(X))
    n_test = round(len(X) * test_fraction)
    n_hold = round(len(X) * holdout_fraction)
    test, holdout, train = np.split(X[perm], [n_test, n_test + n_hold])
    logger.info(
        "%s: %d train / %d holdout / %d test rows, %d columns",
        name,
        len(train),
        len(holdout),
        len(test),
        X.shape[1],
    )
    return TabularData(name, columns, target, train, holdout, test)


class Preprocessor:
    """Per-column quantile transform to a standard normal, fit on the training split only.

    Note: inverse_transform maps back through the training quantiles, so synthetic values stay
    within the training range of each column.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._qt: QuantileTransformer | None = None

    def fit(self, X: np.ndarray) -> Preprocessor:
        self._qt = QuantileTransformer(
            output_distribution="normal", n_quantiles=min(1000, len(X)), random_state=self.seed
        ).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._qt.transform(X).astype(np.float32)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return self._qt.inverse_transform(np.asarray(Z, dtype=np.float64))
