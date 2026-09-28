from __future__ import annotations

import numpy as np
import pytest

from datadiffusion.config import ExperimentConfig
from datadiffusion.data import Preprocessor, load_dataset


def test_splits_are_disjoint_deterministic_and_sized():
    a = load_dataset("moons", seed=0, n_samples=1000)
    b = load_dataset("moons", seed=0, n_samples=1000)
    assert np.array_equal(a.train, b.train)
    assert (len(a.train), len(a.holdout), len(a.test)) == (700, 150, 150)
    rows = {tuple(r) for r in a.train}
    assert not rows & {tuple(r) for r in a.holdout} and not rows & {tuple(r) for r in a.test}
    assert a.target is None and a.columns == ["x", "y"]


def test_unknown_dataset():
    with pytest.raises(ValueError):
        load_dataset("iris")


def test_california_joint_target_column():
    try:
        from sklearn.datasets import fetch_california_housing

        fetch_california_housing(download_if_missing=False)
    except OSError:
        pytest.skip("California Housing not in the local sklearn cache")
    data = load_dataset("california")
    assert data.dim == 9 and data.columns[-1] == data.target == "MedHouseVal"


def test_preprocessor_roundtrip_fit_on_train_only():
    data = load_dataset("circles", n_samples=800)
    pre = Preprocessor().fit(data.train)
    z = pre.transform(data.holdout)
    assert z.dtype == np.float32 and abs(float(pre.transform(data.train).mean())) < 0.1
    assert np.allclose(pre.inverse_transform(pre.transform(data.train)), data.train, atol=1e-2)


def test_config_from_toml_and_validation(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('name = "m"\n[data]\ndataset = "moons"\n[train]\nepochs = 3\n[model]\nhidden_size = 32\n')
    cfg = ExperimentConfig.from_toml(p)
    assert cfg.data.dataset == "moons" and cfg.train.epochs == 3 and cfg.model.hidden_size == 32
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.from_toml(tmp_path / "missing.toml")  # v2 silently fell back to defaults
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"train": {"epoch": 3}})
