"""Figures for reports (optional extra `[plots]`: matplotlib)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def marginals(real: np.ndarray, methods: dict[str, np.ndarray], columns: list[str], path: Path) -> Path:
    plt = _plt()
    k = len(columns)
    cols = min(3, k)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)
    for j, name in enumerate(columns):
        ax = axes[j // cols, j % cols]
        bins = np.histogram_bin_edges(real[:, j], bins=40)
        ax.hist(real[:, j], bins=bins, density=True, alpha=0.5, label="real (holdout)", color="black")
        for label, rows_ in methods.items():
            with np.errstate(invalid="ignore", divide="ignore"):  # a column entirely outside the bins
                ax.hist(rows_[:, j], bins=bins, density=True, histtype="step", linewidth=1.4, label=label)
        ax.set_title(name, fontsize=9)
    for ax in axes.flat[k:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def correlations(real: np.ndarray, methods: dict[str, np.ndarray], columns: list[str], path: Path) -> Path:
    plt = _plt()
    panels = {"real (holdout)": real, **methods}
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.4), squeeze=False)
    for ax, (label, rows_) in zip(axes[0], panels.items(), strict=True):
        im = ax.imshow(np.corrcoef(rows_, rowvar=False), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(label, fontsize=9)
        ax.set_xticks(range(len(columns)), columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(columns)), columns, fontsize=6)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def scatter2d(real: np.ndarray, methods: dict[str, np.ndarray], path: Path) -> Path:
    plt = _plt()
    panels = {"real (holdout)": real, **methods}
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.2), squeeze=False)
    for ax, (label, rows_) in zip(axes[0], panels.items(), strict=True):
        ax.scatter(rows_[:, 0], rows_[:, 1], s=2, alpha=0.5)
        ax.set_title(label, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
