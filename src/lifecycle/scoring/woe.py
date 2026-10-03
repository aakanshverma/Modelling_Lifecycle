"""Weight-of-Evidence / Information-Value binning, shared by every scorecard stage."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BinResult:
    feature: str
    bin_edges: np.ndarray
    woe_by_bin: pd.Series
    iv: float


def _bin_edges(series: pd.Series, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(series.quantile(quantiles).to_numpy())
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def fit_woe(series: pd.Series, target: pd.Series, n_bins: int = 5) -> BinResult:
    """Fit quantile bins on `series` and compute WOE/IV against a binary `target`."""
    edges = _bin_edges(series, n_bins)
    binned = pd.cut(series, bins=edges, duplicates="drop")

    df = pd.DataFrame({"bin": binned, "target": target})
    total_good = (df["target"] == 0).sum()
    total_bad = (df["target"] == 1).sum()

    grouped = df.groupby("bin", observed=True)["target"].agg(
        good=lambda s: (s == 0).sum(), bad=lambda s: (s == 1).sum()
    )
    # Laplace smoothing avoids div-by-zero / log(0) on sparse bins.
    good_dist = (grouped["good"] + 0.5) / (total_good + 0.5)
    bad_dist = (grouped["bad"] + 0.5) / (total_bad + 0.5)
    woe = np.log(good_dist / bad_dist)
    iv = float(((good_dist - bad_dist) * woe).sum())

    return BinResult(feature=series.name, bin_edges=edges, woe_by_bin=woe, iv=iv)


def apply_woe(series: pd.Series, bin_result: BinResult) -> pd.Series:
    """Transform a raw feature into its fitted WOE values."""
    binned = pd.cut(series, bins=bin_result.bin_edges, duplicates="drop")
    mapped = binned.map(bin_result.woe_by_bin)
    return mapped.fillna(0.0).astype(float).rename(f"{bin_result.feature}_woe")


def fit_woe_frame(X: pd.DataFrame, y: pd.Series, n_bins: int = 5) -> dict[str, BinResult]:
    """Fit WOE bins for every column in X. Drops features below the usual IV floor (0.02)."""
    results = {}
    for col in X.columns:
        fit = fit_woe(X[col], y, n_bins=n_bins)
        if fit.iv >= 0.02:
            results[col] = fit
    return results


def transform_woe_frame(X: pd.DataFrame, bin_results: dict[str, BinResult]) -> pd.DataFrame:
    return pd.concat([apply_woe(X[col], fit) for col, fit in bin_results.items()], axis=1)
