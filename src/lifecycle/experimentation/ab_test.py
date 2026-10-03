"""Generic champion/challenger A/B test evaluation.

The same statistical test applies whether you're comparing two
acquisition cutoffs, two collections contact strategies, or two
cross-sell offer scripts — this is the standing framework a risk team
runs continuously in production, not a one-off analysis. Everything
else in this repo (reject inference, uplift modeling, PSI monitoring)
tells you what to try next; this is how you prove the change actually
worked before rolling it out to the whole portfolio.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ABTestResult:
    champion_mean: float
    challenger_mean: float
    lift: float  # challenger - champion
    relative_lift: float  # lift / champion_mean
    p_value: float
    significant: bool  # p_value < alpha
    champion_n: int
    challenger_n: int


def assign_arm(index: pd.Index, challenger_rate: float = 0.5, seed: int = 0) -> pd.Series:
    """Randomly assign each row to the 'champion' (current policy) or 'challenger' arm."""
    rng = np.random.default_rng(seed)
    arm = rng.choice(
        ["champion", "challenger"], size=len(index), p=[1 - challenger_rate, challenger_rate]
    )
    return pd.Series(arm, index=index, name="arm")


def evaluate_ab_test(outcome: pd.Series, arm: pd.Series, alpha: float = 0.05) -> ABTestResult:
    """Welch's two-sample t-test — the standard approximation for comparing both
    continuous metrics and 0/1 outcome rates between a champion and challenger arm.
    """
    champion_vals = outcome[arm == "champion"]
    challenger_vals = outcome[arm == "challenger"]

    champion_mean = float(champion_vals.mean())
    challenger_mean = float(challenger_vals.mean())
    lift = challenger_mean - champion_mean
    relative_lift = lift / champion_mean if champion_mean != 0 else float("nan")

    with warnings.catch_warnings():
        # A degenerate arm (e.g. a "contact everyone" policy, which has zero variance)
        # triggers scipy's precision-loss warning but still returns a valid p-value.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        _, p_value = stats.ttest_ind(challenger_vals, champion_vals, equal_var=False)

    return ABTestResult(
        champion_mean=champion_mean,
        challenger_mean=challenger_mean,
        lift=lift,
        relative_lift=relative_lift,
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        champion_n=len(champion_vals),
        challenger_n=len(challenger_vals),
    )
