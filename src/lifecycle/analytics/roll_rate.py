"""Roll-rate transition matrices and vintage curves for the Behavioral stage.

These are the two standard tools a behavioral/portfolio risk team runs
alongside the behavioral scorecard: the roll-rate matrix tells you how
accounts migrate between DPD buckets month to month (used for loss
forecasting), and the vintage curve tells you how fast a booking cohort
reaches NPA, which is the usual way to compare whether tightening or
loosening the acquisition cutoff actually changed portfolio quality.
"""
from __future__ import annotations

import pandas as pd

from lifecycle.data.synthetic import DPD_BUCKETS


def compute_transition_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Empirical month-over-month DPD bucket transition matrix across the whole panel."""
    panel = panel.sort_values(["applicant_id", "month_on_book"])
    next_bucket = panel.groupby("applicant_id")["dpd_bucket"].shift(-1)
    pairs = panel.assign(next_bucket=next_bucket).dropna(subset=["next_bucket"])

    matrix = pd.crosstab(pairs["dpd_bucket"], pairs["next_bucket"], normalize="index")
    return matrix.reindex(index=DPD_BUCKETS, columns=DPD_BUCKETS, fill_value=0.0)


def compute_vintage_curve(
    panel: pd.DataFrame, score_band: pd.Series, bad_bucket: str = "npa_90_plus"
) -> pd.DataFrame:
    """Cumulative ever-bad rate by month-on-book, segmented by an origination score band.

    `score_band` must be indexed by applicant_id (e.g. acquisition score deciles).
    """
    panel = panel.merge(score_band.rename("score_band"), left_on="applicant_id", right_index=True)
    panel = panel.sort_values(["applicant_id", "month_on_book"])

    is_bad = panel["dpd_bucket"] == bad_bucket
    ever_bad = is_bad.groupby(panel["applicant_id"]).cummax()

    curve = (
        panel.assign(ever_bad=ever_bad.to_numpy())
        .groupby(["score_band", "month_on_book"])["ever_bad"]
        .mean()
        .unstack("score_band")
    )
    return curve
