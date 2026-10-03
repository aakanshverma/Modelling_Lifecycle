"""Stage 2 — Behavioral scorecard (the "B-score").

Once a customer has spent time on the books, re-scores them using how they
actually behaved (repayment, bounces, utilization drift) blended with a
bureau refresh. This score is the fork in the journey: it decides whether
the customer gets routed to Collections or to Cross-sell.
"""
from __future__ import annotations

import pandas as pd

from lifecycle.scoring.scorecard import ScorecardModel, fit_scorecard

FEATURES = [
    "max_dpd_last_3m",
    "num_bounces_last_6m",
    "utilization_trend",
    "payment_to_due_ratio",
    "bureau_refresh_score",
    "months_on_book",
]
TARGET = "behavioral_bad_flag"


def fit(onbook: pd.DataFrame) -> ScorecardModel:
    return fit_scorecard("behavioral", onbook[FEATURES], onbook[TARGET])


def score(model: ScorecardModel, onbook: pd.DataFrame) -> pd.DataFrame:
    scored = model.predict(onbook[FEATURES])
    return pd.concat([onbook[["applicant_id"]], scored], axis=1)


def route(scored: pd.DataFrame, cutoff: int = 600) -> pd.Series:
    """Below cutoff -> underperforming, goes to Collections. At/above -> Cross-sell eligible."""
    return pd.Series(
        ["cross_sell" if s >= cutoff else "collections" for s in scored["behavioral_score"]],
        index=scored.index,
    )
