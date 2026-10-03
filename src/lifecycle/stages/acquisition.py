"""Stage 1 — Acquisition / Application scorecard (the "A-score").

Scores a prospect at the moment they apply, using bureau data (CIBIL-style)
plus the application form, to decide approve / refer / reject before
anything is booked onto the portfolio.
"""
from __future__ import annotations

import pandas as pd

from lifecycle.scoring.scorecard import ScorecardModel, fit_scorecard

FEATURES = [
    "cibil_score",
    "bureau_vintage_months",
    "num_trades",
    "dpd_30_plus_last_6m",
    "credit_utilization",
    "existing_emi_to_income",
]
TARGET = "default_flag"

# Phase 4: bureau features plus alternative-data features (UPI/utility-bill proxies —
# see data/synthetic.generate_alt_data_features), used for the thin-file lift comparison.
ALT_FEATURES = FEATURES + ["upi_avg_monthly_inflow", "upi_bounce_rate", "utility_ontime_payment_rate"]


def fit(applicants: pd.DataFrame) -> ScorecardModel:
    return fit_scorecard("acquisition", applicants[FEATURES], applicants[TARGET])


def fit_with_altdata(applicants: pd.DataFrame) -> ScorecardModel:
    """Same target, bureau features plus alternative-data features. `applicants` must
    already have the alt-data columns merged in (see generate_alt_data_features)."""
    return fit_scorecard("acquisition_altdata", applicants[ALT_FEATURES], applicants[TARGET])


def score(model: ScorecardModel, applicants: pd.DataFrame) -> pd.DataFrame:
    scored = model.predict(applicants[FEATURES])
    return pd.concat([applicants[["applicant_id"]], scored], axis=1)


def decide(scored: pd.DataFrame, approve_cutoff: int = 590, refer_cutoff: int = 560) -> pd.Series:
    """approve_cutoff/refer_cutoff are scorecard points — higher score = lower risk."""
    score_col = scored["acquisition_score"]
    decision = pd.Series("reject", index=scored.index)
    decision[score_col >= refer_cutoff] = "refer"
    decision[score_col >= approve_cutoff] = "approve"
    return decision
