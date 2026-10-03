"""Stage 3b — Cross-sell scorecard (the "X-score").

For customers the behavioral score routed into Cross-sell: predicts
propensity to take up a new product, ranked into deciles so campaigns can
target the top tiers first.
"""
from __future__ import annotations

import pandas as pd

from lifecycle.scoring.scorecard import ScorecardModel, fit_scorecard

FEATURES = [
    "monthly_income",
    "months_on_book",
    "credit_utilization",
    "bureau_refresh_score",
    "existing_emi_to_income",
]
TARGET = "accepted_cross_sell"

OFFER_BY_DECILE = {
    range(8, 10): "top_tier_offer",  # e.g. pre-approved top-up loan
    range(4, 8): "standard_offer",
    range(0, 4): "no_offer",
}


def fit(cross_sell_pop: pd.DataFrame) -> ScorecardModel:
    return fit_scorecard("cross_sell", cross_sell_pop[FEATURES], cross_sell_pop[TARGET])


def score(model: ScorecardModel, cross_sell_pop: pd.DataFrame) -> pd.DataFrame:
    scored = model.predict(cross_sell_pop[FEATURES])
    return pd.concat([cross_sell_pop[["applicant_id"]], scored], axis=1)


def offer_tier(scored: pd.DataFrame) -> pd.Series:
    def _bucket(decile: int) -> str:
        for deciles, label in OFFER_BY_DECILE.items():
            if decile in deciles:
                return label
        return "standard_offer"

    return scored["cross_sell_decile"].apply(_bucket)
