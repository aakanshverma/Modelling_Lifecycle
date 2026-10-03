"""Stage 3a — Collections scorecard (the "C-score").

For customers the behavioral score routed into Collections: predicts
propensity to self-cure, so collections effort is tiered (soft contact vs
field/legal) instead of treated uniformly.
"""
from __future__ import annotations

import pandas as pd

from lifecycle.scoring.scorecard import ScorecardModel, fit_scorecard

FEATURES = [
    "max_dpd_last_3m",
    "num_bounces_last_6m",
    "payment_to_due_ratio",
    "bureau_refresh_score",
]
TARGET = "self_cure_flag"

STRATEGY_BY_DECILE = {
    range(7, 10): "soft_contact",  # high self-cure propensity
    range(3, 7): "tele_calling",
    range(0, 3): "field_or_legal",  # low self-cure propensity
}


def fit(collections_pop: pd.DataFrame) -> ScorecardModel:
    return fit_scorecard("collections", collections_pop[FEATURES], collections_pop[TARGET])


def score(model: ScorecardModel, collections_pop: pd.DataFrame) -> pd.DataFrame:
    scored = model.predict(collections_pop[FEATURES])
    return pd.concat([collections_pop[["applicant_id"]], scored], axis=1)


def strategy(scored: pd.DataFrame) -> pd.Series:
    def _bucket(decile: int) -> str:
        for deciles, label in STRATEGY_BY_DECILE.items():
            if decile in deciles:
                return label
        return "tele_calling"

    return scored["collections_decile"].apply(_bucket)
