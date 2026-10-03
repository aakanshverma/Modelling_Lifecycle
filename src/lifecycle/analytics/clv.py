"""A simple Customer Lifetime Value estimate used to weight Next-Best-Offer ranking.

This is a transparent heuristic (expected tenure x expected monthly margin
x a risk discount), not a full survival/hazard model — that's a
deliberate scope choice: NBO ranking only needs customers ordered
sensibly by value, and a heuristic that's easy to explain to a business
stakeholder is more useful here than a more "accurate" black box.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_MARGIN_RATE = 0.02  # illustrative monthly margin as a fraction of income
DEFAULT_EXPECTED_TENURE_MONTHS = 24


def estimate_clv(
    X: pd.DataFrame,
    behavioral_pd: pd.Series,
    margin_rate: float = DEFAULT_MARGIN_RATE,
    expected_tenure_months: int = DEFAULT_EXPECTED_TENURE_MONTHS,
) -> pd.Series:
    """Estimated CLV = monthly_income x margin_rate x expected_tenure x (1 - risk discount).

    `behavioral_pd` is the current behavioral-scorecard PD — a riskier
    customer has a shorter expected realized relationship, so their CLV is
    discounted accordingly.
    """
    risk_discount = (1 - behavioral_pd).clip(lower=0.05)
    clv = X["monthly_income"] * margin_rate * expected_tenure_months * risk_discount
    return clv.rename("estimated_clv")
