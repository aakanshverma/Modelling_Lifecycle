"""Stage 3b, upgraded — Next-Best-Offer ranking across the cross-sell catalog.

Phase 1's `cross_sell.py` predicts accept/reject propensity for a single
product. A real cross-sell team ranks the *whole* catalog per customer and
weighs it by expected value (acceptance propensity x product margin x
CLV), so a high-CLV customer isn't pointed at a low-margin product just
because they're slightly more likely to accept it, and a thin-margin
product isn't pushed onto someone who'll barely use it.
"""
from __future__ import annotations

import pandas as pd

from lifecycle.analytics.clv import estimate_clv
from lifecycle.scoring.scorecard import ScorecardModel, fit_scorecard

PRODUCTS = ["top_up_loan", "credit_card", "insurance"]

# Illustrative monthly margin rate per product, as a fraction of estimated CLV.
# Scaled so products with lower typical acceptance aren't automatically dominated by
# one high-margin product — the customer's features should decide the winner, not a
# fixed margin ranking that makes the catalog comparison trivial.
PRODUCT_MARGIN_RATE = {
    "top_up_loan": 0.030,
    "credit_card": 0.040,
    "insurance": 0.025,
}

FEATURES = [
    "monthly_income",
    "months_on_book",
    "credit_utilization",
    "bureau_refresh_score",
    "existing_emi_to_income",
    "age",
]

TARGET_BY_PRODUCT = {
    "top_up_loan": "accept_top_up_loan",
    "credit_card": "accept_credit_card",
    "insurance": "accept_insurance",
}


def fit_product_models(cross_sell_pop: pd.DataFrame) -> dict[str, ScorecardModel]:
    return {
        product: fit_scorecard(f"nbo_{product}", cross_sell_pop[FEATURES], cross_sell_pop[target])
        for product, target in TARGET_BY_PRODUCT.items()
    }


def score_products(models: dict[str, ScorecardModel], X: pd.DataFrame) -> pd.DataFrame:
    scored = {
        f"{product}_accept_pd": model.predict(X[FEATURES])[f"{model.name}_pd"]
        for product, model in models.items()
    }
    return pd.DataFrame(scored, index=X.index)


def rank_next_best_offer(product_scores: pd.DataFrame, clv: pd.Series) -> pd.DataFrame:
    """For each customer, pick the product maximizing expected value = accept_pd x margin x CLV."""
    expected_value = pd.DataFrame(
        {
            product: product_scores[f"{product}_accept_pd"] * PRODUCT_MARGIN_RATE[product] * clv
            for product in PRODUCTS
        },
        index=product_scores.index,
    )
    return pd.DataFrame(
        {
            "next_best_offer": expected_value.idxmax(axis=1),
            "expected_value": expected_value.max(axis=1),
        }
    )


def risk_adjusted_offer_amount(
    X: pd.DataFrame, behavioral_pd: pd.Series, income_multiple: float = 3.0
) -> pd.Series:
    """Illustrative top-up-loan sizing: a multiple of monthly income, discounted by
    current behavioral risk — ties the offer itself back to the behavioral score,
    rather than sizing every approved cross-sell customer identically.
    """
    base_amount = X["monthly_income"] * income_multiple
    risk_discount = (1 - behavioral_pd).clip(lower=0.2)
    return (base_amount * risk_discount).round(-3).rename("risk_adjusted_offer_amount")
