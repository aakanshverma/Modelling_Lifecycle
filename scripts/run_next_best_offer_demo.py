#!/usr/bin/env python
"""Demonstrate Next-Best-Offer ranking across the cross-sell product catalog.

Usage:
    python scripts/run_next_best_offer_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from lifecycle.analytics.clv import estimate_clv  # noqa: E402
from lifecycle.data.synthetic import generate_applicants, generate_nbo_outcomes, generate_onbook_behavior  # noqa: E402
from lifecycle.stages import acquisition, behavioral  # noqa: E402
from lifecycle.stages.next_best_offer import (  # noqa: E402
    fit_product_models,
    rank_next_best_offer,
    risk_adjusted_offer_amount,
    score_products,
)


def main() -> None:
    applicants = generate_applicants(20_000, seed=42)

    acquisition_model = acquisition.fit(applicants)
    acquisition_scored = acquisition.score(acquisition_model, applicants)
    booked = applicants[acquisition.decide(acquisition_scored) == "approve"]

    onbook = generate_onbook_behavior(booked, seed=43)
    behavioral_model = behavioral.fit(onbook)
    behavioral_scored = behavioral.score(behavioral_model, onbook)
    onbook = onbook.merge(behavioral_scored[["applicant_id", "behavioral_pd"]], on="applicant_id")
    onbook["route"] = behavioral.route(behavioral_scored).to_numpy()

    cross_sell_pop = onbook[onbook["route"] == "cross_sell"].reset_index(drop=True)
    cross_sell_pop = generate_nbo_outcomes(cross_sell_pop, seed=47)

    print("=" * 60)
    print("NEXT-BEST-OFFER — cross-sell segment")
    print("=" * 60)
    print(f"\nCross-sell segment size: {len(cross_sell_pop):,}")

    models = fit_product_models(cross_sell_pop)
    print("\nPer-product acceptance model train AUC:")
    for product, model in models.items():
        print(f"  {product:12s} {model.train_auc:.3f}")

    scores = score_products(models, cross_sell_pop)
    clv = estimate_clv(cross_sell_pop, cross_sell_pop["behavioral_pd"])
    ranked = rank_next_best_offer(scores, clv)

    print(f"\nEstimated CLV: mean={clv.mean():,.0f}  median={clv.median():,.0f}")
    print("\nNext-best-offer distribution:")
    print(ranked["next_best_offer"].value_counts().to_string())

    profile = pd.concat(
        [cross_sell_pop[["age", "monthly_income", "existing_emi_to_income"]], ranked], axis=1
    )
    print("\nAverage profile by recommended offer:")
    print(
        profile.groupby("next_best_offer")[["age", "monthly_income", "existing_emi_to_income"]]
        .mean()
        .round(1)
        .to_string()
    )

    top_up_mask = ranked["next_best_offer"] == "top_up_loan"
    offer_amount = risk_adjusted_offer_amount(
        cross_sell_pop.loc[top_up_mask], cross_sell_pop.loc[top_up_mask, "behavioral_pd"]
    )
    print(
        f"\nRisk-adjusted top-up-loan offer amount "
        f"(for the {top_up_mask.sum()} customers it was recommended to):"
    )
    print(offer_amount.describe().round(0).to_string())


if __name__ == "__main__":
    main()
