#!/usr/bin/env python
"""Benchmark an XGBoost challenger against each stage's WOE+logistic champion.

Usage:
    python scripts/run_challenger_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lifecycle.data.synthetic import (  # noqa: E402
    generate_applicants,
    generate_collections_outcomes,
    generate_cross_sell_outcomes,
    generate_onbook_behavior,
)
from lifecycle.scoring.challenger import compare_champion_challenger, fit_challenger  # noqa: E402
from lifecycle.scoring.scorecard import fit_scorecard  # noqa: E402
from lifecycle.stages import acquisition, behavioral, collections, cross_sell  # noqa: E402

STAGES = [
    (acquisition, "acquisition"),
    (behavioral, "behavioral"),
    (collections, "collections"),
    (cross_sell, "cross_sell"),
]


def main() -> None:
    applicants = generate_applicants(20_000, seed=42)
    onbook = generate_onbook_behavior(applicants, seed=43)
    collections_pop = generate_collections_outcomes(onbook, seed=44)
    cross_sell_pop = generate_cross_sell_outcomes(onbook, seed=45)

    holdout_applicants = generate_applicants(20_000, seed=142)
    holdout_onbook = generate_onbook_behavior(holdout_applicants, seed=143)
    holdout_collections = generate_collections_outcomes(holdout_onbook, seed=144)
    holdout_cross_sell = generate_cross_sell_outcomes(holdout_onbook, seed=145)

    datasets = {
        "acquisition": (applicants, holdout_applicants),
        "behavioral": (onbook, holdout_onbook),
        "collections": (collections_pop, holdout_collections),
        "cross_sell": (cross_sell_pop, holdout_cross_sell),
    }

    print("=" * 70)
    print("CHAMPION (WOE + logistic) vs CHALLENGER (XGBoost) — holdout AUC / KS")
    print("=" * 70)

    for module, name in STAGES:
        train, holdout = datasets[name]
        champion = fit_scorecard(name, train[module.FEATURES], train[module.TARGET])
        challenger = fit_challenger(f"{name}_xgb", train[module.FEATURES], train[module.TARGET])

        cmp = compare_champion_challenger(
            champion, challenger, holdout[module.FEATURES], holdout[module.TARGET]
        )
        print(f"\n{name}:")
        print(f"  champion   AUC={cmp.champion_auc:.4f}  KS={cmp.champion_ks:.4f}")
        print(f"  challenger AUC={cmp.challenger_auc:.4f}  KS={cmp.challenger_ks:.4f}")

        importance = challenger.shap_importance(holdout[module.FEATURES])
        print(f"  top SHAP feature: {importance.index[0]} ({importance.iloc[0]:.3f})")


if __name__ == "__main__":
    main()
