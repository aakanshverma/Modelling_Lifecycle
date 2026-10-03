#!/usr/bin/env python
"""Demonstrate uplift modeling finding heterogeneity a blanket contact policy would miss.

Runs a randomized collections-contact experiment, fits a T-learner uplift
model, and validates its ranking against the ground-truth treatment
effect (only computable because the data is synthetic).

Usage:
    python scripts/run_uplift_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from lifecycle.data.synthetic import (  # noqa: E402
    generate_applicants,
    generate_collections_experiment,
    generate_onbook_behavior,
    true_collections_uplift,
)
from lifecycle.scoring.uplift import assign_uplift_segment, fit_uplift_model  # noqa: E402
from lifecycle.stages.collections import FEATURES  # noqa: E402


def main() -> None:
    applicants = generate_applicants(20_000, seed=42)
    onbook = generate_onbook_behavior(applicants, seed=43)
    experiment = generate_collections_experiment(onbook, seed=46)

    print("=" * 60)
    print("UPLIFT MODELING — collections contact experiment")
    print("=" * 60)
    print(f"\nContacted: {experiment['contacted'].mean():.1%} of the segment (randomized)")
    print(f"Self-cure rate | contacted:     {experiment.loc[experiment['contacted'] == 1, 'self_cure_flag'].mean():.1%}")
    print(f"Self-cure rate | not contacted: {experiment.loc[experiment['contacted'] == 0, 'self_cure_flag'].mean():.1%}")
    print(
        "-> the average effect looks small/unremarkable. That's exactly why a blanket\n"
        "   'does contact help' read is misleading — see the heterogeneity below."
    )

    model = fit_uplift_model(experiment[FEATURES], experiment["contacted"], experiment["self_cure_flag"])
    predicted_uplift = model.predict_uplift(onbook[FEATURES])
    true_uplift = true_collections_uplift(onbook)

    df = pd.DataFrame({"predicted": predicted_uplift, "true": true_uplift})
    df["decile"] = pd.qcut(df["predicted"], 10, labels=False, duplicates="drop")

    print("\nTrue uplift by predicted-uplift decile (should rise monotonically):")
    print(df.groupby("decile")["true"].mean().round(4).to_string())
    print(f"\nSpearman correlation (predicted vs. true uplift): {df['predicted'].corr(df['true'], method='spearman'):.3f}")

    segment = assign_uplift_segment(predicted_uplift)
    segment_true = pd.DataFrame({"segment": segment, "true_uplift": true_uplift})
    print("\nSegment sizes and true average uplift per segment:")
    print(segment.value_counts().to_string())
    print(segment_true.groupby("segment")["true_uplift"].mean().round(4).to_string())
    print(
        "\n'persuadable' should average a clearly positive true uplift (worth contacting);\n"
        "'sleeping_dog' should average negative (contact backfires — avoid aggressive outreach)."
    )


if __name__ == "__main__":
    main()
