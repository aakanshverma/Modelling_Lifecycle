#!/usr/bin/env python
"""Demonstrate reject inference closing the gap between a naive and an oracle model.

Simulates a prior, crude policy that only ever booked cibil_score >= 700
applicants (so only they have an observed outcome), then compares three
acquisition models evaluated on a fresh, fully-labeled holdout population:

  - baseline:  fit only on the known-outcome population (what you'd ship
               without reject inference — biased toward the old policy)
  - augmented: fit on known + parceling-inferred labels for the rest
  - oracle:    fit on true labels for everyone (impossible in real life;
               only available here because the data is synthetic — this
               is the benchmark reject inference is trying to approach)

Usage:
    python scripts/run_reject_inference_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from lifecycle.data.synthetic import generate_applicants  # noqa: E402
from lifecycle.scoring.reject_inference import run_reject_inference  # noqa: E402
from lifecycle.scoring.scorecard import fit_scorecard  # noqa: E402
from lifecycle.stages.acquisition import FEATURES, TARGET  # noqa: E402


def main() -> None:
    train = generate_applicants(20_000, seed=100)
    holdout = generate_applicants(20_000, seed=200)

    known_mask = train["cibil_score"] >= 700
    known, unknown = train[known_mask], train[~known_mask]

    print("=" * 60)
    print("REJECT INFERENCE — baseline vs augmented vs oracle")
    print("=" * 60)
    print(f"\nPrior policy booked {known_mask.mean():.1%} of applicants (cibil_score >= 700)")
    print(f"Known-population bad rate:   {known[TARGET].mean():.1%}")
    print(f"True bad rate of the never-booked population (hidden in real life): "
          f"{unknown[TARGET].mean():.1%}")

    result = run_reject_inference(known[FEATURES], known[TARGET], unknown[FEATURES], seed=1)
    print(f"Parceling-inferred bad rate for that population: {result.inferred_bad_rate:.1%}")

    oracle_model = fit_scorecard("oracle", train[FEATURES], train[TARGET])

    print("\nHoldout AUC (fresh population, true labels):")
    for label, model in [
        ("baseline (naive)", result.baseline_model),
        ("augmented (reject-inferred)", result.augmented_model),
        ("oracle (full true labels)", oracle_model),
    ]:
        pd_hat = model.predict(holdout[FEATURES])[f"{model.name}_pd"]
        auc = roc_auc_score(holdout[TARGET], pd_hat)
        print(f"  {label:30s} {auc:.4f}")


if __name__ == "__main__":
    main()
