#!/usr/bin/env python
"""Champion/challenger A/B test: blanket contact vs. uplift-targeted contact.

Champion (current policy): contact every account in the collections segment.
Challenger: contact only the accounts the Phase 3 uplift model flags as
"persuadable". Both arms are evaluated on a fresh, randomly-split
population — this is the rollout test a risk team would actually run
before retiring the blanket policy.

Usage:
    python scripts/run_ab_test_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from lifecycle.data.synthetic import (  # noqa: E402
    generate_applicants,
    generate_collections_experiment,
    generate_onbook_behavior,
    simulate_collections_policy_outcome,
)
from lifecycle.experimentation.ab_test import assign_arm, evaluate_ab_test  # noqa: E402
from lifecycle.scoring.uplift import assign_uplift_segment, fit_uplift_model  # noqa: E402
from lifecycle.stages.collections import FEATURES  # noqa: E402


def main() -> None:
    applicants = generate_applicants(20_000, seed=42)
    onbook = generate_onbook_behavior(applicants, seed=43)

    # Train the uplift model on a historical randomized-contact experiment.
    history = generate_collections_experiment(onbook, seed=46)
    model = fit_uplift_model(history[FEATURES], history["contacted"], history["self_cure_flag"])
    segment = assign_uplift_segment(model.predict_uplift(onbook[FEATURES]))

    # Split a fresh population into champion/challenger arms for the rollout test.
    arm = assign_arm(onbook.index, challenger_rate=0.5, seed=99)
    contacted_policy = pd.Series(0, index=onbook.index)
    contacted_policy[arm == "champion"] = 1  # champion: contact everyone
    contacted_policy[(arm == "challenger") & (segment == "persuadable")] = 1  # targeted only

    self_cure = simulate_collections_policy_outcome(onbook, contacted_policy, seed=46)
    contact_made = pd.Series(contacted_policy.to_numpy(), index=onbook.index)

    cure_result = evaluate_ab_test(self_cure, arm)
    contact_result = evaluate_ab_test(contact_made, arm)

    print("=" * 60)
    print("CHAMPION/CHALLENGER A/B TEST — blanket vs. uplift-targeted contact")
    print("=" * 60)
    print(f"\nChampion (contact everyone):            n={cure_result.champion_n:,}")
    print(f"Challenger (contact persuadables only):  n={cure_result.challenger_n:,}")

    print(f"\nSelf-cure rate:  champion={cure_result.champion_mean:.1%}  challenger={cure_result.challenger_mean:.1%}")
    print(f"  lift={cure_result.lift:+.1%} ({cure_result.relative_lift:+.1%} relative), p={cure_result.p_value:.4f}, significant={cure_result.significant}")

    print(f"\nContact rate:    champion={contact_result.champion_mean:.1%}  challenger={contact_result.challenger_mean:.1%}")
    print(f"  lift={contact_result.lift:+.1%} ({contact_result.relative_lift:+.1%} relative), p={contact_result.p_value:.4g}, significant={contact_result.significant}")

    champion_efficiency = cure_result.champion_mean / contact_result.champion_mean
    challenger_efficiency = cure_result.challenger_mean / contact_result.challenger_mean
    print(f"\nCures per contact made:  champion={champion_efficiency:.2f}  challenger={challenger_efficiency:.2f}")
    print(
        f"\n-> The challenger gives up {abs(cure_result.relative_lift):.1%} of the cure rate while cutting\n"
        f"   contact volume by {abs(contact_result.relative_lift):.1%} — "
        f"{challenger_efficiency / champion_efficiency:.1f}x more cures per contact made.\n"
        "   This is the actual rollout decision: a small, real cure-rate cost against a\n"
        "   large reduction in collections operating cost."
    )


if __name__ == "__main__":
    main()
