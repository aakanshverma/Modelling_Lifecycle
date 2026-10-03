#!/usr/bin/env python
"""Demonstrate PSI catching acquisition score drift after a population shift.

Scores a baseline applicant population, then a "three months later"
population whose credit utilization and income have shifted the way they
would in a mild economic downturn, and reports whether PSI flags it.

Usage:
    python scripts/run_psi_monitoring_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lifecycle.data.synthetic import generate_applicants  # noqa: E402
from lifecycle.monitoring.psi import population_stability_index  # noqa: E402
from lifecycle.stages import acquisition  # noqa: E402


def simulate_downturn(applicants):
    """Mild downturn: utilization creeps up, income growth stalls relative to baseline."""
    drifted = applicants.copy()
    drifted["credit_utilization"] = (drifted["credit_utilization"] * 1.25).clip(0, 1)
    drifted["existing_emi_to_income"] = (drifted["existing_emi_to_income"] * 1.3).clip(0, 1)
    drifted["dpd_30_plus_last_6m"] = drifted["dpd_30_plus_last_6m"] + 1
    return drifted


def main() -> None:
    baseline = generate_applicants(20_000, seed=42)
    model = acquisition.fit(baseline)
    baseline_scored = acquisition.score(model, baseline)

    current_month = generate_applicants(20_000, seed=142)  # same distribution, no drift
    downturn = simulate_downturn(generate_applicants(20_000, seed=242))  # shifted distribution

    print("=" * 60)
    print("PSI MONITORING — acquisition score")
    print("=" * 60)

    for label, population in [("no drift (fresh same-distribution cohort)", current_month),
                               ("after simulated downturn", downturn)]:
        scored = acquisition.score(model, population)
        result = population_stability_index(
            baseline_scored["acquisition_score"], scored["acquisition_score"]
        )
        print(f"\n{label}:")
        print(f"  PSI = {result.psi:.4f}  ->  {result.severity}")
        print(result.detail.round(4).to_string())


if __name__ == "__main__":
    main()
