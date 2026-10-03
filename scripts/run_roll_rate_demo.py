#!/usr/bin/env python
"""Run the behavioral stage's roll-rate matrix and vintage curve analysis.

Usage:
    python scripts/run_roll_rate_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from lifecycle.analytics.roll_rate import compute_transition_matrix, compute_vintage_curve  # noqa: E402
from lifecycle.data.synthetic import generate_dpd_panel  # noqa: E402
from lifecycle.orchestrator import run_lifecycle  # noqa: E402


def main() -> None:
    result = run_lifecycle(n_applicants=20_000, seed=42)
    booked = result.journey[result.journey["acquisition_decision"] == "approve"]

    panel = generate_dpd_panel(booked, risk_col="behavioral_pd", n_months=12, seed=55)

    print("=" * 60)
    print("ROLL-RATE MATRIX (month-over-month DPD bucket transitions)")
    print("=" * 60)
    print(compute_transition_matrix(panel).round(3).to_string())

    score_band = pd.qcut(
        booked.set_index("applicant_id")["acquisition_score"],
        3,
        labels=["low_score", "mid_score", "high_score"],
    )
    vintage = compute_vintage_curve(panel, score_band)

    print("\n" + "=" * 60)
    print("VINTAGE CURVE (cumulative ever-90+ rate by month-on-book, by acquisition score band)")
    print("=" * 60)
    print(vintage.round(3).to_string())
    print(
        "\nLower acquisition score bands should reach a higher NPA rate faster — "
        "this is the usual check that the acquisition cutoff is actually doing its job."
    )


if __name__ == "__main__":
    main()
