#!/usr/bin/env python
"""Run the full NBFC customer lifecycle end-to-end on synthetic data and print a summary.

Usage:
    python scripts/run_lifecycle_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lifecycle.orchestrator import run_lifecycle  # noqa: E402


def main() -> None:
    result = run_lifecycle(n_applicants=20_000, seed=42)
    journey = result.journey

    print("=" * 60)
    print("NBFC CUSTOMER LIFECYCLE — SUMMARY")
    print("=" * 60)

    print(f"\nApplicants scored:        {len(result.applicants):>8,}")
    print("\nStage 1 — Acquisition decision:")
    print(journey["acquisition_decision"].value_counts().to_string())
    print(f"  Train AUC: {result.acquisition_model.train_auc:.3f}")

    booked = journey[journey["acquisition_decision"] == "approve"]
    print(f"\nStage 2 — Booked & behaviorally scored: {len(booked):>8,}")
    print("Behavioral route:")
    print(booked["behavioral_route"].value_counts().to_string())
    print(f"  Train AUC: {result.behavioral_model.train_auc:.3f}")

    collections_pop = booked[booked["behavioral_route"] == "collections"]
    print(f"\nStage 3a — Collections segment: {len(collections_pop):>8,}")
    print("Strategy assigned:")
    print(collections_pop["final_action"].value_counts().to_string())
    print(f"  Train AUC: {result.collections_model.train_auc:.3f}")

    cross_sell_pop = booked[booked["behavioral_route"] == "cross_sell"]
    print(f"\nStage 3b — Cross-sell segment: {len(cross_sell_pop):>8,}")
    print("Offer tier assigned:")
    print(cross_sell_pop["final_action"].value_counts().to_string())
    print(f"  Train AUC: {result.cross_sell_model.train_auc:.3f}")

    out_path = Path(__file__).resolve().parent.parent / "lifecycle_journey.csv"
    journey.to_csv(out_path, index=False)
    print(f"\nFull per-customer journey written to {out_path}")


if __name__ == "__main__":
    main()
