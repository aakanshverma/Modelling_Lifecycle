#!/usr/bin/env python
"""Demonstrate alternative-data lift concentrated in the thin-file segment.

Bureau data (cibil_score etc.) is simulated as noisier/less reliable for
thin-file applicants (little trade history), the real-world reason an
Account-Aggregator-style alternative-data pull (UPI transactions, utility
bill payment) is worth the integration cost. Compares bureau-only vs.
bureau+alt-data acquisition models on both segments.

Usage:
    python scripts/run_altdata_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from lifecycle.data.synthetic import (  # noqa: E402
    degrade_bureau_for_thin_file,
    generate_alt_data_features,
    generate_applicants,
)
from lifecycle.stages import acquisition  # noqa: E402


def main() -> None:
    train_applicants = generate_applicants(20_000, seed=42)
    train = degrade_bureau_for_thin_file(train_applicants, seed=49).merge(
        generate_alt_data_features(train_applicants, seed=48), on="applicant_id"
    )

    holdout_applicants = generate_applicants(20_000, seed=142)
    holdout = degrade_bureau_for_thin_file(holdout_applicants, seed=149).merge(
        generate_alt_data_features(holdout_applicants, seed=148), on="applicant_id"
    )

    print("=" * 60)
    print("ALTERNATIVE DATA — thin-file vs. thick-file acquisition lift")
    print("=" * 60)
    print(f"\nThin-file share of population: {holdout['thin_file'].mean():.1%}")

    bureau_model = acquisition.fit(train)
    altdata_model = acquisition.fit_with_altdata(train)

    for label, mask in [("thin-file", holdout["thin_file"]), ("thick-file", ~holdout["thin_file"])]:
        segment = holdout[mask]
        bureau_pd = bureau_model.predict(segment[acquisition.FEATURES])["acquisition_pd"]
        altdata_pd = altdata_model.predict(segment[acquisition.ALT_FEATURES])["acquisition_altdata_pd"]
        bureau_auc = roc_auc_score(segment["default_flag"], bureau_pd)
        altdata_auc = roc_auc_score(segment["default_flag"], altdata_pd)
        print(
            f"\n{label} (n={len(segment):,}):"
            f"\n  bureau-only AUC      = {bureau_auc:.3f}"
            f"\n  bureau + alt-data AUC = {altdata_auc:.3f}"
            f"\n  lift                  = {altdata_auc - bureau_auc:+.3f}"
        )

    print(
        "\n-> The lift should be concentrated in the thin-file segment, where bureau\n"
        "   history is sparse — that's the real-world case for alternative data.\n"
        "   For thick-file applicants, bureau data already captures most of what\n"
        "   alt data would add, so the lift there should be ~0."
    )


if __name__ == "__main__":
    main()
