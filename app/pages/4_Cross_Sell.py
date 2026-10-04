import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from data_loader import get_nbo_data  # noqa: E402
from theme import OFFER_COLORS  # noqa: E402

st.set_page_config(page_title="Cross-sell — NBFC Lifecycle", layout="wide", page_icon="📊")
st.title("Stage D — Cross-sell (Next-Best-Offer)")
st.caption(
    "For behaviorally-performing customers: ranks a 3-product catalog by expected "
    "value (acceptance propensity × margin × CLV), instead of a single accept/reject call."
)

nbo = get_nbo_data()

col1, col2 = st.columns(2)
col1.metric("Cross-sell segment size", f"{nbo['segment_size']:,}")
col2.metric("Mean estimated CLV", f"₹{nbo['clv'].mean():,.0f}")

st.subheader("Per-product acceptance model quality")
auc_series = pd.Series(nbo["model_aucs"])
fig = px.bar(
    x=auc_series.index,
    y=auc_series.values,
    color=auc_series.index,
    color_discrete_map=OFFER_COLORS,
    labels={"x": "Product", "y": "Train AUC"},
)
fig.update_layout(showlegend=False, yaxis_range=[0.5, 0.8])
st.plotly_chart(fig, width='stretch')

left, right = st.columns(2)
with left:
    ranked_counts = nbo["ranked"]["next_best_offer"].value_counts()
    fig = px.bar(
        x=ranked_counts.index,
        y=ranked_counts.values,
        color=ranked_counts.index,
        color_discrete_map=OFFER_COLORS,
        labels={"x": "Recommended offer", "y": "Customers"},
        title="Next-Best-Offer distribution",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')
with right:
    fig = px.histogram(
        x=nbo["clv"],
        nbins=50,
        labels={"x": "Estimated CLV (₹)"},
        title="Estimated CLV distribution",
        color_discrete_sequence=["#1565C0"],
    )
    st.plotly_chart(fig, width='stretch')

st.subheader("Average profile by recommended offer")
profile_avg = (
    nbo["profile"]
    .groupby("next_best_offer")[["age", "monthly_income", "existing_emi_to_income"]]
    .mean()
    .round(1)
)
st.dataframe(profile_avg, width='stretch')
st.caption(
    "By design, credit card should skew toward younger customers and insurance toward "
    "customers with a heavier EMI burden — ranking by expected value should recover that "
    "segmentation without being told it explicitly."
)

st.divider()
st.subheader("Risk-adjusted top-up-loan offer sizing")
st.caption(
    "For customers recommended the top-up loan: offer amount is a multiple of income, "
    "discounted by the current behavioral PD — riskier customers get a smaller offer."
)
fig = px.histogram(
    x=nbo["offer_amount"],
    nbins=40,
    labels={"x": "Risk-adjusted offer amount (₹)"},
    color_discrete_sequence=["#1565C0"],
)
st.plotly_chart(fig, width='stretch')
