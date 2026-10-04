import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from data_loader import get_lifecycle_result, get_roll_rate_data  # noqa: E402
from theme import ROUTE_COLORS, SEQUENTIAL_SCALE  # noqa: E402

st.set_page_config(page_title="Behavioral — NBFC Lifecycle", layout="wide", page_icon="📊")
st.title("Stage B — Behavioral")
st.caption(
    "Once a customer has spent time on the books, re-scores them on actual repayment "
    "behavior blended with a bureau refresh. This score decides the fork: Collections or Cross-sell."
)

result = get_lifecycle_result()
booked = result.journey[result.journey["acquisition_decision"] == "approve"]

col1, col2, col3 = st.columns(3)
col1.metric("Booked population", f"{len(booked):,}")
col2.metric("Routed to Collections", f"{(booked['behavioral_route'] == 'collections').mean():.1%}")
col3.metric("Routed to Cross-sell", f"{(booked['behavioral_route'] == 'cross_sell').mean():.1%}")

left, right = st.columns([2, 1])
with left:
    fig = px.histogram(
        booked,
        x="behavioral_score",
        color="behavioral_route",
        color_discrete_map=ROUTE_COLORS,
        nbins=60,
        barmode="stack",
        title="Behavioral score distribution by route",
    )
    st.plotly_chart(fig, width='stretch')
with right:
    route_counts = booked["behavioral_route"].value_counts()
    fig = px.pie(
        values=route_counts.values,
        names=route_counts.index,
        color=route_counts.index,
        color_discrete_map=ROUTE_COLORS,
        title="Routing split",
        hole=0.5,
    )
    st.plotly_chart(fig, width='stretch')

st.divider()
st.subheader("Roll-rate matrix — month-over-month DPD bucket transitions")
st.caption(
    "Simulated via a severity-blended Markov chain: DPD can only climb one bucket per "
    "month, but curing can jump back several at once (paying off the full overdue amount)."
)
roll_rate = get_roll_rate_data()
tm = roll_rate["transition_matrix"]
fig = go.Figure(
    go.Heatmap(
        z=tm.values,
        x=list(tm.columns),
        y=list(tm.index),
        colorscale=SEQUENTIAL_SCALE,
        text=[[f"{v:.1%}" for v in row] for row in tm.values],
        texttemplate="%{text}",
        hovertemplate="from %{y} → %{x}: %{z:.1%}<extra></extra>",
    )
)
fig.update_layout(
    xaxis_title="next month's bucket",
    yaxis_title="this month's bucket",
    yaxis_autorange="reversed",
    height=420,
)
st.plotly_chart(fig, width='stretch')

st.subheader("Vintage curve — cumulative ever-90+ rate by acquisition score band")
st.caption(
    "The standard check that the acquisition cutoff is doing its job: a lower score "
    "band should reach NPA faster than a higher one."
)
vintage = roll_rate["vintage_curve"]
fig = go.Figure()
band_colors = {"low_score": "#C62828", "mid_score": "#F9A825", "high_score": "#2E7D32"}
for band in vintage.columns:
    fig.add_trace(
        go.Scatter(
            x=vintage.index,
            y=vintage[band],
            mode="lines+markers",
            name=band,
            line=dict(color=band_colors.get(band), width=2),
        )
    )
fig.update_layout(
    xaxis_title="month on book",
    yaxis_title="cumulative ever-90+ rate",
    yaxis_tickformat=".0%",
    height=420,
)
st.plotly_chart(fig, width='stretch')
