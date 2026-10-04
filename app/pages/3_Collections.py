import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from data_loader import get_collections_propensity_data, get_uplift_and_ab_test  # noqa: E402
from theme import ARM_COLORS, STRATEGY_COLORS, UPLIFT_SEGMENT_COLORS  # noqa: E402

st.set_page_config(page_title="Collections — NBFC Lifecycle", layout="wide", page_icon="📊")
st.title("Stage C — Collections")
st.caption(
    "For behaviorally-underperforming customers: self-cure propensity decides the "
    "collections strategy tier; uplift modeling decides who's actually worth contacting."
)

propensity = get_collections_propensity_data()
col1, col2 = st.columns(2)
col1.metric("Collections segment size", f"{propensity['segment_size']:,}")
col2.metric("Self-cure model train AUC", f"{propensity['train_auc']:.3f}")

left, right = st.columns(2)
with left:
    decile = propensity["decile_actual_cure_rate"].sort_index()
    fig = px.bar(
        x=decile.index,
        y=decile.values,
        labels={"x": "Collections score decile (0=worst)", "y": "Actual self-cure rate"},
        title="Decile validation: actual self-cure rate rises with score decile",
        color_discrete_sequence=["#1565C0"],
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, width='stretch')
with right:
    strategy_counts = propensity["strategy_counts"]
    fig = px.bar(
        x=strategy_counts.index,
        y=strategy_counts.values,
        color=strategy_counts.index,
        color_discrete_map=STRATEGY_COLORS,
        labels={"x": "Strategy tier", "y": "Accounts"},
        title="Collections strategy assignment (by self-cure decile)",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')

st.divider()
st.subheader("Uplift modeling: who's actually worth contacting")
st.caption(
    "Self-cure propensity alone can't tell a 'sure thing' (cures regardless) from a "
    "'persuadable' (cures *because* you contacted them) — a T-learner uplift model can."
)

uplift_data = get_uplift_and_ab_test()
left, right = st.columns(2)
with left:
    decile_means = uplift_data["decile_means"].sort_index()
    fig = px.bar(
        x=decile_means.index,
        y=decile_means.values,
        labels={"x": "Predicted-uplift decile", "y": "True uplift (ground truth)"},
        title="True uplift rises with predicted-uplift decile",
        color_discrete_sequence=["#1565C0"],
    )
    fig.update_layout(yaxis_tickformat=".1%")
    st.plotly_chart(fig, width='stretch')
with right:
    segment_means = uplift_data["segment_means"]
    fig = px.bar(
        x=segment_means.index,
        y=segment_means.values,
        color=segment_means.index,
        color_discrete_map=UPLIFT_SEGMENT_COLORS,
        labels={"x": "Uplift segment", "y": "Average true uplift"},
        title="Persuadable vs. sleeping-dog: average true uplift",
    )
    fig.add_hline(y=0, line_color="#424242", line_width=1)
    fig.update_layout(showlegend=False, yaxis_tickformat=".1%")
    st.plotly_chart(fig, width='stretch')

segment_counts = uplift_data["segment_counts"]
st.caption(
    "Segment sizes — "
    + ", ".join(f"**{label}**: {count:,}" for label, count in segment_counts.items())
)

st.divider()
st.subheader("Champion/challenger rollout: blanket contact vs. uplift-targeted contact")
cure_result = uplift_data["cure_result"]
contact_result = uplift_data["contact_result"]

m1, m2, m3 = st.columns(3)
m1.metric(
    "Self-cure rate",
    f"{cure_result.challenger_mean:.1%}",
    f"{cure_result.lift:+.1%} vs. champion",
)
m2.metric(
    "Contact rate",
    f"{contact_result.challenger_mean:.1%}",
    f"{contact_result.lift:+.1%} vs. champion",
)
efficiency_champion = cure_result.champion_mean / contact_result.champion_mean
efficiency_challenger = cure_result.challenger_mean / contact_result.challenger_mean
m3.metric("Cures per contact made", f"{efficiency_challenger:.2f}", f"vs. {efficiency_champion:.2f} champion")

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=["Self-cure rate", "Contact rate"],
        y=[cure_result.champion_mean, contact_result.champion_mean],
        name="champion (contact everyone)",
        marker_color=ARM_COLORS["champion"],
    )
)
fig.add_trace(
    go.Bar(
        x=["Self-cure rate", "Contact rate"],
        y=[cure_result.challenger_mean, contact_result.challenger_mean],
        name="challenger (targeted contact)",
        marker_color=ARM_COLORS["challenger"],
    )
)
fig.update_layout(barmode="group", yaxis_tickformat=".0%", title="Randomized rollout test (fresh split)")
st.plotly_chart(fig, width='stretch')

st.caption(
    f"The challenger gives up {abs(cure_result.relative_lift):.1%} of the cure rate (p={cure_result.p_value:.3f}) "
    f"while cutting contact volume by {abs(contact_result.relative_lift):.1%} (p={contact_result.p_value:.2g}) — "
    f"{efficiency_challenger / efficiency_champion:.1f}x more cures per contact made."
)
