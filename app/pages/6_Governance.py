import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Governance — NBFC Lifecycle", layout="wide", page_icon="📊")
st.title("Model Governance")
st.caption(
    "The model inventory, validation approach, monitoring cadence, and RBI digital-lending "
    "alignment this repo is built to satisfy — rendered here rather than left in a file "
    "nobody opens."
)

governance_path = Path(__file__).resolve().parent.parent.parent / "GOVERNANCE.md"
content = governance_path.read_text()
# Drop the top-level title — the page already has one — so the markdown starts at the
# first real section instead of a duplicate heading.
body = content.split("\n", 2)[-1] if content.startswith("# ") else content
st.markdown(body)
