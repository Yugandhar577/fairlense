"""FairLens Streamlit landing page. Launch with `streamlit run dashboard/app.py`."""
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
from components.data import load_all
from components.text import DISCLAIMER, METHOD_TEXT

st.set_page_config(page_title="FairLens", page_icon="⚖️", layout="wide")
st.title("FairLens")
st.subheader("A comparative benchmarking framework for fairness evaluation and bias mitigation")
st.info("Use the pages in the sidebar to inspect precomputed experiment results. Opening this dashboard never retrains a model.")
try:
    performance, fairness, config = load_all()
except FileNotFoundError as error:
    st.error(str(error))
    st.code("python run_pipeline.py")
    st.stop()

left, middle, right = st.columns(3)
left.metric("Experiment arms", len(performance))
middle.metric("Models", performance["model"].nunique())
right.metric("Fairness group rows", len(fairness))
st.markdown(DISCLAIMER)
st.markdown(METHOD_TEXT)
