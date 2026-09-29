from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dashboard"))
from components.charts import bar_chart
from components.data import load_all
from components.text import DISCLAIMER

st.set_page_config(page_title="FairLens | Performance", layout="wide")
st.title("Model Performance")
try:
    performance, _, _ = load_all()
except FileNotFoundError as error:
    st.error(str(error)); st.stop()
model = st.selectbox("Model", performance["model"].unique())
state = st.selectbox("Experiment state", performance["mitigation_state"].unique())
row = performance[(performance.model == model) & (performance.mitigation_state == state)].iloc[0]
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
columns = st.columns(len(metrics))
for column, metric in zip(columns, metrics):
    column.metric(metric.replace("_", " ").upper(), f"{row[metric]:.3f}")
st.pyplot(bar_chart(performance[performance.model == model].melt(id_vars=["model", "mitigation_state"], value_vars=metrics),
                    "variable", "value", "mitigation_state", f"{model}: performance by experiment state"), clear_figure=True)
st.markdown(DISCLAIMER)
