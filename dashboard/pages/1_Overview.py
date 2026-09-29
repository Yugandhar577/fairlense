from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dashboard"))
from components.data import load_all
from components.text import DISCLAIMER, METHOD_TEXT

st.set_page_config(page_title="FairLens | Overview", layout="wide")
st.title("Overview")
try:
    performance, fairness, config = load_all()
except FileNotFoundError as error:
    st.error(str(error)); st.stop()
report = config["preprocessing_report"]
st.write("FairLens evaluates predictive performance and several definitions of demographic group disparity on the UCI Adult Income dataset, including gender × age-group intersections.")
st.markdown("""**Methodology**  
Adult Dataset → cleaning and training-only preprocessing → common train/test split → 3 baseline models → performance and fairness measurement → training/validation-only mitigation → final untouched-test evaluation → Results Store → dashboard""")
columns = st.columns(4)
columns[0].metric("Input records", report["input_rows"])
columns[1].metric("Training records", report["training_samples"])
columns[2].metric("Test records", report["test_samples"])
columns[3].metric("Encoded features", report["feature_count"])
st.subheader("Experiment scope")
st.write({"models": config["models"], "sensitive_attributes": config["sensitive_attributes"], "mitigations": config["mitigations"], "threshold_constraint": config["threshold_constraint"], "mitigation_target": "gender"})
st.markdown(DISCLAIMER)
st.markdown(METHOD_TEXT)
