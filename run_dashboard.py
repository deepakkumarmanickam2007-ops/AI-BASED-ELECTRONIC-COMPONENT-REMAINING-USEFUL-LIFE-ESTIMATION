"""Optional Streamlit dashboard for the trained RUL model."""
from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/"src"))
from features import engineer_features

st.set_page_config(page_title="Electronic Component RUL Estimator", page_icon="🔧", layout="wide")
st.title("AI-Based Electronic Component Remaining Useful Life Estimation")
st.caption("Educational prototype using a synthetic aging dataset — not validated for real hardware.")
model_path = ROOT/"outputs"/"best_rul_model.joblib"
data_path = ROOT/"data"/"component_aging.csv"

if not model_path.exists():
    st.warning("Model not found. Run `python run_project.py` in the project folder first.")
    st.stop()

bundle = joblib.load(model_path)
st.write(f"**Loaded model:** {bundle['model_name']}")
tab1, tab2 = st.tabs(["Explore dataset", "Estimate RUL"])
with tab1:
    if data_path.exists():
        df = pd.read_csv(data_path)
        a, b, c = st.columns(3)
        a.metric("Observations", f"{len(df):,}")
        b.metric("Components", f"{df.component_id.nunique():,}")
        c.metric("Dataset type", "Synthetic")
        st.dataframe(df.head(100), use_container_width=True)
        st.line_chart(df.groupby("cycle")["degradation_index"].mean())
with tab2:
    st.write("Enter one current observation. Rolling/trend features are initialized from this observation, so predictions are more reliable when a real input file includes prior time-series measurements.")
    with st.form("predict_form"):
        component_id = st.text_input("Component ID", "NEW_001")
        cycle = st.number_input("Age / cycle", min_value=0, value=400, step=25)
        ambient = st.number_input("Ambient temperature (°C)", value=27.0)
        case = st.number_input("Case temperature (°C)", value=52.0)
        load = st.slider("Load (%)", 0, 100, 65)
        voltage = st.number_input("Input voltage (V)", value=12.0)
        ripple = st.number_input("Ripple (mV)", min_value=0.0, value=45.0)
        vibration = st.number_input("Vibration (g)", min_value=0.0, value=0.2)
        esr = st.number_input("ESR (mΩ)", min_value=0.1, value=30.0)
        cap = st.number_input("Capacitance (µF)", min_value=1.0, value=950.0)
        efficiency = st.number_input("Efficiency (%)", min_value=0.0, max_value=100.0, value=93.0)
        degradation = st.number_input("Estimated degradation index", min_value=0.0, value=0.4, step=0.05)
        submitted = st.form_submit_button("Estimate remaining useful life")
    if submitted:
        one = pd.DataFrame([{
            "component_id": component_id, "cycle": cycle, "ambient_temp_c": ambient,
            "case_temp_c": case, "load_pct": load, "input_voltage_v": voltage,
            "ripple_mv": ripple, "vibration_g": vibration, "esr_mohm": esr,
            "capacitance_uf": cap, "efficiency_pct": efficiency,
            "degradation_index": degradation, "failed": 0, "rul_cycles": 0,
            "dataset_type": "user_input"
        }])
        engineered = engineer_features(one)
        prediction = max(0, float(bundle["model"].predict(engineered[bundle["features"]])[0]))
        st.metric("Estimated RUL", f"{prediction:,.0f} simulated cycles")
        st.info("This is a model estimate from synthetic training data, not a maintenance recommendation.")
