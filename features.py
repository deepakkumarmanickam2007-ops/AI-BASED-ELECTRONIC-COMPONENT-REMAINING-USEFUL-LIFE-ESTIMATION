"""Feature engineering for component aging observations."""
import pandas as pd

RAW_NUMERIC = [
    "cycle", "ambient_temp_c", "case_temp_c", "load_pct", "input_voltage_v",
    "ripple_mv", "vibration_g", "esr_mohm", "capacitance_uf",
    "efficiency_pct", "degradation_index"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Physics-inspired stress indicators (simplified, for educational use).
    out["thermal_stress"] = (out["case_temp_c"] - 40).clip(lower=0)
    out["load_stress"] = (out["load_pct"] / 100.0) ** 2
    out["ripple_stress"] = out["ripple_mv"] / 100.0
    out["age_normalized"] = out["cycle"] / (out.groupby("component_id")["cycle"].transform("max") + 1)
    out["esr_to_cap_ratio"] = out["esr_mohm"] / out["capacitance_uf"].clip(lower=1)
    out["efficiency_loss"] = 100 - out["efficiency_pct"]
    # Sort first to ensure rolling statistics are temporal.
    out = out.sort_values(["component_id", "cycle"]).reset_index(drop=True)
    for col in ["esr_mohm", "capacitance_uf", "efficiency_pct", "case_temp_c", "ripple_mv"]:
        out[f"{col}_rolling_mean3"] = (
            out.groupby("component_id")[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
        )
        out[f"{col}_trend"] = out.groupby("component_id")[col].transform(
            lambda s: s.diff().fillna(0)
        )
    return out

FEATURES = [
    "cycle", "ambient_temp_c", "case_temp_c", "load_pct", "input_voltage_v",
    "ripple_mv", "vibration_g", "esr_mohm", "capacitance_uf", "efficiency_pct",
    "degradation_index", "thermal_stress", "load_stress", "ripple_stress",
    "age_normalized", "esr_to_cap_ratio", "efficiency_loss",
    "esr_mohm_rolling_mean3", "esr_mohm_trend",
    "capacitance_uf_rolling_mean3", "capacitance_uf_trend",
    "efficiency_pct_rolling_mean3", "efficiency_pct_trend",
    "case_temp_c_rolling_mean3", "case_temp_c_trend",
    "ripple_mv_rolling_mean3", "ripple_mv_trend"
]
