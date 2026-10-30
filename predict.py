"""Predict RUL for new component measurement rows."""
from pathlib import Path
import joblib
import pandas as pd
from features import engineer_features

def predict_file(input_csv, model_path="outputs/best_rul_model.joblib", output_csv="outputs/new_predictions.csv"):
    bundle = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    # If temporal columns exist, engineer features in order.
    if "component_id" not in df.columns:
        df["component_id"] = "NEW_COMPONENT"
    if "rul_cycles" not in df.columns:
        df["rul_cycles"] = 0
    df = engineer_features(df)
    X = df[bundle["features"]].copy()
    pred = bundle["model"].predict(X).clip(min=0)
    result = df[["component_id", "cycle"]].copy()
    result["predicted_rul_cycles"] = pred.round(1)
    result.to_csv(output_csv, index=False)
    return result

if __name__ == "__main__":
    example = Path("data/component_aging.csv")
    if not Path("outputs/best_rul_model.joblib").exists():
        raise SystemExit("Run python run_project.py first.")
    df = pd.read_csv(example)
    sample = df[df["component_id"] == df["component_id"].iloc[0]].drop(columns=["rul_cycles", "failed"])
    sample.to_csv("outputs/sample_unseen_input.csv", index=False)
    print(predict_file("outputs/sample_unseen_input.csv"))
