"""Train and evaluate RUL regression models with component-level holdout."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from features import engineer_features, FEATURES

def train_project(data_path="data/component_aging.csv", output_dir="outputs", seed=42):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    df = engineer_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True))
    y = df["rul_cycles"].astype(float)
    groups = df["component_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=seed)
    train_idx, test_idx = next(splitter.split(df[FEATURES], y, groups))
    X_train, X_test = df.iloc[train_idx][FEATURES], df.iloc[test_idx][FEATURES]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=18, min_samples_leaf=2,
            random_state=seed, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=220, learning_rate=0.045, max_depth=2,
            loss="huber", random_state=seed
        )
    }
    rows, predictions, fitted = [], {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_test), 0, None)
        fitted[name] = model
        rows.append({
            "model": name,
            "MAE_cycles": mean_absolute_error(y_test, pred),
            "RMSE_cycles": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2": r2_score(y_test, pred),
            "train_rows": len(train_idx), "test_rows": len(test_idx),
            "heldout_components": int(df.iloc[test_idx]["component_id"].nunique())
        })
        predictions[name] = pred

    metrics = pd.DataFrame(rows).sort_values("MAE_cycles")
    metrics.to_csv(outdir/"metrics.csv", index=False)
    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted[best_name]
    joblib.dump({"model": best_model, "features": FEATURES, "model_name": best_name},
                outdir/"best_rul_model.joblib")

    result = df.iloc[test_idx][["component_id", "cycle", "rul_cycles", "degradation_index"]].copy()
    result["predicted_rul_cycles"] = predictions[best_name]
    result["error_cycles"] = result["predicted_rul_cycles"] - result["rul_cycles"]
    result.to_csv(outdir/"predictions_test.csv", index=False)

    # Plot predicted vs actual.
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions[best_name], alpha=0.45)
    lim = max(float(y_test.max()), float(np.max(predictions[best_name])))
    plt.plot([0, lim], [0, lim], linestyle="--")
    plt.xlabel("Actual RUL (simulated cycles)")
    plt.ylabel("Predicted RUL (simulated cycles)")
    plt.title(f"Predicted vs Actual RUL — {best_name}")
    plt.tight_layout()
    plt.savefig(outdir/"predicted_vs_actual.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(result["error_cycles"], bins=30, edgecolor="black")
    plt.axvline(0, linestyle="--")
    plt.xlabel("Prediction error (predicted − actual cycles)")
    plt.ylabel("Number of observations")
    plt.title("RUL prediction residuals")
    plt.tight_layout()
    plt.savefig(outdir/"residuals.png", dpi=160)
    plt.close()

    imp = pd.DataFrame({"feature": FEATURES, "importance": best_model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(outdir/"feature_importance.csv", index=False)
    top = imp.head(15).sort_values("importance")
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Feature importance")
    plt.title(f"Top 15 features — {best_name}")
    plt.tight_layout()
    plt.savefig(outdir/"feature_importance.png", dpi=160)
    plt.close()

    metadata = {
        "best_model": best_name,
        "feature_count": len(FEATURES),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_components": int(df.iloc[train_idx]["component_id"].nunique()),
        "test_components": int(df.iloc[test_idx]["component_id"].nunique()),
        "target": "rul_cycles",
        "dataset_provenance": "synthetic_simulation"
    }
    (outdir/"run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("\nEvaluation metrics (held-out component IDs):")
    print(metrics.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Outputs saved to: {outdir.resolve()}")
    return metrics

if __name__ == "__main__":
    train_project()
