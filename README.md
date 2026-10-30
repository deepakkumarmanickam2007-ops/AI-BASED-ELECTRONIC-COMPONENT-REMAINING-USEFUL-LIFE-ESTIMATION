# AI-Based Electronic Component Remaining Useful Life (RUL) Estimation

A complete, reproducible machine-learning capstone project that estimates electronic-component degradation and remaining useful life (RUL) from time-series operating measurements.

## Important dataset note
This package includes a **synthetic educational dataset** generated from a documented degradation simulator. It is suitable for demonstrating preprocessing, feature engineering, regression, evaluation, plots, and a dashboard. It is **not real laboratory/field data** and the model's scores must not be presented as validated real-world component-life predictions. For a research-grade deployment, replace it with a suitable public or measured aging dataset and retrain.

## Project objectives
- Generate or load component aging time-series data.
- Engineer rolling, trend, stress, and cumulative-aging features.
- Train and compare Random Forest and Gradient Boosting regressors.
- Evaluate with MAE, RMSE, and R² on held-out component IDs.
- Visualize predicted versus actual RUL and feature importance.
- Explain operating conditions that influence degradation.
- Demonstrate predictions on unseen operating-condition samples.

## Project structure
```text
AI_Electronic_Component_RUL_Project/
├── README.md
├── requirements.txt
├── run_project.py
├── run_dashboard.py
├── src/
│   ├── generate_dataset.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── data/
│   ├── component_aging.csv
│   └── feature_dictionary.md
├── outputs/
│   ├── metrics.csv
│   ├── predictions_test.csv
│   ├── feature_importance.csv
│   ├── predicted_vs_actual.png
│   ├── residuals.png
│   └── feature_importance.png
└── docs/
    ├── project_report.md
    └── viva_questions.md
```

## Requirements
- Python 3.10 or later recommended
- Internet is not required after dependencies are installed

## Setup and run
### Windows
```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_project.py
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pip install -r requirements.txt
python run_project.py
```

The runner regenerates the dataset, trains both models, saves metrics and graphs under `outputs/`, and saves the selected model under `outputs/`.

### Optional interactive dashboard
```bash
streamlit run run_dashboard.py
```
The dashboard loads the trained model and allows a user to explore a sample or enter operating measurements.

## Model design
The prediction target is `rul_cycles`: estimated cycles remaining until the simulated component reaches its end-of-life threshold. Rows from the same component are kept together when splitting train and test data to reduce leakage. Random Forest and Gradient Boosting are compared using MAE, RMSE, and R².

## Interpreting results
- **MAE**: average absolute prediction error, in simulated cycles.
- **RMSE**: square-root of average squared error; larger errors receive more penalty.
- **R²**: variance explained by the model (can be negative on poor predictions).
- Feature importance is a model-level association, not proof of causation.

## Academic integrity / limitations
1. Clearly label the bundled data as synthetic.
2. Report the metrics produced by your local run; do not copy unverified example scores.
3. Do not use this prototype to make safety-critical maintenance decisions.
4. A real deployment needs representative component data, domain validation, uncertainty estimates, and periodic retraining.
