"""Build Cardio70k XGBoost artifact for Streamlit deployment."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from heart_cdss.data import read_csv_auto
from heart_cdss.experiment import prepare_dataset
from heart_cdss.models import get_models_and_spaces
from heart_cdss.persist import save_joblib, save_json
from heart_cdss.preprocess import build_preprocessor, normalize_bool_like_columns

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "cardio70k"
ARTIFACTS_DIR = BASE / "artifacts" / "cardio70k"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load best XGBoost params from experiment ──
xgb_json = sorted(RESULTS_DIR.glob("*_xgb.json"))[-1]
with open(xgb_json, encoding="utf-8") as f:
    data = json.load(f)
best_params = {k.replace("model__", ""): v for k, v in data["best_params"].items()}
print(f"Best params: {best_params}")

# ── Load full dataset ──
df = read_csv_auto(BASE / "cardio_train.csv")
X, y = prepare_dataset(df, "cardio70k", "cardio")
print(f"Full dataset: {X.shape}")

# ── Build & fit pipeline ──
preprocessor = build_preprocessor(X)
base_models = get_models_and_spaces(42)
model = base_models["xgb"][0]
model.set_params(**best_params)
pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
pipe.fit(X, y)
print("Pipeline fitted.")

# ── Save artifacts ──
save_joblib(pipe, ARTIFACTS_DIR / "model.joblib")

# Schema
df_X = normalize_bool_like_columns(X)
categorical_cols = df_X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numeric_cols = [c for c in df_X.columns if c not in categorical_cols]
for col in list(numeric_cols):
    s = df_X[col]
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= 10:
        numeric_cols.remove(col)
        categorical_cols.append(col)

schema = {"columns": []}
for col in df_X.columns:
    if col in categorical_cols:
        cats = sorted([x for x in df_X[col].dropna().astype(str).unique().tolist()])
        schema["columns"].append({"name": col, "type": "categorical", "categories": cats})
    else:
        s = pd.to_numeric(df_X[col], errors="coerce")
        schema["columns"].append({
            "name": col, "type": "numeric",
            "min": None if s.dropna().empty else float(s.min()),
            "max": None if s.dropna().empty else float(s.max()),
        })
save_json(schema, ARTIFACTS_DIR / "schema.json")

# Meta
best_row = pd.read_csv(sorted(RESULTS_DIR.glob("*_summary.csv"))[-1])
best_row = best_row[best_row["model"] == "xgb"].iloc[0].to_dict()
meta = {
    "dataset": "cardio70k",
    "target": "cardio",
    "best_model": "xgb",
    "best_run_id": data["run_id"],
    "best_params": best_params,
    "best_row": {k: v for k, v in best_row.items() if not isinstance(v, (pd.Timestamp,))},
}
save_json(meta, ARTIFACTS_DIR / "meta.json")

print(f"Done. Artifacts saved to {ARTIFACTS_DIR}")
