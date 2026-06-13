"""
Figure 4.X — SHAP global explanation for Cardio70k XGBoost model.

Generates two publication-quality figures for the Explainability tab:
  1. Beeswarm (summary)  — per-patient SHAP distribution + feature-value colour coding
  2. Bar (importance)    — mean |SHAP| ranking
Output: charts/global/fig_shap_beeswarm.png  +  charts/global/fig_shap_bar.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from heart_cdss.data import read_csv_auto
from heart_cdss.experiment import prepare_dataset
from heart_cdss.models import get_models_and_spaces
from heart_cdss.preprocess import build_preprocessor

CHARTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CHARTS_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results" / "cardio70k"
OUT_DIR = CHARTS_DIR / "global"
SEED = 42

# ── Clinical feature-name mapping  (same as app.py waterfall) ──
CLINICAL_NAMES: dict[str, str] = {
    "age":           "Age (Years)",
    "gender":        "Gender",
    "height":        "Height (cm)",
    "weight":        "Weight (kg)",
    "ap_hi":         "Systolic Blood Pressure (mmHg)",
    "ap_lo":         "Diastolic Blood Pressure (mmHg)",
    "cholesterol":   "Cholesterol Level",
    "gluc":          "Glucose Level",
    "smoke":         "Smoking Status",
    "alco":          "Alcohol Intake",
    "active":        "Physical Activity",
    "bmi":           "BMI",
    "age_years":     "Age (Years)",
    "pulse_pressure":"Pulse Pressure",
    "map":           "Mean Arterial Pressure",
}


def _load_best_xgb_params() -> dict[str, float]:
    for path in sorted(RESULTS_DIR.glob("*_xgb.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {k.replace("model__", ""): v for k, v in data["best_params"].items()}
    raise FileNotFoundError("No XGBoost result JSON found")


def _safe_name(col: str) -> str:
    """Map one-hot encoded feature names to clinical labels."""
    if "__" in col:
        _, feature_val = col.split("__", 1)
    else:
        feature_val = col

    parts = feature_val.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and parts[0] in CLINICAL_NAMES:
        label = CLINICAL_NAMES[parts[0]]
        return f"{label}={parts[1]}"

    return CLINICAL_NAMES.get(feature_val, feature_val)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load & prepare data ──
    df = read_csv_auto(PROJECT_DIR / "datasets" / "cardio_train.csv")
    X, y = prepare_dataset(df, "cardio70k", "cardio")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y,
    )

    # ── Build pipeline with best XGBoost params ──
    preprocessor = build_preprocessor(X_train)
    best_params = _load_best_xgb_params()
    base_models = get_models_and_spaces(SEED)
    model = base_models["xgb"][0]
    pipe_params = {f"model__{k}": v for k, v in best_params.items()}
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.set_params(**pipe_params)
    pipe.fit(X_train, y_train)

    # ── Preprocess test data & get feature names ──
    X_test_transformed = pipe.named_steps["preprocess"].transform(X_test)
    raw_names = pipe.named_steps["preprocess"].get_feature_names_out()
    feature_names = [_safe_name(str(n)) for n in raw_names]

    # ── SHAP TreeExplainer ──
    xgb_model = pipe.named_steps["model"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    # ═══════════════════════════════════════════════════════════════
    # Figure 1 — Beeswarm (summary) plot
    # ═══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 7))
    shap.summary_plot(
        shap_values, X_test_transformed,
        feature_names=feature_names,
        max_display=12,
        show=False,
    )

    ax.set_xlabel("")                                             # removed default label
    ax.set_title("Feature Impact Distribution — Cardio70k XGBoost",
                 fontsize=14, fontweight="bold", pad=12)

    # Legend explaining the built-in colour bar
    legend_elements = [
        Patch(facecolor="#ec2e4a", label="High feature value"),
        Patch(facecolor="#7d7dff", label="Low feature value"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=8.5, frameon=True, framealpha=0.9,
              edgecolor="#cccccc", title="Feature Value",
              title_fontsize=9)

    fig.tight_layout()
    out_beeswarm = OUT_DIR / "fig_shap_beeswarm.png"
    fig.savefig(out_beeswarm, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_beeswarm}")

    # ═══════════════════════════════════════════════════════════════
    # Figure 2 — Mean |SHAP| bar plot
    # ═══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test_transformed,
        feature_names=feature_names,
        max_display=12,
        plot_type="bar",
        show=False,
    )

    ax.set_xlabel("Mean (|SHAP value|)", fontsize=11)
    ax.set_title("Feature Importance Ranking — Cardio70k XGBoost",
                 fontsize=14, fontweight="bold", pad=12)

    fig.tight_layout()
    out_bar = OUT_DIR / "fig_shap_bar.png"
    fig.savefig(out_bar, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_bar}")

    # ── Console: top-12 SHAP importance ──
    mean_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_shap)[::-1]
    print("\nTop-12 SHAP feature importance:")
    print(f"{'Feature':<45} {'Mean|SHAP|':>12}")
    for idx in order[:12]:
        print(f"{feature_names[idx]:<45} {mean_shap[idx]:>12.6f}")

    print("\nDone. Output directory:", OUT_DIR)


if __name__ == "__main__":
    main()
