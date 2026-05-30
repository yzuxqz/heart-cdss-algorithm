"""
Figure 4.X — Local SHAP explanation: 4 figures for two representative cases.
Case A: high-risk (CVD positive) — Waterfall + Force
Case B: low-risk (CVD negative) — Waterfall + Force
Style consistent with global SHAP figures (fig_shap_cardio70k.py).
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
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from heart_cdss.data import read_csv_auto
from heart_cdss.experiment import prepare_dataset
from heart_cdss.models import get_models_and_spaces
from heart_cdss.preprocess import build_preprocessor

CHARTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CHARTS_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results" / "cardio70k"
OUT_DIR = CHARTS_DIR / "cardio70k"

# ── Same constants as fig_shap_cardio70k.py ──
FONT_LABEL = 11
FONT_TITLE = 13

FEATURE_NAMES_CN: dict[str, str] = {
    "age": "Age (days)",
    "gender": "Gender",
    "height": "Height (cm)",
    "weight": "Weight (kg)",
    "ap_hi": "Systolic BP",
    "ap_lo": "Diastolic BP",
    "cholesterol": "Cholesterol",
    "gluc": "Glucose",
    "smoke": "Smoking",
    "alco": "Alcohol",
    "active": "Physical Activity",
}


def _load_best_xgb_params() -> dict[str, float]:
    for path in sorted(RESULTS_DIR.glob("*_xgb.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {k.replace("model__", ""): v for k, v in data["best_params"].items()}
    raise FileNotFoundError("No XGBoost result JSON found")


def _safe_name(col: str) -> str:
    if "__" in col:
        _, feature_val = col.split("__", 1)
    else:
        feature_val = col
    parts = feature_val.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and parts[0] in FEATURE_NAMES_CN:
        label = FEATURE_NAMES_CN[parts[0]]
        return f"{label}={parts[1]}"
    return FEATURE_NAMES_CN.get(feature_val, feature_val)


def main() -> None:
    seed = 42
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = read_csv_auto(PROJECT_DIR / "cardio_train.csv")
    X, y = prepare_dataset(df, "cardio70k", "cardio")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    best_params = _load_best_xgb_params()
    base_models = get_models_and_spaces(seed)
    model = base_models["xgb"][0]
    pipe_params = {f"model__{k}": v for k, v in best_params.items()}
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.set_params(**pipe_params)
    pipe.fit(X_train, y_train)

    X_test_transformed = pipe.named_steps["preprocess"].transform(X_test)
    raw_names = pipe.named_steps["preprocess"].get_feature_names_out()
    feature_names = [_safe_name(str(n)) for n in raw_names]

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.7383).astype(int)

    xgb_model = pipe.named_steps["model"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[1] if len(expected_value) > 1 else expected_value[0])

    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    tp_mask = (y_test_reset == 1) & (y_pred == 1)
    tp_indices = np.where(tp_mask)[0]
    idx_high = tp_indices[np.argmax(y_proba[tp_mask])]

    tn_mask = (y_test_reset == 0) & (y_pred == 0)
    tn_indices = np.where(tn_mask)[0]
    idx_low = tn_indices[np.argmin(y_proba[tn_mask])]

    cases = [
        ("high_risk", idx_high, "Case A: High-Risk CVD Prediction (True Positive)"),
        ("low_risk", idx_low, "Case B: Low-Risk Healthy Prediction (True Negative)"),
    ]

    for slug, idx, title in cases:
        proba = y_proba[idx]
        exp = shap.Explanation(
            values=shap_values[idx],
            base_values=expected_value,
            data=X_test_transformed[idx],
            feature_names=feature_names,
        )

        # ── Waterfall plot ──
        # Same pattern as global beeswarm: create fig/ax first, then shap draws on plt.gca()
        fig, ax = plt.subplots(figsize=(9, 7))
        shap.plots.waterfall(exp, max_display=10, show=False)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
        fig.tight_layout()
        path_wf = OUT_DIR / f"fig_shap_waterfall_{slug}.png"
        fig.savefig(path_wf, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"Saved: {path_wf}")

        # ── Force plot ──
        fig = shap.plots.force(exp, matplotlib=True, show=False, text_rotation=15)
        fig.set_size_inches(16, 2.8)
        fig.tight_layout(pad=0.5)
        path_fp = OUT_DIR / f"fig_shap_force_{slug}.png"
        fig.savefig(path_fp, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"Saved: {path_fp}")

    print("\nDone.")


if __name__ == "__main__":
    main()
