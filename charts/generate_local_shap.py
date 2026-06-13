"""
Generate local SHAP waterfall plots for thesis Section 4.5.2.
Produces 2 clean waterfall figures for:
  - Case A: High-risk CVD positive patient (True Positive)
  - Case B: Low-risk healthy patient (True Negative)
Output: charts/local/fig_shap_waterfall_high_risk.png
        charts/local/fig_shap_waterfall_low_risk.png
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

# ── Paths ──
CHARTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CHARTS_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results" / "cardio70k"
OUT_DIR = CHARTS_DIR / "local"
SEED = 42
YOUDEN_THRESHOLD = 0.7383

# ── Chinese display names ──
FEATURE_CN: dict[str, str] = {
    "age":        "Age",
    "gender":     "Gender",
    "height":     "Height",
    "weight":     "Weight",
    "ap_hi":      "Systolic BP",
    "ap_lo":      "Diastolic BP",
    "cholesterol":"Cholesterol",
    "gluc":       "Glucose",
    "smoke":      "Smoking",
    "alco":       "Alcohol intake",
    "active":     "Physical activity",
}


def rename_feature(col: str) -> str:
    """Convert one-hot encoded column names to human-readable Chinese labels."""
    if "__" in col:
        # e.g. "gender__1" → "Gender=Male" or "gender__2"
        prefix, val = col.split("__", 1)
        cn = FEATURE_CN.get(prefix, prefix)
        return f"{cn}={val}"
    return FEATURE_CN.get(col, col)


def load_best_xgb_params() -> dict:
    for path in sorted(RESULTS_DIR.glob("*_xgb.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {k.replace("model__", ""): v for k, v in data["best_params"].items()}
    raise FileNotFoundError("No XGBoost result JSON found")


def format_clinical_profile(row: pd.Series, proba: float) -> dict:
    """Extract key clinical values from a raw dataframe row."""
    return {
        "age_years":    round(float(row.get("age", 0)) / 365.25, 1),
        "gender":       "Male" if int(row.get("gender", 1)) == 1 else "Female",
        "height_cm":    int(row.get("height", 0)),
        "weight_kg":    float(row.get("weight", 0)),
        "ap_hi":        int(row.get("ap_hi", 0)),
        "ap_lo":        int(row.get("ap_lo", 0)),
        "cholesterol":  int(row.get("cholesterol", 0)),
        "gluc":         int(row.get("gluc", 0)),
        "smoke":        "Yes" if int(row.get("smoke", 0)) == 1 else "No",
        "alco":         "Yes" if int(row.get("alco", 0)) == 1 else "No",
        "active":       "Yes" if int(row.get("active", 0)) == 1 else "No",
        "predicted_risk": round(proba * 100, 1),
    }


def describe_case(profile: dict, label: str) -> str:
    """Clinical narrative for a patient."""
    risk = "high" if profile["predicted_risk"] > 50 else "low"
    actual = "CVD-positive" if label == "high_risk" else "healthy"
    return (
        f"Patient Profile: {profile['age_years']}-year-old {profile['gender']}, "
        f"BP {profile['ap_hi']}/{profile['ap_lo']} mmHg, "
        f"Cholesterol={profile['cholesterol']}, Glucose={profile['gluc']}, "
        f"BMI≈{round(profile['weight_kg']/((profile['height_cm']/100)**2), 1)}, "
        f"Smoker={'Yes' if profile['smoke']=='Yes' else 'No'}"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load & prepare data ──
    df = read_csv_auto(PROJECT_DIR / "datasets" / "cardio_train.csv")
    X, y = prepare_dataset(df, "cardio70k", "cardio")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y,
    )

    # ── Train best XGBoost ──
    preprocessor = build_preprocessor(X_train)
    best_params = load_best_xgb_params()
    base = get_models_and_spaces(SEED)
    pipe = Pipeline([("preprocess", preprocessor), ("model", base["xgb"][0])])
    pipe.set_params(**{f"model__{k}": v for k, v in best_params.items()})
    pipe.fit(X_train, y_train)

    # ── SHAP explainer ──
    X_test_t = pipe.named_steps["preprocess"].transform(X_test)
    raw_names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    feature_names = [rename_feature(n) for n in raw_names]

    xgb_model = pipe.named_steps["model"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_test_t)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    base_value = float(explainer.expected_value) if not isinstance(explainer.expected_value, (list, np.ndarray)) \
                 else float(explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0])

    # ── Predictions ──
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= YOUDEN_THRESHOLD).astype(int)

    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    # ── Select cases ──
    # High-risk: true positive with highest predicted probability
    tp_mask = (y_test_reset == 1) & (y_pred == 1)
    tp_idx = int(np.where(tp_mask)[0][np.argmax(y_proba[tp_mask])])

    # Low-risk: true negative with lowest predicted probability
    tn_mask = (y_test_reset == 0) & (y_pred == 0)
    tn_idx = int(np.where(tn_mask)[0][np.argmin(y_proba[tn_mask])])

    cases = [
        ("high_risk", tp_idx,   "Case A: High-Risk CVD Prediction (True Positive)"),
        ("low_risk",  tn_idx,   "Case B: Low-Risk Healthy Prediction (True Negative)"),
    ]

    for slug, idx, title in cases:
        proba = float(y_proba[idx])
        profile = format_clinical_profile(X_test_reset.iloc[idx], proba)
        narrative = describe_case(profile, slug)

        # Build SHAP Explanation
        exp = shap.Explanation(
            values=shap_vals[idx],
            base_values=base_value,
            data=X_test_t[idx],
            feature_names=feature_names,
        )

        # ── Waterfall plot ──
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.waterfall(exp, max_display=10, show=False)

        # ── Academic styling: remove SHAP x-label, add legend ──
        ax.set_xlabel("")

        # Extract bar colors for accurate legend
        from matplotlib.patches import Patch
        red_color = "#ff0051"
        blue_color = "#008bfb"
        for patch in ax.patches:
            fc = patch.get_facecolor()
            if len(fc) >= 3:
                if fc[0] > 0.5 and fc[2] < 0.3:
                    red_color = fc
                elif fc[2] > 0.5 and fc[0] < 0.3:
                    blue_color = fc
        legend_elements = [
            Patch(facecolor=red_color, label="Increases CVD risk  (positive SHAP)"),
            Patch(facecolor=blue_color, label="Decreases CVD risk  (negative SHAP)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
                  frameon=True, framealpha=0.9, edgecolor="#cccccc")

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

        # Add clinical narrative text box
        actual_label = "CVD (diseased)" if slug == "high_risk" else "Healthy (no CVD)"
        info_text = (
            f"{profile['age_years']} y/o {profile['gender']} | "
            f"BP: {profile['ap_hi']}/{profile['ap_lo']} mmHg | "
            f"Chol: {profile['cholesterol']} | "
            f"Glucose: {profile['gluc']} | "
            f"Smoker: {profile['smoke']}\n"
            f"Actual: {actual_label} | "
            f"Predicted risk: {profile['predicted_risk']}%"
        )
        fig.text(0.5, -0.02, info_text, ha="center", fontsize=9,
                 style="italic", color="dimgray",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", alpha=0.8))

        fig.tight_layout(rect=[0, 0.05, 1, 1])
        path = OUT_DIR / f"fig_shap_waterfall_{slug}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"Saved: {path}")
        print(f"  {narrative}")
        print()

    print("Done. Output directory:", OUT_DIR)


if __name__ == "__main__":
    main()
