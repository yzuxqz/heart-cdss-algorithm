"""
Figure 4.X — Per-dataset ROC curves. One clean figure per dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from heart_cdss.data import read_csv_auto
from heart_cdss.experiment import prepare_dataset
from heart_cdss.models import get_models_and_spaces
from heart_cdss.preprocess import build_preprocessor

CHARTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CHARTS_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"

COLORS: dict[str, str] = {
    "lgbm":   "#4472C4",
    "logreg":  "#5B9BD5",
    "rf":      "#ED7D31",
    "xgb":     "#70AD47",
    "cat":     "#C00000",
}

MODEL_LABEL: dict[str, str] = {
    "logreg": "Logistic Regression",
    "rf":     "Random Forest",
    "xgb":    "XGBoost",
    "lgbm":   "LightGBM",
    "cat":    "CatBoost",
}

LINE_ORDER = ["lgbm", "logreg", "rf", "xgb", "cat"]

DATASETS = {
    "uci_cleveland": {
        "title": "Cleveland (UCI)  —  N=304",
        "csv": "heart_disease_uci.csv",
        "ds_name": "uci_cleveland",
        "target": "num",
        "zoom_xlim": (-0.01, 0.45),
        "zoom_ylim": (0.55, 1.01),
        "inset_loc": "lower right",
    },
    "framingham": {
        "title": "Framingham  —  N=4,240",
        "csv": "framingham.csv",
        "ds_name": "framingham",
        "target": "TenYearCHD",
        "zoom_xlim": (-0.01, 1.01),
        "zoom_ylim": (-0.01, 1.01),
        "inset_loc": "lower right",
    },
    "cardio70k": {
        "title": "Cardio70k  —  N=68,635",
        "csv": "cardio_train.csv",
        "ds_name": "cardio70k",
        "target": "cardio",
        "zoom_xlim": (-0.01, 0.60),
        "zoom_ylim": (0.40, 1.01),
        "inset_loc": "lower right",
    },
}


def load_best_params(ds_code: str) -> dict[str, dict[str, float]]:
    best: dict[str, dict[str, float]] = {}
    result_dir = RESULTS_DIR / ds_code
    for path in sorted(result_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        model = data["model"]
        params = data["best_params"]
        best[model] = {k.replace("model__", ""): v for k, v in params.items()}
    return best


def plot_roc(scores: dict[str, np.ndarray], y_test: np.ndarray,
             ds_code: str, ds_title: str,
             zoom_xlim, zoom_ylim, inset_loc) -> None:
    out_dir = CHARTS_DIR / ds_code
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))

    # 对角线基准
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.7, alpha=0.5)

    for name in LINE_ORDER:
        fpr, tpr, _ = roc_curve(y_test, scores[name])
        auc_val = roc_auc_score(y_test, scores[name])
        ax.plot(fpr, tpr, color=COLORS[name], linewidth=2.4,
                label=f"{MODEL_LABEL[name]}  (AUC = {auc_val:.3f})")

    ax.set_xlim(*zoom_xlim)
    ax.set_ylim(*zoom_ylim)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR) / Sensitivity", fontsize=11)
    ax.set_title(f"ROC Curve — {ds_title}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92, edgecolor="lightgray")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.set_aspect("auto")

    # 全图缩略图
    inset_ax = inset_axes(ax, width="35%", height="35%", loc=inset_loc,
                          bbox_to_anchor=(0, 0.06, 1, 1),
                          bbox_transform=ax.transAxes)
    inset_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.5, alpha=0.4)
    for name in LINE_ORDER:
        fpr, tpr, _ = roc_curve(y_test, scores[name])
        inset_ax.plot(fpr, tpr, color=COLORS[name], linewidth=1.2)
    inset_ax.set_xlim(0, 1); inset_ax.set_ylim(0, 1)
    inset_ax.set_xticks([0, 0.5, 1]); inset_ax.set_yticks([0, 0.5, 1])
    inset_ax.tick_params(labelsize=6)
    inset_ax.set_title("Full view", fontsize=7, fontweight="bold")

    out_path = out_dir / "fig_roc.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    seed = 42

    for ds_code, cfg in DATASETS.items():
        df = read_csv_auto(PROJECT_DIR / cfg["csv"])
        X, y = prepare_dataset(df, cfg["ds_name"], cfg["target"])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y,
        )

        preprocessor = build_preprocessor(X_train)
        best_params = load_best_params(ds_code)
        base_models = get_models_and_spaces(seed)

        scores: dict[str, np.ndarray] = {}
        for name in LINE_ORDER:
            model = base_models[name][0]
            raw_best = best_params.get(name, {})
            pipe_params = {f"model__{k}": v for k, v in raw_best.items()}
            pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
            pipe.set_params(**pipe_params)
            pipe.fit(X_train, y_train)
            scores[name] = pipe.predict_proba(X_test)[:, 1]

        plot_roc(scores, y_test, ds_code, cfg["title"],
                 cfg["zoom_xlim"], cfg["zoom_ylim"], cfg["inset_loc"])

    print("Done. All ROC curves saved.")


if __name__ == "__main__":
    main()
