"""
Figure 4.X — Four-metric grouped bar chart per dataset.
每个数据集单独一张图：Accuracy / Precision / Recall / F1-Score
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

CHARTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CHARTS_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"

METRICS = {
    "accuracy":  "Accuracy",
    "precision": "Precision",
    "recall":    "Recall",
    "f1":        "F1-Score",
}

METRIC_COLS = {
    "accuracy":  "test_accuracy",
    "precision": "test_precision",
    "recall":    "test_recall",
    "f1":        "test_f1",
}

COLORS = ["#5B9BD5", "#ED7D31", "#70AD47", "#C00000"]
MODEL_ORDER = ["logreg", "rf", "xgb", "lgbm", "cat"]
MODEL_LABELS = ["LogReg", "RF", "XGBoost", "LightGBM", "CatBoost"]

DATASETS = {
    "uci_cleveland": "Cleveland (UCI) — N=304",
    "framingham":    "Framingham — N=4,240",
    "cardio70k":     "Cardio70k — N=68,635",
}


def plot_four_metrics(ds_code: str, ds_title: str) -> None:
    out_dir = CHARTS_DIR / ds_code
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list((RESULTS_DIR / ds_code).glob("*_summary.csv"))
    if not csv_files:
        print(f"  [skip] {ds_code}: no summary CSV")
        return
    df = pd.read_csv(csv_files[0])

    # 提取数据矩阵 5 models × 4 metrics
    data = np.zeros((len(MODEL_ORDER), len(METRICS)))
    for i, model in enumerate(MODEL_ORDER):
        row = df[df["model"] == model]
        if row.empty:
            data[i, :] = np.nan
            continue
        for j, col in enumerate(METRIC_COLS.values()):
            val = row[col].values[0]
            data[i, j] = round(val, 3) if not np.isnan(val) else np.nan

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.19
    n_metrics = len(METRICS)

    for j, (key, label) in enumerate(METRICS.items()):
        offset = (j - (n_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, data[:, j], width,
                      label=label, color=COLORS[j],
                      edgecolor="white", linewidth=0.4)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                        f"{h:.3f}", ha="center", va="bottom",
                        fontsize=7, rotation=90, color="dimgray")

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Accuracy, Precision, Recall & F1-Score — {ds_title}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))

    plt.tight_layout()
    out_path = out_dir / "fig_four_metrics.png"
    fig.savefig(out_path, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    for code, title in DATASETS.items():
        plot_four_metrics(code, title)
    print("Done.")


if __name__ == "__main__":
    main()
