"""
生成论文 Chapter 4 所需图表 / Generate charts for thesis Chapter 4.

输出 / Output:
  - charts/fig_target_distribution.png   (三个数据集的目标变量分布)
  - charts/fig_model_f1.png             (所有数据集-模型 F1 对比)
  - charts/fig_model_roc_auc.png        (所有数据集-模型 ROC-AUC 对比)
  - charts/fig_training_time.png        (训练时间对比)
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中 / Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from heart_cdss.data import read_csv_auto
from heart_cdss.experiment import prepare_dataset

CHARTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CHARTS_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"


# ══════════════════════════════════════════════════════════════════════
# 1. 目标变量分布图 / Target Distribution Bar Chart
# ══════════════════════════════════════════════════════════════════════

def fig_target_distribution() -> None:
    datasets = {
        "Cleveland (UCI)": (PROJECT_DIR / "heart_disease_uci.csv", "uci_cleveland", "num"),
        "Framingham": (PROJECT_DIR / "framingham.csv", "framingham", "TenYearCHD"),
        "Cardio70k": (PROJECT_DIR / "cardio_train.csv", "cardio70k", "cardio"),
    }

    records = []
    for name, (csv_path, ds, target) in datasets.items():
        df = read_csv_auto(csv_path)
        _, y = prepare_dataset(df, ds, target)
        pos = int(y.sum())
        neg = int(len(y) - pos)
        records.append({"Dataset": name, "CVD (Positive)": pos, "Healthy (Negative)": neg})

    df_plot = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df_plot))
    width = 0.35

    bars1 = ax.bar(x - width / 2, df_plot["Healthy (Negative)"], width,
                   label="Healthy (Negative)", color="#4CAF50", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, df_plot["CVD (Positive)"], width,
                   label="CVD (Positive)", color="#F44336", edgecolor="white", linewidth=0.5)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(df_plot["Healthy (Negative)"]) * 0.01,
                f"{h:,}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(df_plot["Healthy (Negative)"]) * 0.01,
                f"{h:,}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Dataset"], fontsize=10)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_title("Target Variable Distribution Across Datasets", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(0, max(df_plot["Healthy (Negative)"]) * 1.18)

    # 添加不平衡比例标签
    for i, row in df_plot.iterrows():
        total = row["CVD (Positive)"] + row["Healthy (Negative)"]
        pct = row["CVD (Positive)"] / total * 100
        ax.text(i, row["Healthy (Negative)"] * 1.10,
                f"{pct:.1f}% / {100-pct:.1f}%", ha="center", fontsize=8,
                color="gray", style="italic")

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "fig_target_distribution.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_target_distribution.png")


# ══════════════════════════════════════════════════════════════════════
# 2. 模型 F1 对比图 / Model F1 Comparison
# ══════════════════════════════════════════════════════════════════════

def fig_model_f1() -> None:
    _model_bar_chart(
        metric_col="test_f1",
        metric_label="Test F1 Score",
        filename="fig_model_f1.png",
        title="Model F1 Score Comparison Across Datasets",
    )


# ══════════════════════════════════════════════════════════════════════
# 3. 模型 ROC-AUC 对比图 / Model ROC-AUC Comparison
# ══════════════════════════════════════════════════════════════════════

def fig_model_roc_auc() -> None:
    _model_bar_chart(
        metric_col="test_roc_auc",
        metric_label="Test ROC-AUC",
        filename="fig_model_roc_auc.png",
        title="Model ROC-AUC Comparison Across Datasets",
    )


def _model_bar_chart(metric_col: str, metric_label: str, filename: str, title: str) -> None:
    """通用模型对比柱状图 / Generic model comparison bar chart."""
    dataset_dirs = {
        "Cleveland (UCI)": RESULTS_DIR / "uci_cleveland",
        "Framingham": RESULTS_DIR / "framingham",
        "Cardio70k": RESULTS_DIR / "cardio70k",
    }

    model_order = ["logreg", "rf", "xgb", "lgbm", "cat"]
    model_labels = ["LogReg", "RF", "XGB", "LGBM", "CatBoost"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_order))
    width = 0.22
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, (ds_name, ds_dir) in enumerate(dataset_dirs.items()):
        csv_files = list(ds_dir.glob("*_summary.csv"))
        if not csv_files:
            print(f"  [skip] No summary CSV in {ds_dir}")
            continue
        summary = pd.read_csv(csv_files[0])
        values = []
        for m in model_order:
            row = summary[summary["model"] == m]
            if row.empty:
                values.append(np.nan)
            else:
                values.append(row[metric_col].values[0])
        bars = ax.bar(x + i * width, values, width, label=ds_name, color=colors[i], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / filename, dpi=300)
    plt.close(fig)
    print(f"Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════
# 4. 训练时间对比图 / Training Time Comparison
# ══════════════════════════════════════════════════════════════════════

def fig_training_time() -> None:
    dataset_dirs = {
        "Cleveland (UCI)": RESULTS_DIR / "uci_cleveland",
        "Framingham": RESULTS_DIR / "framingham",
        "Cardio70k": RESULTS_DIR / "cardio70k",
    }

    model_order = ["logreg", "rf", "xgb", "lgbm", "cat"]
    model_labels = ["LogReg", "RF", "XGB", "LGBM", "CatBoost"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_order))
    width = 0.22
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, (ds_name, ds_dir) in enumerate(dataset_dirs.items()):
        csv_files = list(ds_dir.glob("*_summary.csv"))
        if not csv_files:
            continue
        summary = pd.read_csv(csv_files[0])
        values = []
        for m in model_order:
            row = summary[summary["model"] == m]
            if row.empty or "training_time_s" not in row.columns:
                values.append(np.nan)
            else:
                values.append(row["training_time_s"].values[0])
        bars = ax.bar(x + i * width, values, width, label=ds_name, color=colors[i], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        f"{val:.0f}s", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_ylabel("Training Time (seconds)", fontsize=11)
    ax.set_title("Model Training Time Comparison (n_iter=25 × 5-fold CV)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "fig_training_time.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_training_time.png")


# ══════════════════════════════════════════════════════════════════════
# 5. 皮尔逊相关性热力图 / Pearson Correlation Heatmaps
# ══════════════════════════════════════════════════════════════════════

import seaborn as sns

# 临床可读的特征名映射 / Clinically readable feature name mapping
_FEATURE_LABELS: dict[str, dict[str, str]] = {
    "uci_cleveland": {
        "age": "Age", "sex": "Sex", "cp": "Chest Pain Type",
        "trestbps": "Resting BP", "chol": "Serum Cholesterol",
        "fbs": "Fasting BS > 120", "restecg": "Resting ECG",
        "thalch": "Max Heart Rate", "exang": "Exercise Angina",
        "oldpeak": "ST Depression", "slope": "ST Slope",
        "ca": "# Major Vessels (ca)", "thal": "Thalassemia",
        "target": "CVD (Target)",
    },
    "framingham": {
        "male": "Male", "age": "Age", "education": "Education",
        "currentSmoker": "Current Smoker", "cigsPerDay": "Cigarettes/Day",
        "BPMeds": "BP Medication", "prevalentStroke": "Prevalent Stroke",
        "prevalentHyp": "Prevalent Hypertension", "diabetes": "Diabetes",
        "totChol": "Total Cholesterol", "sysBP": "Systolic BP",
        "diaBP": "Diastolic BP", "BMI": "BMI",
        "heartRate": "Heart Rate", "glucose": "Glucose",
        "target": "CVD (Target)",
    },
    "cardio70k": {
        "age": "Age (days)", "gender": "Gender", "height": "Height (cm)",
        "weight": "Weight (kg)", "ap_hi": "Systolic BP",
        "ap_lo": "Diastolic BP", "cholesterol": "Cholesterol",
        "gluc": "Glucose", "smoke": "Smoking",
        "alco": "Alcohol", "active": "Physical Activity",
        "target": "CVD (Target)",
    },
}


def _get_numeric_corr(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """提取数值列并添加目标变量，计算皮尔逊相关矩阵。"""
    X_num = X.select_dtypes(include=[np.number]).copy()
    X_num["target"] = y.values
    return X_num.corr(method="pearson")


def fig_correlation_heatmaps() -> None:
    """每数据集独立一张热力图 / One standalone heatmap per dataset."""
    datasets = [
        ("Cleveland (UCI)", PROJECT_DIR / "heart_disease_uci.csv", "uci_cleveland", "num"),
        ("Framingham", PROJECT_DIR / "framingham.csv", "framingham", "TenYearCHD"),
        ("Cardio70k", PROJECT_DIR / "cardio_train.csv", "cardio70k", "cardio"),
    ]

    for title, csv_path, ds, target_col in datasets:
        df = read_csv_auto(csv_path)
        X, y = prepare_dataset(df, ds, target_col)
        corr = _get_numeric_corr(X, y)

        # 重命名行列
        label_map = _FEATURE_LABELS.get(ds, {})
        new_labels = [label_map.get(c, c) for c in corr.columns]
        corr.index = new_labels
        corr.columns = new_labels

        n_vars = len(corr)
        fig, ax = plt.subplots(figsize=(max(8, n_vars * 0.7), max(7, n_vars * 0.65)),
                               constrained_layout=True)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        cmap = sns.diverging_palette(240, 10, as_cmap=True)

        sns.heatmap(
            corr, mask=mask, cmap=cmap, center=0,
            vmin=-1, vmax=1, square=True,
            linewidths=0.5, linecolor="white",
            annot=True, fmt=".2f",
            annot_kws={"fontsize": 9},
            cbar_kws={"shrink": 0.75, "label": "Pearson r"},
            ax=ax,
        )
        ax.set_title(f"Pearson Correlation Matrix — {title}\n(n={len(X)})",
                     fontsize=13, fontweight="bold")
        ax.tick_params(axis="both", labelsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

        filename = f"fig_correlation_{ds}.png"
        fig.savefig(CHARTS_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Generating charts...")
    fig_target_distribution()
    fig_model_f1()
    fig_model_roc_auc()
    fig_training_time()
    fig_correlation_heatmaps()
    print("Done. All charts saved to charts/")


if __name__ == "__main__":
    main()
