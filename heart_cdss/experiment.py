from __future__ import annotations

"""
实验管理模块 / Experiment Management Module

中文：
- 协调数据准备、模型训练、超参数调优和评估的完整流程
- 支持多种心脏病数据集（UCI, Framingham, Cardio70k）
- 自动保存实验结果和 SHAP 解释图

English:
- Orchestrates the full pipeline: data prep, training, tuning, and evaluation
- Supports multiple heart disease datasets (UCI, Framingham, Cardio70k)
- Automatically persists results and SHAP explanation outputs
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from heart_cdss.data import read_csv_auto
from heart_cdss.explain import generate_shap_outputs
from heart_cdss.metrics import evaluate, predict_proba_or_score
from heart_cdss.models import get_models_and_spaces
from heart_cdss.preprocess import build_preprocessor, normalize_bool_like_columns


def infer_target_column(columns: list[str]) -> str:
    """
    自动推断目标列名 / Infer target column name automatically.

    中文：
    - 根据预定义的候选名单（target, TenYearCHD 等）匹配数据列
    - 若无法匹配则抛出异常

    English:
    - Matches columns against predefined candidates (target, TenYearCHD, etc.)
    - Raises ValueError if no match is found
    """
    candidates = ["target", "TenYearCHD", "cardio", "num"]
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(
        f"无法自动推断目标列。可用列 / Unable to infer target. Available: {columns[:30]}{'...' if len(columns) > 30 else ''}"
    )


def prepare_dataset(
    df: pd.DataFrame, dataset_name: str, target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    数据集清洗与准备 / Clean and prepare dataset for training.

    中文：
    - 标准化布尔字段
    - 处理特定数据集（如 UCI Cleveland）的子集过滤和目标值二值化
    - 移除 ID 等非特征列

    English:
    - Normalizes boolean-like columns
    - Handles subset filtering and target binarization for specific datasets (e.g., UCI Cleveland)
    - Drops non-feature columns like IDs
    """
    df = normalize_bool_like_columns(df)

    # 特定处理：UCI 数据集仅保留 Cleveland 子集 / Specific handling: Keep only Cleveland subset for UCI
    if dataset_name in {"uci", "uci_cleveland"} and "dataset" in df.columns:
        df = df[df["dataset"].astype(str).str.strip().str.lower() == "cleveland"].copy()

    if target_col not in df.columns:
        raise ValueError(f"目标列不存在 / Target column missing: {target_col}")

    y_raw = df[target_col]

    # 目标值二值化 / Binarize target values
    if dataset_name in {"uci", "uci_cleveland"} and target_col == "num":
        y = (pd.to_numeric(y_raw, errors="coerce").fillna(0) > 0).astype(int)
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        if y.isna().any():
            raise ValueError(f"目标列 {target_col} 存在无法解析为数值的值 / Target contains non-numeric values")
        y = y.astype(int)

    # 移除标识符列 / Drop identifier columns
    drop_cols = [target_col]
    for c in ["id", "ID", "patient_id", "PatientID"]:
        if c in df.columns:
            drop_cols.append(c)

    X = df.drop(columns=drop_cols)
    return X, y


@dataclass(frozen=True)
class RunArgs:
    """
    实验运行参数 / Arguments for running an experiment.
    """
    dataset: str
    csv_path: Path
    target: str | None
    test_size: float
    seed: int
    n_iter: int
    cv_folds: int
    shap: bool
    shap_background: int
    shap_samples: int
    shap_local_index: int


def run_experiment(args: RunArgs) -> Path:
    """
    执行完整的模型实验流程 / Execute the full model experiment pipeline.

    中文：
    - 加载数据并拆分训练/测试集
    - 对所有可用模型进行随机搜索超参数调优
    - 计算评估指标并生成 SHAP 解释图
    - 将详细结果保存为 JSON，汇总结果保存为 CSV

    English:
    - Loads data and splits into train/test sets
    - Performs hyperparameter tuning via RandomizedSearchCV for all models
    - Computes evaluation metrics and generates SHAP explanations
    - Saves detailed results to JSON and summary to CSV
    """
    df = read_csv_auto(args.csv_path)
    target_col = args.target or infer_target_column(df.columns.tolist())
    X, y = prepare_dataset(df, args.dataset, target_col)

    # 训练测试集拆分 / Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    models = get_models_and_spaces(args.seed)

    # 结果保存路径 / Output directory
    out_dir = Path(__file__).resolve().parent.parent / "results" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    rows: list[dict[str, Any]] = []
    for model_name, (model, space) in models.items():
        # 构建流水线：预处理 + 模型 / Build pipeline: Preprocess + Model
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        
        # 随机搜索超参数调优 / Randomized Search for hyperparameter tuning
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=space,
            n_iter=args.n_iter,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0,
            random_state=args.seed,
            refit=True,
        )

        search.fit(X_train, y_train)

        # 评估最佳模型 / Evaluate the best model
        best = search.best_estimator_
        y_score = predict_proba_or_score(best, X_test)
        y_pred = (y_score >= 0.5).astype(int)
        metrics = evaluate(np.asarray(y_test), y_pred, y_score)

        # 生成 SHAP 解释图（如果启用） / Generate SHAP explanations if enabled
        shap_paths: dict[str, str] = {}
        if args.shap:
            bg_n = max(20, int(args.shap_background))
            exp_n = max(20, int(args.shap_samples))
            X_bg = X_train.sample(n=min(bg_n, len(X_train)), random_state=args.seed)
            X_exp = X_test.sample(n=min(exp_n, len(X_test)), random_state=args.seed)
            shap_paths = generate_shap_outputs(
                pipeline=best,
                X_background=X_bg,
                X_explain=X_exp,
                out_dir=out_dir,
                file_prefix=f"{run_id}_{model_name}",
                local_index=int(args.shap_local_index),
            )

        # 构造详细结果负载 / Construct detailed result payload
        payload = {
            "dataset": args.dataset,
            "csv_path": str(args.csv_path),
            "target": target_col,
            "model": model_name,
            "best_params": search.best_params_,
            "cv_best_score_roc_auc": float(search.best_score_),
            "test_metrics": metrics,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "cv_folds": int(args.cv_folds),
            "n_iter": int(args.n_iter),
            "shap": bool(args.shap),
            "shap_paths": shap_paths,
            "run_id": run_id,
        }

        # 保存单模型 JSON 结果 / Save per-model JSON results
        with (out_dir / f"{run_id}_{model_name}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # 准备汇总行 / Prepare summary row
        row = {
            "run_id": run_id,
            "dataset": args.dataset,
            "model": model_name,
            "cv_roc_auc": float(search.best_score_),
            **{f"test_{k}": v for k, v in metrics.items() if k != "confusion_matrix"},
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "cv_folds": int(args.cv_folds),
            "n_iter": int(args.n_iter),
            "shap": bool(args.shap),
        }
        rows.append(row)

    # 保存实验汇总表 / Save experiment summary table
    summary = pd.DataFrame(rows).sort_values(by=["test_roc_auc", "test_f1"], ascending=False)
    summary_path = out_dir / f"{run_id}_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path
