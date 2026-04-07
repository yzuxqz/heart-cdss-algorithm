from __future__ import annotations

"""
命令行接口模块 / Command Line Interface Module

中文：
- 定义实验运行的命令行参数（数据集、模型迭代次数、SHAP 设置等）
- 提供程序主入口点

English:
- Defines CLI arguments for experiment runs (dataset, iterations, SHAP settings, etc.)
- Provides the main entry point for the application
"""

import argparse
from pathlib import Path

from heart_cdss.experiment import RunArgs, run_experiment


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器 / Build the CLI argument parser.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        required=True,
        choices=["uci_cleveland", "uci", "framingham", "cardio70k", "custom"],
        help="数据集名称 / Dataset name",
    )
    p.add_argument("--csv", required=True, help="CSV 文件路径 / Path to CSV file")
    p.add_argument("--target", default=None, help="目标列名 / Target column name")
    p.add_argument(
        "--models",
        default=None,
        help="只跑指定模型(逗号分隔)：logreg,rf,xgb,lgbm,cat / Limit models (comma-separated)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="限制最多使用多少行数据(0=不限制) / Max rows to sample (0=no limit)",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="测试集比例 / Test set ratio")
    p.add_argument("--seed", type=int, default=42, help="随机种子 / Random seed")
    p.add_argument("--n-iter", type=int, default=25, help="超参数搜索迭代次数 / Hyperparameter search iterations")
    p.add_argument("--cv-folds", type=int, default=5, help="交叉验证折数 / CV folds")
    p.add_argument("--shap", action="store_true", help="是否启用 SHAP 解释 / Enable SHAP explanations")
    p.add_argument("--shap-background", type=int, default=200, help="SHAP 背景样本数 / SHAP background samples")
    p.add_argument("--shap-samples", type=int, default=200, help="SHAP 解释样本数 / SHAP explanation samples")
    p.add_argument("--shap-local-index", type=int, default=0, help="局部解释的样本索引 / Sample index for local SHAP")
    return p


def main(argv: list[str] | None = None) -> None:
    """
    主入口函数 / Main entry function.
    """
    p = build_parser()
    ns = p.parse_args(argv)
    models = None
    if ns.models:
        models = tuple([m.strip() for m in str(ns.models).split(",") if m.strip()])
    max_rows = int(ns.max_rows) if int(ns.max_rows) > 0 else None

    # 启动实验 / Start the experiment
    summary_path = run_experiment(
        RunArgs(
            dataset=ns.dataset,
            csv_path=Path(ns.csv),
            target=ns.target,
            models=models,
            max_rows=max_rows,
            test_size=ns.test_size,
            seed=ns.seed,
            n_iter=ns.n_iter,
            cv_folds=ns.cv_folds,
            shap=bool(ns.shap),
            shap_background=int(ns.shap_background),
            shap_samples=int(ns.shap_samples),
            shap_local_index=int(ns.shap_local_index),
        )
    )
    # 输出汇总表路径 / Print summary path
    print(str(summary_path))
