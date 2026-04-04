from __future__ import annotations

"""
评估指标模块 / Metrics Evaluation Module

中文：
- 提供统一的模型预测得分获取接口（支持概率和决策函数）
- 计算多维度的分类评估指标（ROC-AUC, PR-AUC, F1 等）

English:
- Provides a unified interface to extract prediction scores (supports probabilities and decision functions)
- Computes multi-dimensional classification metrics (ROC-AUC, PR-AUC, F1, etc.)
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_proba_or_score(estimator: Any, X: Any) -> np.ndarray:
    """
    获取模型预测概率或得分 / Get prediction probabilities or scores.

    中文：
    - 优先尝试 predict_proba (正类概率)
    - 若不支持，则使用 decision_function 并归一化到 [0, 1] 空间

    English:
    - Prioritizes predict_proba (probability of positive class)
    - Falls back to decision_function and normalizes scores to [0, 1] range
    """
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if isinstance(proba, list):
            proba = np.asarray(proba)
        # 通常返回第二列（正类概率） / Usually return the second column (positive class prob)
        return proba[:, 1]
    
    if hasattr(estimator, "decision_function"):
        score = estimator.decision_function(X)
        score = np.asarray(score).ravel()
        # 简单 Min-Max 归一化 / Simple Min-Max normalization
        score = (score - score.min()) / (score.max() - score.min() + 1e-12)
        return score
    
    raise ValueError("模型不支持 predict_proba / decision_function (Model lacks scoring methods)")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    """
    计算分类评估指标 / Compute classification metrics.

    中文：
    - 包含：准确率、精确率、召回率、F1、ROC-AUC、PR-AUC 和 混淆矩阵
    - 返回结果均转换为 Python 原生类型以便 JSON 序列化

    English:
    - Includes: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix
    - Converts results to native Python types for JSON serialization
    """
    metrics: dict[str, Any] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics
