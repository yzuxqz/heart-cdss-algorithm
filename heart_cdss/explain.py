from __future__ import annotations

"""
模型解释模块 / Model Explanation Module

中文：
- 集成 SHAP (SHapley Additive exPlanations) 库
- 生成全局特征重要性（条形图、蜂群图）
- 生成单样本预测解释（瀑布图）

English:
- Integrates SHAP (SHapley Additive exPlanations) library
- Generates global feature importance (Bar plots, Beeswarm plots)
- Generates local instance explanations (Waterfall plots)
"""

from pathlib import Path
from typing import Any

import numpy as np


def _optional_import_shap() -> Any | None:
    """
    可选导入 SHAP / Optional SHAP import.

    中文：若未安装 shap 库则返回 None，避免强制依赖。
    English: Returns None if shap is not installed to avoid hard dependency.
    """
    try:
        import shap
        return shap
    except Exception:
        return None


def _get_feature_names(preprocessor: Any, fallback_n_features: int) -> list[str]:
    """
    获取预处理后的特征名称 / Get feature names after preprocessing.

    中文：尝试从 ColumnTransformer 获取输出列名，失败则使用索引。
    English: Tries to get output names from ColumnTransformer; falls back to indices.
    """
    try:
        names = preprocessor.get_feature_names_out()
        return [str(x) for x in names]
    except Exception:
        return [f"feature_{i}" for i in range(fallback_n_features)]


def _ensure_2d(x: Any) -> np.ndarray:
    """
    确保数据为 2D NumPy 数组 / Ensure data is a 2D NumPy array.

    中文：支持稀疏矩阵转换为密集矩阵。
    English: Supports converting sparse matrices to dense arrays.
    """
    if isinstance(x, np.ndarray):
        return x
    try:
        return x.toarray()
    except Exception:
        return np.asarray(x)


def _select_positive_class_if_needed(explanation: Any) -> Any:
    """
    如果是多分类（含二分类概率），选择正类 / Select positive class for binary/multiclass explanation.

    中文：处理 SHAP 返回的 3D 数组 [样本, 特征, 类别]，提取正类（索引 1）。
    English: Handles 3D arrays [samples, features, classes] and extracts positive class (index 1).
    """
    values = getattr(explanation, "values", None)
    if values is None:
        return explanation
    if isinstance(values, np.ndarray) and values.ndim == 3 and values.shape[-1] == 2:
        return explanation[..., 1]
    return explanation


def generate_shap_outputs(
    *,
    pipeline: Any,
    X_background,
    X_explain,
    out_dir: Path,
    file_prefix: str,
    local_index: int = 0,
) -> dict[str, str]:
    """
    生成并保存 SHAP 解释图 / Generate and save SHAP explanation plots.

    中文：
    - 使用背景数据集初始化解释器
    - 生成全局蜂群图和条形图
    - 为指定索引的样本生成局部瀑布图
    - 返回保存的文件路径字典

    English:
    - Initializes explainer with background data
    - Generates global Beeswarm and Bar plots
    - Generates local Waterfall plot for a specific sample index
    - Returns a dict of saved file paths
    """
    shap = _optional_import_shap()
    if shap is None:
        return {}

    import matplotlib
    # 使用 Agg 后端，无需 GUI 即可保存图片 / Use Agg backend for headless image saving
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    # 转换数据到特征空间 / Transform data to feature space
    X_bg_t = _ensure_2d(pre.transform(X_background))
    X_ex_t = _ensure_2d(pre.transform(X_explain))

    feature_names = _get_feature_names(pre, fallback_n_features=X_ex_t.shape[1])

    # 创建 SHAP 解释器 / Create SHAP explainer
    explainer = shap.Explainer(model, X_bg_t, feature_names=feature_names)
    explanation = explainer(X_ex_t)
    explanation = _select_positive_class_if_needed(explanation)

    paths: dict[str, str] = {}

    # 1. 蜂群图 / Beeswarm plot
    beeswarm_path = out_dir / f"{file_prefix}_shap_beeswarm.png"
    plt.figure()
    shap.plots.beeswarm(explanation, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=200)
    plt.close()
    paths["beeswarm"] = str(beeswarm_path)

    # 2. 条形图 / Bar plot
    bar_path = out_dir / f"{file_prefix}_shap_bar.png"
    plt.figure()
    shap.plots.bar(explanation, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()
    paths["bar"] = str(bar_path)

    # 3. 局部瀑布图 / Local waterfall plot
    if 0 <= local_index < len(X_ex_t):
        local_path = out_dir / f"{file_prefix}_shap_waterfall_{local_index}.png"
        plt.figure()
        shap.plots.waterfall(explanation[local_index], show=False, max_display=30)
        plt.tight_layout()
        plt.savefig(local_path, dpi=200)
        plt.close()
        paths["waterfall"] = str(local_path)

    return paths
