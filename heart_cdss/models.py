from __future__ import annotations

"""
模型构建模块 / Model Building Module

中文：
- 提供统一的模型实例化接口
- 定义不同模型的超参数搜索空间
- 支持集成多种机器学习库（XGBoost, LightGBM, CatBoost）

English:
- Provides a unified interface for model instantiation
- Defines hyperparameter search spaces for various models
- Supports integration of multiple ML libraries (XGBoost, LightGBM, CatBoost)
"""

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _try_import(module: str, name: str) -> Any | None:
    """
    尝试动态导入模块 / Safely attempt to import a module.

    中文：
    - 避免因未安装可选依赖（如 xgboost）而导致程序崩溃
    - 返回导入的对象或 None

    English:
    - Prevents crashes if optional dependencies (e.g., xgboost) are missing
    - Returns the imported object or None
    """
    try:
        m = __import__(module, fromlist=[name])
        return getattr(m, name)
    except Exception:
        return None


# 动态加载可选的梯度提升库 / Dynamically load optional boosting libraries
XGBClassifier = _try_import("xgboost", "XGBClassifier")
LGBMClassifier = _try_import("lightgbm", "LGBMClassifier")
CatBoostClassifier = _try_import("catboost", "CatBoostClassifier")


def make_model(model_name: str, random_state: int) -> Any:
    """
    根据名称创建模型实例 / Create a model instance by name.

    中文：
    - 支持 logreg (逻辑回归), rf (随机森林), xgb, lgbm, cat
    - 预设了适用于不平衡数据的 class_weight 等参数

    English:
    - Supports logreg, rf, xgb, lgbm, cat
    - Presets parameters like class_weight for imbalanced data
    """
    if model_name == "logreg":
        return LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
        )
    if model_name == "rf":
        return RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    if model_name == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost 未安装 / xgboost not installed")
        return XGBClassifier(
            random_state=random_state,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
        )
    if model_name == "lgbm":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm 未安装 / lightgbm not installed")
        return LGBMClassifier(
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "cat":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost 未安装 / catboost not installed")
        return CatBoostClassifier(
            random_seed=random_state,
            loss_function="Logloss",
            verbose=False,
        )
    raise ValueError(f"未知模型 / Unknown model: {model_name}")


def get_models_and_spaces(random_state: int) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """
    获取所有可用模型及其超参数空间 / Get all available models and their search spaces.

    中文：
    - 返回一个字典，键为模型名称，值为 (模型实例, 参数网格) 的元组
    - 用于自动化模型实验和调优

    English:
    - Returns a dict mapping model names to (instance, param_grid) tuples
    - Used for automated experiments and hyperparameter tuning
    """
    models: dict[str, tuple[Any, dict[str, list[Any]]]] = {}

    # 逻辑回归参数空间 / Logistic Regression space
    models["logreg"] = (
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        {
            "model__C": [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10],
        },
    )

    # 随机森林参数空间 / Random Forest space
    models["rf"] = (
        RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        {
            "model__n_estimators": [300, 600, 900],
            "model__max_depth": [None, 4, 6, 8, 12],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
    )

    # XGBoost 参数空间 / XGBoost space
    if XGBClassifier is not None:
        models["xgb"] = (
            XGBClassifier(
                random_state=random_state,
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [300, 600, 900],
                "model__max_depth": [3, 4, 5, 6],
                "model__learning_rate": [0.01, 0.03, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__min_child_weight": [1, 5, 10],
                "model__reg_lambda": [0.5, 1.0, 2.0],
            },
        )

    # LightGBM 参数空间 / LightGBM space
    if LGBMClassifier is not None:
        models["lgbm"] = (
            LGBMClassifier(
                random_state=random_state,
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [500, 1000, 1500],
                "model__num_leaves": [31, 63, 127],
                "model__learning_rate": [0.01, 0.03, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__min_child_samples": [10, 20, 40],
                "model__reg_lambda": [0.0, 1.0, 2.0],
            },
        )

    # CatBoost 参数空间 / CatBoost space
    if CatBoostClassifier is not None:
        models["cat"] = (
            CatBoostClassifier(
                random_seed=random_state,
                loss_function="Logloss",
                verbose=False,
            ),
            {
                "model__iterations": [500, 1000, 1500],
                "model__depth": [4, 6, 8, 10],
                "model__learning_rate": [0.01, 0.03, 0.1],
                "model__l2_leaf_reg": [1, 3, 10],
            },
        )

    return models
