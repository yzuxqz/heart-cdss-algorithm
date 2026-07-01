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
            is_unbalance=True,
        )
    if model_name == "cat":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost 未安装 / catboost not installed")
        return CatBoostClassifier(
            random_seed=random_state,
            loss_function="Logloss",
            verbose=False,
            auto_class_weights="Balanced",
        )
    raise ValueError(f"未知模型 / Unknown model: {model_name}")


def get_models_and_spaces(random_state: int) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """
    获取所有可用模型及其超参数空间 / Get all available models and their search spaces.

    中文：返回 {模型名: (模型实例, 参数网格)} 字典，供 RandomizedSearchCV 自动调参。
    English: Returns {model_name: (instance, param_grid)} dict for automated RandomizedSearchCV tuning.
    """
    models: dict[str, tuple[Any, dict[str, list[Any]]]] = {}

    # ═══════════════════════════════════════════════════════════════════════
    # 1. 逻辑回归 / Logistic Regression
    # ═══════════════════════════════════════════════════════════════════════
    models["logreg"] = (
        LogisticRegression(
            max_iter=5000,                   # 最大迭代次数；过低可能不收敛 / max iterations; too low may not converge
            class_weight="balanced",         # 自动按正负样本比例加权，缓解类不平衡 / auto-weight inversely to class frequencies
            solver="lbfgs",                  # 拟牛顿法，适合中小规模数据 / quasi-Newton solver, suitable for small-to-medium data
        ),
        {
            # C：正则化强度的倒数 / Inverse of L2 regularisation strength
            # 越小 → 正则化越强 → 系数趋近 0 → 防止过拟合 / Smaller → stronger penalty → coefficients shrink → anti-overfitting
            # 越大 → 正则化越弱 → 模型更灵活、可能过拟合 / Larger → weaker penalty → more flexible, risk of overfitting
            # 搜索范围 0.05~10 覆盖从"强正则化"到"弱正则化"两端
            "model__C": [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10],
        },
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 2. 随机森林 / Random Forest
    # ═══════════════════════════════════════════════════════════════════════
    models["rf"] = (
        RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,                               # 使用全部 CPU 核并行构建 / use all CPU cores
            class_weight="balanced_subsample",       # 每棵树 bootstrap 时按类别比例采样 / weight classes within each bootstrap sample
        ),
        {
            # n_estimators：决策树数量 / number of trees
            # 越多 → 集成越稳定、方差越低，但训练越慢 / more → stabler, lower variance, but slower
            # 超过 900 在这个数据量级收益递减 / diminishing returns beyond ~900 for this data scale
            "model__n_estimators": [300, 600, 900],

            # max_depth：单棵树最大深度 / maximum depth of each tree
            # None → 不限深，树可能长得极深导致过拟合 / no limit, may overfit
            # 4~12 → 限制深度，控制模型复杂度 / cap depth to control complexity
            "model__max_depth": [None, 4, 6, 8, 12],

            # min_samples_split：内部节点继续分裂的最小样本数 / min samples required to split an internal node
            # 2 → 容易分裂，树更复杂 / easy to split → more complex trees
            # 10 → 更难分裂，起正则化作用 / harder to split → regularisation effect
            "model__min_samples_split": [2, 5, 10],

            # min_samples_leaf：叶节点最少样本数 / min samples required at a leaf node
            # 1 → 叶节点可极纯，可能过拟合 / leaf can be very pure → possible overfitting
            # 4 → 迫使每个叶有 ≥4 个样本，起平滑作用 / forces ≥4 samples per leaf → smoothing
            "model__min_samples_leaf": [1, 2, 4],

            # max_features：每次分裂时考虑的特征比例 / fraction of features considered at each split
            # sqrt  → √n_features，scikit-learn 默认 / default, √n_features
            # log2  → log₂(n_features)，更少特征 → 更多随机性 → 更多样化的树
            # None  → 考虑全部特征 / all features
            "model__max_features": ["sqrt", "log2", None],
        },
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 3. XGBoost
    # ═══════════════════════════════════════════════════════════════════════
    if XGBClassifier is not None:
        models["xgb"] = (
            XGBClassifier(
                random_state=random_state,
                tree_method="hist",            # 直方图算法，比 exact 快数倍 / histogram-based, much faster than exact
                eval_metric="logloss",         # 训练时监控对数损失 / monitor log-loss during training
                n_jobs=-1,
            ),
            {
                # n_estimators：提升轮数（树的数量）/ number of boosting rounds (trees)
                # 越多 → 拟合能力越强，但太多会过拟合 / more → stronger fit, but too many overfits
                "model__n_estimators": [300, 600, 900],

                # max_depth：每棵树的最大深度 / maximum tree depth
                # 3~6 是表格数据的典型范围；更深 → 更复杂 → 更多过拟合风险
                # 3–6 is typical for tabular data; deeper → more complex → more overfitting risk
                "model__max_depth": [3, 4, 5, 6],

                # learning_rate（eta）：每次迭代的步长 / step size shrinkage per iteration
                # 0.01 → 学习慢，需要更多轮次但更稳健 / slow learner, needs more rounds, stabler
                # 0.1  → 学习快，但可能跳过最优解 / fast learner, may overshoot optimum
                "model__learning_rate": [0.01, 0.03, 0.1],

                # subsample：每棵树随机采样的训练数据比例 / fraction of training data sampled per tree
                # 1.0 → 用全部数据 / use all data
                # 0.7 → 每棵树只看 70% 数据 → 增加随机性 → 防过拟合 / more randomness → anti-overfitting
                "model__subsample": [0.7, 0.85, 1.0],

                # colsample_bytree：每棵树随机采样的特征比例 / fraction of features sampled per tree
                # 作用类似 subsample 但在特征维度 / similar to subsample but on feature dimension
                "model__colsample_bytree": [0.7, 0.85, 1.0],

                # min_child_weight：叶节点最小样本权重和 / min sum of instance weight in a child
                # 1 → 允许极小的叶节点 / allows very small leaves
                # 10 → 防止学习过于局部化的模式 / prevents learning overly local patterns
                "model__min_child_weight": [1, 5, 10],

                # reg_lambda：L2 正则化系数 / L2 regularisation on leaf weights
                # 0.5 → 弱正则化 / weak regularisation
                # 2.0 → 强正则化，抑制叶子权重过大 / stronger, shrinks leaf weights
                "model__reg_lambda": [0.5, 1.0, 2.0],

                # scale_pos_weight：正类样本权重 / weight for the positive class
                # 1 → 无调整 / no adjustment
                # 7~10 → 对少数类（阳性）赋予更高权重，应对类不平衡
                # 7–10 → up-weight minority (positive) class to handle imbalance
                "model__scale_pos_weight": [1, 3, 5, 7, 10],
            },
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 4. LightGBM
    # ═══════════════════════════════════════════════════════════════════════
    if LGBMClassifier is not None:
        models["lgbm"] = (
            LGBMClassifier(
                random_state=random_state,
                n_jobs=-1,
                is_unbalance=True,             # 自动处理类不平衡 / auto-handle class imbalance
            ),
            {
                # n_estimators：提升轮数。LGBM 收敛快，500~1500 覆盖典型范围
                # boosting rounds; LGBM converges fast, 500–1500 covers typical range
                "model__n_estimators": [500, 1000, 1500],

                # num_leaves：每棵树的叶子数 / number of leaves per tree
                # 31 → 较简单的树 / simpler tree
                # 127 → 更复杂的树，但注意不要超过 2^max_depth / more complex, should not exceed 2^max_depth
                "model__num_leaves": [31, 63, 127],

                # learning_rate：同 XGBoost / same semantics as XGBoost
                "model__learning_rate": [0.01, 0.03, 0.1],

                # subsample：同 XGBoost，训练数据采样比例 / same as XGBoost, data sampling ratio
                "model__subsample": [0.7, 0.85, 1.0],

                # colsample_bytree：同 XGBoost，特征采样比例 / same as XGBoost, feature sampling ratio
                "model__colsample_bytree": [0.7, 0.85, 1.0],

                # min_child_samples：叶节点最少样本数 / min data in a leaf
                # 10 → 对大数据集合理 / reasonable for large datasets
                # 40 → 更强的正则化 / stronger regularisation
                "model__min_child_samples": [10, 20, 40],

                # reg_lambda：L2 正则化 / L2 regularisation
                # 0.0 → 无正则化 / no regularisation
                # 2.0 → 强正则化 / strong regularisation
                "model__reg_lambda": [0.0, 1.0, 2.0],
            },
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 5. CatBoost
    # ═══════════════════════════════════════════════════════════════════════
    if CatBoostClassifier is not None:
        models["cat"] = (
            CatBoostClassifier(
                random_seed=random_state,
                loss_function="Logloss",               # 二分类对数损失 / binary log-loss
                verbose=False,                         # 不打印训练日志 / suppress training output
                auto_class_weights="Balanced",         # 自动平衡类别权重 / auto-balance class weights
            ),
            {
                # iterations：提升轮数（等价于 n_estimators）/ number of boosting iterations
                # CatBoost 通常 500~1500 轮足以收敛 / typically 500–1500 rounds sufficient for convergence
                "model__iterations": [500, 1000, 1500],

                # depth：树的深度 / tree depth
                # CatBoost 使用对称树（symmetric trees），深度直接控制模型复杂度
                # 4 → 较简单、泛化好 / simpler, better generalisation
                # 10 → 复杂、拟合强但可能过拟合 / complex, strong fit, may overfit
                "model__depth": [4, 6, 8, 10],

                # learning_rate：同 XGBoost / same semantics
                "model__learning_rate": [0.01, 0.03, 0.1],

                # l2_leaf_reg：叶节点值的 L2 正则化系数 / L2 regularisation on leaf values
                # 1 → 弱正则化 / weak
                # 10 → 强正则化，防过拟合 / strong, anti-overfitting
                "model__l2_leaf_reg": [1, 3, 10],
            },
        )

    return models
