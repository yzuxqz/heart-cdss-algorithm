from __future__ import annotations

"""
预处理模块 / Preprocessing module

中文：
- 执行布尔型字符串标准化（true/false/0/1）
- 构建数值与类别特征的联合预处理管线

English:
- Normalizes bool-like textual columns
- Builds a joint preprocessing pipeline for numeric and categorical features
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def normalize_bool_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    布尔文本标准化 / Normalize bool-like textual columns.

    中文：
    - 将字符串列中仅包含 true/false/0/1 的字段映射为整数布尔值
    - 减少类别编码噪声，提升后续建模稳定性

    English:
    - Maps textual true/false/0/1 patterns into integer-like boolean values
    - Reduces categorical noise before feature engineering
    """
    df = df.copy()
    for col in df.columns:
        s = df[col]
        # 强制将非 object 的字符串列转换为 object 以便于统一处理 / Coerce string types to object for uniform handling
        if pd.api.types.is_string_dtype(s.dtype) and s.dtype != "object":
            df[col] = s.astype("object")
            s = df[col]
        
        # 检查是否为布尔类别的文本 / Check for boolean-like textual patterns
        if s.dtype == "object":
            unique = set(str(v).strip().lower() for v in s.dropna().unique())
            if unique.issubset({"true", "false", "0", "1"}):
                # 映射并转换为可为空的整数类型 / Map and convert to nullable Int64
                df[col] = (
                    s.astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"true": 1, "false": 0, "1": 1, "0": 0})
                    .astype("Int64")
                )
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    构建特征预处理器 / Build feature preprocessor.

    中文：
    - 数值特征：中位数填补 + 标准化
    - 类别特征：众数填补 + One-Hot 编码
    - 对低基数整数列（如 0/1/2）按类别处理

    English:
    - Numeric: median imputation + standard scaling
    - Categorical: most-frequent imputation + one-hot encoding
    - Low-cardinality integer columns are treated as categorical features
    """
    # 初始分类：基于数据类型 / Initial classification based on dtypes
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # 启发式规则：低基数（<=10个唯一值）的整数列视为分类变量 / Heuristic: Treat low-cardinality integers as categorical
    for col in list(numeric_cols):
        s = X[col]
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= 10:
            numeric_cols.remove(col)
            categorical_cols.append(col)

    # 数值处理流水线 / Numerical pipeline
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 分类处理流水线 / Categorical pipeline
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 合并预处理器 / Combine into ColumnTransformer
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
