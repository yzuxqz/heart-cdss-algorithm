"""
数据预处理脚本 / Data Preprocessing Script

中文：
- 对三个数据集执行完整预处理管线并保存到 processed_datasets/
- 生成可用于 EDA 分析的干净 CSV 文件
- 不涉及模型训练，仅做数据清洗和特征工程

English:
- Runs the full preprocessing pipeline on all three datasets and saves to processed_datasets/
- Produces clean CSV files ready for EDA analysis
- No model training, only data cleaning and feature engineering
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from heart_cdss.data import read_csv_auto
from heart_cdss.experiment import prepare_dataset
from heart_cdss.preprocess import build_preprocessor


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "processed_datasets"

DATASETS = [
    {
        "code": "uci_cleveland",
        "csv": "datasets/heart_disease_uci.csv",
        "target": "num",
        "prepare_name": "uci_cleveland",
    },
    {
        "code": "framingham",
        "csv": "datasets/framingham.csv",
        "target": "TenYearCHD",
        "prepare_name": "framingham",
    },
    {
        "code": "cardio70k",
        "csv": "datasets/cardio_train.csv",
        "target": "cardio",
        "prepare_name": "cardio70k",
    },
]


def preprocess_and_save(code: str, csv_path: Path, target: str, prepare_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"Preprocessing: {code}")
    print(f"{'='*60}")

    # 1. 读取原始数据
    df = read_csv_auto(csv_path)
    print(f"  Raw: {len(df)} rows")

    # 2. 数据准备（过滤子集、二值化目标、去ID列、去异常值）
    X, y = prepare_dataset(df, prepare_name, target)
    print(f"  After prepare_dataset: {len(X)} rows, {len(X.columns)} features")
    print(f"  Target distribution: {y.value_counts().to_dict()}")

    # 3. 构建并应用预处理器（填补 + 标准化 + One-Hot）
    preprocessor = build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)

    # 4. 获取特征名
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    # 5. 组装 DataFrame
    df_processed = pd.DataFrame(X_transformed, columns=feature_names)
    df_processed["target"] = y.values

    print(f"  After pipeline: {df_processed.shape[0]} rows x {df_processed.shape[1]} cols")

    # 6. 保存
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = OUT_DIR / f"{code}_preprocessed.csv"
    df_processed.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"  Saved: {csv_out}")

    # 7. 保存特征映射表
    mapping_rows = []
    for i, name in enumerate(feature_names):
        mapping_rows.append({"index": i, "feature_name": name})
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_out = OUT_DIR / f"{code}_feature_mapping.csv"
    mapping_df.to_csv(mapping_out, index=False, encoding="utf-8-sig")
    print(f"  Mapping saved: {mapping_out}")


def main() -> None:
    for ds in DATASETS:
        preprocess_and_save(
            code=ds["code"],
            csv_path=BASE_DIR / ds["csv"],
            target=ds["target"],
            prepare_name=ds["prepare_name"],
        )

    print(f"\n{'='*60}")
    print(f"All datasets preprocessed. Output: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
