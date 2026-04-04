from __future__ import annotations

"""
系统构件构建脚本 / System Artifacts Builder Script

中文：
- 从实验结果（results/）中挑选最佳模型
- 重新在完整数据集上训练流水线
- 生成元数据和数据架构，并将所有构件保存到 artifacts/ 目录供应用部署使用

English:
- Picks the best models from experiment results (results/)
- Re-fits the pipeline on the full dataset
- Generates metadata/schema and saves all artifacts to artifacts/ for application deployment
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from heart_cdss.data import read_csv_auto
from heart_cdss.experiment import prepare_dataset
from heart_cdss.models import make_model
from heart_cdss.persist import load_json, save_joblib, save_json
from heart_cdss.preprocess import build_preprocessor, normalize_bool_like_columns


@dataclass(frozen=True)
class DatasetConfig:
    code: str
    csv_path: Path
    target: str
    dataset_name_for_prepare: str


def _latest_summary_path(results_dir: Path) -> Path:
    paths = sorted(results_dir.glob("*_summary.csv"), key=lambda p: p.name, reverse=True)
    if not paths:
        raise FileNotFoundError(f"没有找到 summary.csv: {results_dir}")
    return paths[0]


def _pick_best_model(summary_path: Path) -> dict[str, Any]:
    df = pd.read_csv(summary_path)
    if "test_roc_auc" not in df.columns:
        raise ValueError(f"summary 缺少 test_roc_auc 列: {summary_path}")
    df = df.sort_values(by=["test_roc_auc", "test_f1"], ascending=False)
    row = df.iloc[0].to_dict()
    return row


def _load_summary_rows(summary_path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(summary_path)
    if "model" not in df.columns:
        raise ValueError(f"summary 缺少 model 列: {summary_path}")
    return df.to_dict(orient="records")


def _load_best_params(results_dir: Path, run_id: str, model: str) -> dict[str, Any]:
    json_path = results_dir / f"{run_id}_{model}.json"
    payload = load_json(json_path)
    best_params = payload.get("best_params", {})
    stripped: dict[str, Any] = {}
    for k, v in best_params.items():
        if k.startswith("model__"):
            stripped[k.removeprefix("model__")] = v
    return stripped


def _infer_schema(df_X: pd.DataFrame) -> dict[str, Any]:
    df_X = normalize_bool_like_columns(df_X)
    categorical_cols = df_X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in df_X.columns if c not in categorical_cols]
    for col in list(numeric_cols):
        s = df_X[col]
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= 10:
            numeric_cols.remove(col)
            categorical_cols.append(col)

    schema: dict[str, Any] = {"columns": []}
    for col in df_X.columns:
        if col in categorical_cols:
            cats = sorted([x for x in df_X[col].dropna().astype(str).unique().tolist()])
            schema["columns"].append({"name": col, "type": "categorical", "categories": cats})
        else:
            s = pd.to_numeric(df_X[col], errors="coerce")
            schema["columns"].append(
                {
                    "name": col,
                    "type": "numeric",
                    "min": None if s.dropna().empty else float(s.min()),
                    "max": None if s.dropna().empty else float(s.max()),
                }
            )
    return schema


def _fit_pipeline_for_model(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    best_params: dict[str, Any],
    random_state: int,
) -> Pipeline:
    pre = build_preprocessor(X)
    model = make_model(model_name, random_state=random_state)
    if best_params:
        model.set_params(**best_params)
    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
    pipe.fit(X, y)
    return pipe


def build_for_dataset(cfg: DatasetConfig, base_dir: Path) -> Path:
    results_dir = base_dir / "results" / cfg.code
    summary_path = _latest_summary_path(results_dir)
    best_row = _pick_best_model(summary_path)
    run_id = str(best_row["run_id"])
    model_name = str(best_row["model"])

    best_params = _load_best_params(results_dir, run_id, model_name)

    df = read_csv_auto(cfg.csv_path)
    X, y = prepare_dataset(df, cfg.dataset_name_for_prepare, cfg.target)

    out_dir = base_dir / "artifacts" / cfg.code
    model_path = out_dir / "model.joblib"
    schema_path = out_dir / "schema.json"
    meta_path = out_dir / "meta.json"

    pipe = _fit_pipeline_for_model(
        X=X,
        y=y,
        model_name=model_name,
        best_params=best_params,
        random_state=42,
    )
    save_joblib(pipe, model_path)
    save_json(_infer_schema(X), schema_path)

    model_entries: list[dict[str, Any]] = []
    for row in _load_summary_rows(summary_path):
        m = str(row.get("model"))
        if not m or m == "nan":
            continue
        try:
            params = _load_best_params(results_dir, run_id, m)
            model_pipe = _fit_pipeline_for_model(
                X=X,
                y=y,
                model_name=m,
                best_params=params,
                random_state=42,
            )
            m_dir = out_dir / "models" / m
            save_joblib(model_pipe, m_dir / "model.joblib")
            save_json(
                {
                    "dataset": cfg.code,
                    "csv_path": str(cfg.csv_path),
                    "target": cfg.target,
                    "run_id": run_id,
                    "model": m,
                    "best_params": params,
                    "row": row,
                },
                m_dir / "meta.json",
            )
            model_entries.append({"model": m, "path": str(m_dir / "model.joblib")})
        except Exception:
            continue

    meta = {
        "dataset": cfg.code,
        "csv_path": str(cfg.csv_path),
        "target": cfg.target,
        "best_from_summary": str(summary_path),
        "best_run_id": run_id,
        "best_model": model_name,
        "best_params": best_params,
        "best_row": best_row,
        "models_dir": str(out_dir / "models"),
        "available_models": model_entries,
    }
    save_json(meta, meta_path)
    return model_path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    datasets = [
        DatasetConfig(
            code="uci_cleveland",
            csv_path=base_dir / "heart_disease_uci.csv",
            target="num",
            dataset_name_for_prepare="uci_cleveland",
        ),
        DatasetConfig(
            code="framingham",
            csv_path=base_dir / "framingham.csv",
            target="TenYearCHD",
            dataset_name_for_prepare="framingham",
        ),
        DatasetConfig(
            code="cardio70k",
            csv_path=base_dir / "cardio_train.csv",
            target="cardio",
            dataset_name_for_prepare="cardio70k",
        ),
    ]

    for cfg in datasets:
        model_path = build_for_dataset(cfg, base_dir)
        print(str(model_path))


if __name__ == "__main__":
    main()
