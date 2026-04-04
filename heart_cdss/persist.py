from __future__ import annotations

"""
持久化模块 / Persistence Module

中文：
- 提供模型（Joblib）和元数据（JSON）的保存与加载工具
- 统一处理文件编码和目录创建

English:
- Provides utilities for saving and loading models (Joblib) and metadata (JSON)
- Uniformly handles file encoding and directory creation
"""

import json
from pathlib import Path
from typing import Any

import joblib


def save_joblib(obj: Any, path: Path) -> None:
    """保存对象为 Joblib 文件 / Save object as Joblib file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    """加载 Joblib 文件 / Load Joblib file."""
    return joblib.load(path)


def save_json(data: dict[str, Any], path: Path) -> None:
    """保存字典为 JSON 文件 / Save dict as JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    """加载 JSON 文件 / Load JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))
