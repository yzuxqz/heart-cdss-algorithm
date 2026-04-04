from __future__ import annotations

"""
审计模块 / Audit Module

中文：
- 提供简单的事件记录功能
- 将操作日志追加到 CSV 文件中，包含 UTC 时间戳

English:
- Provides simple event logging functionality
- Appends operational logs to a CSV file with UTC timestamps
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _utc_now_iso() -> str:
    """获取当前 UTC 时间的 ISO 格式字符串 / Get current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def append_event_csv(path: Path, event: dict[str, Any]) -> None:
    """
    追加事件到 CSV 文件 / Append an event to a CSV file.

    中文：
    - 确保目标目录存在
    - 自动添加 ts_utc 时间戳
    - 若文件不存在则创建并写入表头，否则追加数据

    English:
    - Ensures the parent directory exists
    - Automatically adds a 'ts_utc' timestamp
    - Creates file with header if missing, otherwise appends data
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(event)
    row.setdefault("ts_utc", _utc_now_iso())
    df = pd.DataFrame([row])
    
    # 写入文件 / Write to file
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, mode="w", header=True, index=False, encoding="utf-8-sig")
