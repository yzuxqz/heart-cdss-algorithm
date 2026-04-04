from __future__ import annotations

"""
数据读取模块 / Data I/O module

中文：
- 负责自动识别 CSV 分隔符（逗号或分号）
- 统一读取数据并处理常见缺失值标记

English:
- Detects CSV delimiter automatically (comma or semicolon)
- Loads tabular data with unified NA handling
"""

from pathlib import Path

import pandas as pd


def guess_csv_sep(path: Path) -> str:
    """
    猜测 CSV 分隔符 / Guess CSV delimiter.

    中文：
    - 读取文件头部前 4KB，检查第一行中 ';' 和 ',' 的出现次数
    - 若分号更多则判定为 ';'，否则使用 ','

    English:
    - Reads first 4KB and compares ';' vs ',' counts in the first line
    - Returns ';' if semicolon appears more often, otherwise ','
    """
    with path.open("rb") as f:
        head = f.read(4096)
    text = head.decode("utf-8", errors="ignore")
    first_line = text.splitlines()[0] if text.splitlines() else text
    if first_line.count(";") > first_line.count(","):
        return ";"
    return ","


def read_csv_auto(path: Path) -> pd.DataFrame:
    """
    自动读取 CSV / Read CSV with auto delimiter and NA normalization.

    中文：
    - 调用 guess_csv_sep 自动判断分隔符
    - 将常见字符串 NA/N-A/空字符串按缺失值处理

    English:
    - Uses guess_csv_sep for delimiter inference
    - Normalizes common textual NA tokens into missing values
    """
    sep = guess_csv_sep(path)
    return pd.read_csv(
        path,
        sep=sep,
        na_values=["NA", "Na", "na", "N/A", "n/a", ""],
        keep_default_na=True,
    )
