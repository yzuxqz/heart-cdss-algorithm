from __future__ import annotations

"""
报告生成模块 / Reporting Module

中文：
- 使用 ReportLab 生成 PDF 诊断报告
- 支持文本自动换行、多页排版
- 支持在报告中嵌入 SHAP 解释图

English:
- Generates PDF diagnostic reports using ReportLab
- Supports automatic text wrapping and multi-page layout
- Supports embedding SHAP explanation images in reports
"""

from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def _wrap_lines(text: str, max_len: int) -> list[str]:
    """
    文本换行处理 / Wrap text lines.

    中文：将长文本按指定字符长度拆分为多行。
    English: Splits long text into multiple lines based on max character length.
    """
    out: list[str] = []
    for line in str(text).splitlines():
        if not line:
            out.append("")
            continue
        cur = ""
        for ch in line:
            if len(cur) >= max_len:
                out.append(cur)
                cur = ""
            cur += ch
        if cur:
            out.append(cur)
    return out


def generate_pdf_report(
    *,
    out_path: Path,
    title: str,
    meta: dict[str, Any],
    input_row: dict[str, Any],
    predictions: list[dict[str, Any]],
    shap_image_paths: list[Path] | None = None,
) -> Path:
    """
    生成 PDF 诊断报告 / Generate PDF diagnostic report.

    中文：
    - 创建 A4 画布并设置边距
    - 绘制标题、元数据、输入参数和预测结果
    - 若提供 SHAP 图片路径，则在新页中嵌入图片
    - 返回生成的 PDF 文件路径

    English:
    - Creates A4 canvas and sets margins
    - Draws title, metadata, input parameters, and predictions
    - Embeds SHAP images in new pages if paths are provided
    - Returns the generated PDF file path
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4

    x0 = 36
    y = height - 48

    # 1. 标题 / Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x0, y, title[:120])
    y -= 22

    # 2. 元数据（版本、时间等） / Metadata
    c.setFont("Helvetica", 10)
    for k, v in meta.items():
        for line in _wrap_lines(f"{k}: {v}", 110):
            if y < 72:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 48
            c.drawString(x0, y, line)
            y -= 12

    y -= 6
    # 3. 输入数据 / Input Data
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0, y, "Input")
    y -= 16
    c.setFont("Helvetica", 9)

    for k, v in input_row.items():
        for line in _wrap_lines(f"{k}: {v}", 120):
            if y < 72:
                c.showPage()
                c.setFont("Helvetica", 9)
                y = height - 48
            c.drawString(x0, y, line)
            y -= 11

    y -= 8
    # 4. 预测结果 / Predictions
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0, y, "Predictions")
    y -= 16
    c.setFont("Helvetica", 9)
    for p in predictions:
        line = f"model={p.get('model')}  proba={p.get('proba')}  threshold={p.get('threshold')}  label={p.get('label')}"
        for w in _wrap_lines(line, 120):
            if y < 72:
                c.showPage()
                c.setFont("Helvetica", 9)
                y = height - 48
            c.drawString(x0, y, w)
            y -= 11

    # 5. 嵌入 SHAP 图片 / Embed SHAP images
    if shap_image_paths:
        for img_path in shap_image_paths:
            if not img_path.exists():
                continue
            c.showPage()
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x0, height - 48, f"SHAP: {img_path.name}")
            y_img_top = height - 80
            max_w = width - 2 * x0
            max_h = height - 140
            try:
                img = ImageReader(str(img_path))
                iw, ih = img.getSize()
                # 保持宽高比缩放 / Scale while preserving aspect ratio
                scale = min(max_w / float(iw), max_h / float(ih))
                dw = float(iw) * scale
                dh = float(ih) * scale
                c.drawImage(img, x0, y_img_top - dh, width=dw, height=dh, preserveAspectRatio=True)
            except Exception:
                c.setFont("Helvetica", 10)
                c.drawString(x0, y_img_top, f"Failed to embed image: {img_path}")

    c.save()
    return out_path
