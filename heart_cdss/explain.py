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


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size or x.size < 3:
        return None
    if float(np.nanstd(x)) == 0.0 or float(np.nanstd(y)) == 0.0:
        return None
    c = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(c):
        return None
    return c


def _direction_hint(corr: float | None) -> str:
    if corr is None:
        return "方向不稳定/数据不足"
    if corr >= 0.15:
        return "值越大 → 风险更高"
    if corr <= -0.15:
        return "值越大 → 风险更低"
    return "方向弱/可能非线性"


def _format_float(x: Any, nd: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return "NA"
    if np.isnan(v):
        return "NA"
    return f"{v:.{nd}f}"


def _top_global_feature_table(
    *,
    shap_values: np.ndarray,
    X_values: np.ndarray,
    feature_names: list[str],
    top_k: int = 10,
) -> list[tuple[str, float, str]]:
    if shap_values.ndim != 2:
        return []
    mean_abs = np.nanmean(np.abs(shap_values), axis=0)
    order = np.argsort(-mean_abs)[: int(top_k)]
    rows: list[tuple[str, float, str]] = []
    for i in order.tolist():
        name = feature_names[i] if 0 <= i < len(feature_names) else f"feature_{i}"
        corr = _safe_corr(X_values[:, i], shap_values[:, i])
        rows.append((str(name), float(mean_abs[i]), _direction_hint(corr)))
    return rows


def _local_feature_table(
    *,
    shap_row: np.ndarray,
    X_row: np.ndarray,
    feature_names: list[str],
    top_k: int = 8,
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]]]:
    shap_row = np.asarray(shap_row).ravel()
    X_row = np.asarray(X_row).ravel()
    idx = np.argsort(-np.abs(shap_row))[: int(top_k)]
    pos: list[tuple[str, float, float]] = []
    neg: list[tuple[str, float, float]] = []
    for i in idx.tolist():
        name = feature_names[i] if 0 <= i < len(feature_names) else f"feature_{i}"
        sv = float(shap_row[i])
        xv = float(X_row[i]) if i < X_row.size else float("nan")
        if sv >= 0:
            pos.append((str(name), xv, sv))
        else:
            neg.append((str(name), xv, sv))
    pos.sort(key=lambda t: -abs(t[2]))
    neg.sort(key=lambda t: -abs(t[2]))
    return pos, neg


def _attach_explanation_panel(
    *,
    fig: Any,
    title: str,
    body_lines: list[str],
    footer_lines: list[str] | None = None,
    bottom_space: float = 0.22,
) -> None:
    footer_lines = footer_lines or []
    fig.set_size_inches(12, 7.6)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.subplots_adjust(bottom=float(bottom_space), top=0.90)
    text = "\n".join(body_lines + ([""] if footer_lines else []) + footer_lines)
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=9)


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
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

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
    fig = plt.gcf()
    shap_vals = np.asarray(getattr(explanation, "values", np.asarray([])))
    global_rows = _top_global_feature_table(
        shap_values=shap_vals,
        X_values=X_ex_t,
        feature_names=feature_names,
        top_k=10,
    )
    body = [
        "阅读指南（Beeswarm）：每个点=一个样本；横轴=SHAP 值（对模型输出的边际贡献）。",
        "SHAP > 0 推高阳性(高风险)倾向；SHAP < 0 拉低阳性(低风险)倾向。",
        "颜色表示特征值大小（红=高，蓝=低）。点分布越“宽”，说明不同人群影响差异越大。",
        "",
        "Top 重要特征（按 mean|SHAP| 排序，附方向提示）：",
    ]
    for name, mean_abs, hint in global_rows:
        body.append(f"- {name}: mean|SHAP|={_format_float(mean_abs)}；{hint}")
    footer = [
        "注意：不同模型的 SHAP 单位可能不同（概率/对数几率/原始分数）。跨模型建议主要比较“排序”而不是数值大小。",
        "方向提示来自“特征值 vs SHAP 值”的线性相关，仅用于辅助理解（非因果）。",
    ]
    _attach_explanation_panel(
        fig=fig,
        title=f"SHAP Beeswarm • {file_prefix}",
        body_lines=body,
        footer_lines=footer,
        bottom_space=0.27,
    )
    plt.tight_layout(rect=[0, 0.18, 1, 0.93])
    plt.savefig(beeswarm_path, dpi=200)
    plt.close()
    paths["beeswarm"] = str(beeswarm_path)

    # 2. 条形图 / Bar plot
    bar_path = out_dir / f"{file_prefix}_shap_bar.png"
    plt.figure()
    shap.plots.bar(explanation, show=False, max_display=30)
    fig = plt.gcf()
    body = [
        "阅读指南（Bar）：条形长度=mean(|SHAP|)，表示该特征对预测的平均影响强度（越大越重要）。",
        "这是“重要性”而不是“风险方向”；方向需要结合 Beeswarm/Dependence 来看。",
        "",
        "Top 重要特征（含方向提示，便于快速汇报/写论文）：",
    ]
    for name, mean_abs, hint in global_rows:
        body.append(f"- {name}: mean|SHAP|={_format_float(mean_abs)}；{hint}")
    footer = [
        "建议写作表述：用 Bar 给出重要性排序，用 Beeswarm 展示分布与高低值对风险的推动/抑制趋势。",
    ]
    _attach_explanation_panel(
        fig=fig,
        title=f"SHAP Global Importance • {file_prefix}",
        body_lines=body,
        footer_lines=footer,
        bottom_space=0.26,
    )
    plt.tight_layout(rect=[0, 0.17, 1, 0.93])
    plt.savefig(bar_path, dpi=200)
    plt.close()
    paths["bar"] = str(bar_path)

    # 3. 全局热力图 / Global heatmap
    try:
        heatmap_path = out_dir / f"{file_prefix}_shap_heatmap.png"
        plt.figure()
        shap.plots.heatmap(explanation, show=False, max_display=30)
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=200)
        plt.close()
        paths["heatmap"] = str(heatmap_path)
    except Exception:
        pass

    # 4. 全局决策图 / Global decision plot
    try:
        values = np.asarray(getattr(explanation, "values", np.asarray([])))
        base_values = np.asarray(getattr(explanation, "base_values", np.asarray([0.0])))
        if values.ndim == 2 and values.shape[0] > 0:
            decision_path = out_dir / f"{file_prefix}_shap_decision.png"
            base = float(np.nanmean(base_values)) if base_values.size > 0 else 0.0
            plt.figure(figsize=(12, 7))
            shap.decision_plot(
                base_value=base,
                shap_values=values,
                features=X_ex_t,
                feature_names=feature_names,
                show=False,
                ignore_warnings=True,
            )
            plt.tight_layout()
            plt.savefig(decision_path, dpi=200)
            plt.close()
            paths["decision"] = str(decision_path)
    except Exception:
        pass

    # 5. 依赖图（Top 特征）/ Dependence plot (top feature)
    try:
        dep_rows = _top_global_feature_table(
            shap_values=np.asarray(getattr(explanation, "values", np.asarray([]))),
            X_values=X_ex_t,
            feature_names=feature_names,
            top_k=2,
        )
        if dep_rows:
            dep_name = dep_rows[0][0]
            dep_path = out_dir / f"{file_prefix}_shap_dependence_top1.png"
            plt.figure(figsize=(9, 6))
            shap.plots.scatter(explanation[:, dep_name], color=explanation, show=False)
            plt.tight_layout()
            plt.savefig(dep_path, dpi=200)
            plt.close()
            paths["dependence_top1"] = str(dep_path)
    except Exception:
        pass

    # 6. 交互图（Top1 by Top2 color）/ Interaction-like scatter
    try:
        dep_rows = _top_global_feature_table(
            shap_values=np.asarray(getattr(explanation, "values", np.asarray([]))),
            X_values=X_ex_t,
            feature_names=feature_names,
            top_k=2,
        )
        if len(dep_rows) >= 2:
            f1 = dep_rows[0][0]
            f2 = dep_rows[1][0]
            inter_path = out_dir / f"{file_prefix}_shap_interaction_top1_top2.png"
            plt.figure(figsize=(9, 6))
            shap.plots.scatter(explanation[:, f1], color=explanation[:, f2], show=False)
            plt.tight_layout()
            plt.savefig(inter_path, dpi=200)
            plt.close()
            paths["interaction_top1_top2"] = str(inter_path)
    except Exception:
        pass

    # 7. 交互式 HTML（dependence / interaction）/ Interactive HTML plots
    try:
        import pandas as pd
        import plotly.express as px

        dep_rows = _top_global_feature_table(
            shap_values=np.asarray(getattr(explanation, "values", np.asarray([]))),
            X_values=X_ex_t,
            feature_names=feature_names,
            top_k=2,
        )
        if dep_rows:
            top1 = dep_rows[0][0]
            idx1 = feature_names.index(top1) if top1 in feature_names else 0
            df_dep = pd.DataFrame(
                {
                    "feature_value": X_ex_t[:, idx1],
                    "shap_value": np.asarray(getattr(explanation, "values", np.asarray([])))[:, idx1],
                }
            )
            dep_html_path = out_dir / f"{file_prefix}_shap_dependence_top1_interactive.html"
            fig_dep = px.scatter(
                df_dep,
                x="feature_value",
                y="shap_value",
                title=f"SHAP Dependence (interactive): {top1}",
                opacity=0.75,
            )
            fig_dep.update_layout(template="plotly_white", height=520)
            fig_dep.write_html(dep_html_path, include_plotlyjs="cdn")
            paths["dependence_top1_html"] = str(dep_html_path)

        if len(dep_rows) >= 2:
            top1 = dep_rows[0][0]
            top2 = dep_rows[1][0]
            idx1 = feature_names.index(top1) if top1 in feature_names else 0
            idx2 = feature_names.index(top2) if top2 in feature_names else 1
            values_arr = np.asarray(getattr(explanation, "values", np.asarray([])))
            df_inter = pd.DataFrame(
                {
                    "feature_top1_value": X_ex_t[:, idx1],
                    "shap_top1_value": values_arr[:, idx1],
                    "feature_top2_value": X_ex_t[:, idx2],
                }
            )
            inter_html_path = out_dir / f"{file_prefix}_shap_interaction_top1_top2_interactive.html"
            fig_inter = px.scatter(
                df_inter,
                x="feature_top1_value",
                y="shap_top1_value",
                color="feature_top2_value",
                title=f"SHAP Interaction-like (interactive): {top1} colored by {top2}",
                opacity=0.75,
                color_continuous_scale="Viridis",
            )
            fig_inter.update_layout(template="plotly_white", height=520)
            fig_inter.write_html(inter_html_path, include_plotlyjs="cdn")
            paths["interaction_top1_top2_html"] = str(inter_html_path)
    except Exception:
        pass

    # 8. 局部瀑布图 / Local waterfall plot
    if 0 <= local_index < len(X_ex_t):
        local_path = out_dir / f"{file_prefix}_shap_waterfall_{local_index}.png"
        plt.figure()
        shap.plots.waterfall(explanation[local_index], show=False, max_display=30)
        fig = plt.gcf()
        try:
            proba = None
            if hasattr(pipeline, "predict_proba"):
                proba = float(pipeline.predict_proba(X_explain)[local_index, 1])
        except Exception:
            proba = None
        shap_row = np.asarray(getattr(explanation[local_index], "values", np.asarray([])))
        pos, neg = _local_feature_table(
            shap_row=shap_row,
            X_row=X_ex_t[local_index],
            feature_names=feature_names,
            top_k=10,
        )
        body = [
            "阅读指南（Waterfall）：从基线(base value)出发，逐项叠加每个特征的 SHAP 贡献，得到最终输出。",
            "红色(正贡献)把预测推向高风险；蓝色(负贡献)把预测拉向低风险。",
        ]
        if proba is not None:
            body.append(f"该样本预测风险概率：{_format_float(proba)}（模型输出可能与 SHAP 计算单位不同，仅做参考）")
        body.append("")
        body.append("主要正向驱动（推高风险）：")
        if pos:
            for name, xv, sv in pos[:6]:
                body.append(f"- {name}: value={_format_float(xv)}；SHAP={_format_float(sv)}")
        else:
            body.append("- 无明显正向驱动项（或已被截断）")
        body.append("主要负向抑制（降低风险）：")
        if neg:
            for name, xv, sv in neg[:6]:
                body.append(f"- {name}: value={_format_float(xv)}；SHAP={_format_float(sv)}")
        else:
            body.append("- 无明显负向抑制项（或已被截断）")
        footer = [
            "提示：若出现 one-hot 特征（如 xxx=1），说明该类别取值对预测有显著影响。",
        ]
        _attach_explanation_panel(
            fig=fig,
            title=f"SHAP Local Explanation • {file_prefix} • index={local_index}",
            body_lines=body,
            footer_lines=footer,
            bottom_space=0.33,
        )
        plt.tight_layout(rect=[0, 0.22, 1, 0.93])
        plt.savefig(local_path, dpi=200)
        plt.close()
        paths["waterfall"] = str(local_path)

        # 9. 局部 Force 图 / Local force plot
        try:
            force_path = out_dir / f"{file_prefix}_shap_force_{local_index}.png"
            plt.figure(figsize=(12, 3))
            local_exp = explanation[local_index]
            shap.force_plot(
                base_value=getattr(local_exp, "base_values", 0.0),
                shap_values=getattr(local_exp, "values", np.asarray([])),
                features=getattr(local_exp, "data", None),
                feature_names=feature_names,
                matplotlib=True,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(force_path, dpi=200)
            plt.close()
            paths["force"] = str(force_path)
        except Exception:
            pass

        # 10. 局部 Force 交互 HTML / Local interactive force plot
        try:
            local_exp = explanation[local_index]
            force_html_path = out_dir / f"{file_prefix}_shap_force_{local_index}_interactive.html"
            force_obj = shap.force_plot(
                base_value=getattr(local_exp, "base_values", 0.0),
                shap_values=getattr(local_exp, "values", np.asarray([])),
                features=getattr(local_exp, "data", None),
                feature_names=feature_names,
                matplotlib=False,
            )
            shap.save_html(str(force_html_path), force_obj)
            paths["force_html"] = str(force_html_path)
        except Exception:
            pass

    return paths
