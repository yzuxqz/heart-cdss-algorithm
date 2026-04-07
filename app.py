from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from heart_cdss.data import read_csv_auto
from heart_cdss.explain import generate_shap_outputs
from heart_cdss.experiment import prepare_dataset
from heart_cdss.persist import load_joblib, load_json
from heart_cdss.reporting import generate_pdf_report
from heart_cdss.audit import append_event_csv


DATASETS = {
    "UCI Cleveland (Current Heart Disease)": "uci_cleveland",
    "Framingham (10-year CHD)": "framingham",
    "Cardio 70k (CVD)": "cardio70k",
}


DATASET_CONFIG = {
    "uci_cleveland": {
        "csv": "heart_disease_uci.csv",
        "target": "num",
        "prepare_name": "uci_cleveland",
        "task": "current heart disease (binary)",
    },
    "framingham": {
        "csv": "framingham.csv",
        "target": "TenYearCHD",
        "prepare_name": "framingham",
        "task": "10-year CHD (binary)",
    },
    "cardio70k": {
        "csv": "cardio_train.csv",
        "target": "cardio",
        "prepare_name": "cardio70k",
        "task": "cardiovascular disease (binary)",
    },
}


def _artifact_dir(dataset_code: str) -> Path:
    return Path(__file__).resolve().parent / "artifacts" / dataset_code


@st.cache_resource
def _load_model_and_schema(dataset_code: str):
    """
    鍔犺浇妯″瀷銆佹暟鎹灦鏋勫拰鍏冩暟鎹?/ Load model, schema, and metadata.

    涓枃锛氫粠 artifacts 鐩綍鍔犺浇宸茶缁冪殑 Joblib 妯″瀷銆丣SON 鏋舵瀯鍜屽厓鏁版嵁銆?    English: Loads trained Joblib model, JSON schema, and meta from artifacts directory.
    """
    ad = _artifact_dir(dataset_code)
    model_path = ad / "model.joblib"
    schema_path = ad / "schema.json"
    meta_path = ad / "meta.json"
    if not model_path.exists():
        return None, None, None
    return load_joblib(model_path), load_json(schema_path), load_json(meta_path)


def _load_model_by_name(dataset_code: str, model_name: str):
    ad = _artifact_dir(dataset_code)
    p = ad / "models" / model_name / "model.joblib"
    if not p.exists():
        return None
    return load_joblib(p)


def _available_models(meta: dict) -> list[str]:
    entries = meta.get("available_models") or []
    out: list[str] = []
    for e in entries:
        m = e.get("model")
        if m and m not in out:
            out.append(str(m))
    return out


def _results_dir(dataset_code: str) -> Path:
    return Path(__file__).resolve().parent / "results" / dataset_code


def _inject_branding() -> None:
    st.markdown(
        """
        <style>
        .stApp {
          background:
            radial-gradient(1200px 600px at 10% 10%, rgba(94, 234, 212, 0.10), transparent 60%),
            radial-gradient(900px 500px at 90% 0%, rgba(59, 130, 246, 0.12), transparent 55%),
            radial-gradient(900px 500px at 90% 90%, rgba(244, 114, 182, 0.10), transparent 55%),
            linear-gradient(180deg, rgba(3, 7, 18, 1), rgba(2, 6, 23, 1));
        }
        footer { visibility: hidden; }
        .block-container { padding-top: 4.5rem; }
        .hc-title {
          font-weight: 800;
          letter-spacing: -0.02em;
          font-size: 1.65rem;
          margin: 0 0 .2rem 0;
        }
        .hc-subtitle {
          opacity: .75;
          margin: 0 0 1rem 0;
          font-size: .95rem;
        }
        .hc-badge {
          display: inline-block;
          padding: .25rem .55rem;
          border-radius: 999px;
          font-size: .78rem;
          margin-right: .4rem;
          border: 1px solid rgba(148, 163, 184, .25);
          background: rgba(15, 23, 42, .55);
        }
        .hc-card {
          border: 1px solid rgba(148, 163, 184, .18);
          background: rgba(2, 6, 23, .55);
          border-radius: 14px;
          padding: 1rem 1rem .85rem 1rem;
        }
        .hc-metric {
          font-size: 2.2rem;
          font-weight: 800;
          line-height: 1;
          margin: .25rem 0 .4rem 0;
        }
        .hc-meter {
          height: 10px;
          border-radius: 999px;
          background: rgba(148, 163, 184, .20);
          overflow: hidden;
        }
        .hc-meter > div {
          height: 100%;
          border-radius: 999px;
          background: linear-gradient(90deg, rgba(34,197,94,1), rgba(59,130,246,1), rgba(239,68,68,1));
        }
        .hc-note {
          opacity: .75;
          font-size: .85rem;
          margin-top: .5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _latest_summary_path(dataset_code: str) -> Path | None:
    rd = _results_dir(dataset_code)
    paths = sorted(rd.glob("*_summary.csv"), key=lambda p: p.name, reverse=True)
    return paths[0] if paths else None


@st.cache_data
def _load_summary_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def _load_raw_X(dataset_code: str) -> pd.DataFrame:
    base = Path(__file__).resolve().parent
    cfg = DATASET_CONFIG[dataset_code]
    df = read_csv_auto(base / cfg["csv"])
    X, _y = prepare_dataset(df, cfg["prepare_name"], cfg["target"])
    return X


def _format_probability(p: float) -> str:
    return f"{p:.4f}"


def _risk_label(p: float, threshold: float) -> str:
    return "High Risk" if p >= threshold else "Low Risk"


def _render_risk_meter(proba: float, threshold: float) -> None:
    """
    娓叉煋椋庨櫓搴﹂噺琛?/ Render risk risk meter UI.

    涓枃锛氬湪 Streamlit 涓樉绀哄喅绛栭槇鍊笺€佸綋鍓嶉闄╁緱鍒嗗強褰╄壊椋庨櫓鏉°€?    English: Displays decision threshold, risk score, and color-coded risk meter in Streamlit.
    """
    label = _risk_label(proba, threshold)
    pct = int(round(min(max(proba, 0.0), 1.0) * 100))
    st.markdown(
        f"""
        <div class="hc-card">
          <div class="hc-badge">Threshold: {threshold:.2f}</div>
          <div class="hc-badge">Decision: {label}</div>
          <div class="hc-metric">{proba:.4f}</div>
          <div class="hc-meter"><div style="width:{pct}%;"></div></div>
          <div class="hc-note">Risk probability scale (0 鈫?1). You can adjust the threshold in the sidebar.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_value_help(dataset_code: str, name: str) -> tuple[str, str | None]:
    if dataset_code == "cardio70k":
        labels = {
            "age": ("Age (years)", None),
            "gender": ("Gender", "1=Female, 2=Male"),
            "height": ("Height (cm)", None),
            "weight": ("Weight (kg)", None),
            "ap_hi": ("Systolic BP (mmHg)", None),
            "ap_lo": ("Diastolic BP (mmHg)", None),
            "cholesterol": ("Cholesterol", "1=normal, 2=above normal, 3=well above normal"),
            "gluc": ("Glucose", "1=normal, 2=above normal, 3=well above normal"),
            "smoke": ("Smoking", "0=No, 1=Yes"),
            "alco": ("Alcohol intake", "0=No, 1=Yes"),
            "active": ("Physical activity", "0=No, 1=Yes"),
        }
        if name in labels:
            return labels[name]
    if dataset_code == "framingham":
        labels = {
            "male": ("Sex (male=1)", None),
            "currentSmoker": ("Current smoker", None),
            "cigsPerDay": ("Cigarettes per day", None),
            "BPMeds": ("BP medication", None),
            "prevalentStroke": ("Previous stroke", None),
            "prevalentHyp": ("Prevalent hypertension", None),
            "totChol": ("Total cholesterol", None),
            "sysBP": ("Systolic BP", None),
            "diaBP": ("Diastolic BP", None),
            "BMI": ("BMI", None),
            "heartRate": ("Heart rate", None),
            "glucose": ("Glucose", None),
        }
        if name in labels:
            return labels[name]
    return name, None


def _selectbox_with_labels(label: str, options: list, label_map: dict[str, str], help_text: str | None):
    options_str = [str(o) for o in options]

    def _fmt(x: str) -> str:
        return label_map.get(x, x)

    selected = st.selectbox(label, options=options_str, format_func=_fmt, help=help_text)
    return selected


def _build_single_input(schema: dict, dataset_code: str) -> pd.DataFrame:
    values: dict[str, object] = {}
    cols = schema.get("columns", [])
    left, right = st.columns(2)
    halves = [cols[::2], cols[1::2]]
    for pane, group in zip([left, right], halves, strict=False):
        with pane:
            for col in group:
                name = col["name"]
                label, help_text = _build_value_help(dataset_code, name)
                if col["type"] == "numeric":
                    min_v = col.get("min")
                    max_v = col.get("max")
                    default = min_v if min_v is not None else 0.0
                    if dataset_code == "cardio70k" and name == "age":
                        default_years = float(default) / 365.0 if default is not None else 50.0
                        values[name] = float(
                            st.number_input(
                                label,
                                value=float(default_years),
                                min_value=0.0,
                                max_value=150.0,
                                step=1.0,
                                help="Model expects age in days; UI uses years and converts automatically.",
                            )
                            * 365.0
                        )
                    else:
                        values[name] = st.number_input(
                            label,
                            value=float(default),
                            min_value=None if min_v is None else float(min_v),
                            max_value=None if max_v is None else float(max_v),
                            step=1.0,
                            help=help_text,
                        )
                else:
                    cats = col.get("categories") or []
                    if not cats:
                        values[name] = st.text_input(label, value="", help=help_text)
                    else:
                        if dataset_code == "cardio70k" and name in {"cholesterol", "gluc"}:
                            label_map = {"1": "1 - normal", "2": "2 - above normal", "3": "3 - well above normal"}
                            values[name] = _selectbox_with_labels(label, cats, label_map, help_text)
                        elif dataset_code == "cardio70k" and name == "gender":
                            label_map = {"1": "1 - Female", "2": "2 - Male"}
                            values[name] = _selectbox_with_labels(label, cats, label_map, help_text)
                        elif dataset_code == "cardio70k" and name in {"smoke", "alco", "active"}:
                            label_map = {"0": "0 - No", "1": "1 - Yes"}
                            values[name] = _selectbox_with_labels(label, cats, label_map, help_text)
                        else:
                            values[name] = st.selectbox(label, options=[str(x) for x in cats], index=0, help=help_text)
    return pd.DataFrame([values])


def _read_uploaded_csv(uploaded) -> pd.DataFrame:
    head = uploaded.getvalue()[:4096]
    text = head.decode("utf-8", errors="ignore")
    first_line = text.splitlines()[0] if text.splitlines() else text
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    uploaded.seek(0)
    return pd.read_csv(
        uploaded,
        sep=sep,
        na_values=["NA", "Na", "na", "N/A", "n/a", ""],
        keep_default_na=True,
    )


def _schema_columns(schema: dict) -> list[str]:
    return [c["name"] for c in schema.get("columns", [])]


def _predict_batch(model, df_in: pd.DataFrame, threshold: float) -> pd.DataFrame:
    proba = model.predict_proba(df_in)[:, 1]
    out = df_in.copy()
    out["risk_proba"] = proba
    out["risk_label"] = np.where(out["risk_proba"] >= threshold, "High Risk", "Low Risk")
    out["predicted_class"] = (out["risk_proba"] >= threshold).astype(int)
    return out


def _render_model_card(meta: dict, dataset_code: str) -> None:
    st.sidebar.subheader("Model Card")
    st.sidebar.write("Task:", DATASET_CONFIG[dataset_code]["task"])
    st.sidebar.write("Best model:", meta.get("best_model"))
    st.sidebar.write("Best run:", meta.get("best_run_id"))
    best_row = meta.get("best_row") or {}
    if best_row:
        st.sidebar.write("Test ROC-AUC:", best_row.get("test_roc_auc"))
        st.sidebar.write("Test F1:", best_row.get("test_f1"))
        st.sidebar.write("Test Accuracy:", best_row.get("test_accuracy"))


def _list_shap_images(dataset_code: str) -> list[Path]:
    rd = _results_dir(dataset_code)
    return sorted(rd.glob("**/*shap*.png"), key=lambda p: p.name, reverse=True)


def _render_shap_gallery(dataset_code: str) -> None:
    imgs = _list_shap_images(dataset_code)
    st.write("SHAP image folder:", str(_results_dir(dataset_code)))
    if not imgs:
        st.info("No SHAP images yet. Please click SHAP generation buttons (Local / Global).")
        return

    cols = st.columns(3)
    for i, p in enumerate(imgs[:18]):
        with cols[i % 3]:
            st.image(str(p), caption=p.name, use_container_width=True)
            st.download_button(
                label="Download",
                data=p.read_bytes(),
                file_name=p.name,
                key=f"dl_{dataset_code}_{p.name}",
            )


def _list_shap_html(dataset_code: str) -> list[Path]:
    rd = _results_dir(dataset_code)
    return sorted(rd.glob("**/*shap*interactive*.html"), key=lambda p: p.name, reverse=True)


def _render_shap_html_gallery(dataset_code: str, *, key_prefix: str = "xai") -> None:
    htmls = _list_shap_html(dataset_code)
    if not htmls:
        st.info("No interactive SHAP HTML yet. Generate SHAP first.")
        return
    for i, p in enumerate(htmls[:8]):
        st.markdown(f"**{p.name}**")
        components.html(p.read_text(encoding="utf-8", errors="ignore"), height=560, scrolling=True)
        key_safe = f"{dataset_code}_{i}_{p.parent.name}_{p.name}"
        st.download_button(
            label="Download HTML",
            data=p.read_bytes(),
            file_name=p.name,
            key=f"dl_html_{key_prefix}_{key_safe}",
        )


def _model_for_explain(dataset_code: str, meta: dict, model_name: str | None, fallback_model):
    if not model_name:
        return fallback_model
    if str(model_name) == str(meta.get("best_model")):
        return fallback_model
    m = _load_model_by_name(dataset_code, str(model_name))
    return fallback_model if m is None else m




    _inject_branding()

    st.markdown('<div class="hc-title">Heart CDSS - Multi-Dataset - XAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hc-subtitle">A clinical-style decision support prototype: multi-dataset benchmarking, deployment-ready pipelines, SHAP explanations, batch scoring, and report export.</div>',
        unsafe_allow_html=True,
    )
    st.set_page_config(page_title="Heart CDSS", layout="wide", initial_sidebar_state="expanded")
    st.title("Clinical Decision Support System (Multi-Dataset)")
    dataset_label = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
    dataset_code = DATASETS[dataset_label]

    st.session_state["role"] = role

    model, schema, meta = _load_model_and_schema(dataset_code)
    if model is None:
        st.error("鏈壘鍒板凡璁粌妯″瀷銆傝鍏堣繍琛岋細py build_system_artifacts.py")
        return
    _render_model_card(meta, dataset_code)

    st.sidebar.subheader("Decision Settings")
    threshold = st.sidebar.slider("Decision Threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01)

    tab_predict, tab_batch, tab_models, tab_xai, tab_report = st.tabs(
        ["Predict", "Batch Predict", "Model Comparison", "Explainability", "Report"]
    )

    with tab_predict:
        st.subheader("Single Patient Prediction")

        models_for_ui = _available_models(meta)
        selected_model = st.selectbox(
            "Model for prediction",
            options=models_for_ui if models_for_ui else [meta.get("best_model")],
            index=0,
        )
        model_for_pred = model if selected_model == meta.get("best_model") else _load_model_by_name(dataset_code, str(selected_model))
        if model_for_pred is None:
            model_for_pred = model

        with st.form(key=f"single_{dataset_code}"):
            X_input = _build_single_input(schema, dataset_code)
            submitted = st.form_submit_button("Predict")

        if submitted:
            proba = float(model_for_pred.predict_proba(X_input)[:, 1][0])
            st.session_state["last_input"] = X_input
            st.session_state["last_proba"] = proba
            st.session_state["last_dataset"] = dataset_code
            st.session_state["last_model"] = str(selected_model)

            append_event_csv(
                Path(__file__).resolve().parent / "logs" / "predictions.csv",
                {
                    "dataset": dataset_code,
                    "model": str(selected_model),
                    "role": role,
                    "threshold": float(threshold),
                    "risk_proba": float(proba),
                    "risk_label": _risk_label(proba, threshold),
                },
            )

        if st.session_state.get("last_dataset") == dataset_code and "last_proba" in st.session_state:
            proba = float(st.session_state["last_proba"])
            _render_risk_meter(proba, threshold)

            with st.expander("Explain this prediction (SHAP waterfall)", expanded=False):
                if st.button("Generate local SHAP", key=f"btn_local_{dataset_code}"):
                    X_bg = _load_raw_X(dataset_code).sample(n=min(200, len(_load_raw_X(dataset_code))), random_state=42)
                    out_dir = _results_dir(dataset_code) / "shap_app"
                    model_to_explain = _model_for_explain(dataset_code, meta, st.session_state.get("last_model"), model)
                    paths = generate_shap_outputs(
                        pipeline=model_to_explain,
                        X_background=X_bg,
                        X_explain=st.session_state["last_input"],
                        out_dir=out_dir,
                        file_prefix=f"single_{dataset_code}_{st.session_state.get('last_model', 'best')}",
                        local_index=0,
                    )
                    st.session_state["local_shap_paths"] = paths
                p = st.session_state.get("local_shap_paths") or {}
                if p.get("waterfall"):
                    st.image(p["waterfall"], caption="SHAP waterfall", use_container_width=True)

            if models_for_ui:
                st.subheader("Multi-Model Comparison (Same Input)")
                compare_models = st.multiselect(
                    "Models to compare",
                    options=models_for_ui,
                    default=models_for_ui[: min(5, len(models_for_ui))],
                )
                if st.button("Run Comparison"):
                    X_cmp = st.session_state["last_input"]
                    rows = []
                    for m in compare_models:
                        m_obj = model if m == meta.get("best_model") else _load_model_by_name(dataset_code, str(m))
                        if m_obj is None:
                            continue
                        p = float(m_obj.predict_proba(X_cmp)[:, 1][0])
                        rows.append(
                            {
                                "model": str(m),
                                "risk_proba": p,
                                "risk_label": _risk_label(p, threshold),
                            }
                        )
                    if rows:
                        df_cmp = pd.DataFrame(rows).sort_values(by="risk_proba", ascending=False)
                        st.dataframe(df_cmp, use_container_width=True)
                        st.session_state["last_comparison"] = df_cmp

    with tab_batch:
        st.subheader("Batch Prediction (Upload CSV)")
        uploaded = st.file_uploader("Upload CSV file", type=["csv", "txt"])
        if uploaded is not None:
            df_up = _read_uploaded_csv(uploaded)
            required = _schema_columns(schema)
            missing = [c for c in required if c not in df_up.columns]
            if missing:
                st.error(f"缂哄皯鍒? {missing[:10]}{'...' if len(missing) > 10 else ''}")
            else:
                df_in = df_up[required].copy()
                if dataset_code == "cardio70k" and "age" in df_in.columns:
                    st.info("Cardio70k 的 age 需要是 days（与原始数据一致）。如果你上传的是 years，请先转换。")
                limit = st.number_input("Max rows to score", min_value=10, max_value=5000, value=500, step=10)
                df_in = df_in.head(int(limit))
                if st.button("Run Batch Prediction"):
                    scored = _predict_batch(model, df_in, threshold)
                    st.session_state["batch_scored"] = scored

        if "batch_scored" in st.session_state:
            scored = st.session_state["batch_scored"]
            st.dataframe(scored, use_container_width=True, height=420)
            csv_bytes = scored.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Download Predictions CSV", data=csv_bytes, file_name=f"{dataset_code}_predictions.csv")
            idx = st.number_input("Row index for SHAP explanation", min_value=0, max_value=max(0, len(scored) - 1), value=0, step=1)
            if st.button("Explain selected row (SHAP)"):
                X_row = scored[_schema_columns(schema)].iloc[[int(idx)]]
                X_bg = _load_raw_X(dataset_code).sample(n=min(200, len(_load_raw_X(dataset_code))), random_state=42)
                out_dir = _results_dir(dataset_code) / "shap_app"
                paths = generate_shap_outputs(
                    pipeline=model,
                    X_background=X_bg,
                    X_explain=X_row,
                    out_dir=out_dir,
                    file_prefix=f"batch_{dataset_code}",
                    local_index=0,
                )
                if not paths:
                    st.error("SHAP 鏈畨瑁呮垨鐢熸垚澶辫触")
                else:
                    st.image(paths.get("waterfall"), caption="Waterfall", use_container_width=True)

    with tab_models:
        st.subheader("Latest Experiment Summary")
        sp = _latest_summary_path(dataset_code)
        if sp is None:
            st.warning("未找到该数据集的 summary.csv，请先运行实验。")
        else:
            df_sum = _load_summary_df(str(sp))
            st.write("Summary file:", str(sp))
            st.dataframe(df_sum.sort_values(by=["test_roc_auc", "test_f1"], ascending=False), use_container_width=True)
            st.download_button(
                "Download Summary CSV",
                data=Path(sp).read_bytes(),
                file_name=Path(sp).name,
            )

    with tab_xai:
        st.subheader("Explainability (SHAP)")
        st.write("SHAP 输出会保存到 results/<dataset>/ 目录下，同时你可以在下面的 Gallery 直接查看/下载。")
        st.divider()

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Global SHAP")
            models_for_ui = _available_models(meta)
            m_choice = st.selectbox(
                "Model to explain",
                options=models_for_ui if models_for_ui else [meta.get("best_model")],
                index=0,
                key=f"global_model_{dataset_code}",
            )
            bg_n = st.slider("Background samples", min_value=50, max_value=800, value=200, step=50)
            exp_n = st.slider("Explain samples", min_value=50, max_value=800, value=200, step=50)
            if st.button("Generate global SHAP (beeswarm + bar)", key=f"btn_global_{dataset_code}"):
                X_raw = _load_raw_X(dataset_code)
                X_bg = X_raw.sample(n=min(int(bg_n), len(X_raw)), random_state=42)
                X_exp = X_raw.sample(n=min(int(exp_n), len(X_raw)), random_state=7)
                out_dir = _results_dir(dataset_code) / "shap_global"
                model_to_explain = _model_for_explain(dataset_code, meta, str(m_choice), model)
                paths = generate_shap_outputs(
                    pipeline=model_to_explain,
                    X_background=X_bg,
                    X_explain=X_exp,
                    out_dir=out_dir,
                    file_prefix=f"global_{dataset_code}_{m_choice}",
                    local_index=0,
                )
                st.session_state["global_shap_paths"] = paths
            p = st.session_state.get("global_shap_paths") or {}
            if p.get("bar"):
                st.image(p["bar"], caption="Global bar", use_container_width=True)
            if p.get("beeswarm"):
                st.image(p["beeswarm"], caption="Global beeswarm", use_container_width=True)

        with right:
            st.subheader("Local SHAP")
            st.write("Local SHAP 使用你最后一次预测的输入。")
            if st.button("Generate local SHAP (waterfall)", key=f"btn_local_tab_{dataset_code}"):
                if st.session_state.get("last_dataset") != dataset_code or "last_input" not in st.session_state:
                    st.warning("请先在 Predict 页做一次预测。")
                else:
                    X_bg = _load_raw_X(dataset_code).sample(n=min(200, len(_load_raw_X(dataset_code))), random_state=42)
                    out_dir = _results_dir(dataset_code) / "shap_app"
                    model_to_explain = _model_for_explain(dataset_code, meta, st.session_state.get("last_model"), model)
                    paths = generate_shap_outputs(
                        pipeline=model_to_explain,
                        X_background=X_bg,
                        X_explain=st.session_state["last_input"],
                        out_dir=out_dir,
                        file_prefix=f"single_{dataset_code}_{st.session_state.get('last_model','best')}",
                        local_index=0,
                    )
                    st.session_state["local_shap_paths"] = paths
            p = st.session_state.get("local_shap_paths") or {}
            if p.get("waterfall"):
                st.image(p["waterfall"], caption="Local waterfall", use_container_width=True)

        st.divider()
        st.subheader("SHAP Gallery (saved images)")
        _render_shap_gallery(dataset_code)

    with tab_report:
        st.subheader("Export Report (PDF)")
        st.write("Generate a PDF report for the last single prediction, including optional SHAP images.")
        if st.session_state.get("last_dataset") != dataset_code or "last_input" not in st.session_state:
            st.warning("请先在 Predict 页完成一次预测。")
        else:
            include_shap = st.checkbox("Include SHAP waterfall", value=True)
            include_global = st.checkbox("Include global SHAP (bar + beeswarm)", value=False)
            if st.button("Generate PDF"):
                base = Path(__file__).resolve().parent
                ds = st.session_state["last_dataset"]
                inp = st.session_state["last_input"].iloc[0].to_dict()
                preds = []
                main_model = st.session_state.get("last_model") or meta.get("best_model")
                preds.append(
                    {
                        "model": str(main_model),
                        "proba": float(st.session_state["last_proba"]),
                        "threshold": float(threshold),
                        "label": _risk_label(float(st.session_state["last_proba"]), threshold),
                    }
                )
                if "last_comparison" in st.session_state:
                    for _, r in st.session_state["last_comparison"].iterrows():
                        preds.append(
                            {
                                "model": str(r["model"]),
                                "proba": float(r["risk_proba"]),
                                "threshold": float(threshold),
                                "label": str(r["risk_label"]),
                            }
                        )

                shap_imgs: list[Path] = []
                if include_shap:
                    X_bg = _load_raw_X(ds).sample(n=min(200, len(_load_raw_X(ds))), random_state=42)
                    out_dir = _results_dir(ds) / "shap_report"
                    paths = generate_shap_outputs(
                        pipeline=model,
                        X_background=X_bg,
                        X_explain=st.session_state["last_input"],
                        out_dir=out_dir,
                        file_prefix="report_local",
                        local_index=0,
                    )
                    if paths.get("waterfall"):
                        shap_imgs.append(Path(paths["waterfall"]))
                if include_global:
                    X_bg = _load_raw_X(ds).sample(n=min(200, len(_load_raw_X(ds))), random_state=42)
                    X_exp = _load_raw_X(ds).sample(n=min(200, len(_load_raw_X(ds))), random_state=7)
                    out_dir = _results_dir(ds) / "shap_report"
                    paths = generate_shap_outputs(
                        pipeline=model,
                        X_background=X_bg,
                        X_explain=X_exp,
                        out_dir=out_dir,
                        file_prefix="report_global",
                        local_index=0,
                    )
                    for k in ["bar", "beeswarm"]:
                        if paths.get(k):
                            shap_imgs.append(Path(paths[k]))

                out_path = base / "reports" / f"{ds}_report.pdf"
                pdf_path = generate_pdf_report(
                    out_path=out_path,
                    title="Heart CDSS Report",
                    meta={
                        "dataset": ds,
                        "role": role,
                        "best_model": meta.get("best_model"),
                        "best_run_id": meta.get("best_run_id"),
                    },
                    input_row=inp,
                    predictions=preds,
                    shap_image_paths=shap_imgs,
                )
                st.success(f"Generated: {pdf_path}")
                st.download_button("Download PDF", data=pdf_path.read_bytes(), file_name=pdf_path.name)
def main() -> None:
    st.set_page_config(page_title="Heart CDSS", layout="wide", initial_sidebar_state="expanded")
    _inject_branding()
    st.markdown('<div class="hc-title">Heart CDSS - Multi-Dataset - XAI</div>', unsafe_allow_html=True)

    dataset_label = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
    dataset_code = DATASETS[dataset_label]
    role = st.sidebar.selectbox("Role", ["Doctor", "Student"])
    threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, 0.50, 0.01)
    st.session_state["role"] = role

    model, schema, meta = _load_model_and_schema(dataset_code)
    if model is None:
        st.error("鏈壘鍒版ā鍨嬫枃浠讹紝璇峰厛杩愯: py build_system_artifacts.py")
        return

    _render_model_card(meta, dataset_code)
    tab_predict, tab_xai, tab_xai_interactive = st.tabs(["Predict", "Explainability", "Interactive SHAP"])

    with tab_predict:
        st.subheader("Single Patient Prediction")
        X_input = _build_single_input(schema, dataset_code)
        if st.button("Predict", key=f"predict_{dataset_code}"):
            proba = float(model.predict_proba(X_input)[:, 1][0])
            st.session_state["last_input"] = X_input
            st.session_state["last_dataset"] = dataset_code
            st.session_state["last_proba"] = proba
            _render_risk_meter(proba, threshold)
            append_event_csv(
                Path(__file__).resolve().parent / "logs" / "predictions.csv",
                {
                    "dataset": dataset_code,
                    "role": role,
                    "threshold": float(threshold),
                    "risk_proba": float(proba),
                    "risk_label": _risk_label(proba, threshold),
                },
            )
        elif st.session_state.get("last_dataset") == dataset_code and "last_proba" in st.session_state:
            _render_risk_meter(float(st.session_state["last_proba"]), threshold)

    with tab_xai:
        st.subheader("SHAP")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Global SHAP", key=f"global_{dataset_code}"):
                X_raw = _load_raw_X(dataset_code)
                X_bg = X_raw.sample(n=min(200, len(X_raw)), random_state=42)
                X_exp = X_raw.sample(n=min(200, len(X_raw)), random_state=7)
                paths = generate_shap_outputs(
                    pipeline=model,
                    X_background=X_bg,
                    X_explain=X_exp,
                    out_dir=_results_dir(dataset_code) / "shap_global",
                    file_prefix=f"global_{dataset_code}",
                    local_index=0,
                )
                st.session_state["global_shap_paths"] = paths
            p = st.session_state.get("global_shap_paths") or {}
            if p.get("bar"):
                st.image(p["bar"], caption="Global SHAP Bar", use_container_width=True)
            if p.get("beeswarm"):
                st.image(p["beeswarm"], caption="Global SHAP Beeswarm", use_container_width=True)
        with col2:
            if st.button("Generate Local SHAP", key=f"local_{dataset_code}"):
                if st.session_state.get("last_dataset") != dataset_code or "last_input" not in st.session_state:
                    st.warning("请先在 Predict 页做一次预测。")
                else:
                    X_bg = _load_raw_X(dataset_code).sample(n=min(200, len(_load_raw_X(dataset_code))), random_state=42)
                    paths = generate_shap_outputs(
                        pipeline=model,
                        X_background=X_bg,
                        X_explain=st.session_state["last_input"],
                        out_dir=_results_dir(dataset_code) / "shap_app",
                        file_prefix=f"single_{dataset_code}",
                        local_index=0,
                    )
                    st.session_state["local_shap_paths"] = paths
            p = st.session_state.get("local_shap_paths") or {}
            if p.get("waterfall"):
                st.image(p["waterfall"], caption="Local SHAP Waterfall", use_container_width=True)

        st.divider()
        _render_shap_gallery(dataset_code)
        st.subheader("Interactive SHAP Gallery (saved HTML)")
        _render_shap_html_gallery(dataset_code, key_prefix="tab_xai")

    with tab_xai_interactive:
        st.subheader("Interactive SHAP")
        st.write("This tab shows interactive SHAP HTML charts (zoom / hover / pan).")
        _render_shap_html_gallery(dataset_code, key_prefix="tab_xai_interactive")


if __name__ == "__main__":
    main()
