"""
Heart CDSS — Cardio70k + XGBoost
Clinical decision support with SHAP explainability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from heart_cdss.data import read_csv_auto
from heart_cdss.experiment import prepare_dataset
from heart_cdss.persist import load_joblib, load_json

BASE = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE / "artifacts" / "cardio70k"
CHARTS_DIR = BASE / "charts" / "global"

# ── Constants ──
YOUDEW_THRESHOLD = 0.7383
FEATURE_LABELS: dict[str, tuple[str, str | None]] = {
    "age":         ("Age (years)", "Model uses days internally; UI converts automatically."),
    "gender":      ("Gender", "1 = Female, 2 = Male"),
    "height":      ("Height (cm)", None),
    "weight":      ("Weight (kg)", None),
    "ap_hi":       ("Systolic BP (mmHg)", None),
    "ap_lo":       ("Diastolic BP (mmHg)", None),
    "cholesterol": ("Cholesterol", "1 = Normal, 2 = Above Normal, 3 = Well Above Normal"),
    "gluc":        ("Glucose", "1 = Normal, 2 = Above Normal, 3 = Well Above Normal"),
    "smoke":       ("Smoking", "0 = No, 1 = Yes"),
    "alco":        ("Alcohol intake", "0 = No, 1 = Yes"),
    "active":      ("Physical activity", "0 = No, 1 = Yes"),
}

CAT_LABELS: dict[str, dict[str, str]] = {
    "cholesterol": {"1": "1 — Normal", "2": "2 — Above Normal", "3": "3 — Well Above Normal"},
    "gluc":        {"1": "1 — Normal", "2": "2 — Above Normal", "3": "3 — Well Above Normal"},
    "gender":      {"1": "1 — Female", "2": "2 — Male"},
    "smoke":       {"0": "0 — No", "1": "1 — Yes"},
    "alco":        {"0": "0 — No", "1": "1 — Yes"},
    "active":      {"0": "0 — No", "1": "1 — Yes"},
}


# ═══════════════════════════════════════════════════════════════════════
# Load model
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    model_path = ARTIFACTS_DIR / "model.joblib"
    schema_path = ARTIFACTS_DIR / "schema.json"
    meta_path = ARTIFACTS_DIR / "meta.json"
    if not model_path.exists():
        return None, None, None
    return load_joblib(model_path), load_json(schema_path), load_json(meta_path)


# ═══════════════════════════════════════════════════════════════════════
# UI helpers
# ═══════════════════════════════════════════════════════════════════════

def inject_styles() -> None:
    st.markdown("""
    <style>
    .stApp {
      background: linear-gradient(180deg, #f8fafc, #f1f5f9);
    }
    header, footer { visibility: hidden; }
    .block-container { padding-top: 1rem; }
    .risk-card {
      border: 1px solid #e2e8f0;
      background: #ffffff;
      border-radius: 16px;
      padding: 1.2rem 1.5rem;
      margin: 0.5rem 0 1rem 0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .risk-value {
      font-size: 3rem;
      font-weight: 800;
      line-height: 1;
      margin: 0.3rem 0;
    }
    .risk-label-high {
      color: #dc2626;
      font-weight: 700;
      font-size: 1.05rem;
    }
    .risk-label-low {
      color: #16a34a;
      font-weight: 700;
      font-size: 1.05rem;
    }
    .risk-bar {
      height: 10px;
      border-radius: 999px;
      background: #e2e8f0;
      overflow: hidden;
      margin: 0.4rem 0 0.6rem 0;
    }
    .risk-bar-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #16a34a, #eab308, #ef4444);
    }
    .meta-row {
      display: flex;
      gap: 1rem;
      flex-wrap: wrap;
      margin-top: 0.4rem;
    }
    .meta-chip {
      font-size: 0.78rem;
      padding: 0.2rem 0.6rem;
      border-radius: 999px;
      border: 1px solid #cbd5e1;
      background: #f8fafc;
    }
    </style>
    """, unsafe_allow_html=True)


def risk_label(p: float, t: float) -> str:
    return "High Risk" if p >= t else "Low Risk"


def render_risk_card(proba: float, threshold: float) -> None:
    label = risk_label(proba, threshold)
    pct = int(round(min(max(proba, 0.0), 1.0) * 100))
    label_class = "risk-label-high" if label == "High Risk" else "risk-label-low"
    st.markdown(f"""
    <div class="risk-card">
      <div class="meta-row">
        <span class="meta-chip">Threshold: {threshold:.2f}</span>
        <span class="meta-chip">Youden-optimal</span>
      </div>
      <div class="risk-value">{proba:.4f}</div>
      <div class="{label_class}">Decision: {label}</div>
      <div class="risk-bar"><div class="risk-bar-fill" style="width:{pct}%;"></div></div>
      <div style="font-size:0.78rem; color:#64748b;">
        Risk probability scale (0 = low risk → 1 = high risk)
      </div>
    </div>
    """, unsafe_allow_html=True)


def build_input_form(schema: dict) -> pd.DataFrame:
    values: dict[str, object] = {}
    cols = schema.get("columns", [])
    left, right = st.columns(2)
    for pane, group in zip([left, right], [cols[::2], cols[1::2]]):
        with pane:
            for col in group:
                name = col["name"]
                label, help_text = FEATURE_LABELS.get(name, (name, None))
                if col["type"] == "numeric":
                    if name == "age":
                        values[name] = float(
                            st.number_input(label, value=50.0, min_value=0.0, max_value=120.0, step=1.0, help=help_text)
                        ) * 365.0
                    else:
                        mn = col.get("min")
                        mx = col.get("max")
                        values[name] = st.number_input(
                            label,
                            value=float(mn) if mn is not None else 0.0,
                            min_value=None if mn is None else float(mn),
                            max_value=None if mx is None else float(mx),
                            step=1.0,
                            help=help_text,
                        )
                else:
                    cats = [str(x) for x in (col.get("categories") or [])]
                    if not cats:
                        values[name] = st.text_input(label, value="", help=help_text)
                    else:
                        label_map = CAT_LABELS.get(name, {})
                        values[name] = st.selectbox(
                            label, options=cats,
                            format_func=lambda x: label_map.get(x, x),
                            help=help_text,
                        )
    return pd.DataFrame([values])


# ═══════════════════════════════════════════════════════════════════════
# SHAP — local waterfall
# ═══════════════════════════════════════════════════════════════════════

def generate_local_shap(pipeline, X_input: pd.DataFrame, out_dir: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    pre = pipeline.named_steps["preprocess"]
    model_step = pipeline.named_steps["model"]
    X_t = pre.transform(X_input)

    explainer = shap.TreeExplainer(model_step)
    shap_vals = explainer.shap_values(X_t)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = float(ev[1] if len(ev) > 1 else ev[0])

    # Feature names
    try:
        raw_names = pre.get_feature_names_out()
    except Exception:
        raw_names = [f"feature_{i}" for i in range(X_t.shape[1])]

    cn_map = {v[0]: k for k, v in FEATURE_LABELS.items()}
    feature_names = []
    for n in raw_names:
        n = str(n)
        if "__" in n:
            _, fv = n.split("__", 1)
        else:
            fv = n
        parts = fv.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit() and parts[0] in cn_map:
            label = cn_map[parts[0]]
            feature_names.append(f"{label}={parts[1]}")
        else:
            feature_names.append(fv)

    exp = shap.Explanation(
        values=shap_vals[0],
        base_values=ev,
        data=X_t[0],
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    shap.plots.waterfall(exp, max_display=10, show=False)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
    ax.set_title("SHAP Waterfall — Current Patient", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "local_waterfall.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════
# Main app
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(page_title="Heart CDSS — Cardio70k + XGBoost", layout="wide")
    inject_styles()

    model, schema, meta = load_model()
    if model is None:
        st.error("Model artifacts not found. Please run: `python build_cardio70k_xgb.py`")
        return

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## Heart CDSS")
        st.markdown("*Cardiovascular Disease Risk Assessment*")
        st.divider()

        st.markdown("### Model Card")
        st.write(f"**Dataset:** Cardio70k (N=68,635)")
        st.write(f"**Model:** XGBoost")
        best = meta.get("best_row") or {}
        if best:
            st.write(f"**ROC-AUC:** {best.get('test_roc_auc', 'N/A')}")
            st.write(f"**F1:** {best.get('test_f1', 'N/A')}")
            st.write(f"**Accuracy:** {best.get('test_accuracy', 'N/A')}")
        st.divider()

        st.markdown("### Decision Threshold")
        threshold = st.slider(
            "Threshold", min_value=0.05, max_value=0.95,
            value=YOUDEW_THRESHOLD, step=0.01,
            help="Youden-optimal threshold from experiment. Adjust to trade off sensitivity vs specificity.",
        )
        st.caption(f"Default (Youden): {YOUDEW_THRESHOLD}")

    # ── Title ──
    st.title("Cardiovascular Disease Risk Assessment")
    st.caption("Cardio70k dataset  ·  XGBoost  ·  SHAP Explainability  ·  Batch Scoring")

    tab_predict, tab_explain, tab_batch = st.tabs(["Predict", "Explainability", "Batch Predict"])

    # ════════════════════ Tab 1: Predict ════════════════════
    with tab_predict:
        st.subheader("Single Patient Prediction")

        X_input = build_input_form(schema)

        col_btn, col_spacer = st.columns([1, 3])
        with col_btn:
            submitted = st.button("Run Prediction", type="primary", width="stretch")

        if submitted:
            proba = float(model.predict_proba(X_input)[:, 1][0])
            st.session_state["last_input"] = X_input
            st.session_state["last_proba"] = proba
            st.session_state["shap_gen"] = False

        if "last_proba" in st.session_state:
            render_risk_card(float(st.session_state["last_proba"]), threshold)

            with st.expander("SHAP Explanation (Waterfall)", expanded=True):
                if st.button("Generate Local SHAP Explanation", key="gen_local"):
                    path = generate_local_shap(
                        model, st.session_state["last_input"],
                        BASE / "results" / "cardio70k" / "shap_app",
                    )
                    if path:
                        st.session_state["shap_path"] = str(path)
                        st.session_state["shap_gen"] = True

                if st.session_state.get("shap_gen") and st.session_state.get("shap_path"):
                    st.image(st.session_state["shap_path"], width="stretch")

    # ════════════════════ Tab 2: Explainability ════════════════════
    with tab_explain:
        st.subheader("Model Explainability")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Global SHAP (Beeswarm)")
            beeswarm_path = CHARTS_DIR / "fig_shap_beeswarm.png"
            if beeswarm_path.exists():
                st.image(str(beeswarm_path), width="stretch")
            else:
                st.info("Beeswarm plot not found. Run `charts/fig_shap_cardio70k.py` first.")

        with col_b:
            st.markdown("#### Global SHAP (Bar)")
            bar_path = CHARTS_DIR / "fig_shap_bar.png"
            if bar_path.exists():
                st.image(str(bar_path), width="stretch")
            else:
                st.info("Bar plot not found.")

        st.divider()
        st.caption("Global SHAP explains which features drive the model's decisions across all patients. "
                   "Local SHAP (in the Predict tab) explains a single patient's risk factors.")

    # ════════════════════ Tab 3: Batch Predict ════════════════════
    with tab_batch:
        st.subheader("Batch Prediction (CSV Upload)")
        st.caption("Upload a CSV with the same columns as the Cardio70k dataset. "
                   "Age should be in days (as in the original data).")

        uploaded = st.file_uploader("Upload CSV", type=["csv", "txt"])

        if uploaded is not None:
            head = uploaded.getvalue()[:4096]
            text = head.decode("utf-8", errors="ignore")
            first_line = text.splitlines()[0] if text.splitlines() else text
            sep = ";" if first_line.count(";") > first_line.count(",") else ","
            uploaded.seek(0)
            df_up = pd.read_csv(uploaded, sep=sep, na_values=["NA", "Na", "na", "N/A", "n/a", ""], keep_default_na=True)

            required = [c["name"] for c in schema.get("columns", [])]
            missing = [c for c in required if c not in df_up.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                df_in = df_up[required].copy()
                limit = st.number_input("Max rows to score", min_value=10, max_value=5000, value=200, step=10)
                df_in = df_in.head(int(limit))

                st.write(f"Scoring {len(df_in)} rows...")
                st.dataframe(df_in.head(5), width="stretch")

                if st.button("Run Batch Prediction", type="primary"):
                    proba = model.predict_proba(df_in)[:, 1]
                    scored = df_in.copy()
                    scored["risk_proba"] = proba
                    scored["risk_label"] = np.where(proba >= threshold, "High Risk", "Low Risk")
                    st.session_state["batch_scored"] = scored

        if "batch_scored" in st.session_state:
            scored = st.session_state["batch_scored"]
            n_high = (scored["risk_label"] == "High Risk").sum()
            n_low = len(scored) - n_high
            st.metric("High Risk", n_high)
            st.metric("Low Risk", n_low)
            st.dataframe(scored, width="stretch", height=400)

            csv_bytes = scored.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download Results CSV",
                data=csv_bytes,
                file_name="cardio70k_predictions.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
