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
SCREENSHOTS_DIR = BASE / "screenshots"

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
    footer { visibility: hidden; }
    header [data-testid="stToolbarActions"] { display: none; }
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

    /* Top-right icon buttons */
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

def generate_local_shap(pipeline, X_input: pd.DataFrame, out_dir: Path):
    """Generate a clinical-grade SHAP waterfall figure for a single patient.

    Returns a matplotlib Figure ready for st.pyplot(), or None on failure.
    """
    import copy
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

    # ── 1. Deep-copy SHAP values to prevent mutation of original data ──
    display_shap_values = copy.deepcopy(shap_vals[0])

    # ── 2. Clinical feature-name mapping ──
    CLINICAL_NAMES: dict[str, str] = {
        "age":          "Age (Years)",
        "gender":       "Gender",
        "height":       "Height (cm)",
        "weight":       "Weight (kg)",
        "ap_hi":        "Systolic Blood Pressure (mmHg)",
        "ap_lo":        "Diastolic Blood Pressure (mmHg)",
        "cholesterol":  "Cholesterol Level",
        "gluc":         "Glucose Level",
        "smoke":        "Smoking Status",
        "alco":         "Alcohol Intake",
        "active":       "Physical Activity",
    }

    # Get raw preprocessor feature names
    try:
        raw_names = pre.get_feature_names_out()
    except Exception:
        raw_names = [f"feature_{i}" for i in range(X_t.shape[1])]

    # Translate each raw feature name to its clinical equivalent
    feature_names: list[str] = []
    for n in raw_names:
        n = str(n)
        # Strip sklearn prefix  (e.g. "num__age" → "age", "cat__cholesterol_1" → "cholesterol_1")
        if "__" in n:
            _, fv = n.split("__", 1)
        else:
            fv = n
        # Handle one-hot-encoded features  (e.g. "cholesterol_1" → "Cholesterol Level=1")
        parts = fv.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit() and parts[0] in CLINICAL_NAMES:
            clinical_name = CLINICAL_NAMES[parts[0]]
            feature_names.append(f"{clinical_name}={parts[1]}")
        elif fv in CLINICAL_NAMES:
            feature_names.append(CLINICAL_NAMES[fv])
        else:
            feature_names.append(fv)

    # ── 3. Build display data — convert age from days to years ──
    display_data = X_t[0].copy()

    # Locate the "age" column index
    age_idx: int | None = None
    for i, n in enumerate(raw_names):
        n_str = str(n)
        if "__" in n_str:
            _, fv = n_str.split("__", 1)
        else:
            fv = n_str
        if fv == "age":
            age_idx = i
            break

    if age_idx is not None:
        raw_age_days = float(X_input["age"].iloc[0])
        display_data[age_idx] = round(raw_age_days / 365.25, 1)

    # ── 4. Build SHAP Explanation with display-friendly data ──
    exp = shap.Explanation(
        values=display_shap_values,
        base_values=ev,
        data=display_data,
        feature_names=feature_names,
    )

    # ── 5. Create and style the figure ──
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(exp, max_display=10, show=False)

    # Clean axis — title & caption are set by the caller via st.markdown
    ax.set_xlabel("")
    ax.set_title("")

    # ── Shrink left-side feature labels ──
    for txt in ax.texts:
        # SHAP waterfall places feature name/value labels on the left (x < 0)
        if txt.get_position()[0] < 0:
            txt.set_fontsize(7.5)
        # Also shrink the f(x) and E[f(x)] labels slightly
        elif "f(x)" in txt.get_text() or "E[f(x)]" in txt.get_text():
            txt.set_fontsize(9)

    # ── Legend — placed BELOW the plot to guarantee zero overlap ──
    from matplotlib.patches import Patch
    red_color = "#ff0051"
    blue_color = "#008bfb"
    for patch in ax.patches:
        fc = patch.get_facecolor()
        if len(fc) >= 3:
            if fc[0] > 0.5 and fc[2] < 0.3:
                red_color = fc
            elif fc[2] > 0.5 and fc[0] < 0.3:
                blue_color = fc
    legend_elements = [
        Patch(facecolor=red_color, label="Increases CVD risk  (positive SHAP)"),
        Patch(facecolor=blue_color, label="Decreases CVD risk  (negative SHAP)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=2,
              fontsize=8.5, frameon=True, framealpha=0.9, edgecolor="#cccccc")

    # Reserve space below the plot for the legend
    fig.subplots_adjust(bottom=0.14)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Main app
# ═══════════════════════════════════════════════════════════════════════

def render_user_manual() -> None:
    """Render the User Manual content."""
    st.subheader("User Manual")

    st.markdown("""
    ### Welcome to Heart CDSS

    This Clinical Decision Support System (CDSS) predicts cardiovascular disease (CVD) risk
    using a machine learning model (XGBoost) trained on 68,635 patient records from the
    Kaggle Cardiovascular Disease dataset.

    ---

    ### 1. Single Patient Prediction (Predict Tab)

    **Step 1:** Fill in the patient's clinical data in the input form.

    | Field | What to enter |
    |---|---|
    | Age | Patient's age in years (auto-converted to days) |
    | Gender | 1 = Female, 2 = Male |
    | Height | Height in centimeters |
    | Weight | Weight in kilograms |
    | Systolic BP | Upper blood pressure reading (mmHg) |
    | Diastolic BP | Lower blood pressure reading (mmHg) |
    | Cholesterol | 1 = Normal, 2 = Above Normal, 3 = Well Above Normal |
    | Glucose | 1 = Normal, 2 = Above Normal, 3 = Well Above Normal |
    | Smoking / Alcohol / Activity | 0 = No, 1 = Yes |
    """)

    pred_form_img = SCREENSHOTS_DIR / "01_predict_form.png"
    if pred_form_img.exists():
        st.markdown("**Prediction Input Form:**")
        st.image(str(pred_form_img), width="stretch")

    st.markdown("""
    **Step 2:** Click **\"Run Prediction\"** to see the risk score.

    **Step 3:** Click **\"Generate Local SHAP Explanation\"** to view a waterfall chart
    showing exactly which factors pushed the prediction toward high or low risk.
    """)

    pred_result_img = SCREENSHOTS_DIR / "02_predict_result.png"
    if pred_result_img.exists():
        st.markdown("**Prediction Result with SHAP Waterfall:**")
        st.image(str(pred_result_img), width="stretch")

    st.markdown("""
    **Interpreting the SHAP Waterfall:**
    - **Red bars (right):** Features that increase risk
    - **Blue bars (left):** Features that decrease risk
    - Bar length = strength of the feature's contribution
    - The final value is the model's log-odds prediction

    ---

    ### 2. Decision Threshold

    Use the **slider in the sidebar** to adjust the classification threshold:
    - **Lower threshold (e.g., 0.3):** Catch more at-risk patients (higher recall),
      but more false alarms
    - **Higher threshold (e.g., 0.7):** Fewer false alarms, but may miss some
      at-risk patients
    - **Default (0.7383):** Youden-optimal threshold — best balance of sensitivity
      and specificity

    ---

    ### 3. Model Explainability (Explainability Tab)

    View **global SHAP plots** that show which features drive the model across
    all patients:
    - **Beeswarm plot:** Each dot = one patient. Red = high feature value,
      Blue = low. SHAP > 0 pushes toward high risk.
    - **Bar plot:** Shows average importance of each feature across all patients.
    """)

    explain_img = SCREENSHOTS_DIR / "03_explainability.png"
    if explain_img.exists():
        st.markdown("**Global SHAP Explainability View:**")
        st.image(str(explain_img), width="stretch")

    st.markdown("""
    *Run `python charts/fig_shap_cardio70k.py` before launching the app to
    generate these plots.*

    ---

    ### 4. Batch Prediction (Batch Predict Tab)

    **Step 1:** Prepare a CSV file with the same column names as the Cardio70k dataset.

    **Step 2:** Upload the CSV and click **\"Run Batch Prediction\"**.

    **Step 3:** Download results as a CSV with added columns:
    `risk_proba` (risk probability) and `risk_label` (\"High Risk\" / \"Low Risk\").
    """)

    batch_img = SCREENSHOTS_DIR / "04_batch_predict.png"
    if batch_img.exists():
        st.markdown("**Batch Prediction Interface:**")
        st.image(str(batch_img), width="stretch")

    st.markdown("*Note: Maximum 5,000 rows per batch.*")


def render_about() -> None:
    """Render the About content."""
    st.subheader("About This Project")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("""
        ### Heart CDSS — Ensemble Learning for Early Heart Disease Detection

        This system is part of a Master of Computer Science thesis at
        **Universiti Kebangsaan Malaysia (UKM)**, Faculty of Information Science
        and Technology.

        **Research Objectives:**
        1. Evaluate and compare five ML algorithms (Logistic Regression, Random
           Forest, XGBoost, LightGBM, CatBoost) across three multi-scale clinical
           datasets
        2. Develop an interpretable, web-based CDSS prototype using Streamlit
        3. Provide both global and local model interpretability through SHAP
           visualizations

        **Deployed Model:**
        - **Algorithm:** XGBoost (best performer on Cardio70k)
        - **Training Data:** Kaggle Cardiovascular Disease dataset (N=68,635)
        - **Optimization:** RandomizedSearchCV with 5-fold stratified cross-validation
        - **Threshold:** Youden-optimal (0.7383)
        """)

    with col_right:
        st.markdown("""
        ### Author
        **Xu Qianzhou**

        ### Supervisor
        **Nur Fazidah Elias** *(UKM FTSM)*

        ### Year
        2026

        ---

        ### Tech Stack
        - Python 3.13
        - Streamlit
        - XGBoost
        - SHAP
        - scikit-learn
        """)

    st.divider()
    st.markdown("""
    ### Citation
    Xu Qianzhou. *Ensemble Learning Techniques for Early Heart Disease Detection:
    From Algorithm to Prototype Development.* Master's Thesis,
    Universiti Kebangsaan Malaysia (UKM), 2026.

    ### Disclaimer
    This system is a research prototype intended for **demonstration and educational
    purposes only**. It is not a medical device and should not be used for actual
    clinical diagnosis without proper validation and regulatory approval.
    """)

    st.info("For questions or feedback, please contact: **P158348@siswa.ukm.edu.my**")


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

    # ── Title row with top-right icon buttons ──
    col_title, col_m, col_a = st.columns([7, 0.7, 0.7])
    with col_title:
        st.title("Cardiovascular Disease Risk Assessment")
        st.caption("Cardio70k dataset  ·  XGBoost  ·  SHAP Explainability  ·  Batch Scoring")
    with col_m:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📖", key="ico_manual", help="User Manual"):
            st.session_state["page"] = "manual"
            st.rerun()
    with col_a:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("ℹ️", key="ico_about", help="About"):
            st.session_state["page"] = "about"
            st.rerun()

    # ── Check if a page was requested ──
    page = st.session_state.get("page", None)
    if page == "manual":
        col_back, _ = st.columns([1, 9])
        with col_back:
            if st.button("← Back to App", key="back_manual", width="stretch"):
                st.session_state["page"] = None
                st.rerun()
        render_user_manual()
        return

    if page == "about":
        col_back, _ = st.columns([1, 9])
        with col_back:
            if st.button("← Back to App", key="back_about", width="stretch"):
                st.session_state["page"] = None
                st.rerun()
        render_about()
        return

    # ── Main tabs ──
    tab_predict, tab_explain, tab_batch = st.tabs([
        "Predict", "Explainability", "Batch Predict"
    ])

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
                    fig = generate_local_shap(
                        model, st.session_state["last_input"],
                        BASE / "results" / "cardio70k" / "shap_app",
                    )
                    if fig is not None:
                        st.session_state["shap_fig"] = fig
                        st.session_state["shap_gen"] = True

                if st.session_state.get("shap_gen") and st.session_state.get("shap_fig"):
                    st.markdown("### :bar_chart: SHAP Individual Patient Risk Decomposition")
                    st.caption(
                        "This waterfall plot employs **SHAP (SHapley Additive "
                        "exPlanations)** to decompose how individual clinical "
                        "features contribute to the patient's predicted CVD risk. "
                        "**E[f(x)]** is the cohort baseline (expected model output "
                        "across the population). Each bar shows a feature's marginal "
                        "contribution; their cumulative sum yields the final output "
                        "**f(x)** — the patient's individual risk score **in log-odds "
                        "units**. The predicted probability displayed above is obtained "
                        "by applying the logistic (sigmoid) transform: "
                        "$p = 1 / (1 + e^{-f(x)})$, mapping log-odds from "
                        "$(-\\infty, +\\infty)$ to a risk probability in $[0, 1]$."
                    )
                    st.pyplot(st.session_state["shap_fig"])

    # ════════════════════ Tab 2: Explainability ════════════════════
    with tab_explain:
        st.subheader("Model Explainability")

        st.markdown("### Global Explanation — Cardio70k XGBoost")
        st.caption(
            "These plots employ **SHAP (SHapley Additive exPlanations)** to reveal "
            "which clinical features drive the model's CVD risk predictions across "
            "the entire test cohort (N≈13,727). Unlike the single-patient waterfall "
            "in the Predict tab, these global views aggregate SHAP values from all "
            "patients to identify population-level feature importance patterns."
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Beeswarm (Summary) Plot")
            beeswarm_path = CHARTS_DIR / "fig_shap_beeswarm.png"
            if beeswarm_path.exists():
                st.image(str(beeswarm_path), width="stretch")
            else:
                st.info("Beeswarm plot not found. Run `charts/fig_shap_cardio70k.py` first.")
            st.caption(
                "**How to read:** Each dot represents one patient. The horizontal "
                "position shows the SHAP value — dots to the right of zero indicate "
                "features pushing toward higher CVD risk; dots to the left indicate "
                "features pushing toward lower risk. **Colour** encodes the actual "
                "feature value: red = high, blue = low. A cluster of red dots on the "
                "right means patients with high values of that feature tend to have "
                "elevated risk."
            )

        with col_b:
            st.markdown("#### Bar (Importance) Plot")
            bar_path = CHARTS_DIR / "fig_shap_bar.png"
            if bar_path.exists():
                st.image(str(bar_path), width="stretch")
            else:
                st.info("Bar plot not found.")
            st.caption(
                "**How to read:** Bar length represents the mean absolute SHAP value "
                "across all patients — a direct measure of each feature's average "
                "impact on the model's output. Longer bars indicate features that "
                "consistently influence CVD risk prediction, regardless of direction. "
                "This is the global importance ranking; refer to the Beeswarm plot "
                "for directional (risk-increasing vs. risk-decreasing) patterns."
            )

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
