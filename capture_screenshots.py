"""
Capture screenshots of the Streamlit app for the User Manual.
Run after the Streamlit app is already launched at localhost:8501.
"""
from __future__ import annotations

from pathlib import Path
import time

from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parent
SCREENSHOTS_DIR = BASE / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)

APP_URL = "http://localhost:8501"


def log(msg: str):
    print(msg)


def click_tab(page, tab_name: str):
    """Click a Streamlit tab by exact name."""
    tab = page.get_by_role("tab", name=tab_name, exact=True)
    tab.wait_for(state="visible", timeout=15000)
    tab.click()
    time.sleep(2)


def screenshot(page, filename: str):
    """Take full-page screenshot."""
    path = str(SCREENSHOTS_DIR / filename)
    page.screenshot(path=path, full_page=True)
    log(f"  [OK] {filename}")


def fill_number(page, label: str, value: str):
    """Fill a Streamlit number_input by aria-label."""
    inp = page.locator(f'input[aria-label="{label}"]')
    if inp.count() > 0:
        inp.click()
        inp.fill(value)
        time.sleep(0.1)
    else:
        log(f"  [WARN] number input not found: {label}")


def pick_select(page, nth: int, option_text: str):
    """Pick an option from the n-th selectbox on the page."""
    sel = page.locator('[data-baseweb="select"]').nth(nth)
    sel.click()
    time.sleep(0.4)
    opt = page.locator('li[role="option"]').filter(has_text=option_text).first
    if opt.count() > 0:
        opt.click()
    else:
        # fallback: pick by index within dropdown
        opts = page.locator('li[role="option"]').all()
        log(f"    options available: {[o.inner_text() for o in opts]}")
    time.sleep(0.3)


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page = ctx.new_page()

        log("Loading app...")
        page.goto(APP_URL, wait_until="networkidle", timeout=30000)
        time.sleep(4)

        # ── Tab 1: Predict (empty form) ──
        click_tab(page, "Predict")
        screenshot(page, "01_predict_form.png")

        # ── Tab 1: Predict (filled form + result + SHAP) ──
        log("Filling form...")

        # Numeric inputs (left column, then right)
        fill_number(page, "Age (years)", "58")
        fill_number(page, "Height (cm)", "165")
        fill_number(page, "Systolic BP (mmHg)", "145")
        fill_number(page, "Weight (kg)", "72")
        fill_number(page, "Diastolic BP (mmHg)", "92")

        # Selectboxes in DOM order:
        # [0] cholesterol (left)  [1] smoke (left)  [2] active (left)
        # [3] gender (right)      [4] gluc (right)   [5] alco (right)
        pick_select(page, 0, "2 — Above Normal")   # cholesterol
        pick_select(page, 1, "0 — No")             # smoke
        pick_select(page, 2, "1 — Yes")            # active
        pick_select(page, 3, "1 — Female")         # gender
        pick_select(page, 4, "1 — Normal")         # gluc
        pick_select(page, 5, "0 — No")             # alco

        time.sleep(1)

        # Click Run Prediction
        run_btn = page.locator('button').filter(has_text="Run Prediction").first
        if run_btn.count() > 0:
            run_btn.click()
            log("Clicked Run Prediction, waiting...")
            time.sleep(4)

        # Click Generate Local SHAP Explanation
        gen_shap = page.locator('button').filter(has_text="Generate Local SHAP Explanation").first
        if gen_shap.count() > 0:
            gen_shap.click()
            log("Generating SHAP, waiting...")
            time.sleep(8)

        screenshot(page, "02_predict_result.png")

        # ── Tab 2: Explainability ──
        click_tab(page, "Explainability")
        screenshot(page, "03_explainability.png")

        # ── Tab 3: Batch Predict ──
        click_tab(page, "Batch Predict")
        screenshot(page, "04_batch_predict.png")

        # ── Tab 4: User Manual ──
        click_tab(page, "User Manual")
        screenshot(page, "05_user_manual.png")

        # ── Tab 5: About ──
        click_tab(page, "About")
        screenshot(page, "06_about.png")

        browser.close()
        log(f"\n[DONE] All screenshots saved to {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    main()
