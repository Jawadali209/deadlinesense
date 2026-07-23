"""
DeadlineSense - A Free Web-Based Effort-Aware Deadline Reminder System
MSc Advanced Computer Science Masters Project (7COM1039)
Student: Farhan Ali (23104223)

Run locally with:  streamlit run deadlinesense_app.py
Requires deadlinesense_vectorizer.joblib and deadlinesense_model.joblib
in the same folder (produced by the project notebook, Post-IPR Work 5).
"""

from datetime import date, timedelta

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Core logic (pure functions, unit-tested in test_deadlinesense.py)
# ----------------------------------------------------------------------

def working_days_between(start: date, deadline: date) -> int:
    """Count working days (Mon-Fri) from the day AFTER `start` up to and
    including `deadline`. Returns 0 if the deadline is not after `start`."""
    if deadline <= start:
        return 0
    return int(np.busday_count(start + timedelta(days=1),
                               deadline + timedelta(days=1)))


def remaining_capacity_hours(start: date, deadline: date,
                             hours_per_day: float) -> float:
    """Total working hours available for this task before the deadline."""
    return working_days_between(start, deadline) * hours_per_day


def remaining_effort_hours(predicted_effort: float, progress_pct: float) -> float:
    """Effort still required, given predicted total effort and progress (0-100%)."""
    return predicted_effort * (1.0 - progress_pct / 100.0)


def is_at_risk(predicted_effort: float, progress_pct: float,
               capacity_hours: float) -> bool:
    """Effort-aware warning rule (proposal Section 5):
    warn when remaining predicted effort exceeds remaining capacity."""
    return remaining_effort_hours(predicted_effort, progress_pct) > capacity_hours


def validate_inputs(task_text: str, deadline: date, today: date,
                    hours_per_day: float, progress_pct: float) -> list:
    """Return a list of human-readable validation errors (empty = valid)."""
    errors = []
    if not task_text or not task_text.strip():
        errors.append("Enter a task title or description - the prediction is based on this text.")
    if deadline <= today:
        errors.append("Choose a deadline after today.")
    if not (0.5 <= hours_per_day <= 24):
        errors.append("Hours per day must be between 0.5 and 24.")
    if not (0 <= progress_pct <= 100):
        errors.append("Progress must be between 0 and 100%.")
    return errors


@st.cache_resource
def load_model():
    vectorizer = joblib.load("deadlinesense_vectorizer.joblib")
    model = joblib.load("deadlinesense_model.joblib")
    return vectorizer, model


def predict_effort(vectorizer, model, task_text: str) -> float:
    """Predict total effort in hours for a task description.
    Mirrors the notebook pipeline: TF-IDF -> Ridge on log1p target -> expm1."""
    X = vectorizer.transform([task_text])
    pred = float(np.expm1(model.predict(X))[0])
    return max(pred, 0.0)


# ----------------------------------------------------------------------
# Page setup and styling
# ----------------------------------------------------------------------

st.set_page_config(page_title="DeadlineSense", page_icon="⏱️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* Hero header */
.ds-hero {
    background: linear-gradient(120deg, #1a2980 0%, #26d0ce 100%);
    border-radius: 14px;
    padding: 28px 34px 24px 34px;
    margin-bottom: 6px;
}
.ds-hero h1 {
    color: #ffffff; font-size: 2.3rem; margin: 0 0 4px 0; letter-spacing: -0.5px;
}
.ds-hero p { color: #dff7f6; font-size: 1.02rem; margin: 0; }
.ds-badges span {
    display: inline-block; background: rgba(255,255,255,0.16); color:#ffffff;
    border-radius: 999px; padding: 3px 12px; margin: 10px 6px 0 0; font-size: 0.8rem;
}
/* Verdict banners */
.ds-ontrack, .ds-atrisk {
    border-radius: 12px; padding: 20px 24px; margin: 6px 0 4px 0;
    font-size: 1.25rem; font-weight: 700;
}
.ds-ontrack { background: rgba(33,195,84,0.14); border: 1px solid #21c354; color:#2ee06a; }
.ds-atrisk  { background: rgba(255,75,75,0.12); border: 1px solid #ff4b4b; color:#ff6b6b; }
.ds-sub { font-size: 0.95rem; font-weight: 400; opacity: 0.9; margin-top: 6px; }
/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px; padding: 14px 16px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Sidebar: about the project and the model
# ----------------------------------------------------------------------

with st.sidebar:
    st.title("⏱️ DeadlineSense")
    st.caption("MSc Advanced Computer Science Masters Project (7COM1039)")

    st.markdown("### How it works")
    st.markdown(
        "1. You describe a software task\n"
        "2. A text model predicts its **total effort in hours**\n"
        "3. Your progress and daily hours give your **remaining capacity**\n"
        "4. Warn if *remaining effort > remaining capacity*"
    )

    st.markdown("### Model card")
    st.markdown(
        "- **Data:** 20,862 real tasks, 363 open-source projects (JOSSE dataset)\n"
        "- **Model:** TF-IDF (1-2 grams) + Ridge regression on log-effort\n"
        "- **Validation:** project-grouped 5-fold cross-validation\n"
        "- **MAE:** 2.67h &nbsp;|&nbsp; **RMSE:** 5.56h (6.3% better than baseline)"
    )

    st.markdown("### Privacy & ethics")
    st.caption("Everything runs in your session and nothing is stored. "
               "DeadlineSense evaluates tasks, never people - it is a planning "
               "aid, not a performance monitor.")

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------

st.markdown("""
<div class="ds-hero">
  <h1>DeadlineSense</h1>
  <p>An effort-aware deadline reminder: it warns you because the <b>work looks big</b>,
     not just because the date is close.</p>
  <div class="ds-badges">
    <span>📊 Trained on 20,862 real tasks</span>
    <span>🔍 Explainable - every warning shows its working</span>
    <span>🔒 Nothing stored</span>
    <span>🆓 Free &amp; open source</span>
  </div>
</div>
""", unsafe_allow_html=True)
st.write("")

try:
    vectorizer, model = load_model()
except FileNotFoundError:
    st.error("Model files not found. Place deadlinesense_vectorizer.joblib and "
             "deadlinesense_model.joblib in the same folder as this app, "
             "then reload the page.")
    st.stop()

# ----------------------------------------------------------------------
# Example tasks (one click fills the form)
# ----------------------------------------------------------------------

EXAMPLES = {
    "🐞 Small bug fix": ("Fix null pointer exception in login form",
                         "Login form throws NPE when the email field is empty. "
                         "Add a null check and a unit test."),
    "⚙️ Medium feature": ("Add CSV export to the reporting page",
                          "Users need to export the monthly report table as CSV, "
                          "including filters and column selection."),
    "🏗️ Large feature": ("Implement complete user authentication system",
                          "Registration, login, password reset, session handling, "
                          "database migration and integration tests."),
}

st.markdown("##### Try an example, or describe your own task below")
ex_cols = st.columns(len(EXAMPLES))
for col, (label, (ex_title, ex_desc)) in zip(ex_cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state["title_input"] = ex_title
        st.session_state["desc_input"] = ex_desc

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------

with st.form("task_form"):
    title = st.text_input("Task title", key="title_input",
                          placeholder="e.g. Fix null pointer exception in login form")
    description = st.text_area("Description (optional)", key="desc_input",
                               placeholder="Paste the issue description here for a better estimate",
                               height=110)
    c1, c2, c3 = st.columns(3)
    with c1:
        deadline = st.date_input("Deadline", value=date.today() + timedelta(days=7),
                                 min_value=date.today())
    with c2:
        hours_per_day = st.number_input("Hours per day for this task",
                                        min_value=0.5, max_value=24.0,
                                        value=2.0, step=0.5)
    with c3:
        progress_pct = st.slider("Progress already done", 0, 100, 0, format="%d%%")
    submitted = st.form_submit_button("🔎 Check deadline risk", use_container_width=True,
                                      type="primary")

# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------

if submitted:
    task_text = f"{title.strip()}. {description.strip()}".strip(". ").strip()
    today = date.today()
    errors = validate_inputs(task_text, deadline, today, hours_per_day, float(progress_pct))

    if errors:
        for e in errors:
            st.warning(e)
    else:
        predicted = predict_effort(vectorizer, model, task_text)
        remaining = remaining_effort_hours(predicted, progress_pct)
        days = working_days_between(today, deadline)
        capacity = remaining_capacity_hours(today, deadline, hours_per_day)
        at_risk = is_at_risk(predicted, progress_pct, capacity)
        buffer_h = capacity - remaining

        if at_risk:
            st.markdown(
                f'<div class="ds-atrisk">🚨 AT RISK - remaining work ({remaining:.1f}h) '
                f'exceeds your remaining time ({capacity:.1f}h)'
                f'<div class="ds-sub">Shortfall of {abs(buffer_h):.1f} hours. Consider starting '
                f'earlier, increasing daily hours, splitting the task, or moving the deadline.</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="ds-ontrack">✅ ON TRACK - remaining work ({remaining:.1f}h) '
                f'fits in your remaining time ({capacity:.1f}h)'
                f'<div class="ds-sub">Buffer of {buffer_h:.1f} hours to spare.</div></div>',
                unsafe_allow_html=True)

        # ---- Metric cards ----
        st.write("")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted total effort", f"{predicted:.1f} h",
                  help="Model estimate of the whole task, from its text")
        m2.metric("Remaining effort", f"{remaining:.1f} h",
                  delta=f"-{progress_pct}% already done", delta_color="off")
        m3.metric("Working days left", f"{days}",
                  help="Mon-Fri only; today excluded, deadline included")
        m4.metric("Remaining capacity", f"{capacity:.1f} h",
                  delta=f"{buffer_h:+.1f} h buffer",
                  delta_color="normal" if buffer_h >= 0 else "inverse")

        # ---- Capacity usage bar ----
        used = min(remaining / capacity, 1.0) if capacity > 0 else 1.0
        st.write("")
        st.markdown(f"**Capacity used by this task:** {used*100:.0f}%"
                    if capacity > 0 else "**No working time left before the deadline.**")
        st.progress(used)

        # ---- Detail tabs ----
        tab1, tab2, tab3 = st.tabs(["📋 How this was calculated",
                                    "📈 Effort vs capacity",
                                    "🧠 About the model"])

        with tab1:
            st.markdown(
                f"| Step | Value |\n|---|---|\n"
                f"| Predicted total effort (from task text) | **{predicted:.2f} h** |\n"
                f"| Progress entered | **{progress_pct}%** |\n"
                f"| Remaining effort = {predicted:.2f}h × (1 − {progress_pct/100:.2f}) | **{remaining:.2f} h** |\n"
                f"| Working days until deadline (Mon-Fri) | **{days} days** |\n"
                f"| Remaining capacity = {days} days × {hours_per_day:.1f}h/day | **{capacity:.2f} h** |\n"
                f"| Rule | warn if **{remaining:.2f}h > {capacity:.2f}h** → "
                f"{'**AT RISK**' if at_risk else '**ON TRACK**'} |"
            )

        with tab2:
            chart_df = pd.DataFrame({
                "What": ["Remaining effort (predicted)", "Remaining capacity (yours)"],
                "Hours": [remaining, capacity],
                "Type": ["Effort", "Capacity"],
            })
            chart = (alt.Chart(chart_df)
                     .mark_bar(cornerRadiusEnd=6, height=42)
                     .encode(x=alt.X("Hours:Q", title="Hours"),
                             y=alt.Y("What:N", title="", sort=None),
                             color=alt.Color("Type:N", legend=None,
                                             scale=alt.Scale(domain=["Effort", "Capacity"],
                                                             range=["#ff6b6b" if at_risk else "#26d0ce",
                                                                    "#4d96ff"])),
                             tooltip=["What", alt.Tooltip("Hours:Q", format=".2f")]))
            st.altair_chart(chart, use_container_width=True)
            st.caption("If the effort bar is longer than the capacity bar, the task is at risk.")

        with tab3:
            st.markdown(
                "- **Training data:** 20,862 cleaned tasks from 363 real open-source projects "
                "(JOSSE dataset - Apache, JBoss and Spring issue trackers)\n"
                "- **Pipeline:** task text → TF-IDF (10,000 features, 1-2 grams) → "
                "Ridge regression (α=10) on log-transformed effort → back-transform to hours\n"
                "- **Evaluation:** project-grouped 5-fold cross-validation "
                "(the model is never tested on projects it trained on)\n"
                "- **Accuracy:** MAE 2.67h; RMSE 5.56h, a statistically significant 6.3% "
                "improvement over a median baseline (paired t-test, p < 0.001)\n"
                "- **Known limits:** estimates are least reliable for unusual tasks and "
                "late-stage work; trained on open-source data, so commercial contexts may differ"
            )

        st.caption("Estimates are averages from historical data and can be wrong for any "
                   "individual task - treat the result as a prompt to re-plan, not a verdict. "
                   "This tool evaluates tasks, not people, and stores nothing: all inputs "
                   "stay on your computer.")