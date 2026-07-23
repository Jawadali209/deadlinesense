"""
DeadlineSense - A Free Web-Based Effort-Aware Deadline Reminder System
MSc Advanced Computer Science Masters Project (7COM1039)
Student: Farhan Ali (23104223)

Run locally with:  streamlit run deadlinesense_app.py
Requires deadlinesense_vectorizer.joblib and deadlinesense_model.joblib
in the same folder (produced by the project notebook, Post-IPR Work 5).

Tasks are stored locally in an SQLite database file (deadlinesense.db),
as specified in proposal Section 6.8. Nothing leaves the user's machine.
"""

import sqlite3
from datetime import date, timedelta

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Cross-validated MAE from the project notebook (uncertainty note, proposal S4.4)
MODEL_MAE_HOURS = 2.67

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DEFAULT_WORKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# ----------------------------------------------------------------------
# Core logic (pure functions, unit-tested in test_deadlinesense.py)
# ----------------------------------------------------------------------

def weekmask_from(workdays):
    """Convert a list like ['Mon','Tue'] to a numpy weekmask string '1100000'.
    Empty/None falls back to Monday-Friday (proposal Section 4.1 default)."""
    if not workdays:
        workdays = DEFAULT_WORKDAYS
    return "".join("1" if d in workdays else "0" for d in DAY_ORDER)


def working_days_between(start: date, deadline: date, workdays=None) -> int:
    """Count selected working days from the day AFTER `start` up to and
    including `deadline`. Default working days are Mon-Fri (proposal S4.1).
    Returns 0 if the deadline is not after `start`."""
    if deadline <= start:
        return 0
    return int(np.busday_count(start + timedelta(days=1),
                               deadline + timedelta(days=1),
                               weekmask=weekmask_from(workdays)))


def remaining_capacity_hours(start: date, deadline: date,
                             hours_per_day: float, workdays=None) -> float:
    """Total working hours available for this task before the deadline."""
    return working_days_between(start, deadline, workdays) * hours_per_day


def remaining_effort_hours(predicted_effort: float, progress_pct: float) -> float:
    """Effort still required, given predicted total effort and progress (0-100%)."""
    return predicted_effort * (1.0 - progress_pct / 100.0)


def is_at_risk(predicted_effort: float, progress_pct: float,
               capacity_hours: float) -> bool:
    """Effort-aware warning rule (proposal Section 4.3):
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


# ----------------------------------------------------------------------
# SQLite storage layer (proposal Section 6.8) - integration-tested
# ----------------------------------------------------------------------

def init_db(db_path: str = "deadlinesense.db") -> sqlite3.Connection:
    """Open (and create if needed) the local task database."""
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        deadline TEXT NOT NULL,
        hours_per_day REAL NOT NULL,
        workdays TEXT NOT NULL,
        progress_pct REAL NOT NULL,
        predicted_effort REAL NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    con.commit()
    return con


def save_task(con, title, description, deadline, hours_per_day,
              workdays, progress_pct, predicted_effort) -> int:
    """Insert a task; returns its new id."""
    cur = con.execute(
        "INSERT INTO tasks (title, description, deadline, hours_per_day, "
        "workdays, progress_pct, predicted_effort) VALUES (?,?,?,?,?,?,?)",
        (title, description, deadline.isoformat(), hours_per_day,
         ",".join(workdays if workdays else DEFAULT_WORKDAYS),
         progress_pct, predicted_effort))
    con.commit()
    return cur.lastrowid


def list_tasks(con) -> list:
    """Return all stored tasks as a list of dicts, earliest deadline first."""
    cols = ["id", "title", "description", "deadline", "hours_per_day",
            "workdays", "progress_pct", "predicted_effort"]
    rows = con.execute(f"SELECT {', '.join(cols)} FROM tasks ORDER BY deadline").fetchall()
    return [dict(zip(cols, r)) for r in rows]


def update_task_progress(con, task_id: int, progress_pct: float) -> None:
    con.execute("UPDATE tasks SET progress_pct=? WHERE id=?", (progress_pct, task_id))
    con.commit()


def delete_task(con, task_id: int) -> None:
    con.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    con.commit()


def task_status(task: dict, today: date) -> dict:
    """Recompute a stored task's current risk status (warning display logic)."""
    workdays = task["workdays"].split(",") if task["workdays"] else DEFAULT_WORKDAYS
    deadline = date.fromisoformat(task["deadline"])
    remaining = remaining_effort_hours(task["predicted_effort"], task["progress_pct"])
    capacity = remaining_capacity_hours(today, deadline, task["hours_per_day"], workdays)
    at_risk = is_at_risk(task["predicted_effort"], task["progress_pct"], capacity)
    return {"remaining": remaining, "capacity": capacity, "at_risk": at_risk,
            "status": "AT RISK" if at_risk else "ON TRACK"}


@st.cache_resource
def load_model():
    vectorizer = joblib.load("deadlinesense_vectorizer.joblib")
    model = joblib.load("deadlinesense_model.joblib")
    return vectorizer, model


@st.cache_resource
def get_db():
    return init_db("deadlinesense.db")


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
.ds-hero {
    background: linear-gradient(120deg, #1a2980 0%, #26d0ce 100%);
    border-radius: 14px;
    padding: 28px 34px 24px 34px;
    margin-bottom: 6px;
}
.ds-hero h1 { color: #ffffff; font-size: 2.3rem; margin: 0 0 4px 0; letter-spacing: -0.5px; }
.ds-hero p { color: #dff7f6; font-size: 1.02rem; margin: 0; }
.ds-badges span {
    display: inline-block; background: rgba(255,255,255,0.16); color:#ffffff;
    border-radius: 999px; padding: 3px 12px; margin: 10px 6px 0 0; font-size: 0.8rem;
}
.ds-ontrack, .ds-atrisk {
    border-radius: 12px; padding: 20px 24px; margin: 6px 0 4px 0;
    font-size: 1.25rem; font-weight: 700;
}
.ds-ontrack { background: rgba(33,195,84,0.14); border: 1px solid #21c354; color:#2ee06a; }
.ds-atrisk  { background: rgba(255,75,75,0.12); border: 1px solid #ff4b4b; color:#ff6b6b; }
.ds-sub { font-size: 0.95rem; font-weight: 400; opacity: 0.9; margin-top: 6px; }
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px; padding: 14px 16px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Sidebar
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
    st.caption("Your tasks are stored only in a local file (deadlinesense.db) on "
               "this machine - nothing is sent anywhere. DeadlineSense evaluates "
               "tasks, never people - it is a planning aid, not a performance monitor.")

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
    <span>💾 Tasks saved locally (SQLite)</span>
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

db = get_db()

# ----------------------------------------------------------------------
# Example tasks
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
    workdays = st.multiselect("Working days to include (proposal default: Mon-Fri)",
                              options=DAY_ORDER, default=DEFAULT_WORKDAYS)
    submitted = st.form_submit_button("🔎 Check deadline risk", use_container_width=True,
                                      type="primary")

# ----------------------------------------------------------------------
# Result (kept in session_state so the Save button works after rerun)
# ----------------------------------------------------------------------

if submitted:
    task_text = f"{title.strip()}. {description.strip()}".strip(". ").strip()
    today = date.today()
    errors = validate_inputs(task_text, deadline, today, hours_per_day, float(progress_pct))
    if not workdays:
        errors.append("Select at least one working day.")

    if errors:
        st.session_state.pop("last_result", None)
        for e in errors:
            st.warning(e)
    else:
        predicted = predict_effort(vectorizer, model, task_text)
        st.session_state["last_result"] = {
            "title": title.strip(), "description": description.strip(),
            "deadline": deadline, "hours_per_day": hours_per_day,
            "progress_pct": float(progress_pct), "workdays": workdays,
            "predicted": predicted,
        }

if "last_result" in st.session_state:
    r = st.session_state["last_result"]
    today = date.today()
    predicted = r["predicted"]
    remaining = remaining_effort_hours(predicted, r["progress_pct"])
    days = working_days_between(today, r["deadline"], r["workdays"])
    capacity = remaining_capacity_hours(today, r["deadline"], r["hours_per_day"], r["workdays"])
    at_risk = is_at_risk(predicted, r["progress_pct"], capacity)
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

    # Uncertainty note (proposal Section 4.4, exact wording promised)
    st.info(f"ℹ️ Historical predictions were typically wrong by approximately "
            f"{MODEL_MAE_HOURS:.1f} hours (cross-validated MAE). Treat this as "
            f"guidance, not a guarantee.")

    if st.button("💾 Save this task to My Tasks", use_container_width=True):
        save_task(db, r["title"], r["description"], r["deadline"],
                  r["hours_per_day"], r["workdays"], r["progress_pct"], predicted)
        st.session_state.pop("last_result", None)
        st.success("Task saved to the local database (deadlinesense.db).")
        st.rerun()

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted total effort", f"{predicted:.1f} h",
              help="Model estimate of the whole task, from its text")
    m2.metric("Remaining effort", f"{remaining:.1f} h",
              delta=f"-{r['progress_pct']:.0f}% already done", delta_color="off")
    m3.metric("Working days left", f"{days}",
              help="Only your selected working days; today excluded, deadline included")
    m4.metric("Remaining capacity", f"{capacity:.1f} h",
              delta=f"{buffer_h:+.1f} h buffer",
              delta_color="normal" if buffer_h >= 0 else "inverse")

    used = min(remaining / capacity, 1.0) if capacity > 0 else 1.0
    st.write("")
    st.markdown(f"**Capacity used by this task:** {used*100:.0f}%"
                if capacity > 0 else "**No working time left before the deadline.**")
    st.progress(used)

    tab1, tab2, tab3 = st.tabs(["📋 How this was calculated",
                                "📈 Effort vs capacity",
                                "🧠 About the model"])
    with tab1:
        st.markdown(
            f"| Step | Value |\n|---|---|\n"
            f"| Predicted total effort (from task text) | **{predicted:.2f} h** |\n"
            f"| Progress entered | **{r['progress_pct']:.0f}%** |\n"
            f"| Remaining effort = {predicted:.2f}h × (1 − {r['progress_pct']/100:.2f}) | **{remaining:.2f} h** |\n"
            f"| Selected working days | **{', '.join(r['workdays'])}** |\n"
            f"| Working days until deadline | **{days} days** |\n"
            f"| Remaining capacity = {days} days × {r['hours_per_day']:.1f}h/day | **{capacity:.2f} h** |\n"
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

# ----------------------------------------------------------------------
# My Tasks (SQLite storage - proposal Section 6.8)
# ----------------------------------------------------------------------

st.write("")
st.markdown("## 💾 My Tasks")
tasks = list_tasks(db)

if not tasks:
    st.caption("No saved tasks yet. Check a task above and press "
               "'Save this task to My Tasks'.")
else:
    today = date.today()
    table_rows = []
    for t in tasks:
        s = task_status(t, today)
        table_rows.append({
            "ID": t["id"], "Task": t["title"], "Deadline": t["deadline"],
            "Progress": f"{t['progress_pct']:.0f}%",
            "Remaining effort (h)": round(s["remaining"], 1),
            "Capacity left (h)": round(s["capacity"], 1),
            "Status": ("🚨 " if s["at_risk"] else "✅ ") + s["status"],
        })
    tasks_df = pd.DataFrame(table_rows)
    st.dataframe(tasks_df, use_container_width=True, hide_index=True)

    at_risk_count = sum(1 for row in table_rows if "AT RISK" in row["Status"])
    st.caption(f"{len(tasks)} saved task(s), {at_risk_count} currently at risk. "
               f"Status is recalculated live from today's date.")

    colu, cold, colc = st.columns(3)

    with colu:
        st.markdown("**Update progress**")
        sel = st.selectbox("Task to update", tasks,
                           format_func=lambda t: f"#{t['id']} {t['title'][:40]}",
                           key="upd_sel")
        new_p = st.slider("New progress", 0, 100, int(sel["progress_pct"]),
                          format="%d%%", key="upd_slider")
        if st.button("Update", use_container_width=True):
            update_task_progress(db, sel["id"], float(new_p))
            st.success(f"Task #{sel['id']} updated to {new_p}%.")
            st.rerun()

    with cold:
        st.markdown("**Delete a task**")
        sel_d = st.selectbox("Task to delete", tasks,
                             format_func=lambda t: f"#{t['id']} {t['title'][:40]}",
                             key="del_sel")
        if st.button("Delete", use_container_width=True):
            delete_task(db, sel_d["id"])
            st.success(f"Task #{sel_d['id']} deleted.")
            st.rerun()

    with colc:
        st.markdown("**Backup / export**")
        csv = tasks_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download my tasks (CSV)", data=csv,
                           file_name="deadlinesense_tasks.csv", mime="text/csv",
                           use_container_width=True)
        st.caption("Your own backup - keep it anywhere you like.")

st.caption("Estimates are averages from historical data and can be wrong for any "
           "individual task - treat the result as a prompt to re-plan, not a verdict. "
           "This tool evaluates tasks, not people. All data stays in a local file "
           "on your machine.")