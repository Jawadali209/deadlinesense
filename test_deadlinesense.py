"""
Unit and integration tests for DeadlineSense (proposal Section 6.8).
Run with:  python -m pytest test_deadlinesense.py -v
These tests use worked examples whose correct outputs were calculated by hand
in advance, as specified in the approved proposal.
"""

from datetime import date

from deadlinesense_app import (
    working_days_between,
    remaining_capacity_hours,
    remaining_effort_hours,
    is_at_risk,
    validate_inputs,
    init_db,
    save_task,
    list_tasks,
    update_task_progress,
    delete_task,
    task_status,
)


# ---------------- Working-day counting ----------------

def test_working_days_simple_week():
    # Mon 2026-07-20 -> Fri 2026-07-24: Tue,Wed,Thu,Fri = 4 working days
    assert working_days_between(date(2026, 7, 20), date(2026, 7, 24)) == 4

def test_working_days_across_weekend():
    # Fri 2026-07-24 -> Mon 2026-07-27: only Monday counts = 1
    assert working_days_between(date(2026, 7, 24), date(2026, 7, 27)) == 1

def test_working_days_deadline_on_weekend():
    # Fri 2026-07-24 -> Sun 2026-07-26: no working days in between = 0
    assert working_days_between(date(2026, 7, 24), date(2026, 7, 26)) == 0

def test_working_days_same_day_or_past():
    assert working_days_between(date(2026, 7, 20), date(2026, 7, 20)) == 0
    assert working_days_between(date(2026, 7, 20), date(2026, 7, 19)) == 0

def test_working_days_full_fortnight():
    # Mon 2026-07-20 -> Fri 2026-07-31 inclusive = 9 working days after today
    assert working_days_between(date(2026, 7, 20), date(2026, 7, 31)) == 9


# ---------------- Capacity calculation ----------------

def test_capacity_worked_example():
    # 4 working days x 2.5 h/day = 10.0 h
    assert remaining_capacity_hours(date(2026, 7, 20), date(2026, 7, 24), 2.5) == 10.0

def test_capacity_zero_days():
    assert remaining_capacity_hours(date(2026, 7, 24), date(2026, 7, 26), 8.0) == 0.0


# ---------------- Remaining effort and risk inequality ----------------

def test_remaining_effort():
    assert remaining_effort_hours(10.0, 0) == 10.0
    assert remaining_effort_hours(10.0, 50) == 5.0
    assert remaining_effort_hours(10.0, 100) == 0.0

def test_risk_boundary_exactly_equal_is_not_at_risk():
    # remaining = 5.0h, capacity = 5.0h -> rule is strict '>' so NOT at risk
    assert is_at_risk(predicted_effort=10.0, progress_pct=50, capacity_hours=5.0) is False

def test_risk_when_effort_exceeds_capacity():
    assert is_at_risk(predicted_effort=10.0, progress_pct=50, capacity_hours=4.9) is True

def test_no_risk_when_task_complete():
    assert is_at_risk(predicted_effort=40.0, progress_pct=100, capacity_hours=0.0) is False


# ---------------- Input validation ----------------

def test_validation_rejects_empty_text():
    errs = validate_inputs("", date(2026, 8, 1), date(2026, 7, 20), 2.0, 50)
    assert any("title or description" in e for e in errs)

def test_validation_rejects_past_deadline():
    errs = validate_inputs("task", date(2026, 7, 19), date(2026, 7, 20), 2.0, 50)
    assert any("after today" in e for e in errs)

def test_validation_rejects_bad_hours_and_progress():
    errs = validate_inputs("task", date(2026, 8, 1), date(2026, 7, 20), 0.0, 150)
    assert any("Hours per day" in e for e in errs)
    assert any("Progress" in e for e in errs)

def test_validation_accepts_valid_inputs():
    errs = validate_inputs("Fix login bug", date(2026, 8, 1), date(2026, 7, 20), 2.0, 25)
    assert errs == []


# ---------------- Workday-selection tests (proposal S4.1) ----------------

def test_working_days_custom_include_saturday():
    # Fri 2026-07-24 -> Mon 2026-07-27 with Sat as a working day: Sat + Mon = 2
    assert working_days_between(date(2026, 7, 24), date(2026, 7, 27),
                                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]) == 2

def test_working_days_weekend_only_worker():
    # Fri 2026-07-24 -> Sun 2026-07-26 for a Sat+Sun worker = 2 days
    assert working_days_between(date(2026, 7, 24), date(2026, 7, 26),
                                ["Sat", "Sun"]) == 2

def test_working_days_empty_selection_falls_back_to_monfri():
    # Empty list must behave exactly like the Mon-Fri default
    assert working_days_between(date(2026, 7, 20), date(2026, 7, 24), []) == 4


# ---------------- Integration tests (proposal S6.8) ----------------
# "integration tests confirming task creation, update, storage and warning display"

def make_db(tmp_path):
    return init_db(str(tmp_path / "test_tasks.db"))


def test_integration_task_creation_and_storage(tmp_path):
    con = make_db(tmp_path)
    tid = save_task(con, "Fix login bug", "NPE on empty email",
                    date(2026, 8, 3), 2.0, ["Mon", "Tue", "Wed", "Thu", "Fri"],
                    0.0, 1.93)
    tasks = list_tasks(con)
    assert len(tasks) == 1
    assert tasks[0]["id"] == tid
    assert tasks[0]["title"] == "Fix login bug"
    assert tasks[0]["predicted_effort"] == 1.93


def test_integration_task_update(tmp_path):
    con = make_db(tmp_path)
    tid = save_task(con, "Task", "", date(2026, 8, 3), 2.0,
                    ["Mon", "Tue", "Wed", "Thu", "Fri"], 0.0, 4.0)
    update_task_progress(con, tid, 60.0)
    assert list_tasks(con)[0]["progress_pct"] == 60.0


def test_integration_storage_persists_across_connections(tmp_path):
    db_file = str(tmp_path / "persist.db")
    con1 = init_db(db_file)
    save_task(con1, "Persistent task", "", date(2026, 8, 3), 2.0,
              ["Mon", "Tue", "Wed", "Thu", "Fri"], 25.0, 4.0)
    con1.close()
    con2 = init_db(db_file)  # reopen: data must still be there
    tasks = list_tasks(con2)
    assert len(tasks) == 1 and tasks[0]["progress_pct"] == 25.0


def test_integration_task_delete(tmp_path):
    con = make_db(tmp_path)
    tid = save_task(con, "Task to remove", "", date(2026, 8, 3), 2.0,
                    ["Mon", "Tue", "Wed", "Thu", "Fri"], 0.0, 4.0)
    delete_task(con, tid)
    assert list_tasks(con) == []


def test_integration_warning_display_from_stored_task(tmp_path):
    # End-to-end: store a task, read it back, recompute status like the UI does.
    con = make_db(tmp_path)
    # Big task (10h), 0% done, deadline Mon 2026-07-27 seen from Fri 2026-07-24,
    # 1h/day, Mon-Fri -> capacity = 1h -> must be AT RISK
    save_task(con, "Big feature", "", date(2026, 7, 27), 1.0,
              ["Mon", "Tue", "Wed", "Thu", "Fri"], 0.0, 10.0)
    t = list_tasks(con)[0]
    s = task_status(t, today=date(2026, 7, 24))
    assert s["at_risk"] is True and s["status"] == "AT RISK"
    assert s["capacity"] == 1.0 and s["remaining"] == 10.0
    # After updating progress to 95%, remaining = 0.5h -> ON TRACK
    update_task_progress(con, t["id"], 95.0)
    t2 = list_tasks(con)[0]
    s2 = task_status(t2, today=date(2026, 7, 24))
    assert s2["at_risk"] is False and s2["status"] == "ON TRACK"