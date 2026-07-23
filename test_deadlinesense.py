"""
Unit tests for DeadlineSense core logic (proposal Section 6.8).
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