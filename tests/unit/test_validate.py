"""The contract gate must reject each malformed decision it claims to catch."""
from dataclasses import replace
from pathlib import Path

import pytest

from buyorwait.io_loaders import load_dataset
from buyorwait.types import Decision
from buyorwait.validate import check_all

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ds():
    return load_dataset(ROOT / "dataset")


@pytest.fixture(scope="module")
def good(ds):
    """A valid 'wait' decision for the first request, plus valid rows for the rest."""
    rows = []
    for r in ds.requests:
        rows.append(Decision(
            request_id=r.request_id, amount_safe_to_pay="0",
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended", payment_plan="none",
            earliest_date_for_full_payment="", spending_changes_needed="none",
            decision_explanation=f"Do not proceed. {ds.profiles[r.user_id].home_currency} 1"))
    return rows


def _problems(ds, rows, **changes):
    first = replace(rows[0], **changes)
    return check_all([first, *rows[1:]], ds)


def test_the_baseline_rows_are_valid(ds, good):
    assert check_all(good, ds) == []


def _wait(ds, rows, **extra):
    r = ds.requests[0]
    later = r.request_date.replace(day=min(r.request_date.day + 1, 28))
    fields = dict(affordability_status="affordable_later", recommended_payment_method="wait",
                  earliest_date_for_full_payment=later.isoformat(),
                  payment_plan=f"{later.isoformat()}:{r.requested_amount}")
    fields.update(extra)
    return _problems(ds, rows, **fields)


def test_a_well_formed_wait_passes(ds, good):
    assert _wait(ds, good) == []


@pytest.mark.parametrize("extra, message", [
    ({"earliest_date_for_full_payment": "13/09/2026"}, "not YYYY-MM-DD"),
    ({"payment_plan": "none"}, "needs a payment plan"),
    ({"amount_safe_to_pay": "1e3"}, "not plain money"),
    ({"amount_safe_to_pay": "0.001"}, "not plain money"),
])
def test_malformed_wait_decisions_are_rejected(ds, good, extra, message):
    assert any(message in p for p in _wait(ds, good, **extra))


def test_wait_must_pay_the_full_amount_on_the_earliest_date(ds, good):
    r = ds.requests[0]
    later = r.request_date.replace(day=min(r.request_date.day + 1, 28))
    got = _wait(ds, good, payment_plan=f"{later.isoformat()}:1")
    assert any("full requested amount" in p for p in got)


def test_affordable_later_needs_a_future_date(ds, good):
    r = ds.requests[0]
    got = _wait(ds, good, earliest_date_for_full_payment=r.request_date.isoformat(),
                payment_plan=f"{r.request_date.isoformat()}:{r.requested_amount}")
    assert any("future earliest date" in p for p in got)
