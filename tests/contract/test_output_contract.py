"""Every invariant in the problem statement, asserted on real generated output.

These run against the full 250-row run and gate the CSV write.
"""
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from buyorwait.io_loaders import load_dataset
from buyorwait.pipeline import run
from buyorwait.validate import check_all

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"

STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("out") / "output.csv"
    decisions = run(DATASET, out)
    return load_dataset(DATASET), decisions, out


def test_one_row_per_request_no_extras_no_duplicates(result):
    ds, decisions, _ = result
    got = [d.request_id for d in decisions]
    assert len(got) == len(set(got)) == 250
    assert set(got) == {r.request_id for r in ds.requests}


def test_column_order_is_exact(result):
    _, _, out = result
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header == ("request_id,amount_safe_to_pay,affordability_status,"
                      "recommended_payment_method,payment_plan,"
                      "earliest_date_for_full_payment,spending_changes_needed,"
                      "decision_explanation")


def test_no_contract_violations(result):
    ds, decisions, _ = result
    violations = check_all(decisions, ds)
    assert violations == [], "\n".join(violations[:20])


def test_allowed_values_only(result):
    _, decisions, _ = result
    for d in decisions:
        assert d.affordability_status in STATUSES
        assert d.recommended_payment_method in METHODS


def test_safe_amount_within_bounds(result):
    ds, decisions, _ = result
    by_id = {r.request_id: r for r in ds.requests}
    for d in decisions:
        v = Decimal(d.amount_safe_to_pay)
        assert 0 <= v <= by_id[d.request_id].requested_amount, d.request_id


def test_affordable_now_implies_earliest_equals_request_date(result):
    ds, decisions, _ = result
    by_id = {r.request_id: r for r in ds.requests}
    for d in decisions:
        if d.affordability_status == "affordable_now":
            assert d.earliest_date_for_full_payment == \
                by_id[d.request_id].request_date.isoformat(), d.request_id


def test_not_affordable_has_no_plan_and_no_date(result):
    _, decisions, _ = result
    for d in decisions:
        if d.affordability_status == "not_affordable":
            assert d.payment_plan == "none", d.request_id
            assert d.earliest_date_for_full_payment == "", d.request_id


def test_partial_payment_rules(result):
    ds, decisions, _ = result
    by_id = {r.request_id: r for r in ds.requests}
    for d in decisions:
        if d.recommended_payment_method != "partial_payment":
            continue
        req = by_id[d.request_id]
        assert d.affordability_status == "affordable_with_plan", d.request_id
        assert req.allows_partial_payment, d.request_id
        parts = d.payment_plan.split("|")
        assert len(parts) == 2, d.request_id
        d1, a1 = parts[0].split(":")
        d2, a2 = parts[1].split(":")
        assert d1 == req.request_date.isoformat()
        assert Decimal(a1) == Decimal(d.amount_safe_to_pay)
        assert Decimal(a1) + Decimal(a2) == req.requested_amount, d.request_id
        assert d2 == d.earliest_date_for_full_payment
        assert date.fromisoformat(d2) <= req.desired_completion_date, d.request_id


def test_installment_plans_reproduce_a_supplied_option(result):
    ds, decisions, _ = result
    for d in decisions:
        if d.recommended_payment_method != "installments":
            continue
        plans = set()
        for o in ds.options_by_request[d.request_id]:
            if o.payment_method != "installments":
                continue
            freq = o.payment_frequency_days or 0
            dates = [o.first_payment_date + timedelta(days=freq * i)
                     for i in range(o.number_of_payments)]
            plans.add("|".join(f"{x.isoformat()}:{o.payment_amount_text}" for x in dates))
        assert d.payment_plan in plans, f"{d.request_id}: {d.payment_plan}"


def test_payment_plan_is_chronological_and_well_formed(result):
    _, decisions, _ = result
    for d in decisions:
        if d.payment_plan == "none":
            continue
        dates = []
        for part in d.payment_plan.split("|"):
            day, _, amount = part.partition(":")
            dates.append(date.fromisoformat(day))
            Decimal(amount)                      # raises if malformed
        assert dates == sorted(dates), d.request_id


def test_spending_changes_are_permitted_flexible_and_capped(result):
    ds, decisions, _ = result
    by_req = {r.request_id: r for r in ds.requests}
    for d in decisions:
        if d.spending_changes_needed == "none":
            continue
        parts = d.spending_changes_needed.split("|")
        assert len(parts) <= 3, d.request_id
        seen: set[str] = set()
        profile = ds.profiles[by_req[d.request_id].user_id]
        for part in parts:
            bits = part.split(":")
            assert bits[0] in ("stop", "reduce_to"), d.request_id
            event_id = bits[1]
            assert event_id not in seen, f"{d.request_id}: {event_id} changed twice"
            seen.add(event_id)
            event = ds.events_by_id[event_id]
            assert event.user_id == profile.user_id, d.request_id
            assert event.is_flexible, f"{d.request_id}: {event_id} is fixed"
            assert event.category not in profile.expense_categories_to_protect
            if bits[0] == "stop":
                assert event.category in profile.expense_categories_user_is_willing_to_stop
            else:
                assert event.category in profile.expense_categories_user_is_willing_to_reduce
                assert event.minimum_allowed_amount is not None
                assert Decimal(bits[2]) >= event.minimum_allowed_amount


def test_method_is_one_the_user_accepts(result):
    ds, decisions, _ = result
    by_req = {r.request_id: r for r in ds.requests}
    for d in decisions:
        if d.recommended_payment_method in ("wait", "not_recommended"):
            continue
        profile = ds.profiles[by_req[d.request_id].user_id]
        assert d.recommended_payment_method in profile.payment_methods_user_will_consider, \
            d.request_id


def test_explanations_are_non_empty_and_name_the_currency(result):
    ds, decisions, _ = result
    by_req = {r.request_id: r for r in ds.requests}
    for d in decisions:
        assert d.decision_explanation.strip(), d.request_id
        ccy = ds.profiles[by_req[d.request_id].user_id].home_currency
        assert ccy in d.decision_explanation, d.request_id


def test_status_and_method_are_consistent(result):
    _, decisions, _ = result
    allowed = {
        "affordable_now": {"full_payment"},
        "affordable_with_plan": {"full_payment", "partial_payment", "installments"},
        "affordable_later": {"wait"},
        "not_affordable": {"not_recommended"},
    }
    for d in decisions:
        assert d.recommended_payment_method in allowed[d.affordability_status], \
            f"{d.request_id}: {d.affordability_status} / {d.recommended_payment_method}"
