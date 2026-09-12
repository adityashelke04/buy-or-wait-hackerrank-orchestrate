from datetime import date
from decimal import Decimal

from buyorwait.forecast import build
from buyorwait.fx import RateTable
from buyorwait.ledger import LedgerView
from buyorwait.recurrence import Series
from buyorwait.types import Event, Profile

EMPTY_RATES = RateTable([])


def prof(balance="1000", minimum="200"):
    return Profile(
        user_id="user_x", home_currency="EUR",
        current_available_balance=Decimal(balance),
        minimum_balance_to_keep=Decimal(minimum),
        financial_priorities=(), expense_categories_to_protect=(),
        expense_categories_user_is_willing_to_reduce=(),
        expense_categories_user_is_willing_to_stop=(),
        payment_methods_user_will_consider=("full_payment",),
        max_installment_months=None,
    )


def rent_series(day=1, amount="300"):
    return Series(category="rent", description="Rent", template_event_id="event_r",
                  cadence="monthly", anchor=day, amount=Decimal(amount),
                  direction="debit", flexibility="fixed",
                  minimum_allowed_amount=None, last_seen=date(2024, 2, 1))


def test_curve_opens_at_the_profile_balance():
    view = LedgerView([], prof(balance="1234.56"), EMPTY_RATES)
    curve = build(view, [], date(2024, 3, 1))
    assert curve.opening == Decimal("1234.56")
    assert curve.start == date(2024, 3, 1)
    assert curve.end == date(2024, 5, 29)          # 90 days inclusive of the start


def test_projected_series_appear_as_negative_flows():
    view = LedgerView([], prof(), EMPTY_RATES)
    curve = build(view, [rent_series()], date(2024, 3, 5))
    assert curve.flows == (
        (date(2024, 4, 1), Decimal("-300")),
        (date(2024, 5, 1), Decimal("-300")),
        (date(2024, 6, 1), Decimal("-300")),
    )


def test_a_real_scheduled_event_suppresses_the_projection_on_the_same_day():
    """The confirmed row wins; projecting on top of it would double-charge rent."""
    real = Event(
        event_id="event_r2", user_id="user_x", event_type="expense",
        description="Rent", category="rent", direction="debit",
        amount=Decimal("300"), currency="EUR",
        event_date=date(2024, 4, 1), settlement_date=date(2024, 4, 1),
        status="scheduled", linked_event_id=None, flexibility="fixed",
        minimum_allowed_amount=None,
    )
    view = LedgerView([real], prof(), EMPTY_RATES)
    curve = build(view, [rent_series()], date(2024, 3, 5))
    april = [f for f in curve.flows if f[0] == date(2024, 4, 1)]
    assert april == [(date(2024, 4, 1), Decimal("-300"))], "rent charged once, not twice"


def test_income_series_appears_as_a_positive_flow():
    salary = Series(category="salary", description="Payroll",
                    template_event_id="event_s", cadence="monthly", anchor=15,
                    amount=Decimal("900"), direction="credit", flexibility="fixed",
                    minimum_allowed_amount=None, last_seen=date(2024, 2, 15))
    view = LedgerView([], prof(), EMPTY_RATES)
    curve = build(view, [salary], date(2024, 3, 20))
    assert all(amount > 0 for _, amount in curve.flows)


def test_flows_are_sorted_by_date():
    view = LedgerView([], prof(), EMPTY_RATES)
    salary = Series(category="salary", description="Payroll",
                    template_event_id="event_s", cadence="monthly", anchor=15,
                    amount=Decimal("900"), direction="credit", flexibility="fixed",
                    minimum_allowed_amount=None, last_seen=date(2024, 2, 15))
    curve = build(view, [rent_series(), salary], date(2024, 3, 20))
    assert list(curve.flows) == sorted(curve.flows, key=lambda f: f[0])


def test_nothing_beyond_the_horizon_is_included():
    view = LedgerView([], prof(), EMPTY_RATES)
    curve = build(view, [rent_series()], date(2024, 3, 5), horizon_days=40)
    assert curve.end == date(2024, 4, 13)
    assert all(d <= curve.end for d, _ in curve.flows)


def test_with_extra_merges_and_keeps_order():
    view = LedgerView([], prof(), EMPTY_RATES)
    curve = build(view, [rent_series()], date(2024, 3, 5))
    merged = curve.with_extra([(date(2024, 3, 10), Decimal("-50"))])
    assert list(merged.flows) == sorted(merged.flows, key=lambda f: f[0])
    assert (date(2024, 3, 10), Decimal("-50")) in merged.flows
