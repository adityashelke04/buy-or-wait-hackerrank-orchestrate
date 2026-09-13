from datetime import date
from decimal import Decimal

from buyorwait.fx import RateTable
from buyorwait.recurrence import ESTIMATORS, detect
from buyorwait.types import Event, Profile

EMPTY_RATES = RateTable([])


def ev(event_id, day, amount, category="rent", description="Apartment rent transfer",
       direction="debit", status="settled", flexibility="fixed", minimum=None):
    return Event(
        event_id=event_id, user_id="user_x", event_type="expense",
        description=description, category=category, direction=direction,
        amount=Decimal(amount), currency="EUR",
        event_date=day, settlement_date=day, status=status,
        linked_event_id=None, flexibility=flexibility,
        minimum_allowed_amount=Decimal(minimum) if minimum else None,
    )


def prof():
    return Profile(
        user_id="user_x", home_currency="EUR",
        current_available_balance=Decimal("1000"),
        minimum_balance_to_keep=Decimal("200"),
        financial_priorities=(), expense_categories_to_protect=(),
        expense_categories_user_is_willing_to_reduce=(),
        expense_categories_user_is_willing_to_stop=(),
        payment_methods_user_will_consider=("full_payment",),
        max_installment_months=None,
    )


def test_monthly_series_detected_on_day_of_month():
    events = [
        ev("event_1", date(2024, 1, 1), "467.50"),
        ev("event_2", date(2024, 2, 1), "467.50"),
        ev("event_3", date(2024, 3, 1), "467.50"),
    ]
    series = detect(events, prof(), EMPTY_RATES, as_of=date(2024, 3, 5))
    assert len(series) == 1
    s = series[0]
    assert s.cadence == "monthly"
    assert s.anchor == 1
    assert s.amount == Decimal("467.50")
    assert s.occurrences(date(2024, 3, 5), date(2024, 6, 3)) == [
        date(2024, 4, 1), date(2024, 5, 1), date(2024, 6, 1),
    ]


def test_weekly_series_detected_on_weekday():
    days = [date(2024, 1, 2), date(2024, 1, 9), date(2024, 1, 16), date(2024, 1, 23)]
    events = [ev(f"event_{i}", d, "50", category="transport",
                 description="Commuter pass") for i, d in enumerate(days)]
    series = detect(events, prof(), EMPTY_RATES, as_of=date(2024, 1, 25))
    assert len(series) == 1
    assert series[0].cadence == "weekly"
    assert series[0].anchor == 1                      # Tuesday
    assert series[0].occurrences(date(2024, 1, 25), date(2024, 2, 8)) == [
        date(2024, 1, 30), date(2024, 2, 6),
    ]


def test_two_observations_is_not_enough_to_declare_recurrence():
    events = [
        ev("event_1", date(2024, 1, 1), "467.50"),
        ev("event_2", date(2024, 2, 1), "467.50"),
    ]
    assert detect(events, prof(), EMPTY_RATES, as_of=date(2024, 2, 5)) == []


def test_one_off_purchase_is_never_projected():
    events = [ev("event_1", date(2024, 1, 15), "2000", category="shopping",
                 description="Laptop purchase")]
    assert detect(events, prof(), EMPTY_RATES, as_of=date(2024, 2, 1)) == []


def test_irregular_gaps_do_not_form_a_series():
    days = [date(2024, 1, 3), date(2024, 1, 20), date(2024, 3, 11)]
    events = [ev(f"event_{i}", d, "80", category="healthcare",
                 description="Clinic visit") for i, d in enumerate(days)]
    assert detect(events, prof(), EMPTY_RATES, as_of=date(2024, 3, 20)) == []


def test_variable_amounts_use_the_conservative_estimator():
    days = [date(2024, 1, 2), date(2024, 1, 9), date(2024, 1, 16), date(2024, 1, 23)]
    amounts = ["40", "60", "50", "70"]
    events = [ev(f"event_{i}", d, a, category="groceries",
                 description="Supermarket basket")
              for i, (d, a) in enumerate(zip(days, amounts))]
    s_max = detect(events, prof(), EMPTY_RATES, date(2024, 1, 25), estimator="max")[0]
    s_mean = detect(events, prof(), EMPTY_RATES, date(2024, 1, 25), estimator="mean")[0]
    assert s_max.amount == Decimal("70")
    assert s_mean.amount == Decimal("55")
    assert s_max.amount > s_mean.amount, "max must be the more conservative of the two"


def test_flexibility_metadata_is_carried_onto_the_series():
    days = [date(2024, 1, 10), date(2024, 2, 10), date(2024, 3, 10)]
    events = [ev(f"event_{i}", d, "47", category="streaming",
                 description="Streaming subscription",
                 flexibility="reducible_or_stoppable", minimum="23.5")
              for i, d in enumerate(days)]
    s = detect(events, prof(), EMPTY_RATES, as_of=date(2024, 3, 15))[0]
    assert s.flexibility == "reducible_or_stoppable"
    assert s.minimum_allowed_amount == Decimal("23.5")
    assert s.template_event_id == "event_2", "must point at the most recent occurrence"


def test_cancelled_and_failed_events_never_form_a_series():
    days = [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]
    events = [ev(f"event_{i}", d, "100", status="cancelled") for i, d in enumerate(days)]
    assert detect(events, prof(), EMPTY_RATES, as_of=date(2024, 3, 5)) == []


def test_income_series_is_positive_and_expense_series_is_negative():
    days = [date(2024, 1, 15), date(2024, 2, 15), date(2024, 3, 15)]
    salary = [ev(f"event_{i}", d, "900", category="salary", description="Payroll credit",
                 direction="credit") for i, d in enumerate(days)]
    s = detect(salary, prof(), EMPTY_RATES, as_of=date(2024, 3, 20))[0]
    assert s.direction == "credit"
    assert s.signed_amount == Decimal("900")

    rent = [ev(f"event_r{i}", d, "500") for i, d in enumerate(days)]
    r = detect(rent, prof(), EMPTY_RATES, as_of=date(2024, 3, 20))[0]
    assert r.signed_amount == Decimal("-500")


def test_month_end_anchor_clamps_into_short_months():
    days = [date(2024, 1, 31), date(2024, 3, 31), date(2024, 5, 31)]
    events = [ev(f"event_{i}", d, "100", description="Month end fee")
              for i, d in enumerate(days)]
    s = detect(events, prof(), EMPTY_RATES, as_of=date(2024, 6, 1),
               min_observations=3, allow_any_cadence=True)
    if s:
        occ = s[0].occurrences(date(2024, 1, 31), date(2024, 3, 1))
        assert date(2024, 2, 29) in occ, "day 31 must clamp to the last day of February"


def test_estimators_are_all_registered():
    assert set(ESTIMATORS) == {"last", "mean", "median", "p75", "max", "max3"}


def test_detect_is_deterministic_and_order_independent():
    days = [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]
    events = [ev(f"event_{i}", d, "467.50") for i, d in enumerate(days)]
    a = detect(events, prof(), EMPTY_RATES, as_of=date(2024, 3, 5))
    b = detect(list(reversed(events)), prof(), EMPTY_RATES, as_of=date(2024, 3, 5))
    assert a == b


def test_rotating_descriptions_still_form_one_category_level_series():
    """The dataset rotates the description for everyday spending: groceries turn
    up as 'Supermarket basket', 'Grocery delivery', 'Bulk pantry shop' and so on.
    Grouped by description each variant looks irregular and none is detected,
    yet the category is plainly weekly. Essential variable spending must be
    picked up at the category level.
    """
    names = ["Supermarket basket", "Grocery delivery", "Bulk pantry shop",
             "Fresh food shop", "Weekly produce market", "Neighbourhood grocer"]
    days = [date(2024, 1, 2), date(2024, 1, 9), date(2024, 1, 16),
            date(2024, 1, 23), date(2024, 1, 30), date(2024, 2, 6)]
    events = [ev(f"event_{i}", d, "60", category="groceries", description=n)
              for i, (d, n) in enumerate(zip(days, names))]

    series = detect(events, prof(), EMPTY_RATES, as_of=date(2024, 2, 10))
    assert len(series) == 1, "expected one weekly groceries series"
    assert series[0].category == "groceries"
    assert series[0].cadence == "weekly"


def test_distinct_fixed_commitments_in_one_category_stay_separate():
    """The opposite case must keep working: two different subscriptions share a
    category but recur on different days and must not be merged."""
    music = [ev(f"event_m{i}", date(2024, m, 10), "14", category="subscription",
                description="Music subscription") for i, m in enumerate((1, 2, 3))]
    backup = [ev(f"event_b{i}", date(2024, m, 22), "11", category="subscription",
                 description="Online backup") for i, m in enumerate((1, 2, 3))]

    series = detect(music + backup, prof(), EMPTY_RATES, as_of=date(2024, 3, 25))
    assert len(series) == 2
    assert {s.anchor for s in series} == {10, 22}


def test_category_fallback_does_not_double_count_a_detected_commitment():
    """Rent already forms a description-level series, so the category must not
    also contribute a second one."""
    days = [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]
    events = [ev(f"event_{i}", d, "467.50") for i, d in enumerate(days)]
    series = detect(events, prof(), EMPTY_RATES, as_of=date(2024, 3, 5))
    assert len(series) == 1


def test_income_projection_can_be_disabled():
    """The problem statement says to count CONFIRMED salary and not to invent
    unsupported future income. Whether a recurring salary may be projected
    beyond its confirmed row is therefore a calibration question, so it is an
    explicit switch rather than a baked-in assumption."""
    days = [date(2024, 1, 15), date(2024, 2, 15), date(2024, 3, 15)]
    salary = [ev(f"event_{i}", d, "900", category="salary",
                 description="Payroll credit", direction="credit")
              for i, d in enumerate(days)]
    rent = [ev(f"event_r{i}", d, "500", category="rent") for i, d in
            enumerate([date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)])]

    with_income = detect(salary + rent, prof(), EMPTY_RATES, date(2024, 3, 20),
                         project_income=True)
    assert {s.category for s in with_income} == {"salary", "rent"}

    without = detect(salary + rent, prof(), EMPTY_RATES, date(2024, 3, 20),
                     project_income=False)
    assert {s.category for s in without} == {"rent"}
    assert all(s.direction == "debit" for s in without)


def test_a_one_off_bulk_purchase_does_not_inflate_the_recurring_estimate():
    """'Distinguish recurring expenses from one-time purchases.' A bulk shop of
    41,272 among weekly groceries of about 8,600 is a one-off; letting it into
    the estimate would project a 41,272 grocery bill every week."""
    days = [date(2026, 1, 2), date(2026, 1, 9), date(2026, 1, 16), date(2026, 1, 23),
            date(2026, 1, 30), date(2026, 2, 6)]
    amounts = ["8124.44", "11433.33", "7093.83", "8638.54", "8581.99", "11342.57"]
    # Rotating descriptions, as in the real data, so the category-level pass is
    # the one that forms the series - and the one the bulk shop can leak into.
    names = ["Supermarket basket", "Grocery delivery", "Bulk pantry shop",
             "Fresh food shop", "Weekly produce market", "Neighbourhood grocer"]
    events = [ev(f"event_{i}", d, a, category="groceries", description=n)
              for i, (d, a, n) in enumerate(zip(days, amounts, names))]
    events.append(ev("event_bulk", date(2026, 2, 6), "41272", category="groceries",
                     description="Bulk groceries and pantry purchase"))

    s = detect(events, prof(), EMPTY_RATES, as_of=date(2026, 2, 10), estimator="max")
    groceries = [x for x in s if x.category == "groceries"]
    assert groceries
    assert all(x.amount < Decimal("20000") for x in groceries), \
        f"one-off leaked into the estimate: {[str(x.amount) for x in groceries]}"


def test_outlier_filter_leaves_a_steady_series_untouched():
    days = [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]
    events = [ev(f"event_{i}", d, "467.50") for i, d in enumerate(days)]
    s = detect(events, prof(), EMPTY_RATES, as_of=date(2024, 3, 5), estimator="max")
    assert s[0].amount == Decimal("467.50")


def test_a_confirmed_next_salary_establishes_a_monthly_income_series():
    """A new employee has one prorated payslip and a scheduled 'Next confirmed
    salary'. Too little history for generic detection, yet the salary is
    confirmed - so it recurs monthly on its day at its confirmed amount."""
    first = ev("event_s1", date(2024, 2, 15), "12826", category="salary",
               description="Prorated first salary", direction="credit")
    confirmed = ev("event_s2", date(2024, 3, 15), "23320", category="salary",
                   description="Next confirmed salary", direction="credit",
                   status="scheduled")
    s = [x for x in detect([first, confirmed], prof(), EMPTY_RATES, date(2024, 3, 3))
         if x.category == "salary"]
    assert len(s) == 1
    assert s[0].cadence == "monthly" and s[0].anchor == 15
    assert s[0].amount == Decimal("23320")
    assert date(2024, 4, 15) in s[0].occurrences(date(2024, 3, 15), date(2024, 5, 31))
