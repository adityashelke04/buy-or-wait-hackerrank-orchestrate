from datetime import date
from decimal import Decimal

from buyorwait.forecast import Curve
from buyorwait.simulate import balance_series, is_safe, trough


def curve(opening="1000", flows=()):
    return Curve(start=date(2024, 3, 1), end=date(2024, 5, 29),
                 opening=Decimal(opening), flows=tuple(flows))


def test_trough_of_a_flat_curve_is_the_opening_balance():
    assert trough(curve()) == Decimal("1000")


def test_trough_finds_the_lowest_point_not_the_final_balance():
    c = curve(flows=[(date(2024, 3, 10), Decimal("-800")),
                     (date(2024, 3, 20), Decimal("900"))])
    assert trough(c) == Decimal("200")


def test_payments_are_applied_on_their_dates():
    c = curve(flows=[(date(2024, 4, 1), Decimal("500"))])
    assert trough(c, [(date(2024, 3, 1), Decimal("700"))]) == Decimal("300")


def test_is_safe_is_inclusive_of_the_minimum():
    c = curve(flows=[(date(2024, 3, 10), Decimal("-800"))])
    assert is_safe(c, Decimal("200")) is True          # lands exactly on 200
    assert is_safe(c, Decimal("200.01")) is False


def test_same_day_flows_all_apply_before_the_day_is_measured():
    """A payment and an expense on the same day both count that day."""
    c = curve(flows=[(date(2024, 3, 5), Decimal("-500"))])
    assert trough(c, [(date(2024, 3, 5), Decimal("400"))]) == Decimal("100")


def test_paying_more_never_raises_the_trough():
    c = curve(flows=[(date(2024, 4, 1), Decimal("-100"))])
    low = trough(c, [(date(2024, 3, 1), Decimal("100"))])
    high = trough(c, [(date(2024, 3, 1), Decimal("500"))])
    assert high < low, "the trough must be monotonically decreasing in the payment"


def test_trough_is_linear_in_the_payment_amount():
    """The property the closed-form solver relies on."""
    c = curve(flows=[(date(2024, 4, 1), Decimal("-100")),
                     (date(2024, 4, 15), Decimal("250"))])
    base = trough(c)
    for x in ("0", "50", "123.45", "900"):
        assert trough(c, [(date(2024, 3, 1), Decimal(x))]) == base - Decimal(x)


def test_balance_series_reports_one_point_per_flow_date():
    c = curve(flows=[(date(2024, 3, 10), Decimal("-100")),
                     (date(2024, 3, 20), Decimal("-100"))])
    pts = balance_series(c)
    assert pts[0] == (date(2024, 3, 1), Decimal("1000"))
    assert pts[-1] == (date(2024, 3, 20), Decimal("800"))


def test_balance_series_collapses_same_day_flows_into_one_point():
    c = curve(flows=[(date(2024, 3, 10), Decimal("-100")),
                     (date(2024, 3, 10), Decimal("-50"))])
    pts = balance_series(c)
    assert [d for d, _ in pts] == [date(2024, 3, 1), date(2024, 3, 10)]
    assert pts[-1][1] == Decimal("850")
