import random
from datetime import date, timedelta
from decimal import Decimal

from buyorwait.forecast import Curve
from buyorwait.simulate import is_safe
from buyorwait.solver import (earliest_full_payment_date, safe_amount,
                              safe_amount_by_search)

START = date(2024, 3, 1)


def curve(opening, flows=()):
    return Curve(start=START, end=START + timedelta(days=89),
                 opening=Decimal(opening), flows=tuple(flows))


def test_safe_amount_is_headroom_above_the_minimum():
    assert safe_amount(curve("1000"), Decimal("200"), Decimal("5000"), START) == Decimal("800")


def test_safe_amount_is_capped_at_the_requested_amount():
    assert safe_amount(curve("1000"), Decimal("200"), Decimal("300"), START) == Decimal("300")


def test_safe_amount_accounts_for_a_future_trough_not_just_today():
    c = curve("1000", [(date(2024, 4, 1), Decimal("-600"))])
    assert safe_amount(c, Decimal("200"), Decimal("5000"), START) == Decimal("200")


def test_safe_amount_is_never_negative():
    assert safe_amount(curve("100"), Decimal("200"), Decimal("5000"), START) == Decimal("0")


def test_safe_amount_ignores_income_arriving_after_the_trough():
    """Money that lands later cannot rescue a dip that happens first."""
    c = curve("1000", [(date(2024, 4, 1), Decimal("-900")),
                       (date(2024, 4, 15), Decimal("5000"))])
    assert safe_amount(c, Decimal("0"), Decimal("5000"), START) == Decimal("100")


def test_closed_form_agrees_with_the_binary_search_oracle():
    """The fast arithmetic answer must equal the slow, obviously-correct one."""
    rng = random.Random(0)
    for _ in range(200):
        opening = Decimal(rng.randrange(0, 500_000))
        minimum = Decimal(rng.randrange(0, 100_000))
        requested = Decimal(rng.randrange(1, 500_000))
        flows = tuple(sorted(
            ((START + timedelta(days=rng.randrange(1, 90)),
              Decimal(rng.randrange(-50_000, 50_000)))
             for _ in range(rng.randrange(0, 12))),
            key=lambda f: f[0]))
        c = curve(opening, flows)
        fast = safe_amount(c, minimum, requested, START)
        slow = safe_amount_by_search(c, minimum, requested, START)
        assert abs(fast - slow) <= Decimal("0.00001"), (
            f"mismatch: opening={opening} min={minimum} req={requested} "
            f"fast={fast} slow={slow}")


def test_safe_amount_result_is_always_actually_safe():
    """Whatever it returns must survive the simulator."""
    rng = random.Random(7)
    for _ in range(100):
        opening = Decimal(rng.randrange(0, 200_000))
        minimum = Decimal(rng.randrange(0, 50_000))
        requested = Decimal(rng.randrange(1, 200_000))
        flows = tuple(sorted(
            ((START + timedelta(days=rng.randrange(1, 90)),
              Decimal(rng.randrange(-20_000, 20_000)))
             for _ in range(rng.randrange(0, 8))),
            key=lambda f: f[0]))
        c = curve(opening, flows)
        x = safe_amount(c, minimum, requested, START)
        if x > 0:
            assert is_safe(c, minimum, [(START, x)])


def test_earliest_date_is_request_date_when_affordable_now():
    assert earliest_full_payment_date(curve("1000"), Decimal("200"),
                                      Decimal("500")) == START


def test_earliest_date_waits_for_incoming_money():
    """Opens above the minimum but cannot cover the request until payday."""
    c = curve("250", [(date(2024, 3, 15), Decimal("900"))])
    assert earliest_full_payment_date(c, Decimal("200"), Decimal("500")) == date(2024, 3, 15)


def test_earliest_date_is_the_first_feasible_day_not_merely_a_feasible_one():
    c = curve("250", [(date(2024, 3, 15), Decimal("900"))])
    got = earliest_full_payment_date(c, Decimal("200"), Decimal("500"))
    assert got is not None
    assert is_safe(c, Decimal("200"), [(got, Decimal("500"))])
    assert not is_safe(c, Decimal("200"), [(got - timedelta(days=1), Decimal("500"))])


def test_no_date_is_safe_when_the_balance_is_already_below_the_minimum():
    """A pre-existing breach makes every plan unsafe - the problem statement
    requires the balance never to fall below the minimum, not merely that the
    plan alone does not cause the dip."""
    c = curve("100", [(date(2024, 3, 15), Decimal("900"))])
    assert earliest_full_payment_date(c, Decimal("200"), Decimal("500")) is None
    assert safe_amount(c, Decimal("200"), Decimal("500"), START) == Decimal("0")


def test_earliest_date_is_none_when_never_affordable_in_the_window():
    assert earliest_full_payment_date(curve("100"), Decimal("200"),
                                      Decimal("5000")) is None


def test_earliest_date_requires_safety_for_the_whole_remaining_window():
    """Affording it on the day is not enough if a bill right after would break
    the minimum."""
    c = curve("1000", [(date(2024, 3, 20), Decimal("-400"))])
    got = earliest_full_payment_date(c, Decimal("200"), Decimal("500"))
    assert got is None or is_safe(c, Decimal("200"), [(got, Decimal("500"))])
