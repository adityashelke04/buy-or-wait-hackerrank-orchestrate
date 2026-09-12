from datetime import date
from decimal import Decimal
from pathlib import Path

from buyorwait.fx import RateTable
from buyorwait.io_loaders import load_dataset
from buyorwait.types import Rate

ROOT = Path(__file__).resolve().parents[2]

RATES = [
    Rate(date(2024, 1, 15), "USD", "INR", Decimal("83")),
    Rate(date(2024, 2, 15), "USD", "INR", Decimal("84")),
]


def test_same_currency_is_identity():
    t = RateTable(RATES)
    assert t.convert(Decimal("100"), "INR", "INR", date(2024, 1, 15)) == Decimal("100")


def test_exact_date_uses_that_rate():
    t = RateTable(RATES)
    assert t.convert(Decimal("100"), "USD", "INR", date(2024, 2, 15)) == Decimal("8400")


def test_missing_date_uses_most_recent_earlier_rate_never_a_future_one():
    t = RateTable(RATES)
    assert t.convert(Decimal("100"), "USD", "INR", date(2024, 1, 20)) == Decimal("8300")


def test_before_all_rates_falls_back_to_earliest():
    t = RateTable(RATES)
    assert t.convert(Decimal("100"), "USD", "INR", date(2023, 1, 1)) == Decimal("8300")


def test_after_all_rates_uses_the_latest():
    t = RateTable(RATES)
    assert t.convert(Decimal("100"), "USD", "INR", date(2030, 1, 1)) == Decimal("8400")


def test_reverse_direction_inverts_when_no_direct_pair():
    t = RateTable(RATES)
    got = t.convert(Decimal("8300"), "INR", "USD", date(2024, 1, 15))
    assert abs(got - Decimal("100")) < Decimal("0.01")


def test_unknown_pair_raises_rather_than_silently_returning_zero():
    t = RateTable(RATES)
    try:
        t.convert(Decimal("100"), "JPY", "BRL", date(2024, 1, 15))
    except KeyError:
        return
    raise AssertionError("expected KeyError for an unknown currency pair")


def test_every_real_foreign_event_can_be_converted():
    """Guards against a missing rate silently becoming a zero.

    The dataset holds 140 foreign-currency events. Exactly one of them
    (a USD taxi fare) has a blank amount that only an image can resolve, so 139
    are convertible at load time.
    """
    ds = load_dataset(ROOT / "dataset")
    t = RateTable(ds.rates)
    converted = 0
    awaiting_image = 0
    for user_id, events in ds.events_by_user.items():
        home = ds.profiles[user_id].home_currency
        for e in events:
            if e.currency == home:
                continue
            if e.amount is None:
                awaiting_image += 1
                assert e.event_id in ds.images_by_event
                continue
            out = t.convert(e.amount, e.currency, home, e.settlement_date)
            assert out > 0, f"{e.event_id} converted to {out}"
            converted += 1
    assert converted == 139, f"expected 139 convertible, saw {converted}"
    assert awaiting_image == 1, f"expected 1 awaiting an image, saw {awaiting_image}"
