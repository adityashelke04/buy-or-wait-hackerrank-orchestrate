from datetime import date
from decimal import Decimal
from pathlib import Path

from buyorwait.io_loaders import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"


def test_loads_every_file_with_expected_row_counts():
    ds = load_dataset(DATASET)
    assert len(ds.profiles) == 275
    assert len(ds.requests) == 250
    assert sum(len(v) for v in ds.events_by_user.values()) == 25342
    assert sum(len(v) for v in ds.options_by_request.values()) == 790
    assert sum(len(v) for v in ds.messages_by_user.values()) == 215
    assert len(ds.images_by_event) == 16
    assert len(ds.rates) == 134


def test_types_are_parsed_not_strings():
    ds = load_dataset(DATASET)
    p = ds.profiles["user_01"]
    assert p.home_currency == "ZAR"
    assert p.current_available_balance == Decimal("58481.1")
    assert p.minimum_balance_to_keep == Decimal("18000")
    assert p.payment_methods_user_will_consider == ("full_payment",)
    assert p.max_installment_months is None          # blank means no installments

    r = ds.requests[0]
    assert isinstance(r.request_date, date)
    assert isinstance(r.requested_amount, Decimal)
    assert isinstance(r.allows_partial_payment, bool)


def test_multi_valued_profile_columns_split_on_pipe():
    ds = load_dataset(DATASET)
    p = ds.profiles["user_03"]
    assert p.expense_categories_user_is_willing_to_stop == ("streaming", "cloud_storage")
    assert p.payment_methods_user_will_consider == (
        "full_payment", "partial_payment", "installments")
    assert p.max_installment_months == 2


def test_blank_event_amount_is_none_not_zero():
    ds = load_dataset(DATASET)
    blanks = [e for evs in ds.events_by_user.values() for e in evs if e.amount is None]
    assert len(blanks) == 16
    for e in blanks:
        assert e.event_id in ds.images_by_event, f"{e.event_id} has no image to resolve it"


def test_referential_integrity():
    ds = load_dataset(DATASET)
    for r in ds.requests:
        assert r.user_id in ds.profiles
        assert r.request_id in ds.options_by_request


def test_images_resolve_to_existing_png_files():
    ds = load_dataset(DATASET)
    for ref in ds.images_by_event.values():
        assert ref.path(DATASET).exists(), f"missing {ref.image_id}.png"


def test_events_are_sorted_by_settlement_date():
    ds = load_dataset(DATASET)
    for events in ds.events_by_user.values():
        dates = [e.settlement_date for e in events]
        assert dates == sorted(dates)


def test_payment_options_keep_the_verbatim_amount_text():
    """Installment plans must reproduce the option exactly, so the original
    string is preserved rather than re-rendered from the Decimal."""
    ds = load_dataset(DATASET)
    for opts in ds.options_by_request.values():
        for o in opts:
            assert o.payment_amount_text
            assert Decimal(o.payment_amount_text) == o.payment_amount


def test_event_flexibility_helpers():
    ds = load_dataset(DATASET)
    by_id = ds.events_by_id
    flexible = [e for e in by_id.values() if e.flexibility == "reducible_or_stoppable"]
    assert flexible, "dataset should contain reducible_or_stoppable events"
    e = flexible[0]
    assert e.is_flexible and e.can_stop and e.can_reduce

    fixed = next(e for e in by_id.values() if e.flexibility == "fixed")
    assert not fixed.is_flexible and not fixed.can_stop and not fixed.can_reduce
