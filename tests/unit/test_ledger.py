from datetime import date
from decimal import Decimal
from pathlib import Path

from buyorwait.fx import RateTable
from buyorwait.io_loaders import load_dataset
from buyorwait.ledger import CashEffect, LedgerView, classify, resolve_links
from buyorwait.types import Event, Profile, Rate

ROOT = Path(__file__).resolve().parents[2]
EMPTY_RATES = RateTable([])


def ev(**kw) -> Event:
    base = dict(
        event_id="event_x", user_id="user_x", event_type="expense",
        description="d", category="groceries", direction="debit",
        amount=Decimal("100"), currency="EUR",
        event_date=date(2024, 1, 1), settlement_date=date(2024, 1, 1),
        status="settled", linked_event_id=None, flexibility="fixed",
        minimum_allowed_amount=None,
    )
    base.update(kw)
    return Event(**base)


def prof(**kw) -> Profile:
    base = dict(
        user_id="user_x", home_currency="EUR",
        current_available_balance=Decimal("1000"),
        minimum_balance_to_keep=Decimal("200"),
        financial_priorities=(), expense_categories_to_protect=(),
        expense_categories_user_is_willing_to_reduce=(),
        expense_categories_user_is_willing_to_stop=(),
        payment_methods_user_will_consider=("full_payment",),
        max_installment_months=None,
    )
    base.update(kw)
    return Profile(**base)


# --------------------------------------------------------------- classify ---

def test_settled_debit_counts():
    assert classify(ev(status="settled", direction="debit")) is CashEffect.COUNT


def test_scheduled_credit_counts_confirmed_salary():
    e = ev(status="scheduled", direction="credit", event_type="income", category="salary")
    assert classify(e) is CashEffect.COUNT


def test_pending_debit_is_reserved():
    """Money is committed even though it has not settled."""
    assert classify(ev(status="pending", direction="debit")) is CashEffect.RESERVE


def test_pending_credit_is_ignored():
    """Trap 3: never count a credit that has not landed."""
    e = ev(status="pending", direction="credit", event_type="refund")
    assert classify(e) is CashEffect.IGNORE


def test_cancelled_is_ignored():
    assert classify(ev(status="cancelled")) is CashEffect.IGNORE


def test_failed_is_ignored():
    assert classify(ev(status="failed")) is CashEffect.IGNORE


def test_unrealized_non_cash_valuation_is_ignored():
    """Trap 4: an investment's paper value is not spendable cash."""
    e = ev(status="unrealized", direction="non_cash", event_type="investment_valuation")
    assert classify(e) is CashEffect.IGNORE


def test_blank_amount_needs_resolution_and_is_never_zero():
    """Trap 6."""
    assert classify(ev(amount=None)) is CashEffect.NEEDS_AMOUNT


# ----------------------------------------------------------- resolve_links ---

def test_resolve_links_drops_cancelled_authorization_keeping_settlement():
    """Trap 1: an authorization that was cancelled, plus the charge that settled."""
    auth = ev(event_id="event_a", status="cancelled", amount=Decimal("816.20"))
    settled = ev(event_id="event_b", status="settled", amount=Decimal("816.20"),
                 linked_event_id="event_a")
    out = resolve_links([auth, settled])
    assert [e.event_id for e in out] == ["event_b"]


def test_resolve_links_keeps_both_sides_of_a_charge_and_refund():
    """Trap 2: they net to zero, so BOTH must survive - dropping one is wrong."""
    charge = ev(event_id="event_a", direction="debit", amount=Decimal("583"))
    refund = ev(event_id="event_b", direction="credit", event_type="refund",
                amount=Decimal("583"), linked_event_id="event_a")
    out = resolve_links([charge, refund])
    assert {e.event_id for e in out} == {"event_a", "event_b"}


def test_resolve_links_drops_failed_payment_keeping_the_retry():
    """Trap 5."""
    failed = ev(event_id="event_a", status="failed", amount=Decimal("13800"))
    retry = ev(event_id="event_b", status="scheduled", amount=Decimal("13800"),
               linked_event_id="event_a")
    out = resolve_links([failed, retry])
    assert [e.event_id for e in out] == ["event_b"]


def test_resolve_links_leaves_unlinked_events_alone():
    a, b = ev(event_id="event_a"), ev(event_id="event_b")
    assert len(resolve_links([a, b])) == 2


# --------------------------------------------------------------- LedgerView ---

def test_history_before_request_date_is_not_replayed():
    """current_available_balance already reflects settled history. Re-applying it
    would double-count - the single most dangerous bug in this system."""
    past = ev(event_id="event_p", settlement_date=date(2024, 1, 1), amount=Decimal("500"))
    future = ev(event_id="event_f", settlement_date=date(2024, 3, 1), amount=Decimal("50"))
    view = LedgerView([past, future], prof(), EMPTY_RATES)
    assert view.future_cashflows(date(2024, 2, 1)) == [(date(2024, 3, 1), Decimal("-50"))]


def test_an_event_settling_on_the_request_date_is_already_in_the_balance():
    same_day = ev(event_id="event_s", settlement_date=date(2024, 2, 1),
                  amount=Decimal("40"))
    view = LedgerView([same_day], prof(), EMPTY_RATES)
    assert view.future_cashflows(date(2024, 2, 1)) == []


def test_pending_debit_before_request_date_is_still_reserved():
    """It has not settled, so it is NOT in the balance yet, even though it is old."""
    pending = ev(event_id="event_p", status="pending",
                 settlement_date=date(2024, 1, 20), amount=Decimal("75"))
    view = LedgerView([pending], prof(), EMPTY_RATES)
    assert view.future_cashflows(date(2024, 2, 1)) == [(date(2024, 2, 1), Decimal("-75"))]


def test_credit_is_positive_and_debit_is_negative():
    credit = ev(event_id="event_c", direction="credit", event_type="income",
                category="salary", status="scheduled",
                settlement_date=date(2024, 2, 15), amount=Decimal("900"))
    debit = ev(event_id="event_d", settlement_date=date(2024, 2, 10), amount=Decimal("40"))
    view = LedgerView([credit, debit], prof(), EMPTY_RATES)
    assert view.future_cashflows(date(2024, 2, 1)) == [
        (date(2024, 2, 10), Decimal("-40")),
        (date(2024, 2, 15), Decimal("900")),
    ]


def test_foreign_currency_is_converted_to_home_currency():
    rates = RateTable([Rate(date(2024, 2, 15), "USD", "EUR", Decimal("0.9"))])
    credit = ev(event_id="event_c", direction="credit", event_type="income",
                category="salary", status="scheduled", currency="USD",
                settlement_date=date(2024, 2, 15), amount=Decimal("1000"))
    view = LedgerView([credit], prof(home_currency="EUR"), rates)
    assert view.future_cashflows(date(2024, 2, 1)) == [(date(2024, 2, 15), Decimal("900.0"))]


def test_blank_amount_event_never_reaches_the_cashflows():
    blank = ev(event_id="event_b", amount=None, settlement_date=date(2024, 3, 1))
    view = LedgerView([blank], prof(), EMPTY_RATES)
    assert view.future_cashflows(date(2024, 2, 1)) == []


def test_real_dataset_traps_are_resolved_as_expected():
    """Uses the actual linked pairs in the dataset as fixtures."""
    ds = load_dataset(ROOT / "dataset")

    # Trap 1: cancelled authorization superseded by the settled purchase.
    kept = {e.event_id for e in resolve_links(ds.events_by_user["user_01"])}
    assert "event_100" not in kept, "cancelled authorization should be dropped"
    assert "event_101" in kept, "settled purchase should survive"

    # Trap 2: charge and its reversal both survive and net to zero.
    assert {"event_98", "event_99"} <= kept

    # Trap 4: unrealized valuations are never counted anywhere.
    for events in ds.events_by_user.values():
        for e in events:
            if e.status == "unrealized":
                assert classify(e) is CashEffect.IGNORE
