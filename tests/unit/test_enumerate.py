from datetime import date, timedelta
from decimal import Decimal

from buyorwait.changes import Change
from buyorwait.forecast import Curve
from buyorwait.recurrence import Series
from buyorwait.solver import enumerate_candidates
from buyorwait.types import PaymentOption, Profile, Request

D = Decimal
START = date(2024, 3, 1)


def req(amount="1000", partial=True, deadline=date(2024, 5, 1)):
    return Request("request_x", "user_x", START, "purchase", D(amount), deadline,
                   partial, "text")


def prof(methods=("full_payment",), max_months=None, balance="5000", minimum="200"):
    return Profile("user_x", "EUR", D(balance), D(minimum), (), (), (), (),
                   methods, max_months)


def curve(opening="5000", flows=()):
    return Curve(START, START + timedelta(days=89), D(opening), tuple(flows))


def option(oid, method, amount, n, first, freq, total, sort_key=None):
    return PaymentOption(oid, "request_x", method, D(amount), amount, n, first,
                         freq, D("0"), D(total))


def a_change(event_id="event_1", amount="10"):
    s = Series("streaming", "Sub", event_id, "monthly", 10, D(amount), "debit",
               "stoppable", None, date(2024, 2, 10))
    return Change("stop", event_id, s, None, D(amount))


# ------------------------------------------------------------- eligibility ---

def test_full_payment_offered_only_when_the_user_accepts_it():
    c = curve()
    accepted = enumerate_candidates(req(), prof(methods=("full_payment",)), c, [], [],
                                    D("1000"), START)
    assert any(x.method == "full_payment" for x in accepted)

    declined = enumerate_candidates(req(), prof(methods=("installments",)), c, [], [],
                                    D("1000"), START)
    assert not any(x.method == "full_payment" for x in declined)


def test_installment_option_exceeding_max_months_is_rejected():
    c = curve()
    long_plan = option("payment_option_07", "installments", "100", 18, START, 30, "1800")
    got = enumerate_candidates(req(), prof(methods=("installments",), max_months=7), c,
                               [long_plan], [], D("1000"), START)
    assert not any(x.option_id == "payment_option_07" for x in got)


def test_installment_option_within_max_months_is_offered():
    c = curve()
    short = option("payment_option_05", "installments", "340", 3, START, 30, "1020")
    got = enumerate_candidates(req(), prof(methods=("installments",), max_months=7), c,
                               [short], [], D("1000"), START)
    assert any(x.option_id == "payment_option_05" for x in got)


def test_blank_max_installment_months_rejects_every_installment_option():
    """119 users have a blank value, which the problem statement says means they
    will not consider installments at all."""
    c = curve()
    short = option("payment_option_05", "installments", "340", 3, START, 30, "1020")
    got = enumerate_candidates(req(), prof(methods=("installments",), max_months=None),
                               c, [short], [], D("1000"), START)
    assert not any(x.method == "installments" for x in got)


def test_unsafe_plans_never_appear():
    c = curve(opening="300")
    got = enumerate_candidates(req(), prof(methods=("full_payment",)), c, [], [],
                               D("100"), None)
    assert not any(x.method == "full_payment" for x in got)


# ----------------------------------------------------------------- partial ---

def test_partial_payment_requires_the_request_to_allow_it():
    c = curve(opening="600")
    got = enumerate_candidates(req(partial=False),
                               prof(methods=("partial_payment",)), c, [], [],
                               D("400"), date(2024, 3, 20))
    assert not any(x.method == "partial_payment" for x in got)


def test_partial_payment_splits_into_exactly_two_payments_summing_to_the_request():
    # 600 now, 900 more on the 20th: enough to pay 400 today and 600 later.
    c = curve(opening="600", flows=[(date(2024, 3, 20), D("900"))])
    got = enumerate_candidates(req(), prof(methods=("partial_payment",)), c, [], [],
                               D("400"), date(2024, 3, 20))
    p = next(x for x in got if x.method == "partial_payment")
    assert len(p.payments) == 2
    assert p.payments[0] == (START, D("400"))
    assert p.payments[1] == (date(2024, 3, 20), D("600"))
    assert sum(amt for _, amt in p.payments) == D("1000")
    assert p.status == "affordable_with_plan"


def test_partial_payment_rejected_when_second_payment_misses_the_deadline():
    c = curve(opening="600")
    got = enumerate_candidates(req(deadline=date(2024, 3, 10)),
                               prof(methods=("partial_payment",)), c, [], [],
                               D("400"), date(2024, 3, 20))
    assert not any(x.method == "partial_payment" for x in got)


def test_partial_payment_rejected_when_safe_amount_is_zero_or_full():
    c = curve(opening="600")
    zero = enumerate_candidates(req(), prof(methods=("partial_payment",)), c, [], [],
                                D("0"), date(2024, 3, 20))
    assert not any(x.method == "partial_payment" for x in zero)
    full = enumerate_candidates(req(), prof(methods=("partial_payment",)), c, [], [],
                                D("1000"), date(2024, 3, 20))
    assert not any(x.method == "partial_payment" for x in full)


# -------------------------------------------------------------------- wait ---

def test_wait_is_offered_when_full_payment_becomes_safe_later():
    c = curve(opening="1500", flows=[(date(2024, 3, 15), D("2000"))])
    got = enumerate_candidates(req(), prof(methods=("full_payment",)), c, [], [],
                               D("0"), date(2024, 3, 15))
    w = next(x for x in got if x.method == "wait")
    assert w.payments == ((date(2024, 3, 15), D("1000")),)
    assert w.status == "affordable_later"


def test_wait_is_not_offered_when_the_user_rejects_full_payment():
    c = curve(opening="1500", flows=[(date(2024, 3, 15), D("2000"))])
    got = enumerate_candidates(req(), prof(methods=("installments",)), c, [], [],
                               D("0"), date(2024, 3, 15))
    assert not any(x.method == "wait" for x in got)


def test_wait_is_not_offered_when_the_money_is_already_there():
    c = curve()
    got = enumerate_candidates(req(), prof(), c, [], [], D("1000"), START)
    assert not any(x.method == "wait" for x in got)


# ------------------------------------------------------------------ status ---

def test_full_payment_today_yields_affordable_now_status():
    c = curve()
    got = enumerate_candidates(req(), prof(), c, [], [], D("1000"), START)
    f = next(x for x in got if x.method == "full_payment" and not x.changes)
    assert f.status == "affordable_now"


def test_full_payment_with_a_spending_change_yields_affordable_with_plan():
    """Matches the labeled samples where safe < requested but a change closes
    the gap and the plan still pays in full today.

    Paying 1000 leaves 205; the 10 streaming charge on 10 April then drops the
    balance to 195, under the 200 minimum. Stopping that subscription removes
    the charge and makes the same full payment safe.
    """
    c = curve(opening="1205", flows=[(date(2024, 4, 10), D("-10"))])
    got = enumerate_candidates(req(), prof(), c, [], [a_change()], D("995"), None)

    assert not [x for x in got if x.method == "full_payment" and not x.changes], \
        "full payment must be unsafe without the change"
    with_change = [x for x in got if x.changes and x.method == "full_payment"]
    assert with_change
    assert all(x.status == "affordable_with_plan" for x in with_change)


def test_installments_always_yield_affordable_with_plan():
    c = curve()
    opt = option("payment_option_05", "installments", "340", 3, START, 30, "1020")
    got = enumerate_candidates(req(), prof(methods=("installments",), max_months=7), c,
                               [opt], [], D("1000"), START)
    inst = next(x for x in got if x.method == "installments")
    assert inst.status == "affordable_with_plan"


def test_option_payments_follow_the_frequency_schedule():
    c = curve()
    opt = option("payment_option_05", "installments", "340", 3, START, 30, "1020")
    got = enumerate_candidates(req(), prof(methods=("installments",), max_months=7), c,
                               [opt], [], D("1000"), START)
    inst = next(x for x in got if x.method == "installments")
    assert [d for d, _ in inst.payments] == [START,
                                             START + timedelta(days=30),
                                             START + timedelta(days=60)]
    assert inst.payment_texts == ("340", "340", "340")
    assert inst.total_paid == D("1020")


def test_completes_by_deadline_reflects_the_last_payment_date():
    c = curve()
    late = option("payment_option_09", "installments", "340", 3, START, 60, "1020")
    got = enumerate_candidates(req(deadline=date(2024, 4, 1)),
                               prof(methods=("installments",), max_months=7), c,
                               [late], [], D("1000"), START)
    inst = next(x for x in got if x.method == "installments")
    assert inst.completes_by_deadline is False
