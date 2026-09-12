from datetime import date
from decimal import Decimal

from buyorwait.changes import Change
from buyorwait.ranker import Candidate, rank
from buyorwait.recurrence import Series

D = Decimal


def a_change(event_id="event_1"):
    s = Series("streaming", "Sub", event_id, "monthly", 10, D("10"), "debit",
               "stoppable", None, date(2024, 2, 10))
    return Change("stop", event_id, s, None, D("10"))


def cand(method="full_payment", payments=((date(2024, 3, 1), D("100")),),
         total="100", changes=(), option_id=None, option_sort_key=10**9,
         completes=True, status="affordable_now"):
    return Candidate(method=method, payments=tuple(payments),
                     payment_texts=tuple(str(amt) for _, amt in payments),
                     total_paid=D(total), changes=tuple(changes),
                     option_id=option_id, option_sort_key=option_sort_key,
                     completes_by_deadline=completes, status=status)


def test_rung_1_completing_by_the_deadline_beats_everything():
    late = cand(total="50", completes=False)
    on_time = cand(total="500", completes=True)
    assert rank([late, on_time]) is on_time


def test_rung_2_no_spending_changes_beats_a_plan_that_needs_them():
    with_change = cand(total="100", changes=(a_change(),))
    without = cand(total="100")
    assert rank([with_change, without]) is without


def test_rung_2_fewer_spending_changes_wins():
    two = cand(changes=(a_change("event_1"), a_change("event_2")))
    one = cand(changes=(a_change("event_1"),))
    assert rank([two, one]) is one


def test_rung_3_lower_total_paid_wins():
    cheap = cand(total="100")
    pricey = cand(total="120")
    assert rank([pricey, cheap]) is cheap


def test_rung_4_earlier_start_wins_at_equal_cost():
    early = cand(payments=((date(2024, 3, 1), D("100")),))
    late = cand(payments=((date(2024, 3, 8), D("100")),))
    assert rank([late, early]) is early


def test_rung_5_fewer_payments_wins():
    one = cand(payments=((date(2024, 3, 1), D("100")),))
    three = cand(payments=((date(2024, 3, 1), D("34")),
                           (date(2024, 4, 1), D("33")),
                           (date(2024, 5, 1), D("33"))))
    assert rank([three, one]) is one


def test_rung_6_lowest_option_id_is_the_final_tiebreak():
    a = cand(option_id="payment_option_07", option_sort_key=7)
    b = cand(option_id="payment_option_05", option_sort_key=5)
    assert rank([a, b]) is b


def test_higher_rungs_dominate_lower_ones():
    """A costlier plan that completes on time beats a cheap one that does not."""
    cheap_late = cand(total="10", completes=False, payments=((date(2024, 3, 1), D("10")),))
    dear_on_time = cand(total="999", completes=True,
                        payments=((date(2024, 9, 1), D("999")),))
    assert rank([cheap_late, dear_on_time]) is dear_on_time


def test_empty_candidate_list_returns_none():
    assert rank([]) is None


def test_ranking_is_deterministic_regardless_of_input_order():
    pool = [cand(total="120"), cand(total="100"), cand(total="110")]
    assert rank(pool) is rank(list(reversed(pool))) or \
        rank(pool).total_paid == rank(list(reversed(pool))).total_paid


def test_render_plan_joins_dates_and_verbatim_amounts():
    c = Candidate("installments",
                  ((date(2025, 8, 8), D("15952906.67")),
                   (date(2025, 9, 7), D("15952906.67"))),
                  ("15952906.67", "15952906.67"),
                  D("31905813.34"), (), "payment_option_05", 5, True,
                  "affordable_with_plan")
    assert c.render_plan() == "2025-08-08:15952906.67|2025-09-07:15952906.67"


def test_render_plan_is_none_when_there_are_no_payments():
    assert cand(payments=()).render_plan() == "none"


def test_render_changes_joins_with_a_pipe_or_says_none():
    assert cand().render_changes() == "none"
    c = cand(changes=(a_change("event_1"), a_change("event_2")))
    assert c.render_changes() == "stop:event_1|stop:event_2"
