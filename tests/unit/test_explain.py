from datetime import date
from decimal import Decimal

from buyorwait.changes import Change
from buyorwait.explain import long_date, render
from buyorwait.ranker import Candidate
from buyorwait.recurrence import Series
from buyorwait.types import Profile, Request

D = Decimal


def req(amount="25256", rd=date(2024, 3, 3), dcd=date(2024, 3, 20)):
    return Request("request_x", "user_x", rd, "purchase", D(amount), dcd, True, "t")


def prof(ccy="ZAR", minimum="18000"):
    return Profile("user_x", ccy, D("58481.1"), D(minimum), (), (), (), (),
                   ("full_payment",), None)


def cand(method, payments, changes=(), status="affordable_now"):
    return Candidate(method, tuple(payments),
                     tuple(str(amt) for _, amt in payments),
                     sum((amt for _, amt in payments), D("0")),
                     tuple(changes), None, 10 ** 9, True, status)


def stop_change(desc="Family streaming plan", eid="event_476"):
    s = Series("streaming", desc, eid, "monthly", 10, D("19"), "debit",
               "stoppable", None, date(2025, 12, 10))
    return Change("stop", eid, s, None, D("19"))


def reduce_change(desc="Weekend food delivery", eid="event_989", new="665950"):
    s = Series("dining", desc, eid, "monthly", 10, D("1163530.49"), "debit",
               "reducible", D(new), date(2025, 4, 10))
    return Change("reduce_to", eid, s, D(new), D("100"))


def test_long_date_formats_as_day_month_year():
    assert long_date(date(2024, 3, 3)) == "3 March 2024"
    assert long_date(date(2019, 11, 15)) == "15 November 2019"


def test_t1_full_payment_today():
    c = cand("full_payment", [(date(2024, 3, 3), D("25256"))])
    assert render(c, req(), prof(), D("25256")) == (
        "Pay ZAR 25,256 today. This leaves at least ZAR 18,000 available "
        "over the next 90 days."
    )


def test_t2_full_payment_with_one_stop():
    c = cand("full_payment", [(date(2026, 1, 3), D("620.40"))],
             changes=[stop_change()], status="affordable_with_plan")
    got = render(c, req(amount="620.40", rd=date(2026, 1, 3)),
                 prof("EUR", "800"), D("603.3"))
    assert got == (
        "Stop the family streaming plan, then pay EUR 620.40 today. "
        "This leaves at least EUR 800 available."
    )


def test_t3_full_payment_with_one_reduce():
    c = cand("full_payment", [(date(2025, 5, 3), D("13110000"))],
             changes=[reduce_change()], status="affordable_with_plan")
    got = render(c, req(amount="13110000", rd=date(2025, 5, 3)),
                 prof("IDR", "9500000"), D("12510645"))
    assert got == (
        "Reduce the weekend food delivery to IDR 665,950, then pay "
        "IDR 13,110,000 today. This leaves at least IDR 9,500,000 available."
    )


def test_t4_full_payment_with_a_stop_and_a_reduce():
    c = cand("full_payment", [(date(2026, 4, 3), D("1574.40"))],
             changes=[stop_change("Online backup subscription", "event_1815"),
                      reduce_change("Streaming subscription", "event_1816", "23.5")],
             status="affordable_with_plan")
    got = render(c, req(amount="1574.40", rd=date(2026, 4, 3)), prof("USD", "600"),
                 D("1543.35"))
    assert got == (
        "Stop the online backup subscription and reduce the streaming subscription "
        "to USD 23.50, then pay USD 1,574.40 today. "
        "This leaves at least USD 600 available."
    )


def test_t5_installments():
    c = cand("installments", [(date(2025, 8, 8), D("15952906.67")),
                              (date(2025, 9, 7), D("15952906.67")),
                              (date(2025, 10, 7), D("15952906.67"))],
             status="affordable_with_plan")
    got = render(c, req(amount="46018000", rd=date(2025, 8, 5)),
                 prof("IDR", "29158400"), D("17229139.2"))
    assert got == (
        "Use 3 installments of IDR 15,952,906.67, starting 8 August 2025. "
        "This leaves at least IDR 29,158,400 available."
    )


def test_t6_wait():
    c = cand("wait", [(date(2019, 11, 15), D("5491000"))], status="affordable_later")
    got = render(c, req(amount="5491000", rd=date(2019, 9, 3)),
                 prof("IDR", "2668700"), D("873000"))
    assert got == (
        "Pay IDR 5,491,000 in full on 15 November 2019. Paying earlier would take "
        "the balance below the IDR 2,668,700 minimum."
    )


def test_t7_not_recommended():
    got = render(None, req(amount="15488", rd=date(2025, 11, 6), dcd=date(2026, 1, 12)),
                 prof("ZAR", "13100"), D("737"))
    assert got == (
        "Do not make this payment by 12 January 2026. None of the available options "
        "keeps the ZAR 13,100 minimum protected."
    )


def test_t8_partial_payment():
    c = cand("partial_payment", [(date(2024, 9, 4), D("28820")),
                                 (date(2024, 9, 15), D("10840"))],
             status="affordable_with_plan")
    got = render(c, req(amount="39660", rd=date(2024, 9, 4), dcd=date(2024, 10, 4)),
                 prof("INR", "92800"), D("28820"))
    assert got == (
        "Pay INR 28,820 today and the remaining INR 10,840 on 15 September 2024. "
        "This completes the full request and keeps the INR 92,800 minimum protected."
    )


def test_every_explanation_names_the_home_currency():
    """A contract invariant checked here at the source."""
    cases = [
        cand("full_payment", [(date(2024, 3, 3), D("100"))]),
        cand("installments", [(date(2024, 3, 3), D("50")), (date(2024, 4, 3), D("50"))]),
        cand("wait", [(date(2024, 4, 3), D("100"))]),
        cand("partial_payment", [(date(2024, 3, 3), D("40")), (date(2024, 4, 3), D("60"))]),
        None,
    ]
    for c in cases:
        out = render(c, req(amount="100"), prof("ZAR", "500"), D("40"))
        assert "ZAR" in out and out.strip()
