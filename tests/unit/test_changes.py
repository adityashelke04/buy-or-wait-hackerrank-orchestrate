from datetime import date
from decimal import Decimal

from buyorwait.changes import Change, apply, candidates, combinations
from buyorwait.forecast import Curve
from buyorwait.recurrence import Series
from buyorwait.types import Profile


def series(category, description, flexibility, amount="47", minimum=None, anchor=10,
           event_id=None):
    return Series(category=category, description=description,
                  template_event_id=event_id or f"event_{category}",
                  cadence="monthly", anchor=anchor, amount=Decimal(amount),
                  direction="debit", flexibility=flexibility,
                  minimum_allowed_amount=Decimal(minimum) if minimum else None,
                  last_seen=date(2024, 2, 10))


def prof(stop=(), reduce=(), protect=()):
    return Profile(
        user_id="user_x", home_currency="EUR",
        current_available_balance=Decimal("1000"),
        minimum_balance_to_keep=Decimal("200"),
        financial_priorities=(), expense_categories_to_protect=protect,
        expense_categories_user_is_willing_to_reduce=reduce,
        expense_categories_user_is_willing_to_stop=stop,
        payment_methods_user_will_consider=("full_payment",),
        max_installment_months=None,
    )


def test_only_categories_the_user_permits_are_offered():
    all_series = [
        series("streaming", "Streaming subscription", "stoppable"),
        series("rent", "Apartment rent", "stoppable"),
    ]
    got = candidates(all_series, prof(stop=("streaming",)))
    assert [c.event_id for c in got] == ["event_streaming"]


def test_protected_category_is_never_offered_even_if_listed_as_stoppable():
    all_series = [series("groceries", "Weekly groceries", "stoppable")]
    p = prof(stop=("groceries",), protect=("groceries",))
    assert candidates(all_series, p) == []


def test_fixed_expenses_are_never_offered():
    all_series = [series("streaming", "Streaming subscription", "fixed")]
    assert candidates(all_series, prof(stop=("streaming",))) == []


def test_income_series_is_never_offered_as_a_spending_change():
    s = Series("salary", "Payroll", "event_s", "monthly", 15, Decimal("900"),
               "credit", "stoppable", None, date(2024, 2, 15))
    assert candidates([s], prof(stop=("salary",))) == []


def test_reduce_targets_the_minimum_allowed_amount_exactly():
    """Confirmed against all three labeled samples that use a reduce_to."""
    all_series = [series("dining", "Weekend food delivery", "reducible",
                         amount="1163530.49", minimum="665950")]
    got = candidates(all_series, prof(reduce=("dining",)))
    assert len(got) == 1
    assert got[0].kind == "reduce_to"
    assert got[0].new_amount == Decimal("665950")
    assert got[0].render() == "reduce_to:event_dining:665950"


def test_reduce_is_not_offered_when_the_minimum_is_missing():
    all_series = [series("dining", "Takeaway", "reducible", amount="100", minimum=None)]
    assert candidates(all_series, prof(reduce=("dining",))) == []


def test_reduce_is_not_offered_when_it_would_not_save_anything():
    all_series = [series("dining", "Takeaway", "reducible", amount="50", minimum="50")]
    assert candidates(all_series, prof(reduce=("dining",))) == []


def test_reducible_or_stoppable_offers_both_forms():
    all_series = [series("streaming", "Streaming subscription",
                         "reducible_or_stoppable", amount="47", minimum="23.5")]
    got = candidates(all_series, prof(stop=("streaming",), reduce=("streaming",)))
    assert {c.kind for c in got} == {"stop", "reduce_to"}


def test_stop_renders_in_the_required_format():
    all_series = [series("cloud_storage", "Online backup subscription", "stoppable",
                         amount="11")]
    got = candidates(all_series, prof(stop=("cloud_storage",)))
    assert got[0].render() == "stop:event_cloud_storage"


def test_phrase_lowercases_the_description_for_the_explanation():
    all_series = [series("dining", "Weekend food delivery", "stoppable")]
    got = candidates(all_series, prof(stop=("dining",)))
    assert got[0].phrase == "the weekend food delivery"


def test_combinations_never_mix_stop_and_reduce_on_one_event():
    s = series("streaming", "Streaming subscription", "reducible_or_stoppable",
               amount="47", minimum="23.5")
    got = candidates([s], prof(stop=("streaming",), reduce=("streaming",)))
    produced = list(combinations(got, max_count=3))
    assert produced, "expected at least the two single-change combinations"
    for combo in produced:
        ids = [c.event_id for c in combo]
        assert len(ids) == len(set(ids)), "the same event appears twice in one combination"


def test_combinations_are_capped_at_three():
    all_series = [series(f"cat{i}", f"Sub {i}", "stoppable", anchor=i + 1)
                  for i in range(5)]
    p = prof(stop=tuple(f"cat{i}" for i in range(5)))
    got = candidates(all_series, p)
    produced = list(combinations(got, max_count=3))
    assert produced
    assert all(len(c) <= 3 for c in produced)


def test_combinations_are_ordered_smallest_first():
    all_series = [series(f"cat{i}", f"Sub {i}", "stoppable", anchor=i + 1)
                  for i in range(3)]
    p = prof(stop=tuple(f"cat{i}" for i in range(3)))
    sizes = [len(c) for c in combinations(candidates(all_series, p), max_count=3)]
    assert sizes == sorted(sizes)


def test_apply_removes_a_stopped_series_from_the_curve():
    c = Curve(start=date(2024, 3, 1), end=date(2024, 5, 29), opening=Decimal("1000"),
              flows=((date(2024, 4, 10), Decimal("-47")),))
    s = series("streaming", "Streaming subscription", "stoppable", amount="47")
    change = Change(kind="stop", event_id="event_streaming", series=s, new_amount=None,
                    saving_per_occurrence=Decimal("47"))
    out = apply(c, [change], date(2024, 3, 1), date(2024, 5, 29))
    assert out.flows == ()


def test_apply_rewrites_a_reduced_series_to_the_new_amount():
    c = Curve(start=date(2024, 3, 1), end=date(2024, 5, 29), opening=Decimal("1000"),
              flows=((date(2024, 4, 10), Decimal("-47")),))
    s = series("streaming", "Streaming subscription", "reducible", amount="47",
               minimum="23.5")
    change = Change(kind="reduce_to", event_id="event_streaming", series=s,
                    new_amount=Decimal("23.5"),
                    saving_per_occurrence=Decimal("23.5"))
    out = apply(c, [change], date(2024, 3, 1), date(2024, 5, 29))
    assert out.flows == ((date(2024, 4, 10), Decimal("-23.5")),)


def test_apply_leaves_unrelated_flows_untouched():
    c = Curve(start=date(2024, 3, 1), end=date(2024, 5, 29), opening=Decimal("1000"),
              flows=((date(2024, 4, 1), Decimal("-500")),
                     (date(2024, 4, 10), Decimal("-47"))))
    s = series("streaming", "Streaming subscription", "stoppable", amount="47")
    change = Change("stop", "event_streaming", s, None, Decimal("47"))
    out = apply(c, [change], date(2024, 3, 1), date(2024, 5, 29))
    assert out.flows == ((date(2024, 4, 1), Decimal("-500")),)


def test_apply_with_no_changes_returns_the_curve_unchanged():
    c = Curve(start=date(2024, 3, 1), end=date(2024, 5, 29), opening=Decimal("1000"),
              flows=((date(2024, 4, 10), Decimal("-47")),))
    assert apply(c, [], date(2024, 3, 1), date(2024, 5, 29)) is c


def test_apply_raises_the_trough_because_money_is_saved():
    from buyorwait.simulate import trough
    c = Curve(start=date(2024, 3, 1), end=date(2024, 5, 29), opening=Decimal("100"),
              flows=((date(2024, 4, 10), Decimal("-47")),))
    s = series("streaming", "Streaming subscription", "stoppable", amount="47")
    change = Change("stop", "event_streaming", s, None, Decimal("47"))
    out = apply(c, [change], date(2024, 3, 1), date(2024, 5, 29))
    assert trough(out) > trough(c)


def test_stopping_one_series_leaves_an_identical_other_expense_in_place():
    """Two different 47.00 bills on the 10th. Stopping one must remove ONE flow;
    removing both would overstate the saving and could pass an unsafe plan."""
    music = series("streaming", "Music", "stoppable", event_id="event_music")
    gym = series("fitness", "Gym", "fixed", event_id="event_gym")
    day = date(2024, 3, 10)
    curve = Curve(date(2024, 3, 1), date(2024, 3, 31), Decimal("1000"),
                  ((day, Decimal("-47")), (day, Decimal("-47"))))
    stop = Change("stop", "event_music", music, None, Decimal("47"))
    out = apply(curve, [stop], date(2024, 3, 1), date(2024, 3, 31))
    assert out.flows == ((day, Decimal("-47")),)
    assert gym.template_event_id == "event_gym"


def test_reducing_one_series_rewrites_only_one_of_two_identical_flows():
    stream = series("streaming", "Video", "reducible", minimum="20", event_id="event_video")
    day = date(2024, 3, 10)
    curve = Curve(date(2024, 3, 1), date(2024, 3, 31), Decimal("1000"),
                  ((day, Decimal("-47")), (day, Decimal("-47"))))
    cut = Change("reduce_to", "event_video", stream, Decimal("20"), Decimal("27"))
    out = apply(curve, [cut], date(2024, 3, 1), date(2024, 3, 31))
    assert sorted(a for _, a in out.flows) == [Decimal("-47"), Decimal("-20")]
