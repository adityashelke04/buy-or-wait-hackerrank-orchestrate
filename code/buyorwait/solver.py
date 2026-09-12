"""The two headline numbers: how much is safe today, and when the full amount
becomes safe.

safe_amount has a closed form. Paying X on the request date lowers every
subsequent balance by exactly X, so the trough is linear in X:

    trough(X) = trough(0) - X

The largest safe X therefore satisfies trough(0) - X >= minimum, giving
X* = trough(0) - minimum, clipped to [0, requested]. A binary-search oracle is
kept alongside it and a 200-case property test asserts the two never disagree.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import ROUND_DOWN, Decimal

from .changes import Change, apply as apply_changes, combinations
from .forecast import Curve
from .money import CENTS, fmt_plain
from .ranker import NO_OPTION, Candidate
from .simulate import is_safe, trough
from .types import PaymentOption, Profile, Request


def safe_amount(curve: Curve, minimum: Decimal, requested: Decimal,
                on: date) -> Decimal:
    """Largest amount payable on `on` that keeps the whole window safe.

    Quantized to whole cents, rounding DOWN. The rounding direction matters
    twice over: paying a fraction less is always safe while paying more may not
    be, and an exact two-decimal figure is what lets a two-step partial payment
    sum back to the requested amount instead of drifting by a cent.
    """
    headroom = trough(curve) - minimum
    if headroom <= 0:
        return Decimal("0")
    return min(headroom, requested).quantize(CENTS, rounding=ROUND_DOWN)


def safe_amount_by_search(curve: Curve, minimum: Decimal, requested: Decimal,
                          on: date) -> Decimal:
    """Deliberately slow reference implementation, used to prove the closed form.

    Shipped rather than confined to the tests so the claim stays auditable.
    """
    if not is_safe(curve, minimum, [(on, Decimal("0"))]):
        return Decimal("0")
    if is_safe(curve, minimum, [(on, requested)]):
        return requested
    lo, hi = Decimal("0"), requested
    for _ in range(100):
        mid = (lo + hi) / 2
        if is_safe(curve, minimum, [(on, mid)]):
            lo = mid
        else:
            hi = mid
    return lo


def earliest_full_payment_date(curve: Curve, minimum: Decimal,
                               requested: Decimal) -> date | None:
    """First day in the window on which paying `requested` keeps the curve safe
    for the whole remainder of the window.

    Measures financial capacity only. It ignores which methods the user accepts,
    which is why it can equal the request date even when the recommendation ends
    up being installments.
    """
    day = curve.start
    while day <= curve.end:
        if is_safe(curve, minimum, [(day, requested)]):
            return day
        day += timedelta(days=1)
    return None


# --------------------------------------------------------------------------
# Candidate enumeration
# --------------------------------------------------------------------------

def option_payments(option: PaymentOption) -> list[tuple[date, Decimal]]:
    freq = option.payment_frequency_days or 0
    return [
        (option.first_payment_date + timedelta(days=freq * i), option.payment_amount)
        for i in range(option.number_of_payments)
    ]


def option_months(option: PaymentOption) -> int:
    """How many months the schedule spans, for the max_installment_months check."""
    freq = option.payment_frequency_days or 30
    span_days = freq * max(option.number_of_payments - 1, 0) + freq
    return max(1, round(span_days / 30))


def enumerate_candidates(request: Request, profile: Profile, curve: Curve,
                         options: list[PaymentOption], change_options: list[Change],
                         safe: Decimal, earliest: date | None) -> list[Candidate]:
    """Every eligible, SAFE way to pay. Ineligible or unsafe plans never appear,
    so the ranker only ever orders things the user could actually do."""
    accepted = set(profile.payment_methods_user_will_consider)
    minimum = profile.minimum_balance_to_keep
    deadline = request.desired_completion_date
    out: list[Candidate] = []

    def add(method, payments, texts, total, changes, option_id, sort_key, status):
        working = apply_changes(curve, list(changes), request.request_date, curve.end)
        if not is_safe(working, minimum, list(payments)):
            return
        out.append(Candidate(
            method=method, payments=tuple(payments), payment_texts=tuple(texts),
            total_paid=total, changes=tuple(changes), option_id=option_id,
            option_sort_key=sort_key,
            completes_by_deadline=bool(payments) and payments[-1][0] <= deadline,
            status=status,
        ))

    change_sets: list[tuple[Change, ...]] = [()]
    change_sets.extend(combinations(change_options))

    for changes in change_sets:
        plan_status = "affordable_with_plan" if changes else "affordable_now"

        # 1. Full payment on the request date.
        if "full_payment" in accepted:
            add("full_payment",
                [(request.request_date, request.requested_amount)],
                [fmt_plain(request.requested_amount)],
                request.requested_amount, changes, None, NO_OPTION, plan_status)

        # 2. Each supplied payment option.
        for option in options:
            if option.payment_method == "full_payment":
                if "full_payment" not in accepted:
                    continue
                status = ("affordable_now"
                          if not changes and option.first_payment_date == request.request_date
                          else "affordable_with_plan")
            elif option.payment_method == "installments":
                if "installments" not in accepted:
                    continue
                # A blank max_installment_months means installments are not
                # considered at all - 119 users in the dataset are in this state.
                if profile.max_installment_months is None:
                    continue
                if option_months(option) > profile.max_installment_months:
                    continue
                status = "affordable_with_plan"
            else:
                continue

            payments = option_payments(option)
            add(option.payment_method, payments,
                [option.payment_amount_text] * len(payments),
                option.total_payable_amount, changes,
                option.payment_option_id, option.sort_key, status)

        # 3. Two-step partial payment. Spending changes do not apply: the
        #    labeled samples pair partial payment with no changes.
        if (not changes and "partial_payment" in accepted
                and request.allows_partial_payment
                and earliest is not None and earliest <= deadline
                and Decimal("0") < safe < request.requested_amount):
            remainder = request.requested_amount - safe
            add("partial_payment",
                [(request.request_date, safe), (earliest, remainder)],
                [fmt_plain(safe), fmt_plain(remainder)],
                request.requested_amount, changes, None, NO_OPTION,
                "affordable_with_plan")

        # 4. Wait for the first date the full amount is safe.
        if (not changes and "full_payment" in accepted
                and earliest is not None and earliest > request.request_date):
            add("wait", [(earliest, request.requested_amount)],
                [fmt_plain(request.requested_amount)],
                request.requested_amount, changes, None, NO_OPTION,
                "affordable_later")

    return out
