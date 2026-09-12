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
from decimal import Decimal

from .forecast import Curve
from .simulate import is_safe, trough


def safe_amount(curve: Curve, minimum: Decimal, requested: Decimal,
                on: date) -> Decimal:
    """Largest amount payable on `on` that keeps the whole window safe."""
    headroom = trough(curve) - minimum
    if headroom <= 0:
        return Decimal("0")
    return min(headroom, requested)


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
