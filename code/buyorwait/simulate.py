"""Walk a curve and report the lowest balance it reaches.

Every recommendation this system makes is gated through is_safe(). If this
function is correct, no unsafe plan can escape.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from .forecast import Curve

Payments = list[tuple[date, Decimal]]


def _merged(curve: Curve, payments: Payments | tuple = ()) -> list[tuple[date, Decimal]]:
    # Payments are outflows, so they enter the curve negated.
    extra = [(d, -amount) for d, amount in payments]
    return sorted([*curve.flows, *extra], key=lambda f: f[0])


def balance_series(curve: Curve, payments: Payments | tuple = ()) -> list[tuple[date, Decimal]]:
    """One point per distinct flow date, for charting."""
    points: list[tuple[date, Decimal]] = [(curve.start, curve.opening)]
    balance = curve.opening
    for when, amount in _merged(curve, payments):
        balance += amount
        if points[-1][0] == when:
            points[-1] = (when, balance)
        else:
            points.append((when, balance))
    return points


def trough(curve: Curve, payments: Payments | tuple = ()) -> Decimal:
    balance = curve.opening
    low = balance
    for _, amount in _merged(curve, payments):
        balance += amount
        if balance < low:
            low = balance
    return low


def is_safe(curve: Curve, minimum: Decimal, payments: Payments | tuple = ()) -> bool:
    return trough(curve, payments) >= minimum
