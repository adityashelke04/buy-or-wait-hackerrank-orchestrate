"""Decimal money helpers.

Every amount in this project is a Decimal parsed directly from its string form.
Floats are never used: 0.1 + 0.2 != 0.3 in binary floating point, and a cent of
drift is enough to flip an affordability decision.

Three formatters, deliberately different, each matching the labeled samples:

  fmt_plain    -> amount_safe_to_pay. Minimal representation, trailing zeros
                  stripped: 25256, 620.4, 603.3
  fmt_amount   -> amounts inside payment_plan and reduce_to changes. Whole
                  numbers stay whole, anything else carries exactly two
                  decimals: 25256, 620.40, 23.50
  fmt_currency -> for explanations. Thousands separators and the ISO code, with
                  two decimals whenever there is a fractional part: ZAR 25,256,
                  EUR 620.40
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

CENTS = Decimal("0.01")


def dec(text: str | None) -> Decimal | None:
    """Parse a CSV cell into a Decimal. Blank or unparseable returns None.

    Returning None rather than zero is deliberate: the problem statement says a
    blank amount must never be treated as zero.
    """
    if text is None:
        return None
    s = str(text).strip().replace(",", "")
    if not s:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def q(value: Decimal) -> Decimal:
    """Round half-up to 2 decimal places for internal arithmetic."""
    return value.quantize(CENTS, rounding=ROUND_HALF_UP)


def fmt_plain(value: Decimal) -> str:
    """Minimal representation: 25256.00 -> '25256', 620.40 -> '620.4'."""
    v = q(Decimal(value))
    if v == v.to_integral_value():
        return format(v.to_integral_value(), "f")
    return format(v.normalize(), "f")


def fmt_amount(value: Decimal) -> str:
    """Plan and spending-change amounts: 25256.00 -> '25256', 620.4 -> '620.40'."""
    v = q(Decimal(value))
    if v == v.to_integral_value():
        return format(v.to_integral_value(), "f")
    return format(v, "f")


def fmt_currency(code: str, value: Decimal) -> str:
    """Human formatting for explanations: 'ZAR 25,256', 'EUR 620.40'."""
    v = q(Decimal(value))
    negative = v < 0
    v = abs(v)
    whole = int(v)
    frac = v - whole
    body = f"{whole:,}"
    if frac:
        body += f"{frac:.2f}"[1:]
    return f"{code} {'-' if negative else ''}{body}"
