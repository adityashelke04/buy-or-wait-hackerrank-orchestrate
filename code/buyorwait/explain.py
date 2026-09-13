"""Render the decision_explanation.

All 25 reference explanations reduce to eight sentence patterns, and the figure
they quote is always the user's own minimum_balance_to_keep. This is therefore
deterministic string formatting - perfectly consistent, and it costs no tokens.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from .forecast import HORIZON_DAYS
from .money import fmt_currency
from .ranker import Candidate
from .types import Profile, Request

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


def long_date(d: date) -> str:
    """'3 March 2024' - the form used throughout the reference explanations."""
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def _change_clause(candidate: Candidate, currency: str) -> str:
    """'Stop the x and reduce the y to CUR n, then ' - or '' when there are none.

    The first verb is capitalised because the clause opens the sentence.
    """
    parts: list[str] = []
    for i, c in enumerate(candidate.changes):
        if c.kind == "stop":
            verb = "Stop" if i == 0 else "stop"
            parts.append(f"{verb} {c.phrase}")
        else:
            verb = "Reduce" if i == 0 else "reduce"
            parts.append(f"{verb} {c.phrase} to {fmt_currency(currency, c.new_amount)}")
    if not parts:
        return ""
    return " and ".join(parts) + ", then "


def _partial_was_the_only_route(request: Request, profile: Profile,
                                safe: Decimal) -> bool:
    """True when paying part today was the user's sole eligible way to proceed.

    Derived from all seven not-recommended samples: the 'Although ... is
    available today' wording appears exactly when the request allows partial
    payment, the user accepts it, neither full payment nor any installment plan
    is open to them, and something can be paid today.
    """
    methods = set(profile.payment_methods_user_will_consider)
    installments_open = ("installments" in methods
                         and profile.max_installment_months is not None)
    return (request.allows_partial_payment
            and "partial_payment" in methods
            and "full_payment" not in methods
            and not installments_open
            and safe > 0)


def render(candidate: Candidate | None, request: Request, profile: Profile,
           safe: Decimal) -> str:
    ccy = profile.home_currency
    minimum = fmt_currency(ccy, profile.minimum_balance_to_keep)
    total = fmt_currency(ccy, request.requested_amount)

    # T7 - no safe eligible option exists at all.
    if candidate is None:
        if _partial_was_the_only_route(request, profile, safe):
            # T7b: something is available today, but the only route the user
            # accepts - paying part now - cannot complete within the window.
            return (f"Do not proceed with the {total} request. Although "
                    f"{fmt_currency(ccy, safe)} is available today, the full amount "
                    f"cannot be completed safely within {HORIZON_DAYS} days.")
        return (f"Do not make this payment by "
                f"{long_date(request.desired_completion_date)}. "
                f"None of the available options keeps the {minimum} minimum "
                f"protected.")

    # T5 - installments.
    if candidate.method == "installments":
        first_date, first_amount = candidate.payments[0]
        return (f"Use {len(candidate.payments)} installments of "
                f"{fmt_currency(ccy, first_amount)}, starting "
                f"{long_date(first_date)}. "
                f"This leaves at least {minimum} available.")

    # T8 - two-step partial payment.
    if candidate.method == "partial_payment":
        (_, first), (second_date, second) = candidate.payments
        return (f"Pay {fmt_currency(ccy, first)} today and the remaining "
                f"{fmt_currency(ccy, second)} on {long_date(second_date)}. "
                f"This completes the full request and keeps the {minimum} "
                f"minimum protected.")

    # T6 - wait for a later date.
    if candidate.method == "wait":
        when, amount = candidate.payments[0]
        if when < request.desired_completion_date:
            # T6b: the money is safe before the deadline, so the advice is to
            # wait for that date rather than to pay at the last moment.
            return (f"Wait until {long_date(when)}, then pay "
                    f"{fmt_currency(ccy, amount)} in full. Paying sooner would put "
                    f"the {minimum} minimum at risk.")
        return (f"Pay {fmt_currency(ccy, amount)} in full on {long_date(when)}. "
                f"Paying earlier would take the balance below the {minimum} "
                f"minimum.")

    # T2/T3/T4 - full payment enabled by spending changes.
    clause = _change_clause(candidate, ccy)
    if clause:
        return (f"{clause}pay {total} today. "
                f"This leaves at least {minimum} available.")

    # T1 - plain full payment today.
    return (f"Pay {total} today. This leaves at least {minimum} available "
            f"over the next {HORIZON_DAYS} days.")
