"""Build the 90-day cash-flow curve for one request.

Sources, in precedence order:
  1. Confirmed rows from the ledger (scheduled payments, reserved pending debits)
  2. Projected recurring series

Precedence matters: when a confirmed row already covers a date for a given
category, the projection for that date is suppressed. Otherwise rent would be
charged twice in any month the dataset happens to hold a confirmed row for.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from .ledger import LedgerView
from .recurrence import Series

HORIZON_DAYS = 90


@dataclass(frozen=True)
class Curve:
    start: date
    end: date
    opening: Decimal
    flows: tuple[tuple[date, Decimal], ...]

    def with_extra(self, extra: list[tuple[date, Decimal]]) -> "Curve":
        merged = sorted([*self.flows, *extra], key=lambda f: f[0])
        return Curve(self.start, self.end, self.opening, tuple(merged))


def build(view: LedgerView, series: list[Series], request_date: date,
          horizon_days: int = HORIZON_DAYS) -> Curve:
    end = request_date + timedelta(days=horizon_days - 1)

    confirmed = [(d, amt) for d, amt in view.future_cashflows(request_date) if d <= end]

    # Dates already covered by a confirmed row, keyed by category, so a
    # projection for the same category on the same day is dropped.
    covered = {
        (e.settlement_date, e.category)
        for e in view.events
        if e.settlement_date >= request_date
    }

    projected: list[tuple[date, Decimal]] = []
    for s in series:
        # A bill due ON the request date that has not settled yet is still
        # money going out. It has not settled exactly when the series was last
        # seen before today, so start the window a day early in that case.
        # (from_last series already treat the start date inclusively.)
        after = (request_date - timedelta(days=1)
                 if not s.from_last and s.last_seen < request_date else request_date)
        for when in s.occurrences(after, end):
            if when < request_date:
                continue
            if (when, s.category) in covered:
                continue
            projected.append((when, s.signed_amount))

    flows = sorted([*confirmed, *projected], key=lambda f: (f[0], f[1]))
    return Curve(
        start=request_date,
        end=end,
        opening=view.profile.current_available_balance,
        flows=tuple(flows),
    )
