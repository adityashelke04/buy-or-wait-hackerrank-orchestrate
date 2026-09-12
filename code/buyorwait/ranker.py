"""Choose between safe plans using the problem statement's six-rung tie-break.

   1. Complete the full request by desired_completion_date
   2. Require no spending changes
   3. Minimize the total amount paid
   4. Start payment earlier
   5. Use fewer payments
   6. Lowest payment_option_id

Only SAFE, ELIGIBLE candidates are ever passed in. This module does not re-check
feasibility; it only orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from .changes import Change

FAR_FUTURE = date(9999, 12, 31)
NO_OPTION = 10 ** 9


@dataclass(frozen=True)
class Candidate:
    method: str                                   # full_payment|partial_payment|installments|wait
    payments: tuple[tuple[date, Decimal], ...]
    payment_texts: tuple[str, ...]                # verbatim amount strings for the plan
    total_paid: Decimal
    changes: tuple[Change, ...]
    option_id: str | None
    option_sort_key: int
    completes_by_deadline: bool
    status: str

    @property
    def start_date(self) -> date:
        return self.payments[0][0] if self.payments else FAR_FUTURE

    def render_plan(self) -> str:
        if not self.payments:
            return "none"
        return "|".join(
            f"{when.isoformat()}:{text}"
            for (when, _), text in zip(self.payments, self.payment_texts)
        )

    def render_changes(self) -> str:
        return "|".join(c.render() for c in self.changes) if self.changes else "none"


def _sort_key(c: Candidate):
    return (
        0 if c.completes_by_deadline else 1,   # rung 1
        len(c.changes),                        # rung 2
        c.total_paid,                          # rung 3
        c.start_date,                          # rung 4
        len(c.payments),                       # rung 5
        c.option_sort_key,                     # rung 6
    )


def rank(candidates: list[Candidate]) -> Candidate | None:
    """Lowest sort key wins. min() is stable, so equal candidates resolve to the
    first one enumerated, keeping the whole pipeline deterministic."""
    return min(candidates, key=_sort_key) if candidates else None
