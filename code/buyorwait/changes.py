"""Optional spending changes the user has pre-authorised.

Three constraints, all from the problem statement:
  - only recurring, flexible expenses
  - only categories the user listed as reducible or stoppable, never a protected one
  - at most three changes, and one event may not be both stopped and reduced

One convention read off the labeled samples: reduce_to always targets the
event's minimum_allowed_amount exactly. The agent never invents an intermediate
figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from itertools import combinations as _combos
from typing import Iterator

from .forecast import Curve
from .money import fmt_plain
from .recurrence import Series
from .types import Profile

MAX_CHANGES = 3


@dataclass(frozen=True)
class Change:
    kind: str                      # "stop" | "reduce_to"
    event_id: str
    series: Series
    new_amount: Decimal | None
    saving_per_occurrence: Decimal

    def render(self) -> str:
        if self.kind == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{fmt_plain(self.new_amount)}"

    @property
    def phrase(self) -> str:
        """Human phrase for the explanation: 'the weekend food delivery'."""
        return f"the {self.series.description.strip().lower()}"


def candidates(series: list[Series], profile: Profile) -> list[Change]:
    protected = set(profile.expense_categories_to_protect)
    may_stop = set(profile.expense_categories_user_is_willing_to_stop) - protected
    may_reduce = set(profile.expense_categories_user_is_willing_to_reduce) - protected

    out: list[Change] = []
    for s in sorted(series, key=lambda x: x.template_event_id):
        if s.direction != "debit" or not s.is_flexible:
            continue
        can_stop = s.flexibility in ("stoppable", "reducible_or_stoppable")
        can_reduce = s.flexibility in ("reducible", "reducible_or_stoppable")

        if can_stop and s.category in may_stop:
            out.append(Change("stop", s.template_event_id, s, None, s.amount))
        if (can_reduce and s.category in may_reduce
                and s.minimum_allowed_amount is not None
                and s.minimum_allowed_amount < s.amount):
            out.append(Change("reduce_to", s.template_event_id, s,
                              s.minimum_allowed_amount,
                              s.amount - s.minimum_allowed_amount))
    return out


def combinations(all_changes: list[Change],
                 max_count: int = MAX_CHANGES) -> Iterator[tuple[Change, ...]]:
    """Every subset of size 1..max_count, never repeating an event in a subset.

    Yielded smallest-first so the ranker naturally meets rung 2 (prefer fewer
    spending changes) without needing to sort again.
    """
    for size in range(1, max_count + 1):
        for combo in _combos(all_changes, size):
            ids = [c.event_id for c in combo]
            if len(ids) != len(set(ids)):
                continue
            yield combo


def apply(curve: Curve, chosen: list[Change], request_date: date, end: date) -> Curve:
    """Rewrite the curve with the chosen changes in force from the request date.

    Flows are matched back to their series by (date, signed amount), which is
    exactly how forecast.build emitted them.
    """
    if not chosen:
        return curve

    targets: dict[str, set[tuple[date, Decimal]]] = {}
    for c in chosen:
        occ = c.series.occurrences(request_date - timedelta(days=1), end)
        targets[c.event_id] = {(d, c.series.signed_amount) for d in occ}

    stopped = [c for c in chosen if c.kind == "stop"]
    reduced = [c for c in chosen if c.kind == "reduce_to"]

    flows: list[tuple[date, Decimal]] = []
    for when, amount in curve.flows:
        if any((when, amount) in targets[c.event_id] for c in stopped):
            continue
        for c in reduced:
            if (when, amount) in targets[c.event_id]:
                amount = -c.new_amount
                break
        flows.append((when, amount))

    return Curve(curve.start, curve.end, curve.opening,
                 tuple(sorted(flows, key=lambda f: f[0])))
