"""Decide what each event does to real, spendable cash.

The dataset deliberately represents the same money more than once. The six traps
documented in the design spec are resolved here, before anything downstream sees
the events, so that no later module can double-count:

  1. Authorization -> settlement pair   (count once, the settled one)
  2. Charge -> reversal refund          (both survive; they net to zero)
  3. Pending credit                     (never counted)
  4. Unrealized / non_cash valuation    (never counted as cash)
  5. Failed payment -> scheduled retry  (count the retry only)
  6. Blank amount                       (resolve from an image, never zero)
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum

from .fx import RateTable
from .types import Event, Profile


class CashEffect(Enum):
    COUNT = "count"                 # moves cash on its settlement date
    RESERVE = "reserve"             # committed but not settled: hold the money back
    IGNORE = "ignore"               # never touches spendable cash
    NEEDS_AMOUNT = "needs_amount"   # blank amount; must be resolved from an image


# Statuses that never move spendable cash.
DEAD_STATUSES = frozenset({"cancelled", "failed", "unrealized"})


def classify(event: Event) -> CashEffect:
    if event.direction == "non_cash" or event.status in DEAD_STATUSES:
        return CashEffect.IGNORE
    if event.amount is None:
        return CashEffect.NEEDS_AMOUNT
    if event.status == "pending":
        # Pending debits are committed money; pending credits have not landed.
        return CashEffect.RESERVE if event.direction == "debit" else CashEffect.IGNORE
    if event.status in ("settled", "scheduled"):
        return CashEffect.COUNT
    return CashEffect.IGNORE


def resolve_links(events: list[Event]) -> list[Event]:
    """Drop rows superseded by a linked successor.

    A row is superseded when a later row points at it via linked_event_id AND the
    earlier row is a dead status - a cancelled authorization, a failed payment.

    A refund linked to a charge supersedes nothing: both are real cash movements
    that happen to net out, and dropping either would be wrong.
    """
    superseded: set[str] = set()
    for e in events:
        if e.linked_event_id and e.status not in DEAD_STATUSES:
            superseded.add(e.linked_event_id)
    return [
        e for e in events
        if not (e.event_id in superseded and e.status in DEAD_STATUSES)
    ]


class LedgerView:
    """The cash-relevant view of one user's event history.

    Anchor rule: current_available_balance is the balance ON the request date and
    already includes everything that settled on or before it. Only events
    settling AFTER the request date move the curve - except reserved pending
    debits, which have not settled and so are charged immediately.
    """

    def __init__(self, events: list[Event], profile: Profile, rates: RateTable) -> None:
        self.profile = profile
        self.rates = rates
        self.events = resolve_links(
            sorted(events, key=lambda e: (e.settlement_date, e.event_id)))

    def _home(self, event: Event) -> Decimal:
        amount = self.rates.convert(
            event.amount, event.currency, self.profile.home_currency,
            event.settlement_date)
        return amount if event.direction == "credit" else -amount

    def future_cashflows(self, on_or_after: date) -> list[tuple[date, Decimal]]:
        flows: list[tuple[date, Decimal]] = []
        for e in self.events:
            effect = classify(e)
            if effect is CashEffect.RESERVE:
                # Charge it on the request date at the latest: the money is gone.
                flows.append((max(e.settlement_date, on_or_after), self._home(e)))
            elif effect is CashEffect.COUNT and e.settlement_date > on_or_after:
                flows.append((e.settlement_date, self._home(e)))
        flows.sort(key=lambda t: t[0])
        return flows
