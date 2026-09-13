"""Detect recurring income and expenses from history, then project them forward.

Two rules from the problem statement drive this module:

  "Detect recurrence only when history supports it."
      -> at least MIN_OBSERVATIONS sightings, and a regular cadence

  "Forecast essential variable spending conservatively."
      -> a swappable estimator, chosen by calibration against the labeled
         samples rather than by guesswork
"""
from __future__ import annotations

import calendar
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Callable

from .fx import RateTable
from .ledger import CashEffect, classify, resolve_links
from .types import Event, Profile

MIN_OBSERVATIONS = 3
LOOKBACK_DAYS = 180          # how much history can form a series
RECENT_WINDOW = 6            # how many recent occurrences the estimator sees

WEEKLY_GAP = (5, 9)
MONTHLY_GAP = (26, 35)


def _p75(values: list[Decimal]) -> Decimal:
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * Decimal("0.75")
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


ESTIMATORS: dict[str, Callable[[list[Decimal]], Decimal]] = {
    "last": lambda v: v[-1],
    "mean": lambda v: sum(v) / len(v),
    "median": lambda v: Decimal(str(statistics.median(sorted(v)))),
    "p75": _p75,
    "max": lambda v: max(v),
    "max3": lambda v: max(v[-3:]),
}


CADENCE_AGREEMENT = 0.6      # share of gaps that must sit inside the band
OUTLIER_MULTIPLE = Decimal("3")   # above this multiple of the median = one-off


def _without_one_offs(members: list[Event]) -> list[Event]:
    """Drop one-off purchases from a recurring group before estimating.

    'Distinguish recurring expenses from one-time purchases.' A bulk shop of
    41,272 among weekly groceries of about 8,600 is a one-off; left in, it would
    be projected as the weekly grocery bill. Anything above OUTLIER_MULTIPLE of
    the group median is treated as a one-time event.
    """
    if len(members) < 3:
        return members
    amounts = sorted(e.amount for e in members)
    median = amounts[len(amounts) // 2]
    if median <= 0:
        return members
    kept = [e for e in members if e.amount <= median * OUTLIER_MULTIPLE]
    return kept if len(kept) >= 2 else members


def _fits(gaps: list[int], band: tuple[int, int]) -> bool:
    """A cadence is real only when the gaps are CONSISTENT, not merely when
    their median lands in the band.

    Gaps of 17 and 51 days have a median of 34, which would otherwise be read as
    monthly even though neither gap is anywhere near a month.
    """
    if not gaps:
        return False
    inside = sum(1 for g in gaps if band[0] <= g <= band[1])
    return inside / len(gaps) >= CADENCE_AGREEMENT


def clamp_day(year: int, month: int, day: int) -> date:
    """Day 31 in a 30-day month lands on the last day of that month."""
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


@dataclass(frozen=True)
class Series:
    category: str
    description: str
    template_event_id: str
    cadence: str                     # "monthly" | "weekly"
    anchor: int                      # day-of-month, or weekday 0=Mon
    amount: Decimal                  # home currency, positive magnitude
    direction: str                   # debit | credit
    flexibility: str
    minimum_allowed_amount: Decimal | None
    last_seen: date

    @property
    def signed_amount(self) -> Decimal:
        return self.amount if self.direction == "credit" else -self.amount

    @property
    def is_flexible(self) -> bool:
        return self.flexibility != "fixed"

    def occurrences(self, after: date, until: date) -> list[date]:
        """Projected dates strictly after `after` and on or before `until`."""
        out: list[date] = []
        if self.cadence == "weekly":
            d = after + timedelta(days=1)
            while d.weekday() != self.anchor:
                d += timedelta(days=1)
            while d <= until:
                out.append(d)
                d += timedelta(days=7)
            return out

        year, month = after.year, after.month
        for _ in range(14):
            d = clamp_day(year, month, self.anchor)
            if after < d <= until:
                out.append(d)
            month += 1
            if month == 13:
                year, month = year + 1, 1
        return sorted(out)


def _family(event: Event) -> tuple[str, str]:
    """Group key. Description is part of it because one category can hold
    several distinct commitments - a music subscription and a delivery
    membership are both 'subscription' but recur on different days."""
    return (event.category, event.description.strip().lower())


def detect(events: list[Event], profile: Profile, rates: RateTable,
           as_of: date, estimator: str = "p75",
           min_observations: int = MIN_OBSERVATIONS,
           lookback_days: int = LOOKBACK_DAYS,
           allow_any_cadence: bool = False,
           project_income: bool = True) -> list[Series]:
    """Two passes over the same history.

    Pass 1 groups by (category, description). That keeps distinct fixed
    commitments apart - a music subscription and a delivery membership share a
    category but fall on different days of the month.

    Pass 2 re-groups by category alone, but ONLY for categories that produced
    nothing in pass 1. The dataset rotates the description of everyday spending
    ('Supermarket basket', 'Grocery delivery', 'Bulk pantry shop'), so each
    variant looks irregular on its own while the category is plainly weekly.
    Restricting pass 2 to untouched categories is what stops rent being counted
    twice.
    """
    horizon_start = as_of - timedelta(days=lookback_days)
    usable = [
        e for e in resolve_links(events)
        if classify(e) is CashEffect.COUNT
        and horizon_start <= e.settlement_date <= as_of
    ]

    by_description: dict[tuple[str, str], list[Event]] = defaultdict(list)
    for e in usable:
        by_description[_family(e)].append(e)

    series = _series_from(by_description, profile, rates, estimator,
                          min_observations, allow_any_cadence)

    covered = {s.category for s in series}
    by_category: dict[tuple[str, str], list[Event]] = defaultdict(list)
    for e in usable:
        if e.category not in covered:
            by_category[(e.category, "")].append(e)

    series.extend(_series_from(by_category, profile, rates, estimator,
                               min_observations, allow_any_cadence))
    if not any(s.category == "salary" and s.direction == "credit" for s in series):
        confirmed = _confirmed_salary_series(events, profile, rates, as_of)
        if confirmed is not None:
            series.append(confirmed)

    if not project_income:
        # Only confirmed income rows already in the ledger will count; a
        # recurring salary is not extrapolated past them.
        series = [s for s in series if s.direction != "credit"]
    series.sort(key=lambda s: (s.category, s.template_event_id))
    return series


def _confirmed_salary_series(events: list[Event], profile: Profile,
                             rates: RateTable, as_of: date) -> Series | None:
    """A monthly income series from the most recent confirmed salary.

    Generic detection needs three sightings, but a new employee may have one
    prorated payslip and a scheduled 'Next confirmed salary'. That salary is
    confirmed, so it recurs monthly on its own day at its own amount - which is
    what the labeled samples imply, where the earliest safe date keeps landing
    on payday.
    """
    salaries = [
        e for e in resolve_links(events)
        if e.category == "salary" and e.direction == "credit"
        and classify(e) is CashEffect.COUNT
        and e.settlement_date <= as_of + timedelta(days=45)
    ]
    if not salaries:
        return None
    latest = max(salaries, key=lambda e: (e.settlement_date, e.event_id))
    return Series(
        category="salary", description=latest.description,
        template_event_id=latest.event_id, cadence="monthly",
        anchor=latest.settlement_date.day,
        amount=rates.convert(latest.amount, latest.currency, profile.home_currency,
                             latest.settlement_date),
        direction="credit", flexibility=latest.flexibility,
        minimum_allowed_amount=None, last_seen=latest.settlement_date,
    )


def _series_from(groups: dict[tuple[str, str], list[Event]], profile: Profile,
                 rates: RateTable, estimator: str, min_observations: int,
                 allow_any_cadence: bool) -> list[Series]:
    estimate = ESTIMATORS[estimator]
    series: list[Series] = []
    for (category, _label), members in sorted(groups.items()):
        members = _without_one_offs(members)
        members.sort(key=lambda e: (e.settlement_date, e.event_id))
        if len(members) < min_observations:
            continue

        gaps = [(b.settlement_date - a.settlement_date).days
                for a, b in zip(members, members[1:])]

        if _fits(gaps, WEEKLY_GAP):
            cadence = "weekly"
            anchor = Counter(e.settlement_date.weekday()
                             for e in members).most_common(1)[0][0]
        elif _fits(gaps, MONTHLY_GAP) or allow_any_cadence:
            cadence = "monthly"
            anchor = Counter(e.settlement_date.day for e in members).most_common(1)[0][0]
        else:
            continue                    # irregular: not a dependable commitment

        recent = members[-RECENT_WINDOW:]
        home_amounts = [
            rates.convert(e.amount, e.currency, profile.home_currency,
                          e.settlement_date)
            for e in recent
        ]
        latest = members[-1]
        series.append(Series(
            category=category,
            description=latest.description,
            template_event_id=latest.event_id,
            cadence=cadence,
            anchor=anchor,
            amount=estimate(home_amounts),
            direction=latest.direction,
            flexibility=latest.flexibility,
            minimum_allowed_amount=latest.minimum_allowed_amount,
            last_seen=latest.settlement_date,
        ))
    return series
