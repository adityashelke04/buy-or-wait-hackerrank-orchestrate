"""Dated currency conversion.

Rule: use the rate for the settlement date. If that exact date has no row, use
the most recent EARLIER row. Never a later one - that would be using information
from the future to decide a past cash effect.

An unknown pair raises rather than returning zero, because a silent zero would
quietly delete someone's salary from the forecast.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from datetime import date
from decimal import Decimal

from .types import Rate


class RateTable:
    def __init__(self, rates: list[Rate]) -> None:
        by_pair: dict[tuple[str, str], list[tuple[date, Decimal]]] = defaultdict(list)
        for r in rates:
            by_pair[(r.from_currency, r.to_currency)].append((r.rate_date, r.rate))
        self._by_pair: dict[tuple[str, str], list[tuple[date, Decimal]]] = {}
        self._dates: dict[tuple[str, str], list[date]] = {}
        for pair, series in by_pair.items():
            series.sort(key=lambda t: t[0])
            self._by_pair[pair] = series
            self._dates[pair] = [d for d, _ in series]

    def _lookup(self, frm: str, to: str, on: date) -> Decimal | None:
        series = self._by_pair.get((frm, to))
        if not series:
            return None
        i = bisect_right(self._dates[(frm, to)], on) - 1
        if i < 0:
            i = 0                       # before the first rate: use the earliest
        return series[i][1]

    def convert(self, amount: Decimal, frm: str, to: str, on: date) -> Decimal:
        if frm == to:
            return amount
        direct = self._lookup(frm, to, on)
        if direct is not None:
            return amount * direct
        inverse = self._lookup(to, frm, on)
        if inverse is not None and inverse != 0:
            return amount / inverse
        raise KeyError(f"no exchange rate for {frm}->{to} on {on}")
