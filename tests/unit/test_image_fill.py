"""Resolving the 16 blank amounts from their linked images.

Two independent readers: the vision model chooses which figure the event means,
and label-driven OCR verifies the digits are actually on the page. They must
agree, or the OCR figure wins - it is grounded in text we can point at.
"""
from decimal import Decimal
from pathlib import Path

import pytest

from buyorwait.evidence.extractor import Extractor
from buyorwait.evidence.provider import StubProvider
from buyorwait.io_loaders import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"


def _event(event_id):
    return load_dataset(DATASET).events_by_id[event_id]


def _ex(payload, tmp_path):
    return Extractor(StubProvider(payload), DATASET, cache_path=tmp_path / "c.json")


@pytest.mark.slow
def test_payslip_resolves_to_net_pay(tmp_path):
    ex = _ex(None, tmp_path)          # no model: OCR alone
    assert ex.fill_amount(_event("event_253")) == Decimal("4365000")


@pytest.mark.slow
def test_rent_receipt_resolves_to_the_outstanding_balance_not_the_total(tmp_path):
    """The event says 'Outstanding rent balance'. The receipt shows a
    2,00,000.00 total and a 1,00,000.00 Balance Due; the balance is the answer,
    so the event's own wording has to drive selection."""
    ex = _ex(None, tmp_path)
    assert ex.fill_amount(_event("event_1442")) == Decimal("100000.00")


@pytest.mark.slow
def test_model_figure_is_accepted_when_ocr_confirms_it(tmp_path):
    ex = _ex({"amendments": [{"kind": "amount_fill", "amount": "4365000",
                              "confidence": "high"}]}, tmp_path)
    assert ex.fill_amount(_event("event_253")) == Decimal("4365000")


@pytest.mark.slow
def test_model_figure_is_rejected_when_ocr_cannot_see_it(tmp_path):
    """A hallucinated amount must not reach the forecast."""
    ex = _ex({"amendments": [{"kind": "amount_fill", "amount": "999999999",
                              "confidence": "high"}]}, tmp_path)
    assert ex.fill_amount(_event("event_253")) == Decimal("4365000")


@pytest.mark.slow
def test_every_blank_amount_resolves_to_a_plausible_positive_figure(tmp_path):
    ds = load_dataset(DATASET)
    ex = _ex(None, tmp_path)
    blanks = [e for evs in ds.events_by_user.values() for e in evs if e.amount is None]
    assert len(blanks) == 16
    for e in blanks:
        got = ex.fill_amount(e)
        assert got is not None and got > 0, f"{e.event_id} unresolved"
