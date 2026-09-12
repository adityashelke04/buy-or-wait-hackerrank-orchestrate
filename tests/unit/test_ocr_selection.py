"""Picking the RIGHT number off a document.

A page carries tax reference numbers, pincodes, phone numbers, percentages and
years alongside the figure that matters. Selection has to be driven by the label
next to the number, not by its size.

The three images asserted here belong to SAMPLE requests, and the expected value
is read off the document itself. This measures the extractor; it is not a lookup
table - nothing in code/ contains these values.
"""
from decimal import Decimal
from pathlib import Path

import pytest

from buyorwait.evidence.ocr import (KEYWORDS, is_plausible_money, read_lines,
                                    select_amount)

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "dataset" / "media" / "images"


# ------------------------------------------------------- plausibility ---

def test_reference_numbers_are_not_money():
    assert not is_plausible_money(Decimal("899763619907000"))   # tax ref
    assert not is_plausible_money(Decimal("2582373290"))        # account no


def test_bare_years_are_not_money():
    assert not is_plausible_money(Decimal("2019"), token="2019")
    assert not is_plausible_money(Decimal("2026"), token="2026")


def test_small_percentages_are_not_money():
    assert not is_plausible_money(Decimal("0.24"))
    assert not is_plausible_money(Decimal("3.7"))


def test_real_amounts_are_money():
    assert is_plausible_money(Decimal("4365000"))
    assert is_plausible_money(Decimal("704.05"))
    assert is_plausible_money(Decimal("100000.00"))


# --------------------------------------------------- label-driven pick ---

def test_selects_the_value_on_the_line_after_its_label():
    lines = ["Total Deductions", ": IDR", "415,800", "Net Pay", ": IDR", "4,365,000"]
    assert select_amount(lines, KEYWORDS["net_income"]) == Decimal("4365000")


def test_selects_the_value_on_the_same_line_as_its_label():
    lines = ["Balance Due: 1,00,000.00", "Received By", "Sanjay"]
    assert select_amount(lines, KEYWORDS["outstanding"]) == Decimal("100000.00")


def test_earlier_keywords_outrank_later_ones():
    lines = ["Total", "999.00", "Net Pay", "123.00"]
    assert select_amount(lines, ["net pay", "total"]) == Decimal("123.00")


def test_returns_none_when_no_keyword_matches():
    assert select_amount(["Thank you for your custom"], KEYWORDS["total"]) is None


# ------------------------------------------------- the real documents ---

@pytest.mark.slow
def test_payslip_yields_net_pay_not_total_earnings():
    """image_01 shows Total Earnings 4,780,800 and Net Pay 4,365,000. The event
    says 'August 2019 net salary', so the answer is the net figure."""
    got = select_amount(read_lines(IMAGES / "image_01.png"), KEYWORDS["net_income"])
    assert got == Decimal("4365000")


@pytest.mark.slow
def test_rent_receipt_yields_the_outstanding_balance():
    """image_02 is a rent receipt; the event is 'Outstanding rent balance', and
    the receipt shows Balance Due 1,00,000.00 against a 2,00,000.00 total."""
    got = select_amount(read_lines(IMAGES / "image_02.png"), KEYWORDS["outstanding"])
    assert got == Decimal("100000.00")


@pytest.mark.slow
def test_telecom_bill_yields_this_months_total_not_the_previous_balance():
    """image_05 shows a previous balance of 3,543.54 that was already paid. The
    event is 'Outstanding telecom bill', which is the 704.05 now due."""
    got = select_amount(read_lines(IMAGES / "image_05.png"), KEYWORDS["total"])
    assert got == Decimal("704.05")


@pytest.mark.slow
def test_every_dataset_image_yields_a_plausible_amount_under_some_keyword_set():
    """No image may be left without a defensible figure."""
    unresolved = []
    for png in sorted(IMAGES.glob("*.png")):
        lines = read_lines(png)
        if not any(select_amount(lines, kws) for kws in KEYWORDS.values()):
            unresolved.append(png.name)
    assert unresolved == [], f"no labelled amount found in: {unresolved}"
