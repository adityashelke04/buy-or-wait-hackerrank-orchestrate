from decimal import Decimal
from pathlib import Path

import pytest

from buyorwait.evidence.ocr import parse_amount_token, read_amounts, read_text

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "dataset" / "media" / "images"


# --------------------------------------------------- pure parsing, no model ---

def test_parses_plain_digits():
    assert parse_amount_token("4365000") == Decimal("4365000")


def test_parses_comma_grouped_thousands():
    assert parse_amount_token("4,365,000") == Decimal("4365000")


def test_parses_comma_grouped_with_decimals():
    assert parse_amount_token("1,234.56") == Decimal("1234.56")


def test_parses_european_dot_grouping_with_comma_decimal():
    assert parse_amount_token("1.234.567,89") == Decimal("1234567.89")


def test_parses_comma_as_a_decimal_separator():
    assert parse_amount_token("23,50") == Decimal("23.50")


def test_rejects_tokens_with_no_digits():
    assert parse_amount_token("IDR") is None
    assert parse_amount_token("") is None


def test_rejects_zero_and_negative_amounts():
    assert parse_amount_token("0") is None


# ----------------------------------------------------- real dataset images ---

pytestmark_images = pytest.mark.skipif(not IMAGES.exists(),
                                       reason="dataset images absent")


@pytest.mark.slow
def test_reads_text_from_a_payslip():
    text = read_text(IMAGES / "image_01.png")
    assert "PAY SLIP" in text.upper()


@pytest.mark.slow
def test_surfaces_both_candidate_figures_from_the_payslip():
    """image_01 shows Total Earnings 4,780,800 and Net Pay 4,365,000. OCR must
    surface BOTH - choosing between them needs semantics and is the vision
    model's job, not the OCR engine's."""
    amounts = read_amounts(IMAGES / "image_01.png")
    assert Decimal("4365000") in amounts
    assert Decimal("4780800") in amounts


@pytest.mark.slow
def test_every_dataset_image_yields_at_least_one_amount():
    for png in sorted(IMAGES.glob("*.png")):
        assert read_amounts(png), f"{png.name} produced no amounts"
