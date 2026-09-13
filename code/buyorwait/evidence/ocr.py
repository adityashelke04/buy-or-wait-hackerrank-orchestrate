"""OCR cross-check using RapidOCR - the ONNXRuntime build of PP-OCR (v6 models).

Chosen over Tesseract because it is markedly more accurate on structured
documents such as payslips and invoices, installs with plain pip on Windows with
no system binary and no Paddle toolchain, is Apache-2.0, and runs on CPU in a
fraction of a second per page - so it never competes with the vision model for
GPU memory.

Its job is NOT to decide which number is the right one. That needs semantics
("net salary", not "total earnings") and belongs to the vision model. OCR
supplies the candidate digits so a hallucinated figure can be caught.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path

# 1,234,567.89 / 1.234.567,89 / 4780800
_NUMBER = re.compile(r"\d[\d,. ]*\d|\d")


@lru_cache(maxsize=1)
def _engine():
    from rapidocr import RapidOCR
    return RapidOCR()


@lru_cache(maxsize=64)
def read_text(png: Path) -> str:
    from .usage import USAGE
    result = _engine()(str(png))
    USAGE.record(provider="rapidocr (local, ONNXRuntime)",
                 model="PP-OCRv6 detection + recognition",
                 input_tokens=0, output_tokens=0)
    lines = getattr(result, "txts", None) or []
    return "\n".join(str(line) for line in lines)


def parse_amount_token(token: str) -> Decimal | None:
    """Turn one OCR token into a positive Decimal, or None.

    Handles both separator conventions. When a token holds both '.' and ',',
    whichever appears LAST is the decimal separator. When it holds only commas,
    a trailing group of exactly two digits is a decimal, anything else is
    thousands grouping.
    """
    t = token.strip().replace(" ", "")
    if not t or not any(ch.isdigit() for ch in t):
        return None

    if "," in t and "." in t:
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        tail = t.rsplit(",", 1)[-1]
        t = t.replace(",", ".") if len(tail) == 2 else t.replace(",", "")

    t = t.rstrip(".")
    try:
        value = Decimal(t)
    except InvalidOperation:
        return None
    return value if value > 0 else None


def read_amounts(png: Path) -> list[Decimal]:
    """Every plausible monetary figure on the page, largest first."""
    seen: dict[Decimal, None] = {}
    for token in _NUMBER.findall(read_text(Path(png))):
        value = parse_amount_token(token)
        if value is not None:
            seen[value] = None
    return sorted(seen, reverse=True)


# ---------------------------------------------------------------------------
# Choosing the RIGHT number
#
# A page carries tax reference numbers, account numbers, pincodes, phone
# numbers, percentages and years alongside the one figure that matters. Size is
# no guide: on image_01 the largest number is a tax reference and the smallest
# is a 0.24% contribution rate. The label beside the number is the only reliable
# signal, so selection is driven by it.
# ---------------------------------------------------------------------------

# Ordered by priority within each concept: an earlier keyword wins.
KEYWORDS: dict[str, list[str]] = {
    "net_income": ["net pay", "net salary", "net amount", "take home",
                   "transferred to", "gaji bersih", "net"],
    "outstanding": ["balance due", "outstanding", "amount due", "due amount",
                    "amount payable", "payable", "balance"],
    # "total" outranks "amount due" deliberately. Bills carry date qualifiers
    # such as "Amount due till 06-Feb-2026" and "Amount due after ...", which
    # label a DATE rather than the sum owed; the canonical field is the total.
    "total": ["total amount to be", "grand total", "total amount", "net payable",
              "total due", "total", "amount due"],
}

MAX_MONEY_DIGITS = 9         # the largest real amount in the dataset is 9 digits
                             # (136,691,818.94); longer runs are reference,
                             # account or phone numbers
MIN_MONEY = Decimal("5")     # below this it is a rate or a count, not a bill
LABEL_WINDOW = 3             # how many lines after a label may hold its value


def read_lines(png: Path) -> list[str]:
    return [ln.strip() for ln in read_text(Path(png)).splitlines() if ln.strip()]


def is_plausible_money(value: Decimal, token: str | None = None) -> bool:
    """Reject the things on a page that look numeric but are not amounts."""
    if value <= 0:
        return False
    whole = int(value)
    if len(str(whole)) > MAX_MONEY_DIGITS:
        return False                                  # tax / account reference
    if value < MIN_MONEY:
        return False                                  # a percentage or a count
    if token is not None and re.fullmatch(r"(19|20)\d{2}", token.strip()):
        return False                                  # a bare year
    return True


def _amounts_in(line: str) -> list[Decimal]:
    out: list[Decimal] = []
    for token in _NUMBER.findall(line):
        value = parse_amount_token(token)
        if value is not None and is_plausible_money(value, token):
            out.append(value)
    return out


def select_amount(lines: list[str], keywords: list[str]) -> Decimal | None:
    """The amount belonging to the highest-priority label present.

    Looks on the label's own line first, then the next few lines, because these
    documents commonly put the label, a currency marker and the figure on
    separate lines.
    """
    lowered = [ln.lower() for ln in lines]
    for keyword in keywords:
        for i, line in enumerate(lowered):
            if keyword not in line:
                continue
            here = _amounts_in(lines[i])
            if here:
                return here[0]
            for j in range(i + 1, min(i + 1 + LABEL_WINDOW, len(lines))):
                nearby = _amounts_in(lines[j])
                if nearby:
                    return nearby[0]
    return None


def select_for_event(png: Path, direction: str, description: str,
                     category: str) -> Decimal | None:
    """Pick the figure that matches what the event says it is."""
    lines = read_lines(png)
    text = f"{description} {category}".lower()

    order: list[str]
    if direction == "credit" or "salary" in text or "payroll" in text:
        order = ["net_income", "total"]
    elif any(w in text for w in ("outstanding", "balance", "due", "payable")):
        order = ["outstanding", "total"]
    else:
        order = ["total", "outstanding"]

    for concept in order:
        got = select_amount(lines, KEYWORDS[concept])
        if got is not None:
            return got

    # Nothing labelled: fall back to the largest plausible figure on the page.
    candidates = [v for v in read_amounts(png) if is_plausible_money(v)]
    return max(candidates) if candidates else None
