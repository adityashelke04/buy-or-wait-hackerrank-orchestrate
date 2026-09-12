"""Model-free fallback: a multilingual pattern parser over message text.

This PARSES message content. It holds no mapping from a request or event id to
an answer - it reads the same words a model would, using patterns rather than
weights, and is measured by the same tests.

It exists so the pipeline runs with no model, no key and no network, and so the
model backends have a floor to be compared against.
"""
from __future__ import annotations

import re
from pathlib import Path

FENCE_OPEN = "<<<UNTRUSTED_DATA>>>"
FENCE_CLOSE = "<<<END_UNTRUSTED_DATA>>>"

# Unsettled money. Checked FIRST: if income is not yet confirmed, nothing else
# in the message may be acted on. English and Indonesian.
_UNSETTLED = re.compile(
    r"(still pending|is pending|not (?:yet )?(?:been )?approved|"
    r"have not been approved|has not reached|pending final|awaiting|"
    r"can change until|belum disetujui|masih menunggu|belum sampai|"
    r"belum dikonfirmasi|masih berjalan)", re.I)

_AMOUNT = r"([\d][\d.,]*\d|\d)"

_SALARY_NEW = re.compile(
    r"(?:salary|pay|gaji)[^.]{0,80}?"
    r"(?:rises? to|increases? to|is now|now|reduced to|is reduced to|will be|"
    r"naik menjadi|turun menjadi|menjadi|adalah)\s*"
    r"(?:[A-Z]{3}\s*)?" + _AMOUNT, re.I)

_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")

_DATE_CONFIRMED = re.compile(
    r"(?:expected on|credit date is|confirmed credit date is|resumes on|"
    r"dikonfirmasi pada|akan dibayarkan pada|mulai)\s*(\d{4}-\d{2}-\d{2})", re.I)

_PERCENT = re.compile(
    r"(?:increases?|raises?|naik)[^.]{0,40}?by\s*(\d+(?:\.\d+)?)\s*%", re.I)


def _number(text: str) -> str | None:
    """Normalise an amount token to a plain decimal string."""
    t = text.strip()
    if "," in t and "." in t:
        t = (t.replace(".", "").replace(",", ".")
             if t.rfind(",") > t.rfind(".") else t.replace(",", ""))
    elif "," in t:
        tail = t.rsplit(",", 1)[-1]
        t = t.replace(",", ".") if len(tail) == 2 else t.replace(",", "")
    t = t.rstrip(".")
    try:
        float(t)
    except ValueError:
        return None
    return t


def _untrusted_body(prompt: str) -> str:
    if FENCE_OPEN in prompt:
        return prompt.split(FENCE_OPEN)[-1].split(FENCE_CLOSE)[0]
    return prompt


class RuleProvider:
    name = "rule"

    def complete_json(self, prompt: str, schema: dict):
        body = _untrusted_body(prompt)
        amendments: list[dict] = []

        # Unsettled money short-circuits everything else. This is the
        # conservative reading the problem statement requires.
        if _UNSETTLED.search(body):
            return {"amendments": [{"kind": "no_change", "confidence": "high"}]}

        m = _SALARY_NEW.search(body)
        if m:
            value = _number(m.group(1))
            if value is not None:
                item = {"kind": "salary_change", "amount": value,
                        "confidence": "high"}
                d = _DATE.search(body)
                if d:
                    item["effective_date"] = d.group(1)
                amendments.append(item)

        p = _PERCENT.search(body)
        if p:
            amendments.append({
                "kind": "rate_change",
                "multiplier": str(1 + float(p.group(1)) / 100),
                "confidence": "medium",
            })

        if not amendments:
            d = _DATE_CONFIRMED.search(body)
            if d:
                amendments.append({"kind": "salary_date_change",
                                   "effective_date": d.group(1),
                                   "confidence": "high"})

        if not amendments:
            amendments.append({"kind": "no_change", "confidence": "high"})
        return {"amendments": amendments}

    def read_image(self, path: Path, prompt: str, schema: dict):
        """There is no model in this backend, so there is no second reader.

        Returning None is honest: the extractor then relies on its own
        label-driven OCR, which is grounded in text on the page rather than in
        a guess about which number looks biggest.
        """
        return None
