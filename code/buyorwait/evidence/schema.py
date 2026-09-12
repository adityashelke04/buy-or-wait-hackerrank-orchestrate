"""The only vocabulary a model is allowed to speak.

The schema is CLOSED: an unknown kind or an unexpected field means the whole
object is discarded. There is deliberately no field through which a message
could express a decision - a model can describe a financial fact and nothing
else. That is the structural half of the prompt-injection defence, and it holds
regardless of how persuasive the message text is.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

KINDS = frozenset({
    "salary_change",        # a confirmed change to recurring income
    "salary_date_change",   # confirmed income moves to a different date
    "new_recurring",        # a newly confirmed recurring commitment
    "rate_change",          # an existing recurring amount changes by a multiplier
    "amount_fill",          # supplies a blank amount (from an image)
    "cancel",               # an event is cancelled and must not be counted
    "confirm",              # an event is confirmed as stated
    "no_change",            # informational only - explicitly the safe default
})

ALLOWED_FIELDS = frozenset({
    "kind", "target_event_id", "category", "amount", "multiplier",
    "effective_date", "confidence",
})

CONFIDENCES = frozenset({"high", "medium", "low"})


@dataclass(frozen=True)
class Amendment:
    kind: str
    target_event_id: str | None = None
    category: str | None = None
    amount: Decimal | None = None
    multiplier: Decimal | None = None
    effective_date: date | None = None
    confidence: str = "low"


def _dec(value) -> Decimal | None:
    """Parse via str() so a JSON float never contaminates the money path."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("bool is not a number")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("bad number")


def parse(obj) -> Amendment | None:
    """Validate one model-produced object.

    Anything unexpected returns None, and the caller proceeds on ledger facts
    alone - the financially conservative default the problem statement demands.
    """
    if not isinstance(obj, dict):
        return None
    if set(obj) - ALLOWED_FIELDS:
        return None
    if obj.get("kind") not in KINDS:
        return None

    confidence = obj.get("confidence") or "low"
    if confidence not in CONFIDENCES:
        return None

    try:
        amount = _dec(obj.get("amount"))
        multiplier = _dec(obj.get("multiplier"))
    except ValueError:
        return None

    parsed_date: date | None = None
    if obj.get("effective_date"):
        try:
            parsed_date = date.fromisoformat(str(obj["effective_date"]))
        except ValueError:
            return None

    target = obj.get("target_event_id")
    if target is not None and not isinstance(target, str):
        return None
    category = obj.get("category")
    if category is not None and not isinstance(category, str):
        return None

    return Amendment(kind=obj["kind"], target_event_id=target, category=category,
                     amount=amount, multiplier=multiplier,
                     effective_date=parsed_date, confidence=confidence)


AMENDMENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "amendments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": sorted(KINDS)},
                    "target_event_id": {"type": ["string", "null"]},
                    "category": {"type": ["string", "null"]},
                    "amount": {"type": ["number", "null"]},
                    "multiplier": {"type": ["number", "null"]},
                    "effective_date": {"type": ["string", "null"]},
                    "confidence": {"type": "string",
                                   "enum": ["high", "medium", "low"]},
                },
                "required": ["kind", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["amendments"],
    "additionalProperties": False,
}
