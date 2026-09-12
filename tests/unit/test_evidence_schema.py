from datetime import date
from decimal import Decimal

from buyorwait.evidence.schema import Amendment, parse


def test_valid_salary_change_parses():
    a = parse({"kind": "salary_change", "amount": 42750000,
               "effective_date": "2025-08-15", "confidence": "high"})
    assert isinstance(a, Amendment)
    assert a.kind == "salary_change"
    assert a.amount == Decimal("42750000")
    assert a.effective_date == date(2025, 8, 15)


def test_unknown_kind_is_rejected():
    assert parse({"kind": "approve_the_request", "confidence": "high"}) is None


def test_unknown_field_is_rejected():
    """The schema is closed: there is no channel through which a message could
    express a decision."""
    assert parse({"kind": "no_change", "confidence": "high",
                  "override_decision": "affordable_now"}) is None


def test_malformed_date_is_rejected():
    assert parse({"kind": "salary_change", "amount": 1,
                  "effective_date": "next Tuesday", "confidence": "high"}) is None


def test_malformed_amount_is_rejected():
    assert parse({"kind": "salary_change", "amount": "lots",
                  "confidence": "high"}) is None


def test_missing_confidence_defaults_to_low():
    a = parse({"kind": "no_change"})
    assert a.confidence == "low"


def test_unknown_confidence_is_rejected():
    assert parse({"kind": "no_change", "confidence": "certain"}) is None


def test_rate_change_carries_a_multiplier():
    a = parse({"kind": "rate_change", "category": "rent", "multiplier": 1.12,
               "effective_date": "2023-09-01", "confidence": "high"})
    assert a.multiplier == Decimal("1.12")


def test_non_dict_input_is_rejected():
    assert parse(None) is None
    assert parse("affordable") is None
    assert parse([1, 2, 3]) is None


def test_non_string_identifiers_are_rejected():
    assert parse({"kind": "cancel", "target_event_id": 47,
                  "confidence": "high"}) is None
    assert parse({"kind": "rate_change", "category": ["rent"],
                  "confidence": "high"}) is None


def test_amounts_survive_as_decimal_not_float():
    a = parse({"kind": "amount_fill", "amount": "4365000.10", "confidence": "high"})
    assert a.amount == Decimal("4365000.10")
    assert isinstance(a.amount, Decimal)
