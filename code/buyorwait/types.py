"""Typed records for every dataset row.

Frozen dataclasses: once built, a record cannot be mutated. Any adjustment - an
amendment derived from a message, an amount resolved from an image - produces a
NEW record via dataclasses.replace, so the original dataset values stay
auditable and no module can silently change data another module relies on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...]
    expense_categories_to_protect: tuple[str, ...]
    expense_categories_user_is_willing_to_reduce: tuple[str, ...]
    expense_categories_user_is_willing_to_stop: tuple[str, ...]
    payment_methods_user_will_consider: tuple[str, ...]
    max_installment_months: int | None


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str                 # debit | credit | non_cash
    amount: Decimal | None         # None means "resolve from the linked image"
    currency: str
    event_date: date
    settlement_date: date
    status: str                    # settled|pending|scheduled|cancelled|failed|unrealized
    linked_event_id: str | None
    flexibility: str               # fixed|reducible|stoppable|reducible_or_stoppable
    minimum_allowed_amount: Decimal | None

    @property
    def is_flexible(self) -> bool:
        return self.flexibility != "fixed"

    @property
    def can_stop(self) -> bool:
        return self.flexibility in ("stoppable", "reducible_or_stoppable")

    @property
    def can_reduce(self) -> bool:
        return self.flexibility in ("reducible", "reducible_or_stoppable")


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str            # full_payment | installments
    payment_amount: Decimal
    payment_amount_text: str       # verbatim from the CSV, written back unchanged
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: int | None
    financing_fee: Decimal
    total_payable_amount: Decimal

    @property
    def sort_key(self) -> int:
        return int(self.payment_option_id.rsplit("_", 1)[-1])


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    sent_at: str
    source_type: str
    message_text: str


@dataclass(frozen=True)
class ImageRef:
    image_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None

    def path(self, dataset_dir: Path) -> Path:
        return Path(dataset_dir) / "media" / "images" / f"{self.image_id}.png"


@dataclass(frozen=True)
class Rate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: Decimal


@dataclass(frozen=True)
class Dataset:
    profiles: dict[str, Profile]
    events_by_user: dict[str, list[Event]]
    requests: list[Request]
    options_by_request: dict[str, list[PaymentOption]]
    messages_by_user: dict[str, list[Message]]
    images_by_event: dict[str, ImageRef]
    rates: list[Rate]
    events_by_id: dict[str, Event] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    request_id: str
    amount_safe_to_pay: str
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str

    COLUMNS = (
        "request_id", "amount_safe_to_pay", "affordability_status",
        "recommended_payment_method", "payment_plan",
        "earliest_date_for_full_payment", "spending_changes_needed",
        "decision_explanation",
    )

    def as_row(self) -> dict[str, str]:
        return {c: getattr(self, c) for c in self.COLUMNS}
