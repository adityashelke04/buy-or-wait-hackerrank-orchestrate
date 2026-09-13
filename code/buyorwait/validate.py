"""The contract gate.

Runs every invariant from the problem statement over the finished decisions. A
non-empty result means the run is INVALID and nothing may be written. This is
the last line of defence against submitting a malformed output.csv.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal, InvalidOperation

from .money import fmt_amount
from .types import Dataset, Decision

STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later",
            "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait",
           "not_recommended"}

STATUS_METHODS = {
    "affordable_now": {"full_payment"},
    "affordable_with_plan": {"full_payment", "partial_payment", "installments"},
    "affordable_later": {"wait"},
    "not_affordable": {"not_recommended"},
}


def _installment_plans(ds: Dataset, request_id: str) -> set[str]:
    plans: set[str] = set()
    for o in ds.options_by_request.get(request_id, []):
        if o.payment_method != "installments":
            continue
        freq = o.payment_frequency_days or 0
        dates = [o.first_payment_date + timedelta(days=freq * i)
                 for i in range(o.number_of_payments)]
        # The option's own text, or the same amounts in the plan's number format.
        for text in {o.payment_amount_text, fmt_amount(o.payment_amount)}:
            plans.add("|".join(f"{d.isoformat()}:{text}" for d in dates))
    return plans


def check_all(decisions: list[Decision], ds: Dataset) -> list[str]:
    problems: list[str] = []
    by_req = {r.request_id: r for r in ds.requests}
    seen: set[str] = set()

    for d in decisions:
        rid = d.request_id
        if rid in seen:
            problems.append(f"{rid}: duplicate row")
        seen.add(rid)

        req = by_req.get(rid)
        if req is None:
            problems.append(f"{rid}: not present in requests.csv")
            continue
        profile = ds.profiles[req.user_id]

        if d.affordability_status not in STATUSES:
            problems.append(f"{rid}: bad status {d.affordability_status!r}")
        if d.recommended_payment_method not in METHODS:
            problems.append(f"{rid}: bad method {d.recommended_payment_method!r}")
        if (d.affordability_status in STATUS_METHODS
                and d.recommended_payment_method
                not in STATUS_METHODS[d.affordability_status]):
            problems.append(f"{rid}: {d.affordability_status} cannot pair with "
                            f"{d.recommended_payment_method}")

        try:
            safe = Decimal(d.amount_safe_to_pay)
        except InvalidOperation:
            problems.append(f"{rid}: amount_safe_to_pay not numeric")
            continue
        if not (0 <= safe <= req.requested_amount):
            problems.append(f"{rid}: safe {safe} outside [0, {req.requested_amount}]")
        if "e" in d.amount_safe_to_pay.lower() or safe != safe.quantize(Decimal("0.01")):
            problems.append(f"{rid}: amount_safe_to_pay {d.amount_safe_to_pay!r} is not plain money")

        earliest: date | None = None
        if d.earliest_date_for_full_payment:
            try:
                earliest = date.fromisoformat(d.earliest_date_for_full_payment)
            except ValueError:
                problems.append(f"{rid}: earliest date is not YYYY-MM-DD")
        if earliest is not None and earliest < req.request_date:
            problems.append(f"{rid}: earliest date precedes the request date")
        if d.affordability_status == "affordable_later":
            if earliest is None or earliest <= req.request_date:
                problems.append(f"{rid}: affordable_later needs a future earliest date")

        if d.affordability_status == "affordable_now":
            if d.earliest_date_for_full_payment != req.request_date.isoformat():
                problems.append(f"{rid}: affordable_now needs earliest == request_date")
        if d.affordability_status == "not_affordable":
            if d.payment_plan != "none":
                problems.append(f"{rid}: not_affordable must have plan 'none'")
            if d.earliest_date_for_full_payment != "":
                problems.append(f"{rid}: not_affordable must have empty earliest date")

        if d.payment_plan != "none":
            dates: list[date] = []
            for part in d.payment_plan.split("|"):
                day, _, amount = part.partition(":")
                try:
                    dates.append(date.fromisoformat(day))
                    Decimal(amount)
                except (ValueError, InvalidOperation):
                    problems.append(f"{rid}: malformed plan entry {part!r}")
            if dates != sorted(dates):
                problems.append(f"{rid}: plan is not chronological")
            for part in d.payment_plan.split("|"):
                amount = part.partition(":")[2]
                try:
                    if Decimal(amount) <= 0 or "e" in amount.lower():
                        problems.append(f"{rid}: plan amount {amount!r} is not a positive amount")
                except InvalidOperation:
                    pass                                   # reported above
        elif d.recommended_payment_method != "not_recommended":
            problems.append(f"{rid}: {d.recommended_payment_method} needs a payment plan")

        if d.recommended_payment_method in ("wait", "full_payment") and d.payment_plan != "none":
            if "|" in d.payment_plan:
                problems.append(f"{rid}: {d.recommended_payment_method} must be one payment")
        if d.recommended_payment_method == "wait" and d.payment_plan != "none":
            day, _, amount = d.payment_plan.partition(":")
            if day != d.earliest_date_for_full_payment:
                problems.append(f"{rid}: wait must pay on the earliest safe date")
            try:
                if Decimal(amount) != req.requested_amount:
                    problems.append(f"{rid}: wait must pay the full requested amount")
            except InvalidOperation:
                pass

        if d.recommended_payment_method == "partial_payment":
            if d.affordability_status != "affordable_with_plan":
                problems.append(f"{rid}: partial_payment requires affordable_with_plan")
            if not req.allows_partial_payment:
                problems.append(f"{rid}: partial_payment but the request forbids it")
            parts = d.payment_plan.split("|")
            if len(parts) != 2:
                problems.append(f"{rid}: partial_payment needs exactly 2 payments")
            else:
                d1, a1 = parts[0].split(":")
                d2, a2 = parts[1].split(":")
                if d1 != req.request_date.isoformat():
                    problems.append(f"{rid}: first partial payment not on request_date")
                if Decimal(a1) + Decimal(a2) != req.requested_amount:
                    problems.append(f"{rid}: partial payments do not sum to requested")
                if d2 != d.earliest_date_for_full_payment:
                    problems.append(f"{rid}: second payment not on the earliest date")
                if date.fromisoformat(d2) > req.desired_completion_date:
                    problems.append(f"{rid}: partial payment misses the deadline")

        if d.recommended_payment_method == "installments":
            if d.payment_plan not in _installment_plans(ds, rid):
                problems.append(f"{rid}: installment plan matches no supplied option")

        if d.recommended_payment_method not in ("wait", "not_recommended"):
            if d.recommended_payment_method not in profile.payment_methods_user_will_consider:
                problems.append(f"{rid}: method not among the user's accepted methods")

        if d.spending_changes_needed != "none":
            parts = d.spending_changes_needed.split("|")
            if len(parts) > 3:
                problems.append(f"{rid}: more than three spending changes")
            touched: set[str] = set()
            for part in parts:
                bits = part.split(":")
                if bits[0] not in ("stop", "reduce_to"):
                    problems.append(f"{rid}: bad change verb {bits[0]!r}")
                    continue
                event_id = bits[1]
                if event_id in touched:
                    problems.append(f"{rid}: {event_id} changed twice")
                touched.add(event_id)
                event = ds.events_by_id.get(event_id)
                if event is None:
                    problems.append(f"{rid}: unknown event {event_id}")
                    continue
                if event.user_id != profile.user_id:
                    problems.append(f"{rid}: {event_id} belongs to another user")
                if not event.is_flexible:
                    problems.append(f"{rid}: {event_id} is not flexible")
                if event.category in profile.expense_categories_to_protect:
                    problems.append(f"{rid}: {event_id} is in a protected category")
                if bits[0] == "stop":
                    if event.category not in profile.expense_categories_user_is_willing_to_stop:
                        problems.append(f"{rid}: user will not stop {event.category}")
                else:
                    if event.category not in profile.expense_categories_user_is_willing_to_reduce:
                        problems.append(f"{rid}: user will not reduce {event.category}")
                    if (event.minimum_allowed_amount is None
                            or Decimal(bits[2]) < event.minimum_allowed_amount):
                        problems.append(f"{rid}: reduce below minimum_allowed_amount")

        if not d.decision_explanation.strip():
            problems.append(f"{rid}: empty explanation")
        elif profile.home_currency not in d.decision_explanation:
            problems.append(f"{rid}: explanation omits the home currency")

    for rid in sorted({r.request_id for r in ds.requests} - seen):
        problems.append(f"{rid}: missing from output")
    return problems
