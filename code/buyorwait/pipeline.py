"""Wire every module into one decision per request."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from . import changes as changes_mod
from . import explain, forecast, recurrence, validate
from .fx import RateTable
from .io_loaders import load_dataset, load_requests_file
from .io_writer import write
from .ledger import LedgerView
from .money import fmt_plain
from .ranker import rank
from .solver import earliest_full_payment_date, enumerate_candidates, safe_amount
from .types import Dataset, Decision, Request

DEFAULT_ESTIMATOR = "p75"          # chosen by evaluation/calibrate.py, not by taste


def decide(ds: Dataset, request: Request, rates: RateTable,
           extractor=None, estimator: str = DEFAULT_ESTIMATOR,
           horizon_days: int = forecast.HORIZON_DAYS,
           min_observations: int = recurrence.MIN_OBSERVATIONS,
           project_income: bool = True) -> Decision:
    profile = ds.profiles[request.user_id]
    events = list(ds.events_by_user.get(request.user_id, []))

    if extractor is not None:
        events = extractor.apply(events, request, profile)

    view = LedgerView(events, profile, rates)
    series = recurrence.detect(events, profile, rates, request.request_date,
                               estimator=estimator,
                               min_observations=min_observations,
                               project_income=project_income)
    curve = forecast.build(view, series, request.request_date, horizon_days)

    minimum = profile.minimum_balance_to_keep
    safe = safe_amount(curve, minimum, request.requested_amount, request.request_date)
    earliest = earliest_full_payment_date(curve, minimum, request.requested_amount)

    change_options = changes_mod.candidates(series, profile)
    candidates = enumerate_candidates(
        request, profile, curve,
        ds.options_by_request.get(request.request_id, []),
        change_options, safe, earliest,
    )
    winner = rank(candidates)

    if winner is None:
        return Decision(
            request_id=request.request_id,
            amount_safe_to_pay=fmt_plain(safe),
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment="",
            spending_changes_needed="none",
            decision_explanation=explain.render(None, request, profile, safe),
        )

    return Decision(
        request_id=request.request_id,
        amount_safe_to_pay=fmt_plain(safe),
        affordability_status=winner.status,
        recommended_payment_method=winner.method,
        payment_plan=winner.render_plan(),
        earliest_date_for_full_payment=earliest.isoformat() if earliest else "",
        spending_changes_needed=winner.render_changes(),
        decision_explanation=explain.render(winner, request, profile, safe),
    )


def run(dataset_dir: Path, out_path: Path, extractor=None,
        estimator: str = DEFAULT_ESTIMATOR,
        requests_file: str = "requests.csv",
        horizon_days: int = forecast.HORIZON_DAYS,
        min_observations: int = recurrence.MIN_OBSERVATIONS,
        project_income: bool = True,
        validate_output: bool = True,
        usage_report_path: Path | None = None) -> list[Decision]:
    dataset_dir = Path(dataset_dir)
    ds = load_dataset(dataset_dir)
    rates = RateTable(ds.rates)

    if requests_file == "requests.csv":
        requests = ds.requests
    else:
        requests = load_requests_file(dataset_dir / requests_file)
        validate_output = False        # the gate is defined over the eval set

    decisions = [
        decide(ds, r, rates, extractor, estimator, horizon_days,
               min_observations, project_income)
        for r in requests
    ]

    if validate_output:
        problems = validate.check_all(decisions, ds)
        if problems:
            raise SystemExit(
                "CONTRACT VIOLATIONS - refusing to write output.csv:\n  "
                + "\n  ".join(problems[:30])
            )

    write(decisions, out_path)

    if usage_report_path is not None:
        from .evidence.usage import USAGE
        USAGE.write_report(Path(usage_report_path), requests=len(decisions))
    return decisions
