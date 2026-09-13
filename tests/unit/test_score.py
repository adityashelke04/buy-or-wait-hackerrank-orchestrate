import csv
from pathlib import Path

from evaluation.score import score

ROOT = Path(__file__).resolve().parents[2]

COLS = ["request_id", "amount_safe_to_pay", "affordability_status",
        "recommended_payment_method", "payment_plan",
        "earliest_date_for_full_payment", "spending_changes_needed",
        "decision_explanation"]


def _copy_samples(samples: Path, out: Path, mutate=None) -> None:
    with open(samples, encoding="utf-8") as f, open(out, "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=COLS)
        w.writeheader()
        for i, row in enumerate(csv.DictReader(f)):
            r = {c: row[c] for c in COLS}
            if mutate is not None:
                mutate(i, r)
            w.writerow(r)


def test_perfect_prediction_scores_one(tmp_path):
    """Feeding the samples back as predictions must score 1.0 on every field."""
    samples = ROOT / "dataset" / "sample_requests.csv"
    out = tmp_path / "pred.csv"
    _copy_samples(samples, out)

    report = score(out, samples)
    assert report.rows == 25
    assert report.overall == 1.0
    assert report.per_field["affordability_status"] == 1.0
    assert report.per_field["amount_safe_to_pay"] == 1.0


def test_wrong_status_lowers_only_that_field(tmp_path):
    samples = ROOT / "dataset" / "sample_requests.csv"
    out = tmp_path / "pred.csv"

    def mutate(i, r):
        if i == 0:
            r["affordability_status"] = "not_affordable"

    _copy_samples(samples, out, mutate)

    report = score(out, samples)
    assert report.per_field["affordability_status"] == 24 / 25
    assert report.per_field["amount_safe_to_pay"] == 1.0


def test_amount_within_tolerance_counts_as_a_match(tmp_path):
    samples = ROOT / "dataset" / "sample_requests.csv"
    out = tmp_path / "pred.csv"

    def mutate(i, r):
        if i == 0:
            r["amount_safe_to_pay"] = "25255"        # 25256 off by 0.004%

    _copy_samples(samples, out, mutate)
    assert score(out, samples).per_field["amount_safe_to_pay"] == 1.0


def test_amount_outside_tolerance_is_a_miss(tmp_path):
    samples = ROOT / "dataset" / "sample_requests.csv"
    out = tmp_path / "pred.csv"

    def mutate(i, r):
        if i == 0:
            r["amount_safe_to_pay"] = "12000"

    _copy_samples(samples, out, mutate)
    report = score(out, samples)
    assert report.per_field["amount_safe_to_pay"] == 24 / 25
    assert any(m[1] == "amount_safe_to_pay" for m in report.misses)


def test_explanation_numbers_ignore_trailing_punctuation():
    """'15 June 2024, then' and '15 June 2024. Paying' name the same date."""
    from evaluation.score import _explanation_match
    assert _explanation_match("Wait until 15 June 2024, then pay IDR 12,693,000 in full.",
                              "Pay IDR 12,693,000 in full on 15 June 2024. Paying earlier.")


def test_explanation_numbers_still_distinguish_real_differences():
    from evaluation.score import _explanation_match
    assert not _explanation_match("Pay EUR 620.40 today.", "Pay EUR 603.30 today.")
    assert not _explanation_match("Pay IDR 12,693,000 today.", "Pay IDR 12,693,001 today.")
