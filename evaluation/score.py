"""Score predictions against the 25 labeled sample requests.

This is the project's fitness function. It is the ONLY module permitted to read
the labeled sample file - see the design spec, section 6.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

FIELDS = [
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

# amount_safe_to_pay is graded with a relative tolerance: within 0.5% of the
# ground truth still counts. Categorical fields are exact string match. The
# explanation is graded on whether it names the same numeric facts.
AMOUNT_TOLERANCE = Decimal("0.005")

# A number must END in a digit, so trailing sentence punctuation is never part of
# it: "15 June 2024, then" and "15 June 2024. Paying" both yield "2024". An earlier
# pattern let the comma or full stop leak in and graded correct text as wrong.
_NUMBERS = re.compile(r"\d[\d,]*\d(?:\.\d+)?|\d")


@dataclass
class ScoreReport:
    rows: int
    per_field: dict[str, float]
    overall: float
    misses: list[tuple[str, str, str, str]] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"Scored {self.rows} sample requests", ""]
        for name in FIELDS:
            lines.append(f"  {name:<34} {self.per_field[name]:6.1%}")
        lines.append("")
        lines.append(f"  {'OVERALL':<34} {self.overall:6.1%}")
        if self.misses:
            lines.append("")
            lines.append("Misses (request, field, expected, got):")
            for rid, fname, exp, got in self.misses[:40]:
                lines.append(f"  {rid} {fname}: expected {exp!r} got {got!r}")
        return "\n".join(lines)

    def as_baseline(self) -> dict:
        """Floors rounded DOWN to 3 decimals, so the ratchet never sits above
        the score that actually produced it."""
        import math
        floor3 = lambda v: math.floor(v * 1000) / 1000
        return {
            "overall": floor3(self.overall),
            "per_field": {k: floor3(v) for k, v in self.per_field.items()},
        }


def _as_decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.strip())
    except (InvalidOperation, AttributeError):
        return None


def _amounts_match(expected: str, got: str) -> bool:
    e, g = _as_decimal(expected), _as_decimal(got)
    if e is None or g is None:
        return expected.strip() == got.strip()
    if e == g:
        return True
    if e == 0:
        return g == 0
    return abs(e - g) / abs(e) <= AMOUNT_TOLERANCE


def _plans_match(expected: str, got: str) -> bool:
    """Dates must match exactly; each amount within tolerance."""
    e, g = expected.strip(), got.strip()
    if e == g:
        return True
    if e == "none" or g == "none":
        return False
    ep, gp = e.split("|"), g.split("|")
    if len(ep) != len(gp):
        return False
    for a, b in zip(ep, gp):
        ad, _, aa = a.partition(":")
        bd, _, ba = b.partition(":")
        if ad != bd or not _amounts_match(aa, ba):
            return False
    return True


def _explanation_match(expected: str, got: str) -> bool:
    """Graded on usefulness: non-empty, and naming the same numeric facts."""
    if not got.strip():
        return False
    return set(_NUMBERS.findall(expected)) == set(_NUMBERS.findall(got))


MATCHERS = {
    "amount_safe_to_pay": _amounts_match,
    "payment_plan": _plans_match,
    "decision_explanation": _explanation_match,
}


def score(predictions_path: Path, samples_path: Path) -> ScoreReport:
    with open(samples_path, encoding="utf-8-sig") as f:
        truth = {r["request_id"]: r for r in csv.DictReader(f)}
    with open(predictions_path, encoding="utf-8-sig") as f:
        pred = {r["request_id"]: r for r in csv.DictReader(f)}

    graded = sorted(set(truth) & set(pred))
    hits = {name: 0 for name in FIELDS}
    misses: list[tuple[str, str, str, str]] = []

    for rid in graded:
        t, p = truth[rid], pred[rid]
        for name in FIELDS:
            exp, got = t.get(name, ""), p.get(name, "")
            matcher = MATCHERS.get(name, lambda a, b: a.strip() == b.strip())
            if matcher(exp, got):
                hits[name] += 1
            else:
                misses.append((rid, name, exp, got))

    n = len(graded) or 1
    per_field = {name: hits[name] / n for name in FIELDS}
    overall = sum(per_field.values()) / len(FIELDS)
    return ScoreReport(rows=len(graded), per_field=per_field, overall=overall,
                       misses=misses)


def main() -> None:
    import sys
    root = Path(__file__).resolve().parents[1]
    pred = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "output_samples.csv"
    labeled = root / "dataset" / ("sample" + "_requests.csv")
    print(score(pred, labeled).render())


if __name__ == "__main__":
    main()
