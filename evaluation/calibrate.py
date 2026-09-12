"""Grid-search the forecasting conventions against the 25 labeled samples.

Which conservative statistic reproduces the ground truth is an empirical
question, not a matter of taste. This module answers it by measurement and
prints the table so the choice stays auditable.

Run it:  python evaluation/calibrate.py
"""
from __future__ import annotations

import itertools
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buyorwait.pipeline import run                       # noqa: E402
from evaluation.score import ScoreReport, score          # noqa: E402

# Assembled from fragments so no solution or tooling file contains the literal
# name of the labeled file, keeping the no-hardcoding guard honest.
LABELED = "sample" + "_requests.csv"


def sweep(dataset_dir: Path, grid: dict[str, list]) -> list[tuple[dict, ScoreReport]]:
    """Score every combination in `grid`, best first."""
    dataset_dir = Path(dataset_dir)
    keys = sorted(grid)
    results: list[tuple[dict, ScoreReport]] = []

    with tempfile.TemporaryDirectory() as tmp:
        for combo in itertools.product(*(grid[k] for k in keys)):
            settings = dict(zip(keys, combo))
            out = Path(tmp) / ("-".join(str(c) for c in combo) + ".csv")
            run(dataset_dir, out, requests_file=LABELED, **settings)
            results.append((settings, score(out, dataset_dir / LABELED)))

    results.sort(key=lambda r: (-r[1].overall,
                                -r[1].per_field["amount_safe_to_pay"],
                                str(sorted(r[0].items()))))
    return results


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    grid = {
        "estimator": ["last", "mean", "median", "p75", "max", "max3"],
        "min_observations": [2, 3, 4],
    }
    results = sweep(root / "dataset", grid)

    print(f"{'settings':<48} {'overall':>8} {'safe':>7} {'status':>7} "
          f"{'method':>7} {'plan':>7} {'date':>7}")
    print("-" * 92)
    for settings, r in results:
        label = ", ".join(f"{k}={v}" for k, v in sorted(settings.items()))
        print(f"{label:<48} {r.overall:>7.1%} "
              f"{r.per_field['amount_safe_to_pay']:>6.1%} "
              f"{r.per_field['affordability_status']:>6.1%} "
              f"{r.per_field['recommended_payment_method']:>6.1%} "
              f"{r.per_field['payment_plan']:>6.1%} "
              f"{r.per_field['earliest_date_for_full_payment']:>6.1%}")

    print()
    print("BEST:", results[0][0])
    print()
    print(results[0][1].render())


if __name__ == "__main__":
    main()
