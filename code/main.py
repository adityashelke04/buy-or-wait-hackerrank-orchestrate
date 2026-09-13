"""Buy or Wait? - entry point.

    python code/main.py                      # full run -> output.csv
    python code/main.py --requests-file X.csv --out Y.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from buyorwait.pipeline import run          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description="Buy or Wait? financial decision agent")
    p.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    p.add_argument("--out", type=Path, default=ROOT / "output.csv")
    p.add_argument("--requests-file", default="requests.csv")
    p.add_argument("--estimator", default="p75",
                   help="conservative spend estimator: last|mean|median|p75|max|max3")
    p.add_argument("--backend", default="rule",
                   help="evidence backend: rule (default, local) | cloud | none")
    args = p.parse_args()

    extractor = None
    if args.backend and args.backend != "none":
        from buyorwait.evidence.extractor import Extractor
        extractor = Extractor.from_env(args.dataset, backend=args.backend)

    report = (ROOT / "evaluation" / "usage_report.md"
              if args.requests_file == "requests.csv" else None)
    decisions = run(args.dataset, args.out, extractor=extractor,
                    estimator=args.estimator, requests_file=args.requests_file,
                    usage_report_path=report)
    if report is not None and report.exists():
        # The starter repository also carries code/evaluation/usage_report.md;
        # keep both copies identical so either location is satisfied.
        mirror = ROOT / "code" / "evaluation" / "usage_report.md"
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {len(decisions)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
