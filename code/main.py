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
    p.add_argument("--backend", default=None,
                   help="evidence backend: rule|ollama|cloud (default: none)")
    args = p.parse_args()

    extractor = None
    if args.backend and args.backend != "none":
        from buyorwait.evidence.extractor import Extractor
        extractor = Extractor.from_env(args.dataset, backend=args.backend)

    decisions = run(args.dataset, args.out, extractor=extractor,
                    estimator=args.estimator, requests_file=args.requests_file)
    print(f"wrote {len(decisions)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
