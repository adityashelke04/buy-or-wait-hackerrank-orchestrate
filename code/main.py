"""Buy or Wait? - entry point.

    python code/main.py                                   # full run -> output.csv
    python code/main.py --requests-file X.csv --out Y.csv # any requests-schema file
    python code/main.py --backend cloud                   # optional, needs LLM_API_KEY
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from buyorwait.pipeline import run                    # noqa: E402
from buyorwait.recurrence import ESTIMATORS           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("rule", "cloud", "none")


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from .env into the environment.

    Stdlib only, and a variable already set in the real environment always wins.
    Values are never printed or logged.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")

    p = argparse.ArgumentParser(description="Buy or Wait? financial decision agent")
    p.add_argument("--dataset", type=Path, default=ROOT / "dataset",
                   help="folder holding the challenge CSVs and media/ (default: dataset/)")
    p.add_argument("--out", type=Path, default=ROOT / "output.csv",
                   help="where to write predictions (default: output.csv)")
    p.add_argument("--requests-file", default="requests.csv",
                   help="requests-schema file inside --dataset (default: requests.csv)")
    p.add_argument("--estimator", default="p75", choices=sorted(ESTIMATORS),
                   help="how conservatively to forecast variable spending (default: p75)")
    p.add_argument("--backend", default=None, choices=BACKENDS,
                   help="evidence reader for messages and images: rule (local, default), "
                        "cloud (needs LLM_API_KEY) or none. Falls back to LLM_BACKEND.")
    args = p.parse_args(argv)

    backend = args.backend or os.environ.get("LLM_BACKEND", "rule")
    if backend not in BACKENDS:
        p.error(f"LLM_BACKEND={backend!r} is not one of {', '.join(BACKENDS)}")
    if not (args.dataset / args.requests_file).is_file():
        p.error(f"{args.dataset / args.requests_file} does not exist")

    extractor = None
    if backend != "none":
        from buyorwait.evidence.extractor import Extractor
        try:
            extractor = Extractor.from_env(args.dataset, backend=backend)
        except RuntimeError as exc:                    # e.g. cloud without a key
            p.error(str(exc))

    full_run = args.requests_file == "requests.csv"
    report = ROOT / "evaluation" / "usage_report.md" if full_run else None
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
