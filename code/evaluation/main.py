"""Reproduce the local evaluation in one command.

    python code/evaluation/main.py

Runs the shipped configuration (rule evidence reader + OCR) over the labeled
sample requests in a temporary folder and prints the per-field score from
evaluation/score.py. The labeled answers are read only by
the scorer, after every decision has been made.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

from buyorwait.evidence.extractor import Extractor        # noqa: E402
from buyorwait.evidence.rule_provider import RuleProvider  # noqa: E402
from buyorwait.pipeline import run                        # noqa: E402
from evaluation.calibrate import LABELED                  # noqa: E402
from evaluation.score import score                        # noqa: E402


def main() -> int:
    dataset = ROOT / "dataset"
    with tempfile.TemporaryDirectory() as tmp:
        out = (Path(tmp) / "predictions").with_suffix(".csv")   # a scratch output, not a dataset file
        extractor = Extractor(RuleProvider(), dataset, cache_path=Path(tmp) / "cache.json")
        run(dataset, out, extractor=extractor, requests_file=LABELED)
        print(score(out, dataset / LABELED).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
