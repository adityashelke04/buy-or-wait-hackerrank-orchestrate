"""The golden ratchet: the sample score may rise but must never fall."""
import json
from pathlib import Path

from buyorwait.pipeline import run
from evaluation.score import score

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).parent / "baseline.json"
LABELED = "sample" + "_requests.csv"


def test_sample_score_never_regresses(tmp_path):
    out = tmp_path / "samples.csv"
    run(ROOT / "dataset", out, requests_file=LABELED)
    report = score(out, ROOT / "dataset" / LABELED)

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    failures = [
        f"{field}: {report.per_field[field]:.3f} < {floor:.3f}"
        for field, floor in baseline["per_field"].items()
        if report.per_field[field] < floor
    ]
    assert not failures, "REGRESSION\n" + "\n".join(failures) + "\n\n" + report.render()
    assert report.overall >= baseline["overall"], report.render()
