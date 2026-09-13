from pathlib import Path

import pytest

from buyorwait.pipeline import run

ROOT = Path(__file__).resolve().parents[2]


def test_full_run_produces_250_rows(tmp_path):
    out = tmp_path / "output.csv"
    decisions = run(ROOT / "dataset", out)
    assert len(decisions) == 250
    assert out.exists()


def test_two_runs_are_byte_identical(tmp_path):
    """Determinism is a stated requirement of the challenge."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    run(ROOT / "dataset", a)
    run(ROOT / "dataset", b)
    assert a.read_bytes() == b.read_bytes()


def test_output_has_no_embedded_newlines_that_would_break_the_csv(tmp_path):
    out = tmp_path / "output.csv"
    decisions = run(ROOT / "dataset", out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(decisions) + 1


@pytest.mark.slow
def test_submitted_output_csv_is_exactly_what_the_shipped_configuration_produces(tmp_path):
    """Guards against submitting an output.csv from an older version of the code."""
    from buyorwait.evidence.extractor import Extractor
    from buyorwait.evidence.rule_provider import RuleProvider

    out = tmp_path / "output.csv"
    extractor = Extractor(RuleProvider(), ROOT / "dataset", cache_path=tmp_path / "c.json")
    run(ROOT / "dataset", out, extractor=extractor)
    # Compared line by line: a Windows checkout may turn LF into CRLF.
    fresh = out.read_text(encoding="utf-8").splitlines()
    submitted = (ROOT / "output.csv").read_text(encoding="utf-8").splitlines()
    assert fresh == submitted, "output.csv is stale: regenerate it with python code/main.py"
