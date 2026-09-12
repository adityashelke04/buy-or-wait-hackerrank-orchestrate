from pathlib import Path

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
