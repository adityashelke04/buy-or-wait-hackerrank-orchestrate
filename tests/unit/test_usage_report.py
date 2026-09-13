from pathlib import Path

from buyorwait.evidence.usage import Usage


def test_report_has_every_required_section(tmp_path):
    u = Usage()
    u.record("rapidocr", "PP-OCRv6", 0, 0)
    u.record("rule", "multilingual-parser", 0, 0)
    u.record("cloud", "gemini-2.0-flash", 1200, 80)
    out = tmp_path / "usage_report.md"
    u.write_report(out, requests=250)
    text = out.read_text(encoding="utf-8")
    for needle in ["Provider", "Model", "Calls", "Input tokens", "Output tokens",
                   "Total tokens", "Average tokens per request", "Actual cost",
                   "Per request"]:
        assert needle in text, needle
    assert "PP-OCRv6" in text and "gemini-2.0-flash" in text
    assert "1,280" in text


def test_local_only_run_reports_zero_cost(tmp_path):
    u = Usage()
    u.record("rapidocr", "PP-OCRv6", 0, 0)
    out = tmp_path / "r.md"
    u.write_report(out, requests=250)
    assert "$0.0000" in out.read_text(encoding="utf-8")


def test_report_contains_no_credentials(tmp_path):
    u = Usage()
    u.record("cloud", "gemini-2.0-flash", 10, 10)
    out = tmp_path / "r.md"
    u.write_report(out, requests=1)
    text = out.read_text(encoding="utf-8")
    for forbidden in ["LLM_API_KEY", "api_key", "sk-", "AIza"]:
        assert forbidden not in text


def test_zero_calls_still_produces_a_valid_report(tmp_path):
    out = tmp_path / "r.md"
    Usage().write_report(out, requests=250)
    assert "Model calls: **0**" in out.read_text(encoding="utf-8")


def test_pipeline_writes_the_report_for_a_full_run(tmp_path):
    from buyorwait.pipeline import run
    report = tmp_path / "usage_report.md"
    run(Path(__file__).resolve().parents[2] / "dataset", tmp_path / "o.csv",
        usage_report_path=report)
    assert report.exists()
    assert "Requests processed: **250**" in report.read_text(encoding="utf-8")
