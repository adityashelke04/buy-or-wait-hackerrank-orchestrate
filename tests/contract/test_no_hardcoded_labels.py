"""The solution must never contain a lookup from a request/event id to an answer,
and must never read the labeled samples outside the scorer.

This is a hard requirement of the challenge, enforced mechanically rather than
by good intentions.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOLUTION = ROOT / "code"


def _solution_sources():
    return [p for p in SOLUTION.rglob("*.py") if "__pycache__" not in str(p)]


def test_solution_never_names_the_labeled_sample_file():
    needle = "sample" + "_requests"
    for path in _solution_sources():
        text = path.read_text(encoding="utf-8")
        assert needle not in text, f"{path} names the labeled samples"


def test_solution_contains_no_eval_request_ids():
    """request_26..request_275 are the evaluation set. None may appear in source."""
    pattern = re.compile(r"request_(\d+)")
    for path in _solution_sources():
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            n = int(match.group(1))
            assert not (26 <= n <= 275), f"{path} hardcodes {match.group(0)}"


def test_solution_contains_no_event_id_value_maps():
    """Event ids may appear in tests, never in solution source."""
    pattern = re.compile(r"event_\d+")
    for path in _solution_sources():
        hits = pattern.findall(path.read_text(encoding="utf-8"))
        assert not hits, f"{path} hardcodes event ids: {sorted(set(hits))[:5]}"


def test_solution_opens_only_documented_dataset_files():
    """Guards against reaching for an organizer-only file."""
    allowed = {
        "financial_profiles.csv", "financial_events.csv", "exchange_rates.csv",
        "requests.csv", "request_payment_options.csv", "messages.csv",
        "images.csv", "output.csv",
    }
    pattern = re.compile(r"[\"']([A-Za-z0-9_\-]+\.csv)[\"']")
    for path in _solution_sources():
        for name in pattern.findall(path.read_text(encoding="utf-8")):
            assert name in allowed, f"{path} references undocumented file {name}"
