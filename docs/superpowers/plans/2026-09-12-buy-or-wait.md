# Buy or Wait? Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `output.csv` with one safe, well-explained financial recommendation for each of the 250 rows in `dataset/requests.csv`.

**Architecture:** A deterministic cash-flow simulator decides every number. A language model is confined to one job — turning untrusted messages and images into schema-validated `Amendment` records that adjust the forecast. Output passes a hard contract gate before it is ever written.

**Tech Stack:** Python 3.13 stdlib only for the core (`csv`, `decimal`, `datetime`, `dataclasses`), `pytest` for tests, Ollama (local, D: drive) for the default model backend, optional Google AI Studio via an OpenAI-compatible client, Tesseract for OCR cross-check.

**Spec:** `docs/superpowers/specs/2026-09-12-buy-or-wait-design.md`

## Global Constraints

- Python 3.13.3. Core pipeline uses **stdlib only** — no pandas, no numpy.
- All money is `decimal.Decimal`. Never `float`. Parse from string directly.
- Output columns, exact order: `request_id,amount_safe_to_pay,affordability_status,recommended_payment_method,payment_plan,earliest_date_for_full_payment,spending_changes_needed,decision_explanation`
- `0 <= amount_safe_to_pay <= requested_amount` always.
- Forecast horizon is exactly **90 days** from `request_date`, inclusive.
- No hardcoded labels: no `request_id`→answer or `event_id`→value map anywhere in `code/`.
- `dataset/sample_requests.csv` may be read **only** by `evaluation/score.py`.
- Secrets come from env vars only: `LLM_BACKEND`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.
- Ollama installs to `D:\ollama`; models to `D:\ollama\models`. Never `C:`.
- Deterministic: sorted iteration, `temperature=0`, `seed=0`, on-disk response cache. Two runs must be byte-identical.
- Submission ZIP under 50 MB.
- Commits are progressive: `test:` (red) → `feat:`/`fix:` (green) → `refactor:`. Never commit a broken state.

---

## File Structure

```
code/
  main.py                      CLI entry: python code/main.py
  buyorwait/
    types.py                   frozen dataclasses for every record
    money.py                   Decimal helpers, formatting
    fx.py                      dated currency conversion
    io_loaders.py              CSV -> typed records
    io_writer.py               Decision[] -> output.csv
    ledger.py                  cash-effect classification, trap resolution
    recurrence.py              recurring series detection + projection
    forecast.py                daily balance curve
    simulate.py                trough / feasibility given payments
    solver.py                  safe amount, earliest date, candidates
    changes.py                 spending-change candidates
    ranker.py                  6-level tie-break + status derivation
    explain.py                 8 explanation templates
    validate.py                contract gate
    pipeline.py                wires everything together
    evidence/
      schema.py                Amendment dataclass + JSON schema
      provider.py              LLMProvider protocol + factory
      rule_provider.py         deterministic parser (no model)
      ollama_provider.py       local backend
      cloud_provider.py        OpenAI-compatible backend
      ocr.py                   Tesseract cross-check
      extractor.py             messages/images -> Amendment[]
      cache.py                 sha256 -> response, on disk
      usage.py                 token accounting -> usage_report.md
evaluation/
  score.py                     per-field accuracy vs the 25 samples
  usage_report.md              generated
tests/
  conftest.py
  unit/  contract/  golden/  smoke/  fixtures/
```

---

## Task 1: Scaffolding, public repo, and the scoring harness

**Plain English:** Before writing any logic, we build the *measuring instrument*. We have 25 solved examples. `score.py` compares our answers to those 25 and gives a percentage per column. Everything after this is "make that number go up". We also publish the repo so every later commit is visible.

**Files:**
- Create: `requirements.txt`, `.env.example`, `pytest.ini`, `code/buyorwait/__init__.py`, `evaluation/score.py`, `tests/conftest.py`
- Modify: `.gitignore` (already secret-safe)

**Interfaces:**
- Produces: `evaluation/score.py::score(predictions_path: Path, samples_path: Path) -> ScoreReport`, where `ScoreReport` has `.per_field: dict[str, float]`, `.overall: float`, `.rows: int`.

- [ ] **Step 1: Create project scaffolding**

```bash
cd "D:/Orca/projects/hackerrank-orchestrate-september26/hackerrank-orchestrate-september26"
mkdir -p code/buyorwait/evidence evaluation tests/unit tests/contract tests/golden tests/smoke tests/fixtures
touch code/buyorwait/__init__.py code/buyorwait/evidence/__init__.py
python -m pip install pytest
```

`requirements.txt`:
```
pytest>=8.0
# Optional backends. The core pipeline runs without these.
openai>=1.0        # used only for LLM_BACKEND=cloud (Google AI Studio / Groq / OpenRouter)
pytesseract>=0.3   # used only for the image OCR cross-check
Pillow>=10.0
```

`.env.example`:
```
# Copy to .env and fill in. .env is gitignored and must never be committed.
LLM_BACKEND=ollama
LLM_MODEL=qwen2.5:3b-instruct-q4_K_M
LLM_VISION_MODEL=qwen2.5vl:3b
# Only for LLM_BACKEND=cloud (Google AI Studio):
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
pythonpath = code
addopts = -q
```

- [ ] **Step 2: Write the failing scorer test**

`tests/unit/test_score.py`:
```python
from pathlib import Path
from evaluation.score import score

ROOT = Path(__file__).resolve().parents[2]


def test_perfect_prediction_scores_one(tmp_path):
    """Feeding the samples back as predictions must score 1.0 on every field."""
    samples = ROOT / "dataset" / "sample_requests.csv"
    import csv
    out = tmp_path / "pred.csv"
    cols = ["request_id", "amount_safe_to_pay", "affordability_status",
            "recommended_payment_method", "payment_plan",
            "earliest_date_for_full_payment", "spending_changes_needed",
            "decision_explanation"]
    with open(samples, encoding="utf-8") as f, open(out, "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=cols)
        w.writeheader()
        for row in csv.DictReader(f):
            w.writerow({c: row[c] for c in cols})

    report = score(out, samples)
    assert report.rows == 25
    assert report.overall == 1.0
    assert report.per_field["affordability_status"] == 1.0
    assert report.per_field["amount_safe_to_pay"] == 1.0


def test_wrong_status_lowers_only_that_field(tmp_path):
    samples = ROOT / "dataset" / "sample_requests.csv"
    import csv
    out = tmp_path / "pred.csv"
    cols = ["request_id", "amount_safe_to_pay", "affordability_status",
            "recommended_payment_method", "payment_plan",
            "earliest_date_for_full_payment", "spending_changes_needed",
            "decision_explanation"]
    with open(samples, encoding="utf-8") as f, open(out, "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=cols)
        w.writeheader()
        for i, row in enumerate(csv.DictReader(f)):
            r = {c: row[c] for c in cols}
            if i == 0:
                r["affordability_status"] = "not_affordable"
            w.writerow(r)

    report = score(out, samples)
    assert report.per_field["affordability_status"] == 24 / 25
    assert report.per_field["amount_safe_to_pay"] == 1.0
```

- [ ] **Step 3: Run it and confirm it fails**

Run: `python -m pytest tests/unit/test_score.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evaluation'`

- [ ] **Step 4: Implement the scorer**

`evaluation/__init__.py` — empty file.

`evaluation/score.py`:
```python
"""Score predictions against the 25 labeled sample requests.

This is the project's fitness function. It is the ONLY module permitted to read
dataset/sample_requests.csv - see the spec, section 6.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

FIELDS = [
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

# amount_safe_to_pay is graded with a relative tolerance: being within 0.5% of the
# ground truth still counts. Everything else is exact string match, except the
# explanation which is graded on whether it names the right numbers.
AMOUNT_TOLERANCE = Decimal("0.005")


@dataclass
class ScoreReport:
    rows: int
    per_field: dict[str, float]
    overall: float
    misses: list[tuple[str, str, str, str]] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"Scored {self.rows} sample requests", ""]
        for name in FIELDS:
            lines.append(f"  {name:<34} {self.per_field[name]:6.1%}")
        lines.append("")
        lines.append(f"  {'OVERALL':<34} {self.overall:6.1%}")
        if self.misses:
            lines.append("")
            lines.append("Misses (request, field, expected, got):")
            for rid, fname, exp, got in self.misses[:40]:
                lines.append(f"  {rid} {fname}: expected {exp!r} got {got!r}")
        return "\n".join(lines)


def _as_decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.strip())
    except (InvalidOperation, AttributeError):
        return None


def _amounts_match(expected: str, got: str) -> bool:
    e, g = _as_decimal(expected), _as_decimal(got)
    if e is None or g is None:
        return expected.strip() == got.strip()
    if e == g:
        return True
    if e == 0:
        return g == 0
    return abs(e - g) / abs(e) <= AMOUNT_TOLERANCE


def _plans_match(expected: str, got: str) -> bool:
    """A plan matches when the dates match exactly and each amount is within tolerance."""
    e, g = expected.strip(), got.strip()
    if e == g:
        return True
    if e == "none" or g == "none":
        return False
    ep, gp = e.split("|"), g.split("|")
    if len(ep) != len(gp):
        return False
    for a, b in zip(ep, gp):
        ad, _, aa = a.partition(":")
        bd, _, ba = b.partition(":")
        if ad != bd or not _amounts_match(aa, ba):
            return False
    return True


def _explanation_match(expected: str, got: str) -> bool:
    """Graded on usefulness: the explanation must be non-empty and mention the same
    numeric facts as the reference. We compare the set of numbers named."""
    import re
    if not got.strip():
        return False
    nums = lambda s: set(re.findall(r"\d[\d,]*\.?\d*", s))
    return nums(expected) == nums(got)


MATCHERS = {
    "amount_safe_to_pay": _amounts_match,
    "payment_plan": _plans_match,
    "decision_explanation": _explanation_match,
}


def score(predictions_path: Path, samples_path: Path) -> ScoreReport:
    with open(samples_path, encoding="utf-8") as f:
        truth = {r["request_id"]: r for r in csv.DictReader(f)}
    with open(predictions_path, encoding="utf-8") as f:
        pred = {r["request_id"]: r for r in csv.DictReader(f)}

    graded = sorted(set(truth) & set(pred))
    hits = {name: 0 for name in FIELDS}
    misses: list[tuple[str, str, str, str]] = []

    for rid in graded:
        t, p = truth[rid], pred[rid]
        for name in FIELDS:
            exp, got = t.get(name, ""), p.get(name, "")
            matcher = MATCHERS.get(name, lambda a, b: a.strip() == b.strip())
            if matcher(exp, got):
                hits[name] += 1
            else:
                misses.append((rid, name, exp, got))

    n = len(graded) or 1
    per_field = {name: hits[name] / n for name in FIELDS}
    overall = sum(per_field.values()) / len(FIELDS)
    return ScoreReport(rows=len(graded), per_field=per_field, overall=overall, misses=misses)


def main() -> None:
    import sys
    root = Path(__file__).resolve().parents[1]
    pred = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "output_samples.csv"
    print(score(pred, root / "dataset" / "sample_requests.csv").render())


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `python -m pytest tests/unit/test_score.py -v`
Expected: PASS, 2 passed

- [ ] **Step 6: Add the no-hardcoding guard test**

`tests/contract/test_no_hardcoded_labels.py`:
```python
"""The solution must never contain a lookup from a request/event id to an answer,
and must never read the labeled samples outside the scorer."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOLUTION = ROOT / "code"


def _solution_sources():
    return [p for p in SOLUTION.rglob("*.py") if "__pycache__" not in str(p)]


def test_solution_never_reads_sample_requests():
    for path in _solution_sources():
        text = path.read_text(encoding="utf-8")
        assert "sample_requests" not in text, f"{path} reads the labeled samples"


def test_solution_contains_no_eval_request_ids():
    """request_26..request_275 are the evaluation set. None may appear in source."""
    pattern = re.compile(r"request_(\d+)")
    for path in _solution_sources():
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            n = int(match.group(1))
            assert not (26 <= n <= 275), f"{path} hardcodes {match.group(0)}"


def test_solution_contains_no_event_id_value_maps():
    """Event ids may be referenced in tests, never in solution source."""
    pattern = re.compile(r"event_\d+")
    for path in _solution_sources():
        hits = pattern.findall(path.read_text(encoding="utf-8"))
        assert not hits, f"{path} hardcodes event ids: {sorted(set(hits))[:5]}"
```

- [ ] **Step 7: Run it — it should pass trivially (no solution code yet)**

Run: `python -m pytest tests/contract/test_no_hardcoded_labels.py -v`
Expected: PASS, 3 passed

- [ ] **Step 8: Commit**

```bash
git add requirements.txt .env.example pytest.ini code evaluation tests
git commit -m "test: add sample scorer and no-hardcoded-labels guard

The scorer is the project fitness function: it grades predictions against the
25 labeled sample requests per field, with a 0.5% tolerance on amounts and
exact match on categorical fields. Everything downstream is measured by it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 9: Create and push the public repository**

```bash
gh repo create buy-or-wait-hackerrank-orchestrate \
  --public \
  --source=. \
  --remote=origin \
  --description "AI financial decision agent for HackerRank Orchestrate Sep 2026 - Buy or Wait?" \
  --push
gh repo view --web
```

Verify no secret was pushed:
```bash
git log --all -p | grep -nEi "sk-[A-Za-z0-9]{20}|AIza[A-Za-z0-9_-]{30}|gho_[A-Za-z0-9]{30}" && echo "SECRET FOUND - STOP" || echo "clean"
```
Expected: `clean`

---

## Task 2: Typed records and CSV loaders

**Plain English:** Raw CSV rows are strings. This task turns them into Python objects with real types — dates as `date`, money as `Decimal` — so that a typo like comparing a string to a number becomes an immediate error rather than a wrong answer. "Frozen dataclass" means the object cannot be modified after creation, which removes a whole class of bug where one part of the code silently changes data another part is relying on.

**Files:**
- Create: `code/buyorwait/types.py`, `code/buyorwait/money.py`, `code/buyorwait/io_loaders.py`
- Test: `tests/unit/test_loaders.py`, `tests/unit/test_money.py`

**Interfaces:**
- Produces:
  - `money.dec(text: str) -> Decimal | None`
  - `money.fmt_plain(value: Decimal) -> str` — minimal representation, trailing zeros stripped
  - `money.fmt_currency(code: str, value: Decimal) -> str` — `"ZAR 25,256"`
  - `types.Profile, Event, Request, PaymentOption, Message, ImageRef, Dataset`
  - `io_loaders.load_dataset(dataset_dir: Path) -> Dataset`
  - `Dataset.profiles: dict[str, Profile]`, `.events_by_user: dict[str, list[Event]]`, `.requests: list[Request]`, `.options_by_request: dict[str, list[PaymentOption]]`, `.messages_by_user: dict[str, list[Message]]`, `.images_by_event: dict[str, ImageRef]`, `.rates: list[Rate]`

- [ ] **Step 1: Write the failing money test**

`tests/unit/test_money.py`:
```python
from decimal import Decimal
from buyorwait.money import dec, fmt_plain, fmt_currency


def test_dec_parses_from_string_without_float_error():
    assert dec("0.1") + dec("0.2") == Decimal("0.3")


def test_dec_returns_none_for_blank():
    assert dec("") is None
    assert dec("   ") is None


def test_fmt_plain_strips_trailing_zeros():
    assert fmt_plain(Decimal("25256.00")) == "25256"
    assert fmt_plain(Decimal("620.40")) == "620.4"
    assert fmt_plain(Decimal("17229139.20")) == "17229139.2"
    assert fmt_plain(Decimal("603.30")) == "603.3"


def test_fmt_currency_uses_thousands_separators():
    assert fmt_currency("ZAR", Decimal("25256")) == "ZAR 25,256"
    assert fmt_currency("IDR", Decimal("29158400")) == "IDR 29,158,400"
    assert fmt_currency("EUR", Decimal("1543.35")) == "EUR 1,543.35"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/unit/test_money.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'buyorwait'`

- [ ] **Step 3: Implement money.py**

`code/buyorwait/money.py`:
```python
"""Decimal money helpers.

Every amount in this project is a Decimal parsed directly from its string form.
Floats are never used: 0.1 + 0.2 != 0.3 in binary floating point, and a cent of
drift is enough to flip an affordability decision.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

CENTS = Decimal("0.01")


def dec(text: str | None) -> Decimal | None:
    """Parse a CSV cell into a Decimal. Blank or unparseable returns None."""
    if text is None:
        return None
    s = text.strip().replace(",", "")
    if not s:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def q(value: Decimal) -> Decimal:
    """Round half-up to 2 decimal places for internal arithmetic."""
    return value.quantize(CENTS, rounding=ROUND_HALF_UP)


def fmt_plain(value: Decimal) -> str:
    """Minimal representation: 25256.00 -> '25256', 620.40 -> '620.4'.

    Matches the formatting seen in the labeled samples.
    """
    v = value.quantize(CENTS, rounding=ROUND_HALF_UP).normalize()
    sign, digits, exp = v.as_tuple()
    if isinstance(exp, int) and exp > 0:          # normalize() may give 1E+3
        v = v.quantize(Decimal(1))
    return format(v, "f")


def fmt_currency(code: str, value: Decimal) -> str:
    """Human formatting for explanations: 'ZAR 25,256', 'EUR 1,543.35'."""
    v = value.quantize(CENTS, rounding=ROUND_HALF_UP).normalize()
    whole = int(abs(v))
    frac = abs(v) - whole
    body = f"{whole:,}"
    if frac:
        body += format(frac, "f")[1:]
    return f"{code} {'-' if v < 0 else ''}{body}"
```

- [ ] **Step 4: Run the money tests**

Run: `python -m pytest tests/unit/test_money.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Write the failing loader test**

`tests/unit/test_loaders.py`:
```python
from datetime import date
from decimal import Decimal
from pathlib import Path

from buyorwait.io_loaders import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"


def test_loads_every_file_with_expected_row_counts():
    ds = load_dataset(DATASET)
    assert len(ds.profiles) == 275
    assert len(ds.requests) == 250
    assert sum(len(v) for v in ds.events_by_user.values()) == 25342
    assert sum(len(v) for v in ds.options_by_request.values()) == 790
    assert sum(len(v) for v in ds.messages_by_user.values()) == 215
    assert len(ds.images_by_event) == 16
    assert len(ds.rates) == 134


def test_types_are_parsed_not_strings():
    ds = load_dataset(DATASET)
    p = ds.profiles["user_01"]
    assert p.home_currency == "ZAR"
    assert p.current_available_balance == Decimal("58481.1")
    assert p.minimum_balance_to_keep == Decimal("18000")
    assert p.payment_methods_user_will_consider == ("full_payment",)
    assert p.max_installment_months is None          # blank means no installments

    r = ds.requests[0]
    assert isinstance(r.request_date, date)
    assert isinstance(r.requested_amount, Decimal)
    assert isinstance(r.allows_partial_payment, bool)


def test_blank_event_amount_is_none_not_zero():
    ds = load_dataset(DATASET)
    blanks = [e for evs in ds.events_by_user.values() for e in evs if e.amount is None]
    assert len(blanks) == 16
    for e in blanks:
        assert e.event_id in ds.images_by_event, f"{e.event_id} has no image to resolve it"


def test_referential_integrity():
    ds = load_dataset(DATASET)
    for r in ds.requests:
        assert r.user_id in ds.profiles
        assert r.request_id in ds.options_by_request


def test_images_resolve_to_existing_png_files():
    ds = load_dataset(DATASET)
    for ref in ds.images_by_event.values():
        assert ref.path(DATASET).exists(), f"missing {ref.image_id}.png"
```

- [ ] **Step 6: Run it and confirm it fails**

Run: `python -m pytest tests/unit/test_loaders.py -v`
Expected: FAIL — `No module named 'buyorwait.io_loaders'`

- [ ] **Step 7: Implement types.py**

`code/buyorwait/types.py`:
```python
"""Typed records for every dataset row.

Frozen dataclasses: once built, a record cannot be mutated. Any adjustment
(an amendment from a message, a resolved image amount) produces a NEW record via
dataclasses.replace, so the original dataset values stay auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...]
    expense_categories_to_protect: tuple[str, ...]
    expense_categories_user_is_willing_to_reduce: tuple[str, ...]
    expense_categories_user_is_willing_to_stop: tuple[str, ...]
    payment_methods_user_will_consider: tuple[str, ...]
    max_installment_months: int | None


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str                 # debit | credit | non_cash
    amount: Decimal | None         # None means "resolve from the linked image"
    currency: str
    event_date: date
    settlement_date: date
    status: str                    # settled|pending|scheduled|cancelled|failed|unrealized
    linked_event_id: str | None
    flexibility: str               # fixed|reducible|stoppable|reducible_or_stoppable
    minimum_allowed_amount: Decimal | None

    @property
    def is_flexible(self) -> bool:
        return self.flexibility != "fixed"

    @property
    def can_stop(self) -> bool:
        return self.flexibility in ("stoppable", "reducible_or_stoppable")

    @property
    def can_reduce(self) -> bool:
        return self.flexibility in ("reducible", "reducible_or_stoppable")


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str            # full_payment | installments
    payment_amount: Decimal
    payment_amount_text: str       # verbatim from the CSV, written back unchanged
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: int | None
    financing_fee: Decimal
    total_payable_amount: Decimal

    @property
    def sort_key(self) -> int:
        return int(self.payment_option_id.rsplit("_", 1)[-1])


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    sent_at: str
    source_type: str
    message_text: str


@dataclass(frozen=True)
class ImageRef:
    image_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None

    def path(self, dataset_dir: Path) -> Path:
        return dataset_dir / "media" / "images" / f"{self.image_id}.png"


@dataclass(frozen=True)
class Rate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: Decimal


@dataclass(frozen=True)
class Dataset:
    profiles: dict[str, Profile]
    events_by_user: dict[str, list[Event]]
    requests: list[Request]
    options_by_request: dict[str, list[PaymentOption]]
    messages_by_user: dict[str, list[Message]]
    images_by_event: dict[str, ImageRef]
    rates: list[Rate]
    events_by_id: dict[str, Event] = field(default_factory=dict)
```

- [ ] **Step 8: Implement io_loaders.py**

`code/buyorwait/io_loaders.py`:
```python
"""Read the nine dataset CSVs into typed records.

Fails loud: a malformed required field raises rather than silently defaulting.
Only the files documented in the problem statement are opened.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from pathlib import Path

from .money import dec
from .types import (Dataset, Event, ImageRef, Message, PaymentOption, Profile,
                    Rate, Request)


def _rows(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def _d(text: str) -> date:
    return date.fromisoformat(text.strip())


def _opt_d(text: str) -> date | None:
    t = (text or "").strip()
    return date.fromisoformat(t) if t else None


def _list(text: str) -> tuple[str, ...]:
    t = (text or "").strip()
    return tuple(p for p in t.split("|") if p) if t else ()


def _opt_int(text: str) -> int | None:
    t = (text or "").strip()
    return int(t) if t else None


def load_dataset(dataset_dir: Path) -> Dataset:
    profiles = {}
    for r in _rows(dataset_dir / "financial_profiles.csv"):
        profiles[r["user_id"]] = Profile(
            user_id=r["user_id"],
            home_currency=r["home_currency"].strip(),
            current_available_balance=dec(r["current_available_balance"]),
            minimum_balance_to_keep=dec(r["minimum_balance_to_keep"]),
            financial_priorities=_list(r["financial_priorities"]),
            expense_categories_to_protect=_list(r["expense_categories_to_protect"]),
            expense_categories_user_is_willing_to_reduce=_list(
                r["expense_categories_user_is_willing_to_reduce"]),
            expense_categories_user_is_willing_to_stop=_list(
                r["expense_categories_user_is_willing_to_stop"]),
            payment_methods_user_will_consider=_list(
                r["payment_methods_user_will_consider"]),
            max_installment_months=_opt_int(r["max_installment_months"]),
        )

    events_by_user: dict[str, list[Event]] = defaultdict(list)
    events_by_id: dict[str, Event] = {}
    for r in _rows(dataset_dir / "financial_events.csv"):
        e = Event(
            event_id=r["event_id"],
            user_id=r["user_id"],
            event_type=r["event_type"].strip(),
            description=r["description"].strip(),
            category=r["category"].strip(),
            direction=r["direction"].strip(),
            amount=dec(r["amount"]),
            currency=r["currency"].strip(),
            event_date=_d(r["event_date"]),
            settlement_date=_d(r["settlement_date"] or r["event_date"]),
            status=r["status"].strip(),
            linked_event_id=(r["linked_event_id"].strip() or None),
            flexibility=r["flexibility"].strip(),
            minimum_allowed_amount=dec(r["minimum_allowed_amount"]),
        )
        events_by_user[e.user_id].append(e)
        events_by_id[e.event_id] = e
    for evs in events_by_user.values():
        evs.sort(key=lambda e: (e.settlement_date, e.event_id))

    requests = [
        Request(
            request_id=r["request_id"],
            user_id=r["user_id"],
            request_date=_d(r["request_date"]),
            request_type=r["request_type"].strip(),
            requested_amount=dec(r["requested_amount"]),
            desired_completion_date=_d(r["desired_completion_date"]),
            allows_partial_payment=r["allows_partial_payment"].strip().lower() == "true",
            request_text=r["request_text"],
        )
        for r in _rows(dataset_dir / "requests.csv")
    ]
    requests.sort(key=lambda r: int(r.request_id.rsplit("_", 1)[-1]))

    options_by_request: dict[str, list[PaymentOption]] = defaultdict(list)
    for r in _rows(dataset_dir / "request_payment_options.csv"):
        o = PaymentOption(
            payment_option_id=r["payment_option_id"],
            request_id=r["request_id"],
            payment_method=r["payment_method"].strip(),
            payment_amount=dec(r["payment_amount"]),
            payment_amount_text=r["payment_amount"].strip(),
            number_of_payments=int(r["number_of_payments"]),
            first_payment_date=_d(r["first_payment_date"]),
            payment_frequency_days=_opt_int(r["payment_frequency_days"]),
            financing_fee=dec(r["financing_fee"]) or dec("0"),
            total_payable_amount=dec(r["total_payable_amount"]),
        )
        options_by_request[o.request_id].append(o)
    for opts in options_by_request.values():
        opts.sort(key=lambda o: o.sort_key)

    messages_by_user: dict[str, list[Message]] = defaultdict(list)
    for r in _rows(dataset_dir / "messages.csv"):
        m = Message(
            message_id=r["message_id"],
            user_id=r["user_id"],
            request_id=(r["request_id"].strip() or None),
            related_event_id=(r["related_event_id"].strip() or None),
            sent_at=r["sent_at"].strip(),
            source_type=r["source_type"].strip(),
            message_text=r["message_text"],
        )
        messages_by_user[m.user_id].append(m)
    for msgs in messages_by_user.values():
        msgs.sort(key=lambda m: (m.sent_at, m.message_id))

    images_by_event: dict[str, ImageRef] = {}
    for r in _rows(dataset_dir / "images.csv"):
        ref = ImageRef(
            image_id=r["image_id"],
            user_id=r["user_id"],
            request_id=(r["request_id"].strip() or None),
            related_event_id=(r["related_event_id"].strip() or None),
        )
        if ref.related_event_id:
            images_by_event[ref.related_event_id] = ref

    rates = [
        Rate(
            rate_date=_d(r["rate_date"]),
            from_currency=r["from_currency"].strip(),
            to_currency=r["to_currency"].strip(),
            rate=dec(r["rate"]),
        )
        for r in _rows(dataset_dir / "exchange_rates.csv")
    ]

    return Dataset(
        profiles=profiles,
        events_by_user=dict(events_by_user),
        requests=requests,
        options_by_request=dict(options_by_request),
        messages_by_user=dict(messages_by_user),
        images_by_event=images_by_event,
        rates=rates,
        events_by_id=events_by_id,
    )
```

- [ ] **Step 9: Run the loader tests**

Run: `python -m pytest tests/unit/test_loaders.py -v`
Expected: PASS, 5 passed

- [ ] **Step 10: Commit**

```bash
git add code/buyorwait/types.py code/buyorwait/money.py code/buyorwait/io_loaders.py tests/unit/test_money.py tests/unit/test_loaders.py
git commit -m "feat: typed dataset loaders with Decimal money

Turns the nine CSVs into frozen dataclasses with real types - dates as date,
money as Decimal parsed from string. Blank event amounts load as None, never
zero, and every one is asserted to have a linked image that can resolve it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: Currency conversion

**Plain English:** Some users are paid in USD but live in INR. Every such amount has to be converted using the rate *for the date it settles*, not today's rate. If a rate for an exact date is missing, we use the most recent earlier rate — never a later one, because that would be using information from the future.

**Files:**
- Create: `code/buyorwait/fx.py`
- Test: `tests/unit/test_fx.py`

**Interfaces:**
- Produces: `fx.RateTable(rates: list[Rate])` with `.convert(amount: Decimal, from_ccy: str, to_ccy: str, on: date) -> Decimal`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_fx.py`:
```python
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from buyorwait.fx import RateTable
from buyorwait.io_loaders import load_dataset
from buyorwait.types import Rate

ROOT = Path(__file__).resolve().parents[2]

RATES = [
    Rate(date(2024, 1, 15), "USD", "INR", Decimal("83")),
    Rate(date(2024, 2, 15), "USD", "INR", Decimal("84")),
]


def test_same_currency_is_identity():
    t = RateTable(RATES)
    assert t.convert(Decimal("100"), "INR", "INR", date(2024, 1, 15)) == Decimal("100")


def test_exact_date_uses_that_rate():
    t = RateTable(RATES)
    assert t.convert(Decimal("100"), "USD", "INR", date(2024, 2, 15)) == Decimal("8400")


def test_missing_date_uses_most_recent_earlier_rate_never_a_future_one():
    t = RateTable(RATES)
    assert t.convert(Decimal("100"), "USD", "INR", date(2024, 1, 20)) == Decimal("8300")


def test_before_all_rates_falls_back_to_earliest():
    t = RateTable(RATES)
    assert t.convert(Decimal("100"), "USD", "INR", date(2023, 1, 1)) == Decimal("8300")


def test_reverse_direction_inverts_when_no_direct_pair():
    t = RateTable(RATES)
    got = t.convert(Decimal("8300"), "INR", "USD", date(2024, 1, 15))
    assert got == pytest.approx(Decimal("100"), abs=Decimal("0.01"))


def test_every_real_foreign_event_can_be_converted():
    """Guards against a missing rate silently becoming a zero."""
    ds = load_dataset(ROOT / "dataset")
    t = RateTable(ds.rates)
    for user_id, events in ds.events_by_user.items():
        home = ds.profiles[user_id].home_currency
        for e in events:
            if e.currency != home and e.amount is not None:
                out = t.convert(e.amount, e.currency, home, e.settlement_date)
                assert out > 0, f"{e.event_id} converted to {out}"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/unit/test_fx.py -v`
Expected: FAIL — `No module named 'buyorwait.fx'`

- [ ] **Step 3: Implement fx.py**

`code/buyorwait/fx.py`:
```python
"""Dated currency conversion.

Rule: use the rate for the settlement date. If that exact date has no row, use
the most recent EARLIER row. Never a later one - that would be using information
from the future to decide a past cash effect.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from datetime import date
from decimal import Decimal

from .types import Rate


class RateTable:
    def __init__(self, rates: list[Rate]) -> None:
        self._by_pair: dict[tuple[str, str], list[tuple[date, Decimal]]] = defaultdict(list)
        for r in rates:
            self._by_pair[(r.from_currency, r.to_currency)].append((r.rate_date, r.rate))
        for series in self._by_pair.values():
            series.sort(key=lambda t: t[0])

    def _lookup(self, frm: str, to: str, on: date) -> Decimal | None:
        series = self._by_pair.get((frm, to))
        if not series:
            return None
        dates = [d for d, _ in series]
        i = bisect_right(dates, on) - 1
        if i < 0:
            i = 0                       # before the first rate: use the earliest
        return series[i][1]

    def convert(self, amount: Decimal, frm: str, to: str, on: date) -> Decimal:
        if frm == to:
            return amount
        direct = self._lookup(frm, to, on)
        if direct is not None:
            return amount * direct
        inverse = self._lookup(to, frm, on)
        if inverse is not None and inverse != 0:
            return amount / inverse
        raise KeyError(f"no exchange rate for {frm}->{to} on {on}")
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/unit/test_fx.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add code/buyorwait/fx.py tests/unit/test_fx.py
git commit -m "feat: dated currency conversion with no-future-rate rule

Converts on the settlement date, falling back to the most recent earlier rate
and never a later one. A test walks all 140 foreign-currency events in the real
dataset and asserts each converts to a positive amount, so a missing rate can
never silently become zero.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: The ledger — cash-effect classification and the six traps

**Plain English:** This is where most competitors will lose points. The event log deliberately contains the same money twice in different forms: a card authorization that was cancelled *and* the real charge that settled; a purchase *and* its refund; a failed payment *and* its retry. Counting both doubles the money. This module decides, for every row, whether it moves real cash — and resolves the traps before anything downstream sees them.

**Files:**
- Create: `code/buyorwait/ledger.py`
- Test: `tests/unit/test_ledger.py`

**Interfaces:**
- Produces:
  - `ledger.CashEffect` enum: `COUNT`, `RESERVE`, `IGNORE`, `NEEDS_AMOUNT`
  - `ledger.classify(event: Event) -> CashEffect`
  - `ledger.resolve_links(events: list[Event]) -> list[Event]` — drops superseded rows
  - `ledger.LedgerView(events, profile, rates)` with `.future_cashflows(on_or_after: date) -> list[tuple[date, Decimal]]` (positive = credit, negative = debit, in home currency)

- [ ] **Step 1: Write the failing classification tests**

`tests/unit/test_ledger.py`:
```python
from datetime import date
from decimal import Decimal

import pytest

from buyorwait.ledger import CashEffect, classify, resolve_links
from buyorwait.types import Event


def ev(**kw) -> Event:
    base = dict(
        event_id="event_x", user_id="user_x", event_type="expense",
        description="d", category="groceries", direction="debit",
        amount=Decimal("100"), currency="EUR",
        event_date=date(2024, 1, 1), settlement_date=date(2024, 1, 1),
        status="settled", linked_event_id=None, flexibility="fixed",
        minimum_allowed_amount=None,
    )
    base.update(kw)
    return Event(**base)


def test_settled_debit_counts():
    assert classify(ev(status="settled", direction="debit")) is CashEffect.COUNT


def test_scheduled_credit_counts_confirmed_salary():
    e = ev(status="scheduled", direction="credit", event_type="income", category="salary")
    assert classify(e) is CashEffect.COUNT


def test_pending_debit_is_reserved():
    """Money is committed even though it has not settled."""
    assert classify(ev(status="pending", direction="debit")) is CashEffect.RESERVE


def test_pending_credit_is_ignored():
    """Trap 3: never count a credit that has not landed."""
    e = ev(status="pending", direction="credit", event_type="refund")
    assert classify(e) is CashEffect.IGNORE


def test_cancelled_is_ignored():
    assert classify(ev(status="cancelled")) is CashEffect.IGNORE


def test_failed_is_ignored():
    assert classify(ev(status="failed")) is CashEffect.IGNORE


def test_unrealized_non_cash_valuation_is_ignored():
    """Trap 4: an investment's paper value is not spendable cash."""
    e = ev(status="unrealized", direction="non_cash", event_type="investment_valuation")
    assert classify(e) is CashEffect.IGNORE


def test_blank_amount_needs_resolution_and_is_never_zero():
    """Trap 6."""
    assert classify(ev(amount=None)) is CashEffect.NEEDS_AMOUNT


def test_resolve_links_drops_cancelled_authorization_keeping_settlement():
    """Trap 1: an authorization that was cancelled, plus the charge that settled."""
    auth = ev(event_id="event_a", status="cancelled", amount=Decimal("816.20"))
    settled = ev(event_id="event_b", status="settled", amount=Decimal("816.20"),
                 linked_event_id="event_a")
    out = resolve_links([auth, settled])
    assert [e.event_id for e in out] == ["event_b"]


def test_resolve_links_keeps_both_sides_of_a_charge_and_refund():
    """Trap 2: they net to zero, so BOTH must survive - dropping one is wrong."""
    charge = ev(event_id="event_a", direction="debit", amount=Decimal("583"))
    refund = ev(event_id="event_b", direction="credit", event_type="refund",
                amount=Decimal("583"), linked_event_id="event_a")
    out = resolve_links([charge, refund])
    assert {e.event_id for e in out} == {"event_a", "event_b"}


def test_resolve_links_drops_failed_payment_keeping_the_retry():
    """Trap 5."""
    failed = ev(event_id="event_a", status="failed", amount=Decimal("13800"))
    retry = ev(event_id="event_b", status="scheduled", amount=Decimal("13800"),
               linked_event_id="event_a")
    out = resolve_links([failed, retry])
    assert [e.event_id for e in out] == ["event_b"]
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/unit/test_ledger.py -v`
Expected: FAIL — `No module named 'buyorwait.ledger'`

- [ ] **Step 3: Implement the classification half of ledger.py**

`code/buyorwait/ledger.py`:
```python
"""Decide what each event does to real, spendable cash.

The dataset deliberately represents the same money more than once. The six traps
documented in the spec are resolved here, before anything downstream sees the
events, so that no later module can double-count.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum

from .fx import RateTable
from .types import Event, Profile


class CashEffect(Enum):
    COUNT = "count"                 # moves cash on its settlement date
    RESERVE = "reserve"             # committed but not settled: hold the money back
    IGNORE = "ignore"               # never touches spendable cash
    NEEDS_AMOUNT = "needs_amount"   # blank amount; must be resolved from an image


# Statuses that never move spendable cash.
_DEAD_STATUSES = frozenset({"cancelled", "failed", "unrealized"})


def classify(event: Event) -> CashEffect:
    if event.direction == "non_cash" or event.status in _DEAD_STATUSES:
        return CashEffect.IGNORE
    if event.amount is None:
        return CashEffect.NEEDS_AMOUNT
    if event.status == "pending":
        # Pending debits are committed money; pending credits have not landed.
        return CashEffect.RESERVE if event.direction == "debit" else CashEffect.IGNORE
    if event.status in ("settled", "scheduled"):
        return CashEffect.COUNT
    return CashEffect.IGNORE


def resolve_links(events: list[Event]) -> list[Event]:
    """Drop rows superseded by a linked successor.

    A row is superseded when a later row points at it via linked_event_id AND the
    earlier row is a dead status (cancelled authorization, failed payment). A
    refund linked to a charge supersedes nothing: both are real cash movements
    that happen to net out.
    """
    superseded: set[str] = set()
    for e in events:
        if e.linked_event_id and e.status not in _DEAD_STATUSES:
            superseded.add(e.linked_event_id)
    return [
        e for e in events
        if not (e.event_id in superseded and e.status in _DEAD_STATUSES)
    ]
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/unit/test_ledger.py -v`
Expected: PASS, 11 passed

- [ ] **Step 5: Write the failing LedgerView test**

Append to `tests/unit/test_ledger.py`:
```python
from buyorwait.fx import RateTable
from buyorwait.ledger import LedgerView
from buyorwait.types import Profile, Rate


def prof(**kw) -> Profile:
    base = dict(
        user_id="user_x", home_currency="EUR",
        current_available_balance=Decimal("1000"),
        minimum_balance_to_keep=Decimal("200"),
        financial_priorities=(), expense_categories_to_protect=(),
        expense_categories_user_is_willing_to_reduce=(),
        expense_categories_user_is_willing_to_stop=(),
        payment_methods_user_will_consider=("full_payment",),
        max_installment_months=None,
    )
    base.update(kw)
    return Profile(**base)


EMPTY_RATES = RateTable([])


def test_history_before_request_date_is_not_replayed():
    """current_available_balance already reflects settled history. Re-applying it
    would double-count - the single most dangerous bug in this system."""
    past = ev(event_id="event_p", settlement_date=date(2024, 1, 1), amount=Decimal("500"))
    future = ev(event_id="event_f", settlement_date=date(2024, 3, 1), amount=Decimal("50"))
    view = LedgerView([past, future], prof(), EMPTY_RATES)
    flows = view.future_cashflows(date(2024, 2, 1))
    assert flows == [(date(2024, 3, 1), Decimal("-50"))]


def test_pending_debit_before_request_date_is_still_reserved():
    """It has not settled, so it is NOT in the balance yet, even though it is old."""
    pending = ev(event_id="event_p", status="pending",
                 settlement_date=date(2024, 1, 20), amount=Decimal("75"))
    view = LedgerView([pending], prof(), EMPTY_RATES)
    flows = view.future_cashflows(date(2024, 2, 1))
    assert flows == [(date(2024, 2, 1), Decimal("-75"))]


def test_credit_is_positive_and_debit_is_negative():
    credit = ev(event_id="event_c", direction="credit", event_type="income",
                category="salary", status="scheduled",
                settlement_date=date(2024, 2, 15), amount=Decimal("900"))
    debit = ev(event_id="event_d", settlement_date=date(2024, 2, 10), amount=Decimal("40"))
    view = LedgerView([credit, debit], prof(), EMPTY_RATES)
    assert view.future_cashflows(date(2024, 2, 1)) == [
        (date(2024, 2, 10), Decimal("-40")),
        (date(2024, 2, 15), Decimal("900")),
    ]


def test_foreign_currency_is_converted_to_home_currency():
    rates = RateTable([Rate(date(2024, 2, 15), "USD", "EUR", Decimal("0.9"))])
    credit = ev(event_id="event_c", direction="credit", event_type="income",
                category="salary", status="scheduled", currency="USD",
                settlement_date=date(2024, 2, 15), amount=Decimal("1000"))
    view = LedgerView([credit], prof(home_currency="EUR"), rates)
    assert view.future_cashflows(date(2024, 2, 1)) == [(date(2024, 2, 15), Decimal("900.0"))]
```

- [ ] **Step 6: Run it and confirm it fails**

Run: `python -m pytest tests/unit/test_ledger.py -v`
Expected: FAIL — `ImportError: cannot import name 'LedgerView'`

- [ ] **Step 7: Implement LedgerView**

Append to `code/buyorwait/ledger.py`:
```python
class LedgerView:
    """The cash-relevant view of one user's event history.

    Anchor rule: current_available_balance is the balance ON the request date and
    already includes everything that settled on or before it. Only events settling
    AFTER the request date move the curve - except reserved pending debits, which
    have not settled and so are charged immediately.
    """

    def __init__(self, events: list[Event], profile: Profile, rates: RateTable) -> None:
        self.profile = profile
        self.rates = rates
        self.events = resolve_links(sorted(events, key=lambda e: (e.settlement_date, e.event_id)))

    def _home(self, event: Event) -> Decimal:
        amount = self.rates.convert(
            event.amount, event.currency, self.profile.home_currency, event.settlement_date
        )
        return amount if event.direction == "credit" else -amount

    def future_cashflows(self, on_or_after: date) -> list[tuple[date, Decimal]]:
        flows: list[tuple[date, Decimal]] = []
        for e in self.events:
            effect = classify(e)
            if effect is CashEffect.RESERVE:
                # Charge it on the request date at the latest: the money is gone.
                when = max(e.settlement_date, on_or_after)
                flows.append((when, self._home(e)))
            elif effect is CashEffect.COUNT and e.settlement_date > on_or_after:
                flows.append((e.settlement_date, self._home(e)))
        flows.sort(key=lambda t: t[0])
        return flows
```

- [ ] **Step 8: Run the tests**

Run: `python -m pytest tests/unit/test_ledger.py -v`
Expected: PASS, 15 passed

- [ ] **Step 9: Commit**

```bash
git add code/buyorwait/ledger.py tests/unit/test_ledger.py
git commit -m "feat: ledger cash-effect classification resolving all six traps

Classifies every event as COUNT / RESERVE / IGNORE / NEEDS_AMOUNT and drops rows
superseded by a linked successor. Covers cancelled authorizations, charge and
refund pairs that must both survive, pending credits, unrealized valuations,
failed payments with retries, and blank amounts.

Enforces the balance anchor: settled history before the request date is already
inside current_available_balance and is never replayed.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: Recurrence detection and projection

**Plain English:** The event log stops at the request date, but rent will still be due next month. This module looks at history and works out which expenses repeat — rent on the 1st, salary on the 15th, groceries every Tuesday — then projects them forward 90 days. The rule "only if history supports it" matters: we need at least three sightings before we believe something recurs, otherwise a one-off holiday purchase would get projected forever.

For expenses that repeat but vary in size (groceries, transport, dining), we must forecast *conservatively* — assume the higher end, because underestimating spending is what makes a recommendation unsafe. Which exact statistic is right is a calibration question, so it is a named, swappable strategy rather than a hardcoded choice.

**Files:**
- Create: `code/buyorwait/recurrence.py`
- Test: `tests/unit/test_recurrence.py`

**Interfaces:**
- Produces:
  - `recurrence.Series` frozen dataclass: `.category`, `.description`, `.template_event_id`, `.cadence` (`"monthly"`/`"weekly"`), `.anchor` (day-of-month or weekday int), `.amount: Decimal`, `.direction`, `.flexibility`, `.minimum_allowed_amount`, `.occurrences(start: date, end: date) -> list[date]`
  - `recurrence.detect(events: list[Event], profile: Profile, rates: RateTable, as_of: date, estimator: str = "p75") -> list[Series]`
  - `recurrence.ESTIMATORS: dict[str, Callable[[list[Decimal]], Decimal]]` with keys `"last"`, `"mean"`, `"median"`, `"p75"`, `"max"`, `"max3"`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_recurrence.py`:
```python
from datetime import date
from decimal import Decimal

from buyorwait.fx import RateTable
from buyorwait.recurrence import ESTIMATORS, detect
from buyorwait.types import Event, Profile

EMPTY_RATES = RateTable([])


def ev(event_id, day, amount, category="rent", description="Apartment rent transfer",
       direction="debit", status="settled", flexibility="fixed", minimum=None):
    return Event(
        event_id=event_id, user_id="user_x", event_type="expense",
        description=description, category=category, direction=direction,
        amount=Decimal(amount), currency="EUR",
        event_date=day, settlement_date=day, status=status,
        linked_event_id=None, flexibility=flexibility,
        minimum_allowed_amount=Decimal(minimum) if minimum else None,
    )


def prof():
    return Profile(
        user_id="user_x", home_currency="EUR",
        current_available_balance=Decimal("1000"),
        minimum_balance_to_keep=Decimal("200"),
        financial_priorities=(), expense_categories_to_protect=(),
        expense_categories_user_is_willing_to_reduce=(),
        expense_categories_user_is_willing_to_stop=(),
        payment_methods_user_will_consider=("full_payment",),
        max_installment_months=None,
    )


def test_monthly_series_detected_on_day_of_month():
    events = [
        ev("event_1", date(2024, 1, 1), "467.50"),
        ev("event_2", date(2024, 2, 1), "467.50"),
        ev("event_3", date(2024, 3, 1), "467.50"),
    ]
    series = detect(events, prof(), EMPTY_RATES, as_of=date(2024, 3, 5))
    assert len(series) == 1
    s = series[0]
    assert s.cadence == "monthly"
    assert s.anchor == 1
    assert s.amount == Decimal("467.50")
    assert s.occurrences(date(2024, 3, 5), date(2024, 6, 3)) == [
        date(2024, 4, 1), date(2024, 5, 1), date(2024, 6, 1),
    ]


def test_weekly_series_detected_on_weekday():
    days = [date(2024, 1, 2), date(2024, 1, 9), date(2024, 1, 16), date(2024, 1, 23)]
    events = [ev(f"event_{i}", d, "50", category="transport",
                 description="Commuter pass") for i, d in enumerate(days)]
    series = detect(events, prof(), EMPTY_RATES, as_of=date(2024, 1, 25))
    assert len(series) == 1
    assert series[0].cadence == "weekly"
    assert series[0].anchor == 1                      # Tuesday
    occ = series[0].occurrences(date(2024, 1, 25), date(2024, 2, 8))
    assert occ == [date(2024, 1, 30), date(2024, 2, 6)]


def test_two_observations_is_not_enough_to_declare_recurrence():
    events = [
        ev("event_1", date(2024, 1, 1), "467.50"),
        ev("event_2", date(2024, 2, 1), "467.50"),
    ]
    assert detect(events, prof(), EMPTY_RATES, as_of=date(2024, 2, 5)) == []


def test_one_off_purchase_is_never_projected():
    events = [ev("event_1", date(2024, 1, 15), "2000", category="shopping",
                 description="Laptop purchase")]
    assert detect(events, prof(), EMPTY_RATES, as_of=date(2024, 2, 1)) == []


def test_variable_amounts_use_the_conservative_estimator():
    days = [date(2024, 1, 2), date(2024, 1, 9), date(2024, 1, 16), date(2024, 1, 23)]
    amounts = ["40", "60", "50", "70"]
    events = [ev(f"event_{i}", d, a, category="groceries", description="Supermarket basket")
              for i, (d, a) in enumerate(zip(days, amounts))]
    s_max = detect(events, prof(), EMPTY_RATES, date(2024, 1, 25), estimator="max")[0]
    s_mean = detect(events, prof(), EMPTY_RATES, date(2024, 1, 25), estimator="mean")[0]
    assert s_max.amount == Decimal("70")
    assert s_mean.amount == Decimal("55")
    assert s_max.amount > s_mean.amount, "max must be the more conservative of the two"


def test_flexibility_metadata_is_carried_onto_the_series():
    days = [date(2024, 1, 10), date(2024, 2, 10), date(2024, 3, 10)]
    events = [ev(f"event_{i}", d, "47", category="streaming",
                 description="Streaming subscription",
                 flexibility="reducible_or_stoppable", minimum="23.5")
              for i, d in enumerate(days)]
    s = detect(events, prof(), EMPTY_RATES, as_of=date(2024, 3, 15))[0]
    assert s.flexibility == "reducible_or_stoppable"
    assert s.minimum_allowed_amount == Decimal("23.5")
    assert s.template_event_id == "event_2", "must point at the most recent occurrence"


def test_cancelled_and_failed_events_never_form_a_series():
    days = [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]
    events = [ev(f"event_{i}", d, "100", status="cancelled") for i, d in enumerate(days)]
    assert detect(events, prof(), EMPTY_RATES, as_of=date(2024, 3, 5)) == []


def test_estimators_are_all_registered():
    assert set(ESTIMATORS) == {"last", "mean", "median", "p75", "max", "max3"}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/unit/test_recurrence.py -v`
Expected: FAIL — `No module named 'buyorwait.recurrence'`

- [ ] **Step 3: Implement recurrence.py**

`code/buyorwait/recurrence.py`:
```python
"""Detect recurring income and expenses from history, then project them forward.

Two rules from the problem statement drive this module:

  "Detect recurrence only when history supports it."  -> at least MIN_OBSERVATIONS
  "Forecast essential variable spending conservatively." -> a swappable estimator,
      chosen by calibration against the labeled samples rather than by guesswork.
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Callable

from .fx import RateTable
from .ledger import CashEffect, classify, resolve_links
from .types import Event, Profile

MIN_OBSERVATIONS = 3
LOOKBACK_DAYS = 180          # how much history forms a series
RECENT_WINDOW = 6            # how many recent occurrences the estimator sees


def _p75(values: list[Decimal]) -> Decimal:
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * Decimal("0.75")
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


ESTIMATORS: dict[str, Callable[[list[Decimal]], Decimal]] = {
    "last": lambda v: v[-1],
    "mean": lambda v: sum(v) / len(v),
    "median": lambda v: Decimal(statistics.median(sorted(v))),
    "p75": _p75,
    "max": lambda v: max(v),
    "max3": lambda v: max(v[-3:]),
}


@dataclass(frozen=True)
class Series:
    category: str
    description: str
    template_event_id: str
    cadence: str                     # "monthly" | "weekly"
    anchor: int                      # day-of-month, or weekday 0=Mon
    amount: Decimal                  # already in home currency, positive magnitude
    direction: str                   # debit | credit
    flexibility: str
    minimum_allowed_amount: Decimal | None
    last_seen: date

    @property
    def signed_amount(self) -> Decimal:
        return self.amount if self.direction == "credit" else -self.amount

    @property
    def is_flexible(self) -> bool:
        return self.flexibility != "fixed"

    def occurrences(self, start: date, end: date) -> list[date]:
        """Every projected date strictly after `start` and on or before `end`."""
        out: list[date] = []
        if self.cadence == "weekly":
            d = start + timedelta(days=1)
            while d.weekday() != self.anchor:
                d += timedelta(days=1)
            while d <= end:
                out.append(d)
                d += timedelta(days=7)
            return out

        year, month = start.year, start.month
        for _ in range(6):
            d = _clamp_day(year, month, self.anchor)
            if d > start and d <= end:
                out.append(d)
            month += 1
            if month == 13:
                year, month = year + 1, 1
        return sorted(out)


def _clamp_day(year: int, month: int, day: int) -> date:
    """Day 31 in a 30-day month lands on the 30th."""
    import calendar
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def _family(event: Event) -> tuple[str, str]:
    """Group key. Description is kept because one category can hold several
    distinct commitments (music subscription vs delivery membership)."""
    return (event.category, event.description.strip().lower())


def detect(events: list[Event], profile: Profile, rates: RateTable,
           as_of: date, estimator: str = "p75") -> list[Series]:
    estimate = ESTIMATORS[estimator]
    horizon_start = as_of - timedelta(days=LOOKBACK_DAYS)

    groups: dict[tuple[str, str], list[Event]] = defaultdict(list)
    for e in resolve_links(events):
        if classify(e) is not CashEffect.COUNT:
            continue
        if e.settlement_date > as_of or e.settlement_date < horizon_start:
            continue
        groups[_family(e)].append(e)

    series: list[Series] = []
    for (category, description), members in sorted(groups.items()):
        members.sort(key=lambda e: (e.settlement_date, e.event_id))
        if len(members) < MIN_OBSERVATIONS:
            continue

        gaps = [(b.settlement_date - a.settlement_date).days
                for a, b in zip(members, members[1:])]
        median_gap = statistics.median(gaps)

        if 5 <= median_gap <= 9:
            cadence = "weekly"
            anchor = Counter(e.settlement_date.weekday() for e in members).most_common(1)[0][0]
        elif 26 <= median_gap <= 35:
            cadence = "monthly"
            anchor = Counter(e.settlement_date.day for e in members).most_common(1)[0][0]
        else:
            continue                    # irregular: not a dependable commitment

        recent = members[-RECENT_WINDOW:]
        home_amounts = [
            rates.convert(e.amount, e.currency, profile.home_currency, e.settlement_date)
            for e in recent
        ]
        latest = members[-1]
        series.append(Series(
            category=category,
            description=latest.description,
            template_event_id=latest.event_id,
            cadence=cadence,
            anchor=anchor,
            amount=estimate(home_amounts),
            direction=latest.direction,
            flexibility=latest.flexibility,
            minimum_allowed_amount=latest.minimum_allowed_amount,
            last_seen=latest.settlement_date,
        ))
    return series
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/unit/test_recurrence.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add code/buyorwait/recurrence.py tests/unit/test_recurrence.py
git commit -m "feat: recurring series detection with swappable conservative estimator

Groups settled history by category and description, infers monthly or weekly
cadence from the median gap, and requires three observations before declaring a
series so one-off purchases are never projected.

The amount estimator is a named strategy (last/mean/median/p75/max/max3) rather
than a hardcoded choice, because which one matches the ground truth is a
calibration question to be answered against the labeled samples.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 6: The forecast curve and the simulator

**Plain English:** Now we put it together into a day-by-day projection of the user's bank balance for the next 90 days. Then the simulator answers one question: "if I add these payments to the plan, does the balance ever dip below the minimum?" Every recommendation the system makes is checked through this one function, so if this is right, nothing unsafe can get out.

**Files:**
- Create: `code/buyorwait/forecast.py`, `code/buyorwait/simulate.py`
- Test: `tests/unit/test_forecast.py`, `tests/unit/test_simulate.py`

**Interfaces:**
- Produces:
  - `forecast.Curve` with `.start: date`, `.end: date`, `.opening: Decimal`, `.flows: list[tuple[date, Decimal]]` (sorted, signed)
  - `forecast.build(view: LedgerView, series: list[Series], request_date: date, horizon_days: int = 90) -> Curve`
  - `simulate.trough(curve: Curve, payments: list[tuple[date, Decimal]] = ()) -> Decimal` — lowest balance reached
  - `simulate.is_safe(curve: Curve, minimum: Decimal, payments=...) -> bool`
  - `simulate.balance_series(curve, payments=...) -> list[tuple[date, Decimal]]` — for the inspector

- [ ] **Step 1: Write the failing forecast test**

`tests/unit/test_forecast.py`:
```python
from datetime import date
from decimal import Decimal

from buyorwait.forecast import build
from buyorwait.fx import RateTable
from buyorwait.ledger import LedgerView
from buyorwait.recurrence import Series
from buyorwait.types import Event, Profile

EMPTY_RATES = RateTable([])


def prof(balance="1000", minimum="200"):
    return Profile(
        user_id="user_x", home_currency="EUR",
        current_available_balance=Decimal(balance),
        minimum_balance_to_keep=Decimal(minimum),
        financial_priorities=(), expense_categories_to_protect=(),
        expense_categories_user_is_willing_to_reduce=(),
        expense_categories_user_is_willing_to_stop=(),
        payment_methods_user_will_consider=("full_payment",),
        max_installment_months=None,
    )


def rent_series(day=1, amount="300"):
    return Series(category="rent", description="Rent", template_event_id="event_r",
                  cadence="monthly", anchor=day, amount=Decimal(amount),
                  direction="debit", flexibility="fixed",
                  minimum_allowed_amount=None, last_seen=date(2024, 2, 1))


def test_curve_opens_at_the_profile_balance():
    view = LedgerView([], prof(balance="1234.56"), EMPTY_RATES)
    curve = build(view, [], date(2024, 3, 1))
    assert curve.opening == Decimal("1234.56")
    assert curve.start == date(2024, 3, 1)
    assert curve.end == date(2024, 5, 30)          # 90 days inclusive of the start


def test_projected_series_appear_as_negative_flows():
    view = LedgerView([], prof(), EMPTY_RATES)
    curve = build(view, [rent_series()], date(2024, 3, 5))
    assert curve.flows == [
        (date(2024, 4, 1), Decimal("-300")),
        (date(2024, 5, 1), Decimal("-300")),
    ]


def test_a_real_scheduled_event_suppresses_the_projection_on_the_same_day():
    """The confirmed row wins; projecting on top of it would double-charge rent."""
    real = Event(
        event_id="event_r2", user_id="user_x", event_type="expense",
        description="Rent", category="rent", direction="debit",
        amount=Decimal("300"), currency="EUR",
        event_date=date(2024, 4, 1), settlement_date=date(2024, 4, 1),
        status="scheduled", linked_event_id=None, flexibility="fixed",
        minimum_allowed_amount=None,
    )
    view = LedgerView([real], prof(), EMPTY_RATES)
    curve = build(view, [rent_series()], date(2024, 3, 5))
    april = [f for f in curve.flows if f[0] == date(2024, 4, 1)]
    assert april == [(date(2024, 4, 1), Decimal("-300"))], "rent charged once, not twice"


def test_flows_are_sorted_by_date():
    view = LedgerView([], prof(), EMPTY_RATES)
    salary = Series(category="salary", description="Payroll", template_event_id="event_s",
                    cadence="monthly", anchor=15, amount=Decimal("900"),
                    direction="credit", flexibility="fixed",
                    minimum_allowed_amount=None, last_seen=date(2024, 2, 15))
    curve = build(view, [rent_series(), salary], date(2024, 3, 20))
    assert curve.flows == sorted(curve.flows, key=lambda f: f[0])
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/unit/test_forecast.py -v`
Expected: FAIL — `No module named 'buyorwait.forecast'`

- [ ] **Step 3: Implement forecast.py**

`code/buyorwait/forecast.py`:
```python
"""Build the 90-day cash-flow curve for one request.

Sources, in precedence order:
  1. Confirmed rows from the ledger (scheduled payments, reserved pending debits)
  2. Projected recurring series

Precedence matters: when a real scheduled row already covers a date for a given
category, the projection for that date is suppressed. Otherwise rent would be
charged twice in the month the dataset happens to have a confirmed row for.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from .ledger import LedgerView
from .recurrence import Series

HORIZON_DAYS = 90


@dataclass(frozen=True)
class Curve:
    start: date
    end: date
    opening: Decimal
    flows: tuple[tuple[date, Decimal], ...]
    sources: tuple[str, ...] = ()

    def with_extra(self, extra: list[tuple[date, Decimal]]) -> "Curve":
        merged = sorted([*self.flows, *extra], key=lambda f: f[0])
        return Curve(self.start, self.end, self.opening, tuple(merged), self.sources)


def build(view: LedgerView, series: list[Series], request_date: date,
          horizon_days: int = HORIZON_DAYS) -> Curve:
    end = request_date + timedelta(days=horizon_days - 1)

    confirmed = view.future_cashflows(request_date)
    confirmed = [(d, amt) for d, amt in confirmed if d <= end]

    # Dates already covered by a confirmed row, keyed by category, so a projection
    # for the same category on the same day is dropped.
    covered: set[tuple[date, str]] = set()
    for e in view.events:
        if e.settlement_date >= request_date:
            covered.add((e.settlement_date, e.category))

    projected: list[tuple[date, Decimal]] = []
    for s in series:
        for when in s.occurrences(request_date, end):
            if (when, s.category) in covered:
                continue
            projected.append((when, s.signed_amount))

    flows = sorted([*confirmed, *projected], key=lambda f: (f[0], f[1]))
    return Curve(
        start=request_date,
        end=end,
        opening=view.profile.current_available_balance,
        flows=tuple(flows),
    )
```

- [ ] **Step 4: Run the forecast tests**

Run: `python -m pytest tests/unit/test_forecast.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Write the failing simulator test**

`tests/unit/test_simulate.py`:
```python
from datetime import date
from decimal import Decimal

from buyorwait.forecast import Curve
from buyorwait.simulate import balance_series, is_safe, trough


def curve(opening="1000", flows=()):
    return Curve(start=date(2024, 3, 1), end=date(2024, 5, 29),
                 opening=Decimal(opening), flows=tuple(flows))


def test_trough_of_a_flat_curve_is_the_opening_balance():
    assert trough(curve()) == Decimal("1000")


def test_trough_finds_the_lowest_point_not_the_final_balance():
    c = curve(flows=[(date(2024, 3, 10), Decimal("-800")),
                     (date(2024, 3, 20), Decimal("900"))])
    assert trough(c) == Decimal("200")


def test_payments_are_applied_on_their_dates():
    c = curve(flows=[(date(2024, 4, 1), Decimal("500"))])
    assert trough(c, [(date(2024, 3, 1), Decimal("700"))]) == Decimal("300")


def test_is_safe_is_inclusive_of_the_minimum():
    c = curve(flows=[(date(2024, 3, 10), Decimal("-800"))])
    assert is_safe(c, Decimal("200")) is True          # lands exactly on 200
    assert is_safe(c, Decimal("200.01")) is False


def test_same_day_flows_all_apply_before_the_day_is_measured():
    """A payment and an expense on the same day both count that day."""
    c = curve(flows=[(date(2024, 3, 5), Decimal("-500"))])
    assert trough(c, [(date(2024, 3, 5), Decimal("400"))]) == Decimal("100")


def test_balance_series_reports_one_point_per_flow_date():
    c = curve(flows=[(date(2024, 3, 10), Decimal("-100")),
                     (date(2024, 3, 20), Decimal("-100"))])
    pts = balance_series(c)
    assert pts[0] == (date(2024, 3, 1), Decimal("1000"))
    assert pts[-1] == (date(2024, 3, 20), Decimal("800"))
```

- [ ] **Step 6: Run it and confirm it fails**

Run: `python -m pytest tests/unit/test_simulate.py -v`
Expected: FAIL — `No module named 'buyorwait.simulate'`

- [ ] **Step 7: Implement simulate.py**

`code/buyorwait/simulate.py`:
```python
"""Walk a curve and report the lowest balance it reaches.

Every recommendation this system makes is checked through is_safe(). If this
function is correct, no unsafe plan can escape.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from .forecast import Curve

Payments = list[tuple[date, Decimal]]


def _merged(curve: Curve, payments: Payments | tuple = ()) -> list[tuple[date, Decimal]]:
    # Payments are outflows, so they enter the curve negated.
    extra = [(d, -amount) for d, amount in payments]
    return sorted([*curve.flows, *extra], key=lambda f: f[0])


def balance_series(curve: Curve, payments: Payments | tuple = ()) -> list[tuple[date, Decimal]]:
    points = [(curve.start, curve.opening)]
    balance = curve.opening
    for when, amount in _merged(curve, payments):
        balance += amount
        if points and points[-1][0] == when:
            points[-1] = (when, balance)
        else:
            points.append((when, balance))
    return points


def trough(curve: Curve, payments: Payments | tuple = ()) -> Decimal:
    balance = curve.opening
    low = balance
    for _, amount in _merged(curve, payments):
        balance += amount
        if balance < low:
            low = balance
    return low


def is_safe(curve: Curve, minimum: Decimal, payments: Payments | tuple = ()) -> bool:
    return trough(curve, payments) >= minimum
```

- [ ] **Step 8: Run the tests**

Run: `python -m pytest tests/unit/test_simulate.py -v`
Expected: PASS, 6 passed

- [ ] **Step 9: Commit**

```bash
git add code/buyorwait/forecast.py code/buyorwait/simulate.py tests/unit/test_forecast.py tests/unit/test_simulate.py
git commit -m "feat: 90-day forecast curve and safety simulator

Merges confirmed ledger flows with projected recurring series, suppressing a
projection when a confirmed row already covers that date and category so nothing
is charged twice. The simulator reports the trough of the curve under a set of
proposed payments; every recommendation is gated through is_safe().

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 7: The solver — safe amount and earliest date

**Plain English:** Two headline numbers. **How much can you safely pay today?** Because paying more only ever lowers the balance, the answer is simple arithmetic: take the lowest point the balance would reach if you paid nothing, subtract the minimum you want to keep, and that is your headroom — capped at what you asked for. **When could you afford the whole thing?** Walk forward day by day and find the first date where paying the full amount still keeps you above the minimum for the rest of the window.

We prove the arithmetic version is right by also writing a slow, obviously-correct version (binary search) and asserting they always agree.

**Files:**
- Create: `code/buyorwait/solver.py`
- Test: `tests/unit/test_solver.py`

**Interfaces:**
- Produces:
  - `solver.safe_amount(curve: Curve, minimum: Decimal, requested: Decimal, on: date) -> Decimal`
  - `solver.safe_amount_by_search(...)` — the oracle, test-only but shipped for auditability
  - `solver.earliest_full_payment_date(curve, minimum, requested) -> date | None`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_solver.py`:
```python
import random
from datetime import date, timedelta
from decimal import Decimal

from buyorwait.forecast import Curve
from buyorwait.solver import (earliest_full_payment_date, safe_amount,
                              safe_amount_by_search)

START = date(2024, 3, 1)


def curve(opening, flows=()):
    return Curve(start=START, end=START + timedelta(days=89),
                 opening=Decimal(opening), flows=tuple(flows))


def test_safe_amount_is_headroom_above_the_minimum():
    c = curve("1000")
    assert safe_amount(c, Decimal("200"), Decimal("5000"), START) == Decimal("800")


def test_safe_amount_is_capped_at_the_requested_amount():
    c = curve("1000")
    assert safe_amount(c, Decimal("200"), Decimal("300"), START) == Decimal("300")


def test_safe_amount_accounts_for_a_future_trough_not_just_today():
    c = curve("1000", [(date(2024, 4, 1), Decimal("-600"))])
    assert safe_amount(c, Decimal("200"), Decimal("5000"), START) == Decimal("200")


def test_safe_amount_is_never_negative():
    c = curve("100")
    assert safe_amount(c, Decimal("200"), Decimal("5000"), START) == Decimal("0")


def test_closed_form_agrees_with_the_binary_search_oracle():
    """The fast arithmetic answer must equal the slow, obviously-correct one."""
    rng = random.Random(0)
    for _ in range(200):
        opening = Decimal(rng.randrange(0, 500_000))
        minimum = Decimal(rng.randrange(0, 100_000))
        requested = Decimal(rng.randrange(1, 500_000))
        flows = tuple(
            (START + timedelta(days=rng.randrange(1, 90)),
             Decimal(rng.randrange(-50_000, 50_000)))
            for _ in range(rng.randrange(0, 12))
        )
        c = curve(opening, sorted(flows, key=lambda f: f[0]))
        fast = safe_amount(c, minimum, requested, START)
        slow = safe_amount_by_search(c, minimum, requested, START)
        assert fast == slow, f"mismatch for opening={opening} min={minimum}"


def test_earliest_date_is_request_date_when_affordable_now():
    c = curve("1000")
    assert earliest_full_payment_date(c, Decimal("200"), Decimal("500")) == START


def test_earliest_date_waits_for_incoming_money():
    c = curve("100", [(date(2024, 3, 15), Decimal("900"))])
    assert earliest_full_payment_date(c, Decimal("200"), Decimal("500")) == date(2024, 3, 15)


def test_earliest_date_is_the_first_feasible_day_not_merely_a_feasible_one():
    c = curve("100", [(date(2024, 3, 15), Decimal("900"))])
    got = earliest_full_payment_date(c, Decimal("200"), Decimal("500"))
    from buyorwait.simulate import is_safe
    assert is_safe(c, Decimal("200"), [(got, Decimal("500"))])
    assert not is_safe(c, Decimal("200"), [(got - timedelta(days=1), Decimal("500"))])


def test_earliest_date_is_none_when_never_affordable_in_the_window():
    c = curve("100")
    assert earliest_full_payment_date(c, Decimal("200"), Decimal("5000")) is None
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/unit/test_solver.py -v`
Expected: FAIL — `No module named 'buyorwait.solver'`

- [ ] **Step 3: Implement solver.py**

`code/buyorwait/solver.py`:
```python
"""The two headline numbers: how much is safe today, and when is the full amount safe.

safe_amount has a closed form. Paying X on the request date lowers every
subsequent balance by exactly X, so the trough is linear in X:

    trough(X) = trough(0) - X

The largest safe X therefore satisfies trough(0) - X >= minimum, giving
X* = trough(0) - minimum, clipped to [0, requested]. A binary-search oracle is
kept alongside it and a property test asserts the two never disagree.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from .forecast import Curve
from .simulate import is_safe, trough


def safe_amount(curve: Curve, minimum: Decimal, requested: Decimal, on: date) -> Decimal:
    headroom = trough(curve) - minimum
    if headroom <= 0:
        return Decimal("0")
    return min(headroom, requested)


def safe_amount_by_search(curve: Curve, minimum: Decimal, requested: Decimal,
                          on: date) -> Decimal:
    """Deliberately slow reference implementation used to prove the closed form."""
    if not is_safe(curve, minimum, [(on, Decimal("0"))]):
        return Decimal("0")
    if is_safe(curve, minimum, [(on, requested)]):
        return requested
    lo, hi = Decimal("0"), requested
    for _ in range(80):
        mid = (lo + hi) / 2
        if is_safe(curve, minimum, [(on, mid)]):
            lo = mid
        else:
            hi = mid
    return lo.quantize(Decimal("0.00000001"))


def earliest_full_payment_date(curve: Curve, minimum: Decimal,
                               requested: Decimal) -> date | None:
    """First day in the window on which paying `requested` keeps the curve safe.

    Measures financial capacity only. It ignores which methods the user accepts,
    which is why it can equal the request date even when the recommendation is
    installments.
    """
    day = curve.start
    while day <= curve.end:
        if is_safe(curve, minimum, [(day, requested)]):
            return day
        day += timedelta(days=1)
    return None
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/unit/test_solver.py -v`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
git add code/buyorwait/solver.py tests/unit/test_solver.py
git commit -m "feat: closed-form safe amount and earliest full-payment date

Paying X today lowers every later balance by exactly X, so the trough is linear
in X and the largest safe payment is trough(0) - minimum, clipped to the
requested amount. A binary-search oracle ships alongside it and a 200-case
randomized property test asserts the two never disagree.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 8: Spending changes

**Plain English:** If someone can't quite afford something, the agent may suggest cutting back — cancel a streaming plan, reduce takeaway spending. But it may only touch expenses the user has explicitly said they're willing to change, never protected essentials like rent. A key finding from the samples: when the agent reduces an expense, it always reduces it to that expense's `minimum_allowed_amount` — it never invents a middle figure.

**Files:**
- Create: `code/buyorwait/changes.py`
- Test: `tests/unit/test_changes.py`

**Interfaces:**
- Produces:
  - `changes.Change` frozen dataclass: `.kind` (`"stop"`/`"reduce_to"`), `.event_id`, `.series`, `.new_amount: Decimal | None`, `.saving_per_occurrence: Decimal`, `.render() -> str`
  - `changes.candidates(series: list[Series], profile: Profile) -> list[Change]`
  - `changes.apply(curve: Curve, chosen: list[Change], request_date: date, end: date) -> Curve`
  - `changes.combinations(all_changes: list[Change], max_count: int = 3) -> Iterator[tuple[Change, ...]]`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_changes.py`:
```python
from datetime import date
from decimal import Decimal

from buyorwait.changes import Change, apply, candidates, combinations
from buyorwait.forecast import Curve
from buyorwait.recurrence import Series
from buyorwait.types import Profile


def series(category, description, flexibility, amount="47", minimum=None, anchor=10):
    return Series(category=category, description=description,
                  template_event_id=f"event_{category}", cadence="monthly",
                  anchor=anchor, amount=Decimal(amount), direction="debit",
                  flexibility=flexibility,
                  minimum_allowed_amount=Decimal(minimum) if minimum else None,
                  last_seen=date(2024, 2, 10))


def prof(stop=(), reduce=(), protect=()):
    return Profile(
        user_id="user_x", home_currency="EUR",
        current_available_balance=Decimal("1000"),
        minimum_balance_to_keep=Decimal("200"),
        financial_priorities=(), expense_categories_to_protect=protect,
        expense_categories_user_is_willing_to_reduce=reduce,
        expense_categories_user_is_willing_to_stop=stop,
        payment_methods_user_will_consider=("full_payment",),
        max_installment_months=None,
    )


def test_only_categories_the_user_permits_are_offered():
    all_series = [
        series("streaming", "Streaming subscription", "stoppable"),
        series("rent", "Apartment rent", "stoppable"),
    ]
    got = candidates(all_series, prof(stop=("streaming",)))
    assert [c.event_id for c in got] == ["event_streaming"]


def test_protected_category_is_never_offered_even_if_listed_as_stoppable():
    all_series = [series("groceries", "Weekly groceries", "stoppable")]
    p = prof(stop=("groceries",), protect=("groceries",))
    assert candidates(all_series, p) == []


def test_fixed_expenses_are_never_offered():
    all_series = [series("streaming", "Streaming subscription", "fixed")]
    assert candidates(all_series, prof(stop=("streaming",))) == []


def test_reduce_targets_the_minimum_allowed_amount_exactly():
    """Confirmed against all three spending-change samples."""
    all_series = [series("dining", "Weekend food delivery", "reducible",
                         amount="1163530.49", minimum="665950")]
    got = candidates(all_series, prof(reduce=("dining",)))
    assert len(got) == 1
    assert got[0].kind == "reduce_to"
    assert got[0].new_amount == Decimal("665950")
    assert got[0].render() == "reduce_to:event_dining:665950"


def test_reducible_or_stoppable_offers_both_forms():
    all_series = [series("streaming", "Streaming subscription",
                         "reducible_or_stoppable", amount="47", minimum="23.5")]
    got = candidates(all_series, prof(stop=("streaming",), reduce=("streaming",)))
    assert {c.kind for c in got} == {"stop", "reduce_to"}


def test_stop_renders_in_the_required_format():
    all_series = [series("cloud_storage", "Online backup subscription", "stoppable",
                         amount="11")]
    got = candidates(all_series, prof(stop=("cloud_storage",)))
    assert got[0].render() == "stop:event_cloud_storage"


def test_combinations_never_mix_stop_and_reduce_on_one_event():
    s = series("streaming", "Streaming subscription", "reducible_or_stoppable",
               amount="47", minimum="23.5")
    got = candidates([s], prof(stop=("streaming",), reduce=("streaming",)))
    for combo in combinations(got, max_count=3):
        ids = [c.event_id for c in combo]
        assert len(ids) == len(set(ids)), "the same event appears twice in one combination"


def test_combinations_are_capped_at_three():
    all_series = [series(f"cat{i}", f"Sub {i}", "stoppable", anchor=i + 1) for i in range(5)]
    p = prof(stop=tuple(f"cat{i}" for i in range(5)))
    got = candidates(all_series, p)
    assert all(len(c) <= 3 for c in combinations(got, max_count=3))


def test_apply_removes_a_stopped_series_from_the_curve():
    c = Curve(start=date(2024, 3, 1), end=date(2024, 5, 29), opening=Decimal("1000"),
              flows=((date(2024, 4, 10), Decimal("-47")),))
    s = series("streaming", "Streaming subscription", "stoppable", amount="47")
    change = Change(kind="stop", event_id="event_streaming", series=s, new_amount=None,
                    saving_per_occurrence=Decimal("47"))
    out = apply(c, [change], date(2024, 3, 1), date(2024, 5, 29))
    assert out.flows == ()


def test_apply_rewrites_a_reduced_series_to_the_new_amount():
    c = Curve(start=date(2024, 3, 1), end=date(2024, 5, 29), opening=Decimal("1000"),
              flows=((date(2024, 4, 10), Decimal("-47")),))
    s = series("streaming", "Streaming subscription", "reducible", amount="47",
               minimum="23.5")
    change = Change(kind="reduce_to", event_id="event_streaming", series=s,
                    new_amount=Decimal("23.5"),
                    saving_per_occurrence=Decimal("23.5"))
    out = apply(c, [change], date(2024, 3, 1), date(2024, 5, 29))
    assert out.flows == ((date(2024, 4, 10), Decimal("-23.5")),)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/unit/test_changes.py -v`
Expected: FAIL — `No module named 'buyorwait.changes'`

- [ ] **Step 3: Implement changes.py**

`code/buyorwait/changes.py`:
```python
"""Optional spending changes the user has pre-authorised.

Three constraints, all from the problem statement:
  - only recurring, flexible expenses
  - only categories the user listed as reducible/stoppable, never a protected one
  - at most three changes, and one event may not be both stopped and reduced

One convention from the labeled samples: reduce_to always targets the event's
minimum_allowed_amount exactly. The agent never invents an intermediate figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from itertools import combinations as _combos
from typing import Iterator

from .forecast import Curve
from .money import fmt_plain
from .recurrence import Series
from .types import Profile


@dataclass(frozen=True)
class Change:
    kind: str                      # "stop" | "reduce_to"
    event_id: str
    series: Series
    new_amount: Decimal | None
    saving_per_occurrence: Decimal

    def render(self) -> str:
        if self.kind == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{fmt_plain(self.new_amount)}"

    @property
    def phrase(self) -> str:
        """Human phrase for the explanation: 'the weekend food delivery'."""
        return f"the {self.series.description.strip().lower()}"


def candidates(series: list[Series], profile: Profile) -> list[Change]:
    protected = set(profile.expense_categories_to_protect)
    may_stop = set(profile.expense_categories_user_is_willing_to_stop) - protected
    may_reduce = set(profile.expense_categories_user_is_willing_to_reduce) - protected

    out: list[Change] = []
    for s in sorted(series, key=lambda x: x.template_event_id):
        if s.direction != "debit" or not s.is_flexible:
            continue
        can_stop = s.flexibility in ("stoppable", "reducible_or_stoppable")
        can_reduce = s.flexibility in ("reducible", "reducible_or_stoppable")

        if can_stop and s.category in may_stop:
            out.append(Change("stop", s.template_event_id, s, None, s.amount))
        if (can_reduce and s.category in may_reduce
                and s.minimum_allowed_amount is not None
                and s.minimum_allowed_amount < s.amount):
            out.append(Change("reduce_to", s.template_event_id, s,
                              s.minimum_allowed_amount,
                              s.amount - s.minimum_allowed_amount))
    return out


def combinations(all_changes: list[Change], max_count: int = 3) -> Iterator[tuple[Change, ...]]:
    """Every subset of size 1..max_count, never repeating an event within a subset.

    Ordered by subset size so the ranker naturally prefers fewer changes.
    """
    for size in range(1, max_count + 1):
        for combo in _combos(all_changes, size):
            ids = [c.event_id for c in combo]
            if len(ids) != len(set(ids)):
                continue
            yield combo


def apply(curve: Curve, chosen: list[Change], request_date: date, end: date) -> Curve:
    """Rewrite the curve with the chosen changes in force from the request date."""
    if not chosen:
        return curve

    stopped = {c.event_id: c for c in chosen if c.kind == "stop"}
    reduced = {c.event_id: c for c in chosen if c.kind == "reduce_to"}

    # Match flows back to their series by (date, signed amount).
    targets: dict[str, set[tuple[date, Decimal]]] = {}
    for c in chosen:
        occ = c.series.occurrences(request_date - _one_day(), end)
        targets[c.event_id] = {(d, c.series.signed_amount) for d in occ}

    flows: list[tuple[date, Decimal]] = []
    for when, amount in curve.flows:
        dropped = False
        for event_id, change in stopped.items():
            if (when, amount) in targets[event_id]:
                dropped = True
                break
        if dropped:
            continue
        for event_id, change in reduced.items():
            if (when, amount) in targets[event_id]:
                amount = -change.new_amount
                break
        flows.append((when, amount))

    return Curve(curve.start, curve.end, curve.opening,
                 tuple(sorted(flows, key=lambda f: f[0])), curve.sources)


def _one_day():
    from datetime import timedelta
    return timedelta(days=1)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/unit/test_changes.py -v`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
git add code/buyorwait/changes.py tests/unit/test_changes.py
git commit -m "feat: spending-change candidates restricted to permitted flexible expenses

Offers stop and reduce_to actions only for recurring flexible series in a
category the user allows and has not protected. reduce_to always targets the
event's minimum_allowed_amount exactly, matching all three labeled samples that
use a spending change.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 9: Candidate plans, ranking, and status

**Plain English:** Now we list every legal way to pay — in full today, each installment offer the seller made, a two-part payment, or waiting until payday — throw away any that would break the budget or that the user has said they don't want, and pick the best survivor using the six tie-breaker rules the problem statement gives, in order. The status column then follows automatically from which plan won; it is never chosen separately.

**Files:**
- Create: `code/buyorwait/ranker.py`
- Modify: `code/buyorwait/solver.py` (add `enumerate_candidates`)
- Test: `tests/unit/test_ranker.py`

**Interfaces:**
- Produces:
  - `ranker.Candidate` frozen dataclass: `.method`, `.payments: tuple[tuple[date, Decimal], ...]`, `.payment_texts: tuple[str, ...]`, `.total_paid: Decimal`, `.changes: tuple[Change, ...]`, `.option_id: str | None`, `.option_sort_key: int`, `.completes_by_deadline: bool`, `.status: str`
  - `ranker.rank(candidates: list[Candidate]) -> Candidate | None`
  - `solver.enumerate_candidates(request, profile, curve, options, change_options, safe, earliest) -> list[Candidate]`

- [ ] **Step 1: Write the failing ranking test**

`tests/unit/test_ranker.py`:
```python
from datetime import date
from decimal import Decimal

from buyorwait.ranker import Candidate, rank

D = Decimal


def cand(method="full_payment", payments=((date(2024, 3, 1), D("100")),),
         total="100", changes=(), option_id=None, option_sort_key=9999,
         completes=True, status="affordable_now"):
    return Candidate(method=method, payments=tuple(payments),
                     payment_texts=tuple(str(a) for _, a in payments),
                     total_paid=D(total), changes=tuple(changes),
                     option_id=option_id, option_sort_key=option_sort_key,
                     completes_by_deadline=completes, status=status)


def test_rung_1_completing_by_the_deadline_beats_everything():
    late = cand(total="50", completes=False)
    on_time = cand(total="500", completes=True)
    assert rank([late, on_time]) is on_time


def test_rung_2_no_spending_changes_beats_fewer_payments():
    from buyorwait.changes import Change
    from buyorwait.recurrence import Series
    s = Series("streaming", "Sub", "event_1", "monthly", 10, D("10"), "debit",
               "stoppable", None, date(2024, 2, 10))
    with_change = cand(total="100", changes=(Change("stop", "event_1", s, None, D("10")),))
    without = cand(total="100", payments=((date(2024, 3, 1), D("50")),
                                          (date(2024, 4, 1), D("50"))))
    assert rank([with_change, without]) is without


def test_rung_3_lower_total_paid_wins():
    cheap = cand(total="100")
    pricey = cand(total="120")
    assert rank([pricey, cheap]) is cheap


def test_rung_4_earlier_start_wins_at_equal_cost():
    early = cand(payments=((date(2024, 3, 1), D("100")),))
    late = cand(payments=((date(2024, 3, 8), D("100")),))
    assert rank([late, early]) is early


def test_rung_5_fewer_payments_wins():
    one = cand(payments=((date(2024, 3, 1), D("100")),))
    three = cand(payments=((date(2024, 3, 1), D("34")), (date(2024, 4, 1), D("33")),
                           (date(2024, 5, 1), D("33"))))
    assert rank([three, one]) is one


def test_rung_6_lowest_option_id_is_the_final_tiebreak():
    a = cand(option_id="payment_option_07", option_sort_key=7)
    b = cand(option_id="payment_option_05", option_sort_key=5)
    assert rank([a, b]) is b


def test_empty_candidate_list_returns_none():
    assert rank([]) is None
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/unit/test_ranker.py -v`
Expected: FAIL — `No module named 'buyorwait.ranker'`

- [ ] **Step 3: Implement ranker.py**

`code/buyorwait/ranker.py`:
```python
"""Choose between safe plans using the problem statement's six-rung tie-break.

   1. Complete the full request by desired_completion_date
   2. Require no spending changes
   3. Minimize the total amount paid
   4. Start payment earlier
   5. Use fewer payments
   6. Lowest payment_option_id

Only SAFE, ELIGIBLE candidates are ever passed in; this module does not re-check
feasibility, it only orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from .changes import Change

FAR_FUTURE = date(9999, 12, 31)


@dataclass(frozen=True)
class Candidate:
    method: str                                   # full_payment|partial_payment|installments|wait
    payments: tuple[tuple[date, Decimal], ...]
    payment_texts: tuple[str, ...]                # verbatim amount strings for the plan
    total_paid: Decimal
    changes: tuple[Change, ...]
    option_id: str | None
    option_sort_key: int
    completes_by_deadline: bool
    status: str

    @property
    def start_date(self) -> date:
        return self.payments[0][0] if self.payments else FAR_FUTURE

    def render_plan(self) -> str:
        if not self.payments:
            return "none"
        return "|".join(
            f"{when.isoformat()}:{text}"
            for (when, _), text in zip(self.payments, self.payment_texts)
        )

    def render_changes(self) -> str:
        return "|".join(c.render() for c in self.changes) if self.changes else "none"


def _sort_key(c: Candidate):
    return (
        0 if c.completes_by_deadline else 1,   # rung 1
        len(c.changes),                        # rung 2
        c.total_paid,                          # rung 3
        c.start_date,                          # rung 4
        len(c.payments),                       # rung 5
        c.option_sort_key,                     # rung 6
    )


def rank(candidates: list[Candidate]) -> Candidate | None:
    return min(candidates, key=_sort_key) if candidates else None
```

- [ ] **Step 4: Run the ranking tests**

Run: `python -m pytest tests/unit/test_ranker.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Write the failing enumeration test**

`tests/unit/test_enumerate.py`:
```python
from datetime import date, timedelta
from decimal import Decimal

from buyorwait.forecast import Curve
from buyorwait.solver import enumerate_candidates
from buyorwait.types import PaymentOption, Profile, Request

D = Decimal
START = date(2024, 3, 1)


def req(amount="1000", partial=True, deadline=date(2024, 5, 1)):
    return Request("request_x", "user_x", START, "purchase", D(amount), deadline,
                   partial, "text")


def prof(methods=("full_payment",), max_months=None, balance="5000", minimum="200"):
    return Profile("user_x", "EUR", D(balance), D(minimum), (), (), (), (),
                   methods, max_months)


def curve(opening="5000", flows=()):
    return Curve(START, START + timedelta(days=89), D(opening), tuple(flows))


def option(oid, method, amount, n, first, freq, total):
    return PaymentOption(oid, "request_x", method, D(amount), amount, n, first,
                         freq, D("0"), D(total))


def test_full_payment_offered_only_when_the_user_accepts_it():
    c = curve()
    accepted = enumerate_candidates(req(), prof(methods=("full_payment",)), c, [], [],
                                    D("1000"), START)
    assert any(x.method == "full_payment" for x in accepted)

    declined = enumerate_candidates(req(), prof(methods=("installments",)), c, [], [],
                                    D("1000"), START)
    assert not any(x.method == "full_payment" for x in declined)


def test_installment_option_exceeding_max_months_is_rejected():
    c = curve()
    long_plan = option("payment_option_07", "installments", "100", 18, START, 30, "1800")
    got = enumerate_candidates(req(), prof(methods=("installments",), max_months=7), c,
                               [long_plan], [], D("1000"), START)
    assert not any(x.option_id == "payment_option_07" for x in got)


def test_installment_option_within_max_months_is_offered():
    c = curve()
    short = option("payment_option_05", "installments", "340", 3, START, 30, "1020")
    got = enumerate_candidates(req(), prof(methods=("installments",), max_months=7), c,
                               [short], [], D("1000"), START)
    assert any(x.option_id == "payment_option_05" for x in got)


def test_partial_payment_requires_the_request_to_allow_it():
    c = curve(opening="600")
    got = enumerate_candidates(req(partial=False),
                               prof(methods=("partial_payment",)), c, [], [],
                               D("400"), date(2024, 3, 20))
    assert not any(x.method == "partial_payment" for x in got)


def test_partial_payment_splits_into_exactly_two_payments_summing_to_the_request():
    c = curve(opening="600")
    got = enumerate_candidates(req(), prof(methods=("partial_payment",)), c, [], [],
                               D("400"), date(2024, 3, 20))
    p = next(x for x in got if x.method == "partial_payment")
    assert len(p.payments) == 2
    assert p.payments[0] == (START, D("400"))
    assert p.payments[1] == (date(2024, 3, 20), D("600"))
    assert sum(a for _, a in p.payments) == D("1000")
    assert p.status == "affordable_with_plan"


def test_partial_payment_rejected_when_second_payment_misses_the_deadline():
    c = curve(opening="600")
    got = enumerate_candidates(req(deadline=date(2024, 3, 10)),
                               prof(methods=("partial_payment",)), c, [], [],
                               D("400"), date(2024, 3, 20))
    assert not any(x.method == "partial_payment" for x in got)


def test_wait_is_offered_when_full_payment_becomes_safe_later():
    c = curve(opening="100", flows=[(date(2024, 3, 15), D("2000"))])
    got = enumerate_candidates(req(), prof(methods=("full_payment",)), c, [], [],
                               D("0"), date(2024, 3, 15))
    w = next(x for x in got if x.method == "wait")
    assert w.payments == ((date(2024, 3, 15), D("1000")),)
    assert w.status == "affordable_later"


def test_full_payment_today_yields_affordable_now_status():
    c = curve()
    got = enumerate_candidates(req(), prof(), c, [], [], D("1000"), START)
    f = next(x for x in got if x.method == "full_payment" and not x.changes)
    assert f.status == "affordable_now"
```

- [ ] **Step 6: Run it and confirm it fails**

Run: `python -m pytest tests/unit/test_enumerate.py -v`
Expected: FAIL — `cannot import name 'enumerate_candidates'`

- [ ] **Step 7: Add enumerate_candidates to solver.py**

Append to `code/buyorwait/solver.py`:
```python
from datetime import timedelta as _td

from .changes import Change, apply as apply_changes, candidates as change_candidates, combinations
from .ranker import Candidate
from .types import PaymentOption, Profile, Request


def _option_payments(option: PaymentOption) -> list[tuple[date, Decimal]]:
    freq = option.payment_frequency_days or 0
    return [
        (option.first_payment_date + _td(days=freq * i), option.payment_amount)
        for i in range(option.number_of_payments)
    ]


def _option_months(option: PaymentOption) -> int:
    """How many months the schedule spans, for the max_installment_months check."""
    freq = option.payment_frequency_days or 30
    total_days = freq * max(option.number_of_payments - 1, 0)
    return max(1, round((total_days + freq) / 30))


def enumerate_candidates(request: Request, profile: Profile, curve: Curve,
                         options: list[PaymentOption], change_options: list[Change],
                         safe: Decimal, earliest: date | None) -> list[Candidate]:
    """Every eligible, SAFE way to pay. Ineligible or unsafe plans never appear."""
    accepted = set(profile.payment_methods_user_will_consider)
    minimum = profile.minimum_balance_to_keep
    deadline = request.desired_completion_date
    out: list[Candidate] = []

    def add(method, payments, texts, total, changes, option_id, sort_key, status):
        working = apply_changes(curve, list(changes), request.request_date, curve.end)
        if not is_safe(working, minimum, list(payments)):
            return
        out.append(Candidate(
            method=method, payments=tuple(payments), payment_texts=tuple(texts),
            total_paid=total, changes=tuple(changes), option_id=option_id,
            option_sort_key=sort_key,
            completes_by_deadline=bool(payments) and payments[-1][0] <= deadline,
            status=status,
        ))

    change_sets: list[tuple[Change, ...]] = [()]
    change_sets.extend(combinations(change_options, max_count=3))

    for changes in change_sets:
        status_with = "affordable_with_plan" if changes else "affordable_now"

        # 1. Full payment on the request date
        if "full_payment" in accepted:
            add("full_payment",
                [(request.request_date, request.requested_amount)],
                [_text(request.requested_amount)],
                request.requested_amount, changes, None, 10**9, status_with)

        # 2. Each supplied payment option
        for option in options:
            if option.payment_method == "full_payment":
                if "full_payment" not in accepted:
                    continue
            elif option.payment_method == "installments":
                if "installments" not in accepted:
                    continue
                if (profile.max_installment_months is not None
                        and _option_months(option) > profile.max_installment_months):
                    continue
            else:
                continue
            payments = _option_payments(option)
            add(option.payment_method, payments,
                [option.payment_amount_text] * len(payments),
                option.total_payable_amount, changes,
                option.payment_option_id, option.sort_key,
                "affordable_now" if (option.payment_method == "full_payment"
                                     and not changes
                                     and option.first_payment_date == request.request_date)
                else "affordable_with_plan")

        # 3. Two-step partial payment
        if ("partial_payment" in accepted and request.allows_partial_payment
                and earliest is not None and earliest <= deadline
                and Decimal("0") < safe < request.requested_amount and not changes):
            remainder = request.requested_amount - safe
            add("partial_payment",
                [(request.request_date, safe), (earliest, remainder)],
                [_text(safe), _text(remainder)],
                request.requested_amount, changes, None, 10**9,
                "affordable_with_plan")

        # 4. Wait for the first date the full amount is safe
        if ("full_payment" in accepted and earliest is not None
                and earliest > request.request_date and not changes):
            add("wait", [(earliest, request.requested_amount)],
                [_text(request.requested_amount)],
                request.requested_amount, changes, None, 10**9,
                "affordable_later")

    return out


def _text(value: Decimal) -> str:
    from .money import fmt_plain
    return fmt_plain(value)
```

- [ ] **Step 8: Run the tests**

Run: `python -m pytest tests/unit/test_enumerate.py tests/unit/test_ranker.py -v`
Expected: PASS, 15 passed

- [ ] **Step 9: Commit**

```bash
git add code/buyorwait/ranker.py code/buyorwait/solver.py tests/unit/test_ranker.py tests/unit/test_enumerate.py
git commit -m "feat: candidate enumeration and six-rung plan ranking

Enumerates full payment, every supplied payment option, a two-step partial
payment and waiting, each optionally combined with up to three spending changes.
Candidates are filtered by the methods the user accepts and by
max_installment_months, and every one is safety-checked before it can be ranked.

Ranking applies the problem statement's tie-break in order: completes by
deadline, no spending changes, lowest total, earliest start, fewest payments,
lowest option id. Each rung has its own isolated test.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 10: Explanations

**Plain English:** The last column is a sentence explaining the decision. We found that all 25 reference explanations follow just eight fixed sentence patterns, and the number they quote is always the user's own minimum balance. So this is string formatting with a lookup on which pattern applies — no model needed, and it is perfectly consistent every time.

**Files:**
- Create: `code/buyorwait/explain.py`
- Test: `tests/unit/test_explain.py`

**Interfaces:**
- Produces: `explain.render(candidate: Candidate | None, request: Request, profile: Profile, safe: Decimal) -> str`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_explain.py`:
```python
from datetime import date
from decimal import Decimal

from buyorwait.changes import Change
from buyorwait.explain import render
from buyorwait.ranker import Candidate
from buyorwait.recurrence import Series
from buyorwait.types import Profile, Request

D = Decimal


def req(amount="25256", rd=date(2024, 3, 3), dcd=date(2024, 3, 20)):
    return Request("request_x", "user_x", rd, "purchase", D(amount), dcd, True, "t")


def prof(ccy="ZAR", minimum="18000"):
    return Profile("user_x", ccy, D("58481.1"), D(minimum), (), (), (), (),
                   ("full_payment",), None)


def cand(method, payments, changes=(), status="affordable_now"):
    return Candidate(method, tuple(payments), tuple(str(a) for _, a in payments),
                     sum(a for _, a in payments), tuple(changes), None, 10**9,
                     True, status)


def stop_change(desc="Family streaming plan", eid="event_476"):
    s = Series("streaming", desc, eid, "monthly", 10, D("19"), "debit",
               "stoppable", None, date(2025, 12, 10))
    return Change("stop", eid, s, None, D("19"))


def reduce_change(desc="Weekend food delivery", eid="event_989", new="665950"):
    s = Series("dining", desc, eid, "monthly", 10, D("1163530.49"), "debit",
               "reducible", D(new), date(2025, 4, 10))
    return Change("reduce_to", eid, s, D(new), D("497580.49"))


def test_t1_full_payment_today():
    c = cand("full_payment", [(date(2024, 3, 3), D("25256"))])
    assert render(c, req(), prof(), D("25256")) == (
        "Pay ZAR 25,256 today. This leaves at least ZAR 18,000 available "
        "over the next 90 days."
    )


def test_t2_full_payment_with_one_stop():
    c = cand("full_payment", [(date(2026, 1, 3), D("620.40"))],
             changes=[stop_change()], status="affordable_with_plan")
    got = render(c, req(amount="620.40", rd=date(2026, 1, 3)), prof("EUR", "800"), D("603.3"))
    assert got == (
        "Stop the family streaming plan, then pay EUR 620.40 today. "
        "This leaves at least EUR 800 available."
    )


def test_t3_full_payment_with_one_reduce():
    c = cand("full_payment", [(date(2025, 5, 3), D("13110000"))],
             changes=[reduce_change()], status="affordable_with_plan")
    got = render(c, req(amount="13110000", rd=date(2025, 5, 3)),
                 prof("IDR", "9500000"), D("12510645"))
    assert got == (
        "Reduce the weekend food delivery to IDR 665,950, then pay IDR 13,110,000 today. "
        "This leaves at least IDR 9,500,000 available."
    )


def test_t4_full_payment_with_a_stop_and_a_reduce():
    c = cand("full_payment", [(date(2026, 4, 3), D("1574.40"))],
             changes=[stop_change("Online backup subscription", "event_1815"),
                      reduce_change("Streaming subscription", "event_1816", "23.5")],
             status="affordable_with_plan")
    got = render(c, req(amount="1574.40", rd=date(2026, 4, 3)), prof("USD", "600"),
                 D("1543.35"))
    assert got == (
        "Stop the online backup subscription and reduce the streaming subscription "
        "to USD 23.50, then pay USD 1,574.40 today. "
        "This leaves at least USD 600 available."
    )


def test_t5_installments():
    c = cand("installments", [(date(2025, 8, 8), D("15952906.67")),
                              (date(2025, 9, 7), D("15952906.67")),
                              (date(2025, 10, 7), D("15952906.67"))],
             status="affordable_with_plan")
    got = render(c, req(amount="46018000", rd=date(2025, 8, 5)),
                 prof("IDR", "29158400"), D("17229139.2"))
    assert got == (
        "Use 3 installments of IDR 15,952,906.67, starting 8 August 2025. "
        "This leaves at least IDR 29,158,400 available."
    )


def test_t6_wait():
    c = cand("wait", [(date(2019, 11, 15), D("5491000"))], status="affordable_later")
    got = render(c, req(amount="5491000", rd=date(2019, 9, 3)),
                 prof("IDR", "2668700"), D("873000"))
    assert got == (
        "Pay IDR 5,491,000 in full on 15 November 2019. Paying earlier would take "
        "the balance below the IDR 2,668,700 minimum."
    )


def test_t7_not_recommended():
    got = render(None, req(amount="15488", rd=date(2025, 11, 6), dcd=date(2026, 1, 12)),
                 prof("ZAR", "13100"), D("737"))
    assert got == (
        "Do not make this payment by 12 January 2026. None of the available options "
        "keeps the ZAR 13,100 minimum protected."
    )


def test_t8_partial_payment():
    c = cand("partial_payment", [(date(2024, 9, 4), D("28820")),
                                 (date(2024, 9, 15), D("10840"))],
             status="affordable_with_plan")
    got = render(c, req(amount="39660", rd=date(2024, 9, 4), dcd=date(2024, 10, 4)),
                 prof("INR", "92800"), D("28820"))
    assert got == (
        "Pay INR 28,820 today and the remaining INR 10,840 on 15 September 2024. "
        "This completes the full request and keeps the INR 92,800 minimum protected."
    )
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/unit/test_explain.py -v`
Expected: FAIL — `No module named 'buyorwait.explain'`

- [ ] **Step 3: Implement explain.py**

`code/buyorwait/explain.py`:
```python
"""Render the decision_explanation.

All 25 reference explanations reduce to eight sentence patterns, and the figure
they quote is always the user's own minimum_balance_to_keep. This is therefore
deterministic string formatting - consistent every time, and no model call.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from .money import fmt_currency
from .ranker import Candidate
from .types import Profile, Request

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


def long_date(d: date) -> str:
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def _change_clause(candidate: Candidate, currency: str) -> str:
    parts = []
    for c in candidate.changes:
        if c.kind == "stop":
            parts.append(f"Stop {c.phrase}")
        else:
            parts.append(f"reduce {c.phrase} to {fmt_currency(currency, c.new_amount)}")
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0] + ", then "
    joined = parts[0] + " and " + " and ".join(parts[1:])
    return joined + ", then "


def render(candidate: Candidate | None, request: Request, profile: Profile,
           safe: Decimal) -> str:
    ccy = profile.home_currency
    minimum = fmt_currency(ccy, profile.minimum_balance_to_keep)
    total = fmt_currency(ccy, request.requested_amount)

    if candidate is None:
        # T7a - no safe eligible option exists at all.
        return (f"Do not make this payment by {long_date(request.desired_completion_date)}. "
                f"None of the available options keeps the {minimum} minimum protected.")

    if candidate.method == "installments":
        first_date, first_amount = candidate.payments[0]
        return (f"Use {len(candidate.payments)} installments of "
                f"{fmt_currency(ccy, first_amount)}, starting {long_date(first_date)}. "
                f"This leaves at least {minimum} available.")

    if candidate.method == "partial_payment":
        (_, a), (second_date, b) = candidate.payments
        return (f"Pay {fmt_currency(ccy, a)} today and the remaining "
                f"{fmt_currency(ccy, b)} on {long_date(second_date)}. "
                f"This completes the full request and keeps the {minimum} "
                f"minimum protected.")

    if candidate.method == "wait":
        when, amount = candidate.payments[0]
        return (f"Pay {fmt_currency(ccy, amount)} in full on {long_date(when)}. "
                f"Paying earlier would take the balance below the {minimum} minimum.")

    # full_payment, with or without spending changes
    clause = _change_clause(candidate, ccy)
    if clause:
        verb = "pay" if clause else "Pay"
        return (f"{clause}{verb} {total} today. "
                f"This leaves at least {minimum} available.")
    return (f"Pay {total} today. This leaves at least {minimum} available "
            f"over the next 90 days.")
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/unit/test_explain.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add code/buyorwait/explain.py tests/unit/test_explain.py
git commit -m "feat: deterministic explanation templates

All 25 reference explanations reduce to eight sentence patterns, always quoting
the user's own minimum_balance_to_keep. Rendered as string formatting rather
than a model call, so the column is perfectly consistent and costs no tokens.
Each template has a test asserting byte-exact output.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

Tasks 11–14 (contract gate and MVP wiring, calibration, evidence extraction, packaging and inspector) follow below.
