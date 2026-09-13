# Buy or Wait? — a financial decision agent

HackerRank Orchestrate, September 2026.

For each of the 250 requests in `dataset/requests.csv`, the agent decides whether a user
should pay in full, pay partially, use an installment offer, wait, or not proceed — and
writes a safe, explained recommendation to `output.csv`.

**Local result: 74.3% average field accuracy on the 25 labeled sample requests**
(`python code/evaluation/main.py`). Full run: 250 rows in about 25–30 seconds on a laptop
CPU, fully local, no API key, $0. The official score comes from hidden answers and will
differ; see [Limitations](#known-limitations).

---

## Quick start

Tested on Python 3.13 (Windows). No GPU, network access or API key is needed; the OCR
models ship inside the `rapidocr` wheel.

```bash
pip install -r requirements.txt
python code/main.py
```

This writes `output.csv` to the repository root and the token/cost report to
`evaluation/usage_report.md` (mirrored to `code/evaluation/usage_report.md`).

| Task | Command |
|---|---|
| Produce `output.csv` | `python code/main.py` |
| Score against the 25 labeled samples | `python code/evaluation/main.py` |
| Run every test (331) | `python -m pytest` |
| Fast loop, skipping the OCR and full-run tests | `python -m pytest -m "not slow"` |
| Grid-search forecasting settings | `python evaluation/calibrate.py` |
| Build `code.zip` | `python scripts/package.py` |

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--backend` | `rule` | Evidence reader: `rule` (local), `cloud` (OpenAI-compatible API), `none` |
| `--estimator` | `p75` | How conservatively to forecast variable spending: `last`, `mean`, `median`, `p75`, `max`, `max3` |
| `--requests-file` | `requests.csv` | Any file with the requests schema, inside `--dataset` |
| `--dataset` | `dataset/` | Folder holding the CSVs and `media/images/` |
| `--out` | `output.csv` | Where to write predictions |

Invalid values are rejected before any work starts.

### Optional cloud backend

The submitted run does not use it. To try it, copy `.env.example` to `.env` and set
`LLM_API_KEY` (a Google AI Studio key by default; `LLM_BASE_URL` and `LLM_CLOUD_MODEL`
switch to any OpenAI-compatible provider), then run `python code/main.py --backend cloud`.

`code/main.py` reads `.env` itself with the standard library; a variable already set in the
shell always wins. The key is never printed, logged, cached or placed in a prompt. `.env` is
gitignored and is refused by the packager.

---

## Approach

### Numbers are computed, not generated

Reverse-engineering the 25 labeled samples showed that `amount_safe_to_pay` has this
structure:

```
amount_safe_to_pay = clip(balance − minimum_balance − worst 90-day drawdown, 0, requested)
```

A language model cannot reliably produce a figure like `17229139.2`, so **every number in
`output.csv` comes from a deterministic cash-flow simulator**. Messages and images are read
only to adjust the facts the simulator starts from.

### Pipeline

```
CSVs + PNGs
  → loaders      typed records; money as Decimal, never float
  → ledger       resolves the six double-counting traps
  → evidence     messages and images → validated amendments
  → recurrence   which expenses and income repeat, and how often
  → forecast     day-by-day balance for 90 days
  → solver       safe amount, earliest safe date, every eligible plan
  → ranker       the problem statement's six tie-break rules
  → explain      templated explanation
  → validate     contract gate: refuses to write an invalid CSV
  → output.csv
```

### The six traps the ledger resolves

| Trap | Handling |
|---|---|
| Cancelled authorization + the settled charge | Count once |
| Charge + its refund | Both kept; they net to zero |
| Pending refund / pending income | Never counted |
| Unrealized investment valuation | Not cash; ignored |
| Failed payment + its retry | Count the retry only |
| Blank amount | Resolved from its image, never treated as zero |

### How `amount_safe_to_pay` is computed

Paying X today lowers every later balance by exactly X, so the lowest point of the forecast
is a straight line in X. The largest safe payment is therefore one subtraction:
`lowest balance − minimum`, rounded down to the cent. A slow binary-search version ships
alongside it, and a test generates 200 random balance curves asserting both always agree.

### Choosing the plan

Every eligible plan — full payment today, each supplied option, a two-step partial payment,
waiting for the first safe date, each with zero to three permitted spending changes — is
simulated over the 90-day window. Unsafe or ineligible plans are discarded, and the rest are
ranked by the problem statement's six rules. A spending change only ever touches a
non-protected, flexible expense in a category the user allows, and `reduce_to` always uses
the event's own `minimum_allowed_amount`.

### Output format

Amounts follow the labeled samples exactly: `amount_safe_to_pay` is minimal (`603.3`),
while amounts inside `payment_plan` and `reduce_to` are whole numbers or carry two decimals
(`620.40`, `reduce_to:event_1816:23.50`), even when the source CSV wrote `620.4`.

### Reading the images

All 16 blank amounts are resolved with **RapidOCR** (PP-OCRv6 via ONNXRuntime — Apache-2.0,
pip-only, CPU). Reading the page is the easy part; choosing the right number is not. A
payslip carries a tax reference, an account number and percentage rates beside the figure
that matters, so size is no guide. Selection is driven by the **label** next to each figure,
chosen from the event's own wording — "net salary" selects Net Pay (4,365,000) rather than
Total Earnings (4,780,800); "outstanding rent balance" selects Balance Due rather than the
receipt total.

### Untrusted input

Messages and images are data, never instructions. The defence is structural rather than a
matter of prompt wording: the amendment schema is closed and has **no field in which a
decision could be expressed**, content is fenced as data, and implausible values are
clamped. Tests fire injection attacks through a provider that obediently echoes them and
assert the decision is unchanged.

---

## Quality gates

Test-driven throughout, measured at every step. **331 tests** in four layers:

- **Contract** — every rule in the problem statement, asserted on all 250 rows produced by the
  shipped configuration (OCR included). The same rules run as a gate inside the pipeline, which
  refuses to write an invalid `output.csv`. On its first run it caught a rounding bug that broke
  the rule that a two-step partial payment must sum to the requested amount.
- **Unit** — each module, including the contract gate itself (each check must reject a
  deliberately malformed row).
- **Golden ratchet** — fails the build if the sample score ever drops.
- **Smoke** — two full runs are byte-identical; the committed `output.csv` is exactly what the
  current code produces; the CLI rejects bad input; `code.zip` builds under the size limit
  and passes the sensitive-content scan.

### No hardcoded answers

Enforced by tests, not by intention:

- the solution source is scanned for evaluation request ids and event ids, and fails if any appear
- the labeled sample file is never named anywhere in the solution; only the scorer reads it
- only the documented dataset files are opened

### Submission hygiene

`scripts/package.py` refuses to build `code.zip` if any text file would ship an API key or
token, a private key, an IP address, an email address, an absolute path from a developer
machine, or a filled-in secret. `.env`, `log.txt`, caches, `__pycache__` and `.git` are
never packaged.

---

## How the score moved

Every change was measured on the labeled samples before being kept.

| Change | Score |
|---|---|
| First end-to-end run | 60.6% |
| Detect everyday spending by category — the dataset rotates descriptions ("Supermarket basket", "Bulk pantry shop"), which hid weekly groceries entirely | 65.7% |
| Keep one-off purchases out of recurring estimates | 65.7% with evidence on |
| Project a confirmed next salary monthly — new employees had no income after payday | 69.1% |
| **Income confirmation policy** — stop projecting income after a "Final employer payroll", and never project volatile gig payouts or unconfirmed secondary income | 72.0% |
| Charge bills due on the request date that have not settled | 72.6% |
| Recover 10- and 14-day spending in **essential** (protected) categories only | 73.7% |
| Scorer fix: number matching no longer swallows trailing punctuation *(measurement correction)* | 74.3% |

| Field | Accuracy |
|---|---|
| recommended_payment_method | 92% |
| spending_changes_needed | 88% |
| affordability_status | 84% |
| payment_plan | 84% |
| earliest_date_for_full_payment | 80% |
| decision_explanation | 72% |
| amount_safe_to_pay | 20% |

### Ideas measured and rejected

| Idea | Result | Why it was plausible |
|---|---|---|
| Stop projecting salary beyond the confirmed row | 40.0% | "Do not invent unsupported future income" |
| Fixed-interval cadence for every category | 63.4% when first tried, 64.0% re-measured later | The data really does use exact 7/10/14-day steps |
| Recover 10/14-day spending in all categories | 60.0% | More complete forecast — but it swept in discretionary spending |
| Drop discretionary series the calendar model already finds | 71.4% | Extending the "essential only" finding |

## Known limitations

- **`amount_safe_to_pay` is the weakest field (20% within 0.5%).** Every forecast component is
  identified, but the estimated amounts differ. The reference reserves are round numbers
  (157.00, 452.00, 568.00) while history is noisy, which suggests the generator forecasts from
  hidden base amounts. The forecast under-reserves slightly, so it says *affordable now* a
  little more often than the reference. This was deliberately not tuned further against 25
  rows, to avoid overfitting the 250 evaluation requests.
- **74.3% is optimistic.** Settings were chosen on the same 25 samples they are scored on.
- The rule-based message parser recognises confirmed salary changes, date changes and
  percentage increases in English and Indonesian; other message types leave the ledger
  unchanged, which is the conservative default.

---

## Repository layout

```
code/
  main.py                   entry point: python code/main.py
  evaluation/main.py        one-command local score
  buyorwait/
    types.py money.py fx.py io_loaders.py io_writer.py
    ledger.py recurrence.py forecast.py simulate.py solver.py
    changes.py ranker.py explain.py validate.py pipeline.py
    evidence/               schema, extractor, OCR, cache, providers, usage
evaluation/
  score.py                  per-field accuracy against the labeled samples
  calibrate.py              grid search over forecasting choices
  usage_report.md           token and cost report, generated by the final run
tests/  unit/ contract/ golden/ smoke/
scripts/package.py          builds code.zip and scans it for sensitive content
docs/superpowers/           design spec and implementation plan (historical)
```

Token usage and cost for the final run: [`evaluation/usage_report.md`](evaluation/usage_report.md).
