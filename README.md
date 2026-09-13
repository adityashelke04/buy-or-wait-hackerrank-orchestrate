# Buy or Wait? — a financial decision agent

HackerRank Orchestrate, September 2026.

For each of the 250 requests in `dataset/requests.csv`, the agent decides whether a user
should pay in full, pay partially, use an installment offer, wait, or not proceed — and
writes a safe, explained recommendation to `output.csv`.

**Result on the 25 labeled samples: 74.3% average field accuracy**, measured by
`evaluation/score.py`. Full run: 250 rows in about 27 seconds, fully local, $0.

---

## Quick start

```bash
pip install -r requirements.txt
python code/main.py
```

That writes `output.csv` to the repository root and `evaluation/usage_report.md`.
No API key, no network and no GPU are needed.

Run the tests:

```bash
python -m pytest              # 269 tests, including OCR on the real images
python -m pytest -m "not slow"  # skips the image tests for a faster loop
```

Score against the labeled samples:

```bash
python code/main.py --requests-file sample_requests.csv --out output_samples.csv
python evaluation/score.py output_samples.csv
```

Build the submission archive (fails if it approaches the 50 MB limit):

```bash
python scripts/package.py
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--backend` | `rule` | Evidence reader: `rule` (local), `cloud` (Google AI Studio), `none` |
| `--estimator` | `p75` | How conservatively to forecast variable spending |
| `--requests-file` | `requests.csv` | Any file with the requests schema |
| `--out` | `output.csv` | Where to write predictions |

### Optional cloud backend

Copy `.env.example` to `.env`, set `LLM_API_KEY` to a free Google AI Studio key, then run
`python code/main.py --backend cloud`. Keys are read from the environment only — never
written to disk, logged, or placed in a prompt. `.env` is gitignored.

---

## Approach

### The core decision: numbers are computed, not generated

Reverse-engineering the 25 labeled samples showed that `amount_safe_to_pay` follows exact
arithmetic:

```
amount_safe_to_pay = clip(balance − minimum_balance − worst 90-day drawdown, 0, requested)
```

A language model cannot reliably produce a figure like `17229139.2`. So **every number in
`output.csv` comes from a deterministic cash-flow simulator**, and models are confined to
reading untrusted text and images.

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

The event log deliberately represents the same money more than once:

| Trap | Handling |
|---|---|
| Cancelled authorization + the settled charge | Count once |
| Charge + its refund | Both kept; they net to zero |
| Pending refund / pending income | Never counted |
| Unrealized investment valuation | Not cash; ignored |
| Failed payment + its retry | Count the retry only |
| Blank amount | Resolved from its image, never treated as zero |

### How `amount_safe_to_pay` is computed

Paying X today lowers every later balance by exactly X, so the lowest point of the
forecast is a straight line in X. The largest safe payment is therefore one subtraction:
`lowest balance − minimum`. A slow binary-search version ships alongside it, and a test
generates 200 random balance curves asserting both always agree.

### Reading the images

All 16 blank amounts are resolved with **RapidOCR** (PP-OCRv6 via ONNXRuntime —
Apache-2.0, pip-only, CPU). Reading the page is the easy part; choosing the right number
is not. A payslip carries a tax reference, an account number and percentage rates beside
the figure that matters, so size is no guide. Selection is driven by the **label** next to
each figure, chosen from the event's own wording — "net salary" selects Net Pay
(4,365,000) rather than Total Earnings (4,780,800); "outstanding rent balance" selects
Balance Due rather than the receipt total.

### Untrusted input

Messages and images are data, never instructions. The defence is structural rather than a
matter of prompt wording: the amendment schema is closed and has **no field in which a
decision could be expressed**, content is fenced as data, and implausible values are
clamped. Tests fire injection attacks through a provider that obediently echoes them and
assert the decision is unchanged.

---

## How it was built

Test-driven throughout, measured at every step.

- **Four test layers.** Contract tests assert every rule in the problem statement on all
  250 output rows. Unit tests cover each module. A golden **ratchet** fails the build if the
  sample score ever drops. Smoke tests assert two full runs are byte-identical.
- **Measure, don't guess.** `evaluation/calibrate.py` grid-searches forecasting choices
  against the labeled samples. Two ideas that looked right were rejected because they scored
  lower: stopping salary projection at the confirmed row (40.0%), and a fixed-interval
  cadence model (63.4%).
- **Fail closed.** On its first run the contract gate refused to write anything: rounding had
  broken the rule that a two-step partial payment must sum to the requested amount.

### Findings that moved the score

Every row was measured on the labeled samples before being kept.

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
| affordability_status | 84% |
| payment_plan | 84% |
| earliest_date_for_full_payment | 80% |
| spending_changes_needed | 88% |
| decision_explanation | 72% |
| amount_safe_to_pay | 20% |

### Ideas measured and rejected

| Idea | Result | Why it was plausible |
|---|---|---|
| Stop projecting salary beyond the confirmed row | 40.0% | "Do not invent unsupported future income" |
| Fixed-interval cadence for every category | 64.0% | The data really does use exact 7/10/14-day steps |
| Recover 10/14-day spending in all categories | 60.0% | More complete forecast — but it swept in discretionary spending |
| Drop discretionary series the calendar model already finds | 71.4% | Extending the "essential only" finding |

### Known limitation

`amount_safe_to_pay` is scored to within 0.5%, and remains the weakest field. Every
forecast component is now identified correctly; what differs is the exact estimated
amount. The ground-truth reserves are round numbers (157.00, 452.00, 568.00) while
history is noisy, which suggests the generator forecasts from hidden base amounts. The
residual bias is visible and consistent — the forecast under-reserves slightly, so it says
*affordable now* a little more often than the reference — and was deliberately not tuned
further against 25 rows, to avoid overfitting the 250 evaluation requests.

---

## No hardcoded answers

Enforced by tests, not by intention:

- the source is scanned for evaluation request ids and event ids, and fails if any appear
- `sample_requests.csv` is never named anywhere in `code/`; only the scorer reads it
- only the documented dataset files are opened

---

## Repository layout

```
code/
  main.py                   entry point
  buyorwait/
    types.py money.py fx.py io_loaders.py io_writer.py
    ledger.py recurrence.py forecast.py simulate.py solver.py
    changes.py ranker.py explain.py validate.py pipeline.py
    evidence/               schema, extractor, OCR, cache, providers, usage
evaluation/
  score.py                  per-field accuracy against the labeled samples
  calibrate.py              grid search over forecasting choices
  usage_report.md           generated by the final run
tests/  unit/ contract/ golden/ smoke/
scripts/package.py          builds code.zip
docs/superpowers/           design spec and implementation plan
```

Token usage and cost for the final run: [`evaluation/usage_report.md`](evaluation/usage_report.md).
