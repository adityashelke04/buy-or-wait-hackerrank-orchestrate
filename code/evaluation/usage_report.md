# Token Usage and Cost Report

Covers the final full-dataset run that produced `output.csv`.

- Requests processed: **250**
- Wall-clock duration: **23.7s**
- Cache hits / misses: **0 / 209**

## Per model

| Provider | Model | Calls | Input tokens | Output tokens | Total |
|---|---|---:|---:|---:|---:|
| rapidocr (local, ONNXRuntime) | PP-OCRv6 detection + recognition | 11 | 0 | 0 | 0 |
| rule-based (local, no model) | multilingual message parser | 198 | 0 | 0 | 0 |

## Totals

- Model calls: **209**
- Input tokens: **0**
- Output tokens: **0**
- Total tokens: **0**
- Average tokens per request: **0.0**

## Cost

- Actual cost: **$0.0000**  (everything ran locally; no metered API was called)
- Per request: **$0.000000**
- Cloud-equivalent estimate at Gemini 2.0 Flash list prices: **$0.0000**

## Why the token counts are zero

The final run used only local, non-generative components. RapidOCR is a text-detection and recognition network (PP-OCRv6 via ONNXRuntime): it processes image pixels and has no token vocabulary, so it reports calls rather than tokens. The message parser is rule-based and consumes no tokens. Every number in output.csv is computed by the deterministic simulator, which is why no language model was needed to produce it.

An optional cloud backend (Google AI Studio, Gemini) is implemented behind `--backend cloud`. When used, its real prompt and completion token counts are recorded in this same report.

No API keys, credentials or sensitive configuration are included in this report.
