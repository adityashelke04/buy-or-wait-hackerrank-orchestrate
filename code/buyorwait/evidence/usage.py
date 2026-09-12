"""Token accounting for evaluation/usage_report.md."""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

# Indicative public per-million-token prices, used ONLY for the
# cloud-equivalent comparison. A local run costs nothing.
PRICES = {"gemini-2.0-flash": (0.10, 0.40)}


class Usage:
    def __init__(self) -> None:
        self.calls: dict[tuple[str, str], int] = defaultdict(int)
        self.input_tokens: dict[tuple[str, str], int] = defaultdict(int)
        self.output_tokens: dict[tuple[str, str], int] = defaultdict(int)
        self.started = time.time()
        self.cache_hits = 0
        self.cache_misses = 0

    def record(self, provider: str, model: str, input_tokens: int,
               output_tokens: int) -> None:
        key = (provider, model)
        self.calls[key] += 1
        self.input_tokens[key] += int(input_tokens or 0)
        self.output_tokens[key] += int(output_tokens or 0)

    def write_report(self, path: Path, requests: int) -> None:
        total_calls = sum(self.calls.values())
        total_in = sum(self.input_tokens.values())
        total_out = sum(self.output_tokens.values())
        total = total_in + total_out
        elapsed = time.time() - self.started
        local = all(p == "ollama" for (p, _) in self.calls) if self.calls else True

        est = 0.0
        for (provider, model) in self.calls:
            if model in PRICES:
                pin, pout = PRICES[model]
                est += (self.input_tokens[(provider, model)] / 1e6) * pin
                est += (self.output_tokens[(provider, model)] / 1e6) * pout
        actual = 0.0 if local else est

        lines = [
            "# Token Usage and Cost Report", "",
            "Covers the final full-dataset run that produced `output.csv`.", "",
            f"- Requests processed: **{requests}**",
            f"- Wall-clock duration: **{elapsed:.1f}s**",
            f"- Cache hits / misses: **{self.cache_hits} / {self.cache_misses}**",
            "", "## Per model", "",
            "| Provider | Model | Calls | Input tokens | Output tokens | Total |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for (provider, model) in sorted(self.calls):
            i = self.input_tokens[(provider, model)]
            o = self.output_tokens[(provider, model)]
            lines.append(f"| {provider} | {model} | {self.calls[(provider, model)]} "
                         f"| {i:,} | {o:,} | {i + o:,} |")
        if not self.calls:
            lines.append("| - | - | 0 | 0 | 0 | 0 |")

        lines += [
            "", "## Totals", "",
            f"- Model calls: **{total_calls}**",
            f"- Input tokens: **{total_in:,}**",
            f"- Output tokens: **{total_out:,}**",
            f"- Total tokens: **{total:,}**",
            f"- Average tokens per request: **{total / requests if requests else 0:,.1f}**",
            "", "## Cost", "",
            f"- Actual cost: **${actual:,.4f}**"
            + ("  (models run locally via Ollama; no metered API was used)"
               if local else ""),
            f"- Per request: **${actual / (requests or 1):,.6f}**",
        ]
        if local:
            lines.append(
                "- Cloud-equivalent estimate at Gemini 2.0 Flash list prices: "
                f"**${(total_in / 1e6) * 0.10 + (total_out / 1e6) * 0.40:,.4f}**")
        lines += ["", "No API keys, credentials or sensitive configuration are "
                  "included in this report.", ""]

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")


USAGE = Usage()
