"""Optional cloud backend - Google AI Studio (Gemini) by default.

Uses the OpenAI-compatible endpoint, so Groq, OpenRouter or OpenAI itself work by
changing LLM_BASE_URL and LLM_CLOUD_MODEL. The API key is read from the
environment only; it is never written to disk, logged, or placed in a prompt.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from .usage import USAGE

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-2.0-flash"


class CloudProvider:
    name = "cloud"

    def __init__(self, client=None, model: str | None = None) -> None:
        self.model = model or os.environ.get("LLM_CLOUD_MODEL", DEFAULT_MODEL)
        if client is not None:
            self._client = client
            return
        key = os.environ.get("LLM_API_KEY")
        if not key:
            raise RuntimeError(
                "LLM_API_KEY is not set. Put it in .env, or run with --backend rule.")
        from openai import OpenAI
        self._client = OpenAI(api_key=key,
                              base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL))

    def _chat(self, content):
        try:
            resp = self._client.chat.completions.create(
                model=self.model, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": content}],
            )
        except Exception:
            return None                  # fail safe: ledger facts alone
        usage = getattr(resp, "usage", None)
        USAGE.record(provider="cloud (Google AI Studio)", model=self.model,
                     input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                     output_tokens=getattr(usage, "completion_tokens", 0) or 0)
        try:
            return json.loads(resp.choices[0].message.content)
        except (json.JSONDecodeError, IndexError, TypeError, AttributeError):
            return None

    def complete_json(self, prompt: str, schema: dict):
        return self._chat(prompt)

    def read_image(self, path: Path, prompt: str, schema: dict):
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        return self._chat([
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ])
