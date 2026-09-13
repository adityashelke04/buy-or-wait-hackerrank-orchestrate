"""One interface, four backends.

Swapping the backend never changes the decision logic - only the quality of the
amendments fed into it. That is what lets the same test suite measure all of
them on identical ground.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class LLMProvider(Protocol):
    name: str

    def complete_json(self, prompt: str, schema: dict) -> dict | None: ...
    def read_image(self, path: Path, prompt: str, schema: dict) -> dict | None: ...


class StubProvider:
    """Test double. Returns a fixed payload and counts calls."""
    name = "stub"

    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls = 0

    def complete_json(self, prompt: str, schema: dict):
        self.calls += 1
        return self.payload

    def read_image(self, path: Path, prompt: str, schema: dict):
        self.calls += 1
        return self.payload


def build(backend: str | None = None) -> LLMProvider:
    backend = backend or os.environ.get("LLM_BACKEND", "rule")
    if backend == "cloud":
        from .cloud_provider import CloudProvider
        return CloudProvider()
    from .rule_provider import RuleProvider
    return RuleProvider()
