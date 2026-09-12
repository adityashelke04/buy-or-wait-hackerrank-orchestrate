"""Content-addressed response cache.

Makes a cold run a one-time cost, every rerun instant, and the whole pipeline
reproducible offline - a grader can regenerate output.csv with no model
installed at all. No credential ever enters a prompt, so none can enter the
cache.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_SEP = "\u0000"


class Cache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, object] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}          # a corrupt cache is a cold cache
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(*parts: object) -> str:
        """NUL-joined so ('ab','c') and ('a','bc') cannot collide."""
        blob = _SEP.join(str(p) for p in parts).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def has(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str):
        if key in self._data:
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return None

    def put(self, key: str, value) -> None:
        self._data[key] = value

    def flush(self) -> None:
        self.path.write_text(
            json.dumps(self._data, indent=1, sort_keys=True, ensure_ascii=False),
            encoding="utf-8")
