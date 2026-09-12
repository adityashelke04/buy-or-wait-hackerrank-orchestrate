"""Resolve an event id to its image file without re-reading the whole dataset."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def _index(dataset_dir: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(Path(dataset_dir) / "images.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            related = row["related_event_id"].strip()
            if related:
                out[related] = row["image_id"].strip()
    return out


def image_for_event(dataset_dir: Path, event_id: str) -> Path | None:
    image_id = _index(str(dataset_dir)).get(event_id)
    if not image_id:
        return None
    path = Path(dataset_dir) / "media" / "images" / f"{image_id}.png"
    return path if path.exists() else None
