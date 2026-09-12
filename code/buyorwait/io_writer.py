"""Write output.csv with the exact required columns in the exact required order."""
from __future__ import annotations

import csv
from pathlib import Path

from .types import Decision


def write(decisions: list[Decision], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Decision.COLUMNS),
                                lineterminator="\n")
        writer.writeheader()
        for d in decisions:
            writer.writerow(d.as_row())
