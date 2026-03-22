from __future__ import annotations

import csv
from pathlib import Path

from src.poc.exp5.utils import ensure_dir


def write_scores_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    ensure_dir(Path(path).parent)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

