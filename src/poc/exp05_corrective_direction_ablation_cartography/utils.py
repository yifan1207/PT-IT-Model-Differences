from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def navigate_path(obj: Any, path: str) -> Any:
    """Navigate dotted attribute paths with optional `[idx]` segments."""
    for part in path.split("."):
        if "[" in part:
            attr_name, rest = part.split("[", 1)
            idx = int(rest.rstrip("]"))
            obj = getattr(obj, attr_name)[idx]
        else:
            obj = getattr(obj, part)
    return obj


def sanitise_json(obj: Any) -> Any:
    """Replace non-finite floats recursively so all outputs are valid JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: sanitise_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitise_json(v) for v in obj]
    return obj


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitise_json(data), f, indent=2, ensure_ascii=False)


def save_jsonl(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(sanitise_json(row), ensure_ascii=False) + "\n")


def normalise_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def exact_match_any(prediction: str, golds: list[str]) -> bool:
    pred = normalise_text(prediction)
    return any(pred == normalise_text(g) for g in golds if g)


def starts_with_refusal(text: str) -> bool:
    text_l = text.strip().lower()
    refusal_prefixes = (
        "i can’t",
        "i can't",
        "i cannot",
        "sorry",
        "i’m sorry",
        "im sorry",
        "i won’t",
        "i will not",
        "i can’t help",
        "i can't help",
        "i’m unable",
        "im unable",
    )
    return any(text_l.startswith(p) for p in refusal_prefixes)


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.std(ddof=1))


def top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=np.int64)
    k = min(k, values.size)
    idx = np.argpartition(values, -k)[-k:]
    idx = idx[np.argsort(values[idx])[::-1]]
    return idx.astype(np.int64, copy=False)

