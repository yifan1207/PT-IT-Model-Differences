"""Shared helpers for Exp49 constrained continuation bridge."""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.poc.exp49_constrained_continuation_bridge import DEFAULT_EVENT_KIND


EXP47_DEFAULT_ROOT = Path(
    "results/exp47_same_base_recipe_specificity/"
    "exp47_same_base_recipe_specificity_20260504_0959_a100x24"
)
EXP49_DEFAULT_ROOT = Path("results/exp49_constrained_continuation_bridge")


def json_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            n += 1
    return n


def append_jsonl_gz(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def find_exp20_manifest(exp47_root: Path, model: str, prompt_mode: str = "raw_shared") -> Path:
    candidates = [
        exp47_root / "exp20" / prompt_mode / model / "exp20_validation_records.jsonl",
        exp47_root / "exp20" / prompt_mode / model / "exp20_records.jsonl",
        exp47_root / "exp20" / prompt_mode / model / "records.jsonl.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No Exp47 Exp20 manifest found for model={model} prompt_mode={prompt_mode}; "
        f"tried {', '.join(str(p) for p in candidates)}"
    )


def load_exp47_effect_slices(exp47_root: Path, event_kind: str = DEFAULT_EVENT_KIND) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Map ``(model, prompt_id, readout)`` to Exp47 metadata and support slices."""

    effects_path = exp47_root / "analysis" / "effects.csv"
    if not effects_path.exists():
        raise FileNotFoundError(f"Missing Exp47 effects file: {effects_path}")

    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    with open(effects_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("event_kind") != event_kind:
                continue
            key = (str(row["model"]), str(row["prompt_id"]), str(row["readout"]))
            entry = out.setdefault(
                key,
                {
                    "model": row["model"],
                    "prompt_id": row["prompt_id"],
                    "readout": row["readout"],
                    "recipe_group": row.get("recipe_group"),
                    "category": row.get("category"),
                    "source": row.get("source"),
                    "token_category": row.get("token_category"),
                    "position": safe_int(row.get("position")),
                    "position_ge_3": str(row.get("position_ge_3", "")).lower() == "true",
                    "position_ge_5": str(row.get("position_ge_5", "")).lower() == "true",
                    "pt_token_id": safe_int(row.get("pt_token_id")),
                    "it_token_id": safe_int(row.get("it_token_id")),
                    "pt_token_text": row.get("pt_token_text"),
                    "it_token_text": row.get("it_token_text"),
                    "exp47": {},
                    "slices": set(),
                },
            )
            entry["slices"].add(row.get("slice") or "full_1400")
            if row.get("slice") == "full_1400":
                entry["exp47"] = {
                    "P": safe_float(row.get("P")),
                    "M": safe_float(row.get("M")),
                    "C": safe_float(row.get("C")),
                    "U_PT__L_PT": safe_float(row.get("U_PT__L_PT")),
                    "U_PT__L_IT": safe_float(row.get("U_PT__L_IT")),
                    "U_IT__L_PT": safe_float(row.get("U_IT__L_PT")),
                    "U_IT__L_IT": safe_float(row.get("U_IT__L_IT")),
                }
    for entry in out.values():
        entry["slices"] = sorted(entry["slices"])
    return out


def load_exp47_event_metadata(exp47_root: Path, event_kind: str = DEFAULT_EVENT_KIND) -> dict[tuple[str, str], dict[str, Any]]:
    by_readout = load_exp47_effect_slices(exp47_root, event_kind=event_kind)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for (model, prompt_id, _readout), row in by_readout.items():
        out.setdefault(
            (model, prompt_id),
            {
                "model": model,
                "prompt_id": prompt_id,
                "recipe_group": row.get("recipe_group"),
                "category": row.get("category"),
                "source": row.get("source"),
                "token_category": row.get("token_category"),
                "position": row.get("position"),
                "position_ge_3": row.get("position_ge_3"),
                "position_ge_5": row.get("position_ge_5"),
                "pt_token_id": row.get("pt_token_id"),
                "it_token_id": row.get("it_token_id"),
                "pt_token_text": row.get("pt_token_text"),
                "it_token_text": row.get("it_token_text"),
                "slices": row.get("slices", ["full_1400"]),
            },
        )
    return out


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def bootstrap_ci(
    values: list[float],
    *,
    clusters: list[str] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    clean = [(float(v), str(c) if clusters is not None else str(i)) for i, (v, c) in enumerate(zip(values, clusters or range(len(values)))) if math.isfinite(float(v))]
    if not clean:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "se": None}
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for value, cluster in clean:
        by_cluster[cluster].append(value)
    cluster_ids = sorted(by_cluster)
    cluster_means = np.array([np.mean(by_cluster[c]) for c in cluster_ids], dtype=np.float64)
    mean = float(np.mean(cluster_means))
    if len(cluster_means) < 2 or n_boot <= 0:
        return {"n": len(clean), "n_clusters": len(cluster_ids), "mean": mean, "lo": mean, "hi": mean, "se": 0.0}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample_idx = rng.integers(0, len(cluster_means), size=len(cluster_means))
        draws[idx] = float(np.mean(cluster_means[sample_idx]))
    return {
        "n": len(clean),
        "n_clusters": len(cluster_ids),
        "mean": mean,
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "se": float(np.std(draws, ddof=1)),
    }


def horizon_values(max_tail: int, requested: Iterable[int]) -> list[int]:
    out = sorted({int(x) for x in requested if int(x) >= 0 and int(x) <= max_tail})
    if 0 not in out:
        out.insert(0, 0)
    return out


def cumulative_sum(values: list[float], horizon: int) -> float | None:
    n = int(horizon) + 1
    if len(values) < n:
        return None
    subset = values[:n]
    if any(v is None or not math.isfinite(float(v)) for v in subset):
        return None
    return float(sum(float(v) for v in subset))


def suffix_sum(values: list[float], horizon: int) -> float | None:
    if horizon <= 0:
        return 0.0
    if len(values) < horizon + 1:
        return None
    subset = values[1 : horizon + 1]
    if any(v is None or not math.isfinite(float(v)) for v in subset):
        return None
    return float(sum(float(v) for v in subset))


def stable_hash(text: str) -> int:
    # Small deterministic hash for shuffled-pairing, independent of Python's randomized hash seed.
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h

