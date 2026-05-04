from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


CELL_ORDER = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")
HYBRID_PRIMARY_CELLS = ("U_PT__L_IT", "U_IT__L_IT")


def stable_int(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: Iterable[Any]) -> float | None:
    vals = [x for x in (finite(v) for v in values) if x is not None]
    return float(np.mean(vals)) if vals else None


def lexical_metrics(text: str, *, post_first_text: str | None = None) -> dict[str, Any]:
    """Cheap deterministic assistant/register markers for short continuations."""
    text = text or ""
    post = text if post_first_text is None else (post_first_text or "")

    def one(blob: str, prefix: str) -> dict[str, Any]:
        stripped = blob.strip()
        lower = stripped.lower()
        bullets = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", blob))
        markdown = len(re.findall(r"```|`[^`]+`|\*\*|^#+\s+", blob, flags=re.MULTILINE))
        refusals = len(
            re.findall(
                r"\b(?:sorry|can't|cannot|unable|not able|won't|i can(?:not|'t) help|unsafe|harmful)\b",
                lower,
            )
        )
        answer_opening = bool(
            re.match(
                r"^\s*(?:sure|certainly|of course|here(?:'s| is)?|yes|no|i(?:'d| can| will| would)?|the answer|to\b|in\b)",
                lower,
            )
        )
        direct_answer = bool(re.match(r"^\s*(?:the answer is|answer:|yes|no|sure|certainly|here)", lower))
        newline_count = blob.count("\n")
        words = re.findall(r"\S+", blob)
        structure_score = (
            min(bullets, 3)
            + min(markdown, 2)
            + min(newline_count, 4) * 0.25
            + (1.0 if answer_opening else 0.0)
            + (1.0 if direct_answer else 0.0)
            + min(refusals, 2) * 0.5
        )
        raw_continuation_markers = bool(
            re.match(r"^\s*(?:and|or|but|because|,|\.|;|:|\)|\]|the)\b", lower)
        )
        return {
            f"{prefix}_char_len": len(blob),
            f"{prefix}_word_count": len(words),
            f"{prefix}_newline_count": newline_count,
            f"{prefix}_bullet_count": bullets,
            f"{prefix}_markdown_marker_count": markdown,
            f"{prefix}_refusal_marker_count": refusals,
            f"{prefix}_answer_opening": answer_opening,
            f"{prefix}_direct_answer_opening": direct_answer,
            f"{prefix}_raw_continuation_like_opening": raw_continuation_markers,
            f"{prefix}_lexical_it_like_score": float(structure_score - (0.5 if raw_continuation_markers else 0.0)),
        }

    return {**one(text, "full"), **one(post, "post_first")}


def cluster_bootstrap_ci(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    cluster_key: str = "prompt_id",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    values_by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = finite(row.get(metric))
        if val is None:
            continue
        values_by_cluster[str(row.get(cluster_key, ""))].append(val)
    clusters = sorted(k for k, vals in values_by_cluster.items() if vals)
    if not clusters:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n_prompt_clusters": 0}
    cluster_means = np.asarray([np.mean(values_by_cluster[k]) for k in clusters], dtype=float)
    estimate = float(cluster_means.mean())
    if len(clusters) == 1 or n_boot <= 0:
        return {
            "estimate": estimate,
            "ci_low": estimate,
            "ci_high": estimate,
            "n_prompt_clusters": len(clusters),
        }
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=float)
    n = len(cluster_means)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(cluster_means[idx].mean())
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n_prompt_clusters": len(clusters),
    }


def family_balanced_ci(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    models: list[str],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    per_model = {model: [row for row in rows if row.get("model") == model] for model in models}
    per_model = {model: subset for model, subset in per_model.items() if subset}
    if not per_model:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "n_families": 0,
            "n_prompt_clusters": 0,
            "family_means": {},
        }
    family_means: dict[str, float] = {}
    cluster_values: dict[str, np.ndarray] = {}
    n_clusters = 0
    for model, subset in per_model.items():
        by_cluster: dict[str, list[float]] = defaultdict(list)
        for row in subset:
            val = finite(row.get(metric))
            if val is not None:
                by_cluster[str(row.get("prompt_id", ""))].append(val)
        vals = np.asarray([np.mean(v) for _, v in sorted(by_cluster.items()) if v], dtype=float)
        if vals.size == 0:
            continue
        cluster_values[model] = vals
        family_means[model] = float(vals.mean())
        n_clusters += int(vals.size)
    if not cluster_values:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "n_families": 0,
            "n_prompt_clusters": 0,
            "family_means": {},
        }
    estimate = float(np.mean(list(family_means.values())))
    if n_boot <= 0:
        return {
            "estimate": estimate,
            "ci_low": estimate,
            "ci_high": estimate,
            "n_families": len(cluster_values),
            "n_prompt_clusters": n_clusters,
            "family_means": family_means,
        }
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=float)
    model_names = sorted(cluster_values)
    for i in range(int(n_boot)):
        sampled_family_means = []
        for model in model_names:
            vals = cluster_values[model]
            idx = rng.integers(0, len(vals), size=len(vals))
            sampled_family_means.append(float(vals[idx].mean()))
        boot[i] = float(np.mean(sampled_family_means))
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n_families": len(cluster_values),
        "n_prompt_clusters": n_clusters,
        "family_means": family_means,
    }
