#!/usr/bin/env python3
"""Analyze Exp32 terminal-window MLP write-out records.

The script supports two inputs:

* a new terminal-only Exp21-style run containing B_last3/B_last1 and
  D_last3/D_last1 conditions; and
* an existing Exp21 run with full per-layer rows, used as the full-late
  reference and as an optional CPU-only terminal-window audit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.cross_model.config import get_spec  # noqa: E402
from src.poc.exp20_divergence_token_counterfactual.collect import DEPTH_ABLATION_WINDOWS  # noqa: E402


DENSE5 = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"]
DEFAULT_REFERENCE = Path(
    "results/exp21_productive_opposition/exp21_full_productive_opposition_clean_20260426_053736"
)
WINDOWS = ["terminal_last3", "terminal_last1"]
METRICS = [
    "support_it_token",
    "support_pt_token",
    "margin_writein_it_vs_pt",
    "target_vs_alt_margin",
    "opposition_margin_it_vs_pt",
    "negative_parallel_norm",
    "delta_cosine_mlp",
]
EFFECTS = [
    "B_terminal_minus_A",
    "C_minus_D_terminal",
    "terminal_interaction",
    "terminal_late_weight_effect",
    "pure_IT_minus_PT",
]


def _iter_records(path: Path):
    if not path.exists():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _records_path(root: Path, mode: str, model: str) -> Path:
    return root / mode / model / "records.jsonl.gz"


def _load_records(root: Path, mode: str, model: str) -> list[dict[str, Any]]:
    return list(_iter_records(_records_path(root, mode, model)) or [])


def _window_bounds(model: str, window: str) -> tuple[int, int]:
    spec = get_spec(model)
    if window == "terminal_last3":
        return spec.n_layers - 3, spec.n_layers
    if window == "terminal_last1":
        return spec.n_layers - 1, spec.n_layers
    if window == "exp11_late":
        return DEPTH_ABLATION_WINDOWS[model]["late"]
    raise KeyError(window)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[Any]) -> float | None:
    kept = [_finite(value) for value in values]
    kept = [value for value in kept if value is not None]
    if not kept:
        return None
    return float(np.mean(kept))


def _condition_metric(
    *,
    condition_payload: dict[str, Any],
    model: str,
    window: str,
    metric: str,
) -> float | None:
    # New runs include terminal windows directly. Existing Exp21 records can be
    # re-audited from per-layer rows without GPU recomputation.
    direct = (((condition_payload.get("windows") or {}).get(window) or {}).get(metric))
    if direct is not None:
        return _finite(direct)
    layers = condition_payload.get("layers") or []
    if not layers:
        return None
    start, end = _window_bounds(model, window)
    if metric == "productive_opposition_rate":
        return _mean([1.0 if row.get("productive_opposition") else 0.0 for row in layers[start:end]])
    return _mean([row.get(metric) for row in layers[start:end]])


def _event_conditions(record: dict[str, Any], event_kind: str) -> dict[str, Any]:
    return (((record.get("events") or {}).get(event_kind) or {}).get("conditions") or {})


def _effect_value(
    *,
    record: dict[str, Any],
    model: str,
    event_kind: str,
    window: str,
    metric: str,
    effect: str,
    terminal_condition: str | None,
) -> float | None:
    conditions = _event_conditions(record, event_kind)
    a = conditions.get("A_pt_raw")
    c = conditions.get("C_it_chat")
    if terminal_condition == "last3":
        b = conditions.get("B_last3_raw")
        d = conditions.get("D_last3_ptswap")
    elif terminal_condition == "last1":
        b = conditions.get("B_last1_raw")
        d = conditions.get("D_last1_ptswap")
    else:
        b = conditions.get("B_late_raw")
        d = conditions.get("D_late_ptswap")
    # Phase A audits run on older Exp21 records that contain per-layer rows but
    # not explicit terminal-only graft conditions. In that case, use the
    # original late graft/swap condition and recompute terminal windows from the
    # stored layer rows. Phase B terminal runs override this with B_last*/D_last*.
    if b is None and terminal_condition in {"last3", "last1"}:
        b = conditions.get("B_late_raw")
    if d is None and terminal_condition in {"last3", "last1"}:
        d = conditions.get("D_late_ptswap")
    if not all(isinstance(x, dict) for x in [a, b, c, d]):
        return None
    av = _condition_metric(condition_payload=a, model=model, window=window, metric=metric)
    bv = _condition_metric(condition_payload=b, model=model, window=window, metric=metric)
    cv = _condition_metric(condition_payload=c, model=model, window=window, metric=metric)
    dv = _condition_metric(condition_payload=d, model=model, window=window, metric=metric)
    if any(value is None for value in [av, bv, cv, dv]):
        return None
    assert av is not None and bv is not None and cv is not None and dv is not None
    if effect == "B_terminal_minus_A":
        return bv - av
    if effect == "C_minus_D_terminal":
        return cv - dv
    if effect == "terminal_interaction":
        return (cv - dv) - (bv - av)
    if effect == "terminal_late_weight_effect":
        return 0.5 * ((bv - av) + (cv - dv))
    if effect == "pure_IT_minus_PT":
        return cv - av
    raise KeyError(effect)


def _prompt_effects(
    *,
    records: list[dict[str, Any]],
    model: str,
    event_kind: str,
    window: str,
    metric: str,
    effect: str,
    terminal_condition: str | None,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = _effect_value(
            record=record,
            model=model,
            event_kind=event_kind,
            window=window,
            metric=metric,
            effect=effect,
            terminal_condition=terminal_condition,
        )
        if value is not None and math.isfinite(value):
            grouped[str(record.get("prompt_id"))].append(float(value))
    return {prompt_id: float(np.mean(values)) for prompt_id, values in grouped.items() if values}


def _ci(values: np.ndarray) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap_values(arr: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float | None, float | None]:
    if arr.size == 0 or n_boot <= 0:
        return None, None
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    return _ci(arr[idx].mean(axis=1))


def _scope_models(scope: str, models: list[str]) -> list[str]:
    if scope == "Dense-5 family mean":
        return [m for m in models if m in DENSE5]
    if scope == "Gemma-removed Dense-4":
        return [m for m in models if m in DENSE5 and m != "gemma3_4b"]
    return models


def _reduce(values: list[float], scope: str) -> float:
    arr = np.array(values, dtype=float)
    if scope == "family median":
        return float(np.median(arr))
    if scope == "trimmed mean" and arr.size >= 3:
        arr = np.sort(arr)
        return float(arr[1:-1].mean())
    return float(arr.mean())


def _aggregate(
    *,
    arrays_by_model: dict[str, np.ndarray],
    refs_by_model: dict[str, np.ndarray],
    scope: str,
    models: list[str],
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    kept = [model for model in _scope_models(scope, models) if model in arrays_by_model and model in refs_by_model]
    if not kept:
        return None
    model_means = [float(arrays_by_model[model].mean()) for model in kept]
    ref_means = [float(refs_by_model[model].mean()) for model in kept]
    mean = _reduce(model_means, scope)
    ref = _reduce(ref_means, scope)
    row = {
        "n": int(sum(arrays_by_model[model].size for model in kept)),
        "mean": mean,
        "full_late_reference": ref,
        "retention_fraction": float(mean / ref) if ref else None,
    }
    if n_boot <= 0:
        row.update({"ci_low": None, "ci_high": None, "retention_ci_low": None, "retention_ci_high": None})
        return row
    boot_mean = []
    boot_ret = []
    for _ in range(n_boot):
        draw = []
        draw_ref = []
        for model in kept:
            arr = arrays_by_model[model]
            ref_arr = refs_by_model[model]
            idx = rng.integers(0, arr.size, size=arr.size)
            ref_idx = rng.integers(0, ref_arr.size, size=ref_arr.size)
            draw.append(float(arr[idx].mean()))
            draw_ref.append(float(ref_arr[ref_idx].mean()))
        mean_draw = _reduce(draw, scope)
        ref_draw = _reduce(draw_ref, scope)
        boot_mean.append(mean_draw)
        if ref_draw:
            boot_ret.append(mean_draw / ref_draw)
    lo, hi = _ci(np.array(boot_mean, dtype=float))
    rlo, rhi = _ci(np.array(boot_ret, dtype=float))
    row.update({"ci_low": lo, "ci_high": hi, "retention_ci_low": rlo, "retention_ci_high": rhi})
    return row


def analyze(
    *,
    root: Path,
    reference_root: Path,
    out_dir: Path,
    models: list[str],
    modes: list[str],
    event_kinds: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    quality: dict[str, Any] = {"root_counts": {}, "reference_counts": {}}

    for mode in modes:
        run_records = {model: _load_records(root, mode, model) for model in models}
        ref_records = {model: _load_records(reference_root, mode, model) for model in models}
        quality["root_counts"][mode] = {model: len(records) for model, records in run_records.items()}
        quality["reference_counts"][mode] = {model: len(records) for model, records in ref_records.items()}
        for event_kind in event_kinds:
            for window in WINDOWS:
                terminal = "last3" if window == "terminal_last3" else "last1"
                for metric in METRICS:
                    for effect in EFFECTS:
                        arrays_by_model: dict[str, np.ndarray] = {}
                        refs_by_model: dict[str, np.ndarray] = {}
                        for model in models:
                            effects = _prompt_effects(
                                records=run_records[model],
                                model=model,
                                event_kind=event_kind,
                                window=window,
                                metric=metric,
                                effect=effect,
                                terminal_condition=terminal,
                            )
                            refs = _prompt_effects(
                                records=ref_records[model],
                                model=model,
                                event_kind=event_kind,
                                window="exp11_late",
                                metric=metric,
                                effect="terminal_interaction" if effect == "terminal_interaction" else effect,
                                terminal_condition=None,
                            )
                            common = sorted(set(effects) & set(refs))
                            if common:
                                arr = np.array([effects[prompt_id] for prompt_id in common], dtype=float)
                                ref_arr = np.array([refs[prompt_id] for prompt_id in common], dtype=float)
                                arrays_by_model[model] = arr
                                refs_by_model[model] = ref_arr
                                lo, hi = _bootstrap_values(arr, n_boot, rng)
                                finite_ratio_mask = np.isfinite(arr) & np.isfinite(ref_arr) & (ref_arr != 0)
                                if np.any(finite_ratio_mask):
                                    rlo, rhi = _bootstrap_values(arr[finite_ratio_mask] / ref_arr[finite_ratio_mask], n_boot, rng)
                                else:
                                    rlo, rhi = None, None
                                rows.append(
                                    {
                                        "model": model,
                                        "prompt_mode": mode,
                                        "event_kind": event_kind,
                                        "window": window,
                                        "effect": effect,
                                        "metric": metric,
                                        "n": int(arr.size),
                                        "mean": float(arr.mean()),
                                        "ci_low": lo,
                                        "ci_high": hi,
                                        "full_late_reference": float(ref_arr.mean()),
                                        "retention_fraction": float(arr.mean() / ref_arr.mean()) if ref_arr.mean() else None,
                                        "retention_ci_low": rlo,
                                        "retention_ci_high": rhi,
                                    }
                                )
                        for scope in ["Dense-5 family mean", "Gemma-removed Dense-4", "family median", "trimmed mean"]:
                            row = _aggregate(
                                arrays_by_model=arrays_by_model,
                                refs_by_model=refs_by_model,
                                scope=scope,
                                models=models,
                                n_boot=n_boot,
                                rng=rng,
                            )
                            if row is not None:
                                rows.append(
                                    {
                                        "model": scope,
                                        "prompt_mode": mode,
                                        "event_kind": event_kind,
                                        "window": window,
                                        "effect": effect,
                                        "metric": metric,
                                        **row,
                                    }
                                )

    fields = [
        "model",
        "prompt_mode",
        "event_kind",
        "window",
        "effect",
        "metric",
        "n",
        "mean",
        "ci_low",
        "ci_high",
        "full_late_reference",
        "retention_fraction",
        "retention_ci_low",
        "retention_ci_high",
    ]
    with (out_dir / "exp32_terminal_mlp_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    family_rows = [row for row in rows if row["model"] in {"Dense-5 family mean", "Gemma-removed Dense-4", "family median", "trimmed mean"}]
    with (out_dir / "exp32_terminal_mlp_family_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(family_rows)
    window_rows = [row for row in rows if row["metric"] == "margin_writein_it_vs_pt" and row["effect"] == "terminal_interaction"]
    with (out_dir / "exp32_terminal_mlp_window_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(window_rows)
    with (out_dir / "exp32_terminal_mlp_retention.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([row for row in rows if row["retention_fraction"] is not None])

    summary = {
        "root": str(root),
        "reference_root": str(reference_root),
        "models": models,
        "modes": modes,
        "event_kinds": event_kinds,
        "n_bootstrap": n_boot,
        "quality": quality,
        "effects": rows,
    }
    (out_dir / "exp32_terminal_mlp_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _plot(rows, out_dir / "exp32_terminal_mlp_writeout.png")
    return summary


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    wanted = [
        row
        for row in rows
        if row["model"] in {"Dense-5 family mean", "Gemma-removed Dense-4"}
        and row["prompt_mode"] == "raw_shared"
        and row["event_kind"] == "first_diff"
        and row["effect"] == "terminal_interaction"
        and row["metric"] == "margin_writein_it_vs_pt"
    ]
    if not wanted:
        return
    labels = [f"{row['model']}\n{row['window'].replace('terminal_', '')}" for row in wanted]
    means = np.array([float(row["mean"]) for row in wanted])
    lows = np.array([float(row["ci_low"]) if row["ci_low"] is not None else float(row["mean"]) for row in wanted])
    highs = np.array([float(row["ci_high"]) if row["ci_high"] is not None else float(row["mean"]) for row in wanted])
    ret = np.array([float(row["retention_fraction"]) for row in wanted])
    x = np.arange(len(wanted))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(x, means, color="#4c78a8")
    axes[0].errorbar(x, means, yerr=np.vstack([means - lows, highs - means]), fmt="none", color="black", capsize=3)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_ylabel("MLP IT-vs-PT margin interaction")
    axes[0].set_title("Exp32 terminal MLP interaction")
    axes[1].bar(x, ret, color="#f58518")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_ylabel("Retention vs full-late MLP reference")
    axes[1].set_title("Descriptive retention")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=DENSE5)
    parser.add_argument("--modes", nargs="+", default=["raw_shared"])
    parser.add_argument("--event-kinds", nargs="+", default=["first_diff"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260502)
    args = parser.parse_args()
    out_dir = args.out_dir or (args.root / "analysis")
    summary = analyze(
        root=args.root,
        reference_root=args.reference_root,
        out_dir=out_dir,
        models=args.models,
        modes=args.modes,
        event_kinds=args.event_kinds,
        n_boot=args.n_bootstrap,
        seed=args.seed,
    )
    print(json.dumps({"out_dir": str(out_dir), "quality": summary["quality"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
