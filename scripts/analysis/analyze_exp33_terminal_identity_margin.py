#!/usr/bin/env python3
"""Analyze Exp33 terminal token-identity and margin validation records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DENSE5 = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"]
CONDITIONS = [
    "A_pt_raw",
    "B_mid_raw",
    "B_late_raw",
    "B_last3_raw",
    "B_last1_raw",
    "B_midlate_raw",
    "C_it_chat",
    "D_mid_ptswap",
    "D_late_ptswap",
    "D_last3_ptswap",
    "D_last1_ptswap",
    "D_midlate_ptswap",
]


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("rb") as handle:
        for raw in handle:
            if raw.strip():
                yield json.loads(raw.decode("utf-8", errors="ignore"))


def _load_records(root: Path, mode: str, model: str) -> list[dict[str, Any]]:
    return list(_iter_jsonl(root / mode / model / "exp20_validation_records.jsonl") or [])


def _readout(record: dict[str, Any], event_kind: str) -> dict[str, Any]:
    payload = (record.get("readouts") or {}).get(event_kind)
    return payload if isinstance(payload, dict) else {}


def _class_label(record: dict[str, Any], event_kind: str, condition: str) -> str:
    cls = (((_readout(record, event_kind).get("condition_token_at_step") or {}).get(condition) or {}).get("class"))
    return str(cls or "missing")


def _class_rate(record: dict[str, Any], event_kind: str, condition: str, target: str) -> float | None:
    cls = _class_label(record, event_kind, condition)
    if cls == "missing":
        return None
    return 1.0 if cls == target else 0.0


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _margin(record: dict[str, Any], event_kind: str, condition: str, window: str) -> float | None:
    condition_payload = (((_readout(record, event_kind).get("conditions") or {}).get(condition) or {}))
    payload = (((condition_payload.get("windows") or {}).get(window) or {}).get("it_minus_pt_margin") or {})
    return _finite(payload.get("total_delta"))


def _paired(records: list[dict[str, Any]], fn) -> list[float]:
    out = []
    for record in records:
        value = fn(record)
        if value is not None and math.isfinite(value):
            out.append(float(value))
    return out


def _diff(a, b):
    def inner(record):
        av = a(record)
        bv = b(record)
        if av is None or bv is None:
            return None
        return float(av) - float(bv)

    return inner


def _interaction(ml, m, l, a):
    def inner(record):
        vals = [fn(record) for fn in [ml, m, l, a]]
        if any(value is None for value in vals):
            return None
        mlv, mv, lv, av = [float(value) for value in vals]
        return (mlv - mv) - (lv - av)

    return inner


def _effect_defs(event_kind: str):
    cls = lambda condition, target: (lambda record: _class_rate(record, event_kind, condition, target))
    margin = lambda condition, window: (lambda record: _margin(record, event_kind, condition, window))
    return [
        ("B_last3 IT-token transfer", "identity", "B_last3_raw", "terminal_last3", cls("B_last3_raw", "it")),
        ("B_last1 IT-token transfer", "identity", "B_last1_raw", "terminal_last1", cls("B_last1_raw", "it")),
        ("B_mid IT-token transfer", "identity", "B_mid_raw", "exp11_mid", cls("B_mid_raw", "it")),
        ("B_late IT-token transfer", "identity", "B_late_raw", "exp11_late", cls("B_late_raw", "it")),
        ("B_midlate IT-token transfer", "identity", "B_midlate_raw", "condition_graft_window", cls("B_midlate_raw", "it")),
        ("D_last3 PT-token transfer", "identity", "D_last3_ptswap", "terminal_last3", cls("D_last3_ptswap", "pt")),
        ("D_last1 PT-token transfer", "identity", "D_last1_ptswap", "terminal_last1", cls("D_last1_ptswap", "pt")),
        ("D_mid PT-token transfer", "identity", "D_mid_ptswap", "exp11_mid", cls("D_mid_ptswap", "pt")),
        ("D_late PT-token transfer", "identity", "D_late_ptswap", "exp11_late", cls("D_late_ptswap", "pt")),
        ("D_midlate PT-token transfer", "identity", "D_midlate_ptswap", "condition_graft_window", cls("D_midlate_ptswap", "pt")),
        (
            "PT host margin: last3 minus A",
            "margin",
            "B_last3_raw",
            "terminal_last3",
            _diff(margin("B_last3_raw", "terminal_last3"), margin("A_pt_raw", "terminal_last3")),
        ),
        (
            "PT host margin: last1 minus A",
            "margin",
            "B_last1_raw",
            "terminal_last1",
            _diff(margin("B_last1_raw", "terminal_last1"), margin("A_pt_raw", "terminal_last1")),
        ),
        (
            "PT host margin: mid minus A",
            "margin",
            "B_mid_raw",
            "exp11_mid",
            _diff(margin("B_mid_raw", "exp11_mid"), margin("A_pt_raw", "exp11_mid")),
        ),
        (
            "PT host margin: late minus A",
            "margin",
            "B_late_raw",
            "exp11_late",
            _diff(margin("B_late_raw", "exp11_late"), margin("A_pt_raw", "exp11_late")),
        ),
        (
            "IT host margin: C minus D_last3",
            "margin",
            "D_last3_ptswap",
            "terminal_last3",
            _diff(margin("C_it_chat", "terminal_last3"), margin("D_last3_ptswap", "terminal_last3")),
        ),
        (
            "IT host margin: C minus D_last1",
            "margin",
            "D_last1_ptswap",
            "terminal_last1",
            _diff(margin("C_it_chat", "terminal_last1"), margin("D_last1_ptswap", "terminal_last1")),
        ),
        (
            "IT host margin: C minus D_mid",
            "margin",
            "D_mid_ptswap",
            "exp11_mid",
            _diff(margin("C_it_chat", "exp11_mid"), margin("D_mid_ptswap", "exp11_mid")),
        ),
        (
            "IT host margin: C minus D_late",
            "margin",
            "D_late_ptswap",
            "exp11_late",
            _diff(margin("C_it_chat", "exp11_late"), margin("D_late_ptswap", "exp11_late")),
        ),
        (
            "terminal last3 margin interaction",
            "margin_interaction",
            "last3_2x2",
            "terminal_last3",
            _interaction(
                margin("C_it_chat", "terminal_last3"),
                margin("D_last3_ptswap", "terminal_last3"),
                margin("B_last3_raw", "terminal_last3"),
                margin("A_pt_raw", "terminal_last3"),
            ),
        ),
        (
            "terminal last1 margin interaction",
            "margin_interaction",
            "last1_2x2",
            "terminal_last1",
            _interaction(
                margin("C_it_chat", "terminal_last1"),
                margin("D_last1_ptswap", "terminal_last1"),
                margin("B_last1_raw", "terminal_last1"),
                margin("A_pt_raw", "terminal_last1"),
            ),
        ),
    ]


def _bootstrap(values: list[float], n_boot: int, seed: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if n_boot <= 0:
        return {"n": int(arr.size), "mean": float(arr.mean()), "ci_low": None, "ci_high": None}
    chunks = []
    for start in range(0, n_boot, 256):
        size = min(256, n_boot - start)
        idx = rng.integers(0, arr.size, size=(size, arr.size))
        chunks.append(arr[idx].mean(axis=1))
    boots = np.concatenate(chunks)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": int(arr.size), "mean": float(arr.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def _scope_records(records_by_model: dict[str, list[dict[str, Any]]], scope: str, models: list[str]) -> list[dict[str, Any]]:
    if scope == "dense5":
        kept = [model for model in models if model in DENSE5]
    elif scope == "gemma_removed_dense4":
        kept = [model for model in models if model in DENSE5 and model != "gemma3_4b"]
    else:
        kept = []
    out = []
    for model in kept:
        out.extend(records_by_model.get(model, []))
    return out


def _condition_summary(records: list[dict[str, Any]], event_kind: str) -> dict[str, Any]:
    out = {}
    for condition in CONDITIONS:
        labels = Counter(_class_label(record, event_kind, condition) for record in records if _readout(record, event_kind))
        total = sum(labels.values())
        out[condition] = {
            "count": total,
            "class_counts": dict(labels),
            "class_fractions": {key: value / total for key, value in labels.items()} if total else {},
        }
    return out


def analyze(root: Path, out_dir: Path, models: list[str], modes: list[str], event_kinds: list[str], n_boot: int, seed: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    by_model: dict[str, Any] = {}
    for mode in modes:
        records_by_model = {model: _load_records(root, mode, model) for model in models}
        for model, records in records_by_model.items():
            for event_kind in event_kinds:
                by_model[f"{mode}/{model}/{event_kind}"] = {
                    "n_records": len(records),
                    "event_present": sum(1 for record in records if _readout(record, event_kind)),
                    "conditions": _condition_summary(records, event_kind),
                }
                for idx, (name, kind, condition, window, fn) in enumerate(_effect_defs(event_kind)):
                    rows.append(
                        {
                            "mode": mode,
                            "model": model,
                            "condition": condition,
                            "window": window,
                            "metric": name,
                            "kind": kind,
                            **_bootstrap(_paired(records, fn), n_boot, seed + len(rows) + idx),
                        }
                    )
        for scope in ["dense5", "gemma_removed_dense4"]:
            records = _scope_records(records_by_model, scope, models)
            for event_kind in event_kinds:
                for idx, (name, kind, condition, window, fn) in enumerate(_effect_defs(event_kind)):
                    rows.append(
                        {
                            "mode": mode,
                            "model": scope,
                            "condition": condition,
                            "window": window,
                            "metric": name,
                            "kind": kind,
                            **_bootstrap(_paired(records, fn), n_boot, seed + len(rows) + idx),
                        }
                    )
        # Family-median and trimmed-mean are computed over per-model means.
        for scope in ["family_median", "trimmed_mean"]:
            for event_kind in event_kinds:
                for name, kind, condition, window, fn in _effect_defs(event_kind):
                    means = []
                    for model in [m for m in models if m in DENSE5]:
                        vals = _paired(records_by_model.get(model, []), fn)
                        if vals:
                            means.append(float(np.mean(vals)))
                    if not means:
                        stats = {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
                    else:
                        arr = np.array(sorted(means), dtype=float)
                        if scope == "family_median":
                            value = float(np.median(arr))
                        else:
                            value = float(arr[1:-1].mean()) if arr.size >= 3 else float(arr.mean())
                        stats = {"n": int(arr.size), "mean": value, "ci_low": None, "ci_high": None}
                    rows.append(
                        {
                            "mode": mode,
                            "model": scope,
                            "condition": condition,
                            "window": window,
                            "metric": name,
                            "kind": kind,
                            **stats,
                        }
                    )

    fields = ["mode", "model", "condition", "window", "metric", "kind", "n", "mean", "ci_low", "ci_high"]
    with (out_dir / "exp33_terminal_identity_margin_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    family_rows = [row for row in rows if row["model"] in {"dense5", "gemma_removed_dense4", "family_median", "trimmed_mean"}]
    with (out_dir / "exp33_terminal_identity_margin_family_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(family_rows)
    with (out_dir / "exp33_terminal_identity_margin_position_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([row for row in rows if row["window"].startswith("terminal")])
    summary = {
        "root": str(root),
        "models": models,
        "modes": modes,
        "event_kinds": event_kinds,
        "n_bootstrap": n_boot,
        "by_model": by_model,
        "effects": rows,
    }
    (out_dir / "exp33_terminal_identity_margin_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _plot(rows, out_dir / "exp33_terminal_identity_margin.png")
    return summary


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    wanted_names = {
        "B_last3 IT-token transfer",
        "B_last1 IT-token transfer",
        "B_mid IT-token transfer",
        "B_late IT-token transfer",
        "PT host margin: last3 minus A",
        "PT host margin: last1 minus A",
        "IT host margin: C minus D_last3",
        "IT host margin: C minus D_last1",
    }
    rows = [
        row for row in rows
        if row["mode"] == "raw_shared"
        and row["model"] == "dense5"
        and row["metric"] in wanted_names
        and row.get("mean") is not None
    ]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [row["metric"].replace(": ", ":\n") for row in rows]
    means = np.array([float(row["mean"]) for row in rows])
    lows = np.array([float(row["ci_low"]) if row["ci_low"] is not None else float(row["mean"]) for row in rows])
    highs = np.array([float(row["ci_high"]) if row["ci_high"] is not None else float(row["mean"]) for row in rows])
    colors = ["#4c78a8" if row["kind"] == "identity" else "#f58518" for row in rows]
    y = np.arange(len(rows))
    ax.barh(y, means, color=colors)
    ax.errorbar(means, y, xerr=np.vstack([means - lows, highs - means]), fmt="none", color="black", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Fraction for identity, logits for margin")
    ax.set_title("Exp33 terminal identity and margin, dense5 raw-shared")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=DENSE5)
    parser.add_argument("--modes", nargs="+", default=["raw_shared"])
    parser.add_argument("--event-kinds", nargs="+", default=["first_diff"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260502)
    args = parser.parse_args()
    out_dir = args.out_dir or (args.root / "analysis")
    summary = analyze(args.root, out_dir, args.models, args.modes, args.event_kinds, args.n_bootstrap, args.seed)
    print(json.dumps({"out_dir": str(out_dir), "n_effect_rows": len(summary["effects"])}, indent=2))


if __name__ == "__main__":
    main()
