#!/usr/bin/env python3
"""Analyze Exp20 holdout first-divergence factorial validation records."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DENSE5 = ["gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b"]
PROMPT_MODES = ["raw_shared", "native"]
CONDITIONS = [
    "A_pt_raw",
    "B_early_raw",
    "B_mid_raw",
    "B_late_raw",
    "B_earlymid_raw",
    "B_midlate_raw",
    "C_it_chat",
    "D_early_ptswap",
    "D_mid_ptswap",
    "D_late_ptswap",
    "D_earlymid_ptswap",
    "D_midlate_ptswap",
]


@dataclass(frozen=True)
class Effect:
    mode: str
    model: str
    name: str
    kind: str
    values: list[float]


def _iter_jsonl(path: Path):
    with path.open("rb") as handle:
        for raw in handle:
            if raw.strip():
                yield json.loads(raw.decode("utf-8", errors="ignore"))


def _load_records(root: Path, mode: str, model: str) -> list[dict]:
    path = root / mode / model / "exp20_validation_records.jsonl"
    if not path.exists():
        return []
    return list(_iter_jsonl(path))


def _readout(record: dict) -> dict:
    payload = (record.get("readouts") or {}).get("first_diff")
    return payload if isinstance(payload, dict) else {}


def _class_at(record: dict, condition: str, target: str) -> float | None:
    cls = ((_readout(record).get("condition_token_at_step") or {}).get(condition) or {}).get("class")
    if cls is None or cls == "missing":
        return None
    return 1.0 if cls == target else 0.0


def _class_label(record: dict, condition: str) -> str:
    cls = ((_readout(record).get("condition_token_at_step") or {}).get(condition) or {}).get("class")
    return str(cls or "missing")


def _margin(record: dict, condition: str, window: str = "late_reconciliation") -> float | None:
    condition_payload = ((_readout(record).get("conditions") or {}).get(condition) or {})
    metric = (
        ((condition_payload.get("windows") or {}).get(window) or {})
        .get("it_minus_pt_margin", {})
        .get("total_delta")
    )
    try:
        return float(metric)
    except (TypeError, ValueError):
        return None


def _paired(records: list[dict], fn: Callable[[dict], float | None]) -> list[float]:
    values = []
    for record in records:
        value = fn(record)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return values


def _diff(a: Callable[[dict], float | None], b: Callable[[dict], float | None]) -> Callable[[dict], float | None]:
    def inner(record: dict) -> float | None:
        av = a(record)
        bv = b(record)
        if av is None or bv is None:
            return None
        return float(av) - float(bv)

    return inner


def _interaction(
    ml: Callable[[dict], float | None],
    m: Callable[[dict], float | None],
    l: Callable[[dict], float | None],
    a: Callable[[dict], float | None],
) -> Callable[[dict], float | None]:
    def inner(record: dict) -> float | None:
        values = [fn(record) for fn in (ml, m, l, a)]
        if any(value is None for value in values):
            return None
        ml_v, m_v, l_v, a_v = [float(value) for value in values]
        return (ml_v - m_v) - (l_v - a_v)

    return inner


def _bootstrap(values: list[float], n_boot: int, seed: int) -> dict:
    if not values:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    n = len(values)
    mean = float(np.mean(values))
    boots = [float(np.mean([values[rng.randrange(n)] for _ in range(n)])) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": n, "mean": mean, "ci_low": float(lo), "ci_high": float(hi)}


def _effect_defs() -> list[tuple[str, str, Callable[[dict], float | None]]]:
    it = lambda c: (lambda record: _class_at(record, c, "it"))
    pt = lambda c: (lambda record: _class_at(record, c, "pt"))
    margin = lambda c: (lambda record: _margin(record, c))
    return [
        ("PT host identity: mid+late minus mid", "identity", _diff(it("B_midlate_raw"), it("B_mid_raw"))),
        ("PT host identity: early+mid minus mid", "identity", _diff(it("B_earlymid_raw"), it("B_mid_raw"))),
        ("PT host identity interaction: [ML-M]-[L-A]", "identity_interaction", _interaction(it("B_midlate_raw"), it("B_mid_raw"), it("B_late_raw"), it("A_pt_raw"))),
        ("PT host margin: mid+late minus mid", "margin", _diff(margin("B_midlate_raw"), margin("B_mid_raw"))),
        ("PT host margin: early+mid minus mid", "margin", _diff(margin("B_earlymid_raw"), margin("B_mid_raw"))),
        ("PT host margin interaction: [ML-M]-[L-A]", "margin_interaction", _interaction(margin("B_midlate_raw"), margin("B_mid_raw"), margin("B_late_raw"), margin("A_pt_raw"))),
        ("IT host identity: pure IT minus PT early swap", "identity", _diff(it("C_it_chat"), it("D_early_ptswap"))),
        ("IT host identity: pure IT minus PT late swap", "identity", _diff(it("C_it_chat"), it("D_late_ptswap"))),
        ("IT host identity: pure IT minus PT early+mid swap", "identity", _diff(it("C_it_chat"), it("D_earlymid_ptswap"))),
        ("IT host identity: pure IT minus PT mid+late swap", "identity", _diff(it("C_it_chat"), it("D_midlate_ptswap"))),
        ("IT host margin: pure IT minus PT early swap", "margin", _diff(margin("C_it_chat"), margin("D_early_ptswap"))),
        ("IT host margin: pure IT minus PT mid swap", "margin", _diff(margin("C_it_chat"), margin("D_mid_ptswap"))),
        ("IT host margin: pure IT minus PT late swap", "margin", _diff(margin("C_it_chat"), margin("D_late_ptswap"))),
        ("IT host margin: pure IT minus PT early+mid swap", "margin", _diff(margin("C_it_chat"), margin("D_earlymid_ptswap"))),
        ("IT host margin: pure IT minus PT mid+late swap", "margin", _diff(margin("C_it_chat"), margin("D_midlate_ptswap"))),
        ("IT host margin: PT mid+late swap minus PT mid swap", "margin", _diff(margin("D_midlate_ptswap"), margin("D_mid_ptswap"))),
    ]


def _condition_summary(records: list[dict]) -> dict:
    out = {}
    for condition in CONDITIONS:
        labels = Counter(_class_label(record, condition) for record in records if _readout(record))
        margins = _paired(records, lambda record, c=condition: _margin(record, c))
        total = sum(labels.values())
        out[condition] = {
            "count": total,
            "class_counts": dict(labels),
            "class_fractions": {key: value / total for key, value in labels.items()} if total else {},
            "late_margin_mean": float(np.mean(margins)) if margins else None,
        }
    return out


def analyze(root: Path, models: list[str], n_boot: int, seed: int) -> dict:
    effects = []
    by_model = {}
    pooled_records = {}
    effect_defs = _effect_defs()

    for mode in PROMPT_MODES:
        pooled_records[mode] = []
        for model in models:
            records = _load_records(root, mode, model)
            pooled_records[mode].extend(records)
            present = sum(1 for record in records if _readout(record))
            missing_first_diff = len(records) - present
            by_model[f"{mode}/{model}"] = {
                "n_records": len(records),
                "first_diff_present": present,
                "first_diff_missing": missing_first_diff,
                "conditions": _condition_summary(records),
            }
            for name, kind, fn in effect_defs:
                values = _paired(records, fn)
                effects.append(Effect(mode=mode, model=model, name=name, kind=kind, values=values))

    pooled = {}
    for mode, records in pooled_records.items():
        pooled[mode] = {
            "n_records": len(records),
            "first_diff_present": sum(1 for record in records if _readout(record)),
            "conditions": _condition_summary(records),
        }
        for name, kind, fn in effect_defs:
            values = _paired(records, fn)
            effects.append(Effect(mode=mode, model="dense5", name=name, kind=kind, values=values))

    rows = []
    for idx, effect in enumerate(effects):
        rows.append({
            "mode": effect.mode,
            "model": effect.model,
            "effect": effect.name,
            "kind": effect.kind,
            **_bootstrap(effect.values, n_boot=n_boot, seed=seed + idx),
        })
    return {
        "root": str(root),
        "models": models,
        "by_model": by_model,
        "pooled": pooled,
        "effects": rows,
    }


def write_outputs(summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    with (out_dir / "effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "model", "effect", "kind", "n", "mean", "ci_low", "ci_high"])
        writer.writeheader()
        writer.writerows(summary["effects"])
    _plot(summary, out_dir / "factorial_validation_effects.png")


def _plot(summary: dict, out_path: Path) -> None:
    wanted = {
        "PT host identity: mid+late minus mid",
        "PT host margin: mid+late minus mid",
        "PT host margin interaction: [ML-M]-[L-A]",
        "IT host margin: pure IT minus PT early swap",
        "IT host margin: pure IT minus PT late swap",
        "IT host margin: pure IT minus PT early+mid swap",
        "IT host margin: pure IT minus PT mid+late swap",
    }
    rows = [
        row for row in summary["effects"]
        if row["model"] == "dense5" and row["effect"] in wanted
    ]
    labels = [f"{row['mode']}\n{row['effect'].replace(': ', ': ')}" for row in rows]
    means = np.array([float(row["mean"]) for row in rows])
    lo = np.array([float(row["ci_low"]) for row in rows])
    hi = np.array([float(row["ci_high"]) for row in rows])
    y = np.arange(len(rows))
    colors = ["#5E81AC" if "identity" in row["kind"] else "#BF616A" for row in rows]
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.barh(y, means, color=colors, alpha=0.9)
    ax.errorbar(means, y, xerr=np.vstack([means - lo, hi - means]), fmt="none", ecolor="black", capsize=4)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Effect size: fraction points for identity, logits for margin")
    ax.set_title("Exp20 holdout factorial validation (dense-5)")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=DENSE5)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260425)
    args = parser.parse_args()
    summary = analyze(args.root, args.models, args.n_boot, args.seed)
    write_outputs(summary, args.out_dir or (args.root / "validation_analysis"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
