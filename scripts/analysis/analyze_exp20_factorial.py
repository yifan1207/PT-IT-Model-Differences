#!/usr/bin/env python3
"""Analyze the Exp20 mid+late factorial completion."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
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


@dataclass(frozen=True)
class Effect:
    name: str
    mode: str
    kind: str
    values: list[float]


def _iter_jsonl(path: Path):
    with path.open("rb") as handle:
        for raw in handle:
            if raw.strip():
                yield json.loads(raw.decode("utf-8", errors="ignore"))


def _class_at(record: dict, condition: str, target: str) -> float | None:
    readout = ((record.get("readouts") or {}).get("first_diff") or {})
    cls = ((readout.get("condition_token_at_step") or {}).get(condition) or {}).get("class")
    if cls is None or cls == "missing":
        return None
    return 1.0 if cls == target else 0.0


def _margin(record: dict, condition: str, window: str = "late_reconciliation") -> float | None:
    readout = ((record.get("readouts") or {}).get("first_diff") or {})
    condition_payload = ((readout.get("conditions") or {}).get(condition) or {})
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
    out = []
    for record in records:
        value = fn(record)
        if value is not None and np.isfinite(value):
            out.append(float(value))
    return out


def _diff(metric_a: Callable[[dict], float | None], metric_b: Callable[[dict], float | None]) -> Callable[[dict], float | None]:
    def inner(record: dict) -> float | None:
        a = metric_a(record)
        b = metric_b(record)
        if a is None or b is None:
            return None
        return float(a) - float(b)

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
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(float(np.mean(sample)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": n, "mean": mean, "ci_low": float(lo), "ci_high": float(hi)}


def _load_records(root: Path, mode: str, models: list[str]) -> list[dict]:
    records: list[dict] = []
    for model in models:
        path = root / mode / model / "exp20_records.jsonl"
        if not path.exists():
            continue
        records.extend(_iter_jsonl(path))
    return records


def analyze(root: Path, models: list[str], n_boot: int, seed: int) -> dict:
    effects: list[Effect] = []
    mode_counts = {}
    for mode in PROMPT_MODES:
        records = _load_records(root, mode, models)
        mode_counts[mode] = len(records)
        it = lambda c: (lambda record: _class_at(record, c, "it"))
        pt = lambda c: (lambda record: _class_at(record, c, "pt"))
        margin = lambda c: (lambda record: _margin(record, c))

        definitions = [
            (
                "PT host identity: mid+late minus mid",
                "identity",
                _diff(it("B_midlate_raw"), it("B_mid_raw")),
            ),
            (
                "PT host identity: mid+late minus late",
                "identity",
                _diff(it("B_midlate_raw"), it("B_late_raw")),
            ),
            (
                "PT host identity interaction: [ML-M]-[L-A]",
                "identity_interaction",
                _interaction(it("B_midlate_raw"), it("B_mid_raw"), it("B_late_raw"), it("A_pt_raw")),
            ),
            (
                "IT host PT-token identity: mid+late minus mid",
                "identity",
                _diff(pt("D_midlate_ptswap"), pt("D_mid_ptswap")),
            ),
            (
                "IT host PT-token identity interaction: [ML-M]-[L-C]",
                "identity_interaction",
                _interaction(pt("D_midlate_ptswap"), pt("D_mid_ptswap"), pt("D_late_ptswap"), pt("C_it_chat")),
            ),
            (
                "PT host margin: mid+late minus mid",
                "margin",
                _diff(margin("B_midlate_raw"), margin("B_mid_raw")),
            ),
            (
                "PT host margin interaction: [ML-M]-[L-A]",
                "margin_interaction",
                _interaction(margin("B_midlate_raw"), margin("B_mid_raw"), margin("B_late_raw"), margin("A_pt_raw")),
            ),
            (
                "IT host margin: pure IT minus PT mid+late swap",
                "margin",
                _diff(margin("C_it_chat"), margin("D_midlate_ptswap")),
            ),
            (
                "IT host margin: PT mid+late swap minus PT mid swap",
                "margin",
                _diff(margin("D_midlate_ptswap"), margin("D_mid_ptswap")),
            ),
        ]
        for name, kind, fn in definitions:
            effects.append(Effect(name=name, mode=mode, kind=kind, values=_paired(records, fn)))

    rows = []
    for idx, effect in enumerate(effects):
        summary = _bootstrap(effect.values, n_boot=n_boot, seed=seed + idx)
        rows.append({
            "mode": effect.mode,
            "effect": effect.name,
            "kind": effect.kind,
            **summary,
        })
    return {"root": str(root), "models": models, "mode_counts": mode_counts, "effects": rows}


def write_outputs(summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "factorial_effects.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    csv_path = out_dir / "factorial_effects.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "effect", "kind", "n", "mean", "ci_low", "ci_high"])
        writer.writeheader()
        for row in summary["effects"]:
            writer.writerow(row)
    _plot(summary, out_dir / "exp20_midlate_factorial_effects.png")


def _plot(summary: dict, out_path: Path) -> None:
    rows = [
        row for row in summary["effects"]
        if row["mode"] == "raw_shared"
        and row["effect"] in {
            "PT host identity: mid+late minus mid",
            "PT host identity interaction: [ML-M]-[L-A]",
            "PT host margin: mid+late minus mid",
            "PT host margin interaction: [ML-M]-[L-A]",
            "IT host margin: pure IT minus PT mid+late swap",
        }
    ]
    labels = [row["effect"].replace(": ", ":\n") for row in rows]
    means = np.array([float(row["mean"]) for row in rows])
    lo = np.array([float(row["ci_low"]) for row in rows])
    hi = np.array([float(row["ci_high"]) for row in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    colors = ["#5E81AC" if "identity" in row["kind"] else "#BF616A" for row in rows]
    ax.barh(y, means, color=colors, alpha=0.9)
    ax.errorbar(means, y, xerr=np.vstack([means - lo, hi - means]), fmt="none", ecolor="black", capsize=4)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, labels)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Exp20 mid+late factorial completion (dense-5, raw-shared)")
    ax.set_xlabel("Effect size: fraction points for identity, logits for margin")
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
    parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args()
    out_dir = args.out_dir or (args.root / "factorial_analysis")
    summary = analyze(args.root, args.models, args.n_boot, args.seed)
    write_outputs(summary, out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
