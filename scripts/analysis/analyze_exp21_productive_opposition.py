#!/usr/bin/env python3
"""Analyze Exp21 productive-opposition records."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DENSE5 = ["gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b"]
ALL6 = DENSE5 + ["deepseek_v2_lite"]
PROMPT_MODES = ["native", "raw_shared"]
EVENT_KINDS = ["first_diff", "first_nonformat_diff", "first_assistant_marker_diff"]
PRIMARY_WINDOW = "late_reconciliation"
SOURCE_METRICS = [
    "delta_cosine_mlp",
    "negative_parallel_norm",
    "support_it_token",
    "support_pt_token",
    "support_pipeline_token",
    "alt_top10_delta",
    "productive_opposition_rate",
    "margin_writein_it_vs_pt",
    "opposition_support_it_token",
    "opposition_support_pt_token",
    "opposition_margin_it_vs_pt",
    "remainder_margin_it_vs_pt",
    "target_vs_alt_margin",
]
CONDITIONS_FOR_SUMMARY = [
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
    "B_late_identity",
    "D_late_identity",
    "B_late_rand_resproj_s0",
    "D_late_rand_resproj_s0",
]


@dataclass(frozen=True)
class Effect:
    mode: str
    model: str
    event_kind: str
    effect: str
    metric: str
    values: list[float]


def _iter_records(path: Path):
    if not path.exists():
        return
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_records(root: Path, mode: str, model: str) -> list[dict]:
    return list(_iter_records(root / mode / model / "records.jsonl.gz") or [])


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _mean(values) -> float | None:
    kept = [float(v) for v in values if _finite(v)]
    if not kept:
        return None
    return float(np.mean(kept))


def _value(record: dict, event_kind: str, condition: str, metric: str, window: str = PRIMARY_WINDOW) -> float | None:
    try:
        value = record["events"][event_kind]["conditions"][condition]["windows"][window][metric]
    except KeyError:
        return None
    return float(value) if _finite(value) else None


def _paired(records: list[dict], fn: Callable[[dict], float | None]) -> list[float]:
    out = []
    for record in records:
        value = fn(record)
        if _finite(value):
            out.append(float(value))
    return out


def _diff(
    event_kind: str,
    metric: str,
    condition_a: str,
    condition_b: str,
    window: str = PRIMARY_WINDOW,
) -> Callable[[dict], float | None]:
    def inner(record: dict) -> float | None:
        a = _value(record, event_kind, condition_a, metric, window)
        b = _value(record, event_kind, condition_b, metric, window)
        if a is None or b is None:
            return None
        return a - b

    return inner


def _late_weight_effect(event_kind: str, metric: str) -> Callable[[dict], float | None]:
    def inner(record: dict) -> float | None:
        vals = [
            _value(record, event_kind, "B_late_raw", metric),
            _value(record, event_kind, "A_pt_raw", metric),
            _value(record, event_kind, "C_it_chat", metric),
            _value(record, event_kind, "D_late_ptswap", metric),
        ]
        if any(v is None for v in vals):
            return None
        b, a, c, d = [float(v) for v in vals]
        return 0.5 * ((b - a) + (c - d))

    return inner


def _upstream_effect(event_kind: str, metric: str) -> Callable[[dict], float | None]:
    def inner(record: dict) -> float | None:
        vals = [
            _value(record, event_kind, "D_late_ptswap", metric),
            _value(record, event_kind, "A_pt_raw", metric),
            _value(record, event_kind, "C_it_chat", metric),
            _value(record, event_kind, "B_late_raw", metric),
        ]
        if any(v is None for v in vals):
            return None
        d, a, c, b = [float(v) for v in vals]
        return 0.5 * ((d - a) + (c - b))

    return inner


def _late_interaction(event_kind: str, metric: str) -> Callable[[dict], float | None]:
    def inner(record: dict) -> float | None:
        vals = [
            _value(record, event_kind, "C_it_chat", metric),
            _value(record, event_kind, "D_late_ptswap", metric),
            _value(record, event_kind, "B_late_raw", metric),
            _value(record, event_kind, "A_pt_raw", metric),
        ]
        if any(v is None for v in vals):
            return None
        c, d, b, a = [float(v) for v in vals]
        return (c - d) - (b - a)

    return inner


def _bootstrap(values: list[float], n_boot: int, seed: int) -> dict:
    if not values:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(seed)
    chunks = []
    # Vectorized chunks avoid the old Python inner loop while keeping memory bounded.
    for start in range(0, n_boot, 256):
        size = min(256, n_boot - start)
        idx = rng.integers(0, n, size=(size, n))
        chunks.append(arr[idx].mean(axis=1))
    boots = np.concatenate(chunks) if chunks else np.asarray([arr.mean()])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": n, "mean": float(arr.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def _condition_summary(records: list[dict], event_kind: str) -> dict:
    out = {}
    for condition in CONDITIONS_FOR_SUMMARY:
        payload: dict[str, dict[str, float | None]] = {}
        for window in ["early", "mid_policy", "late_reconciliation", "exp11_early", "exp11_mid", "exp11_late"]:
            payload[window] = {
                metric: _mean(_value(record, event_kind, condition, metric, window) for record in records)
                for metric in SOURCE_METRICS
            }
        winner_counts = Counter()
        support_cats = Counter()
        suppress_cats = Counter()
        for record in records:
            event = (record.get("events") or {}).get(event_kind) or {}
            cond = ((event.get("conditions") or {}).get(condition) or {})
            winner = cond.get("winner_class")
            if winner:
                winner_counts[str(winner)] += 1
            late = ((cond.get("windows") or {}).get(PRIMARY_WINDOW) or {})
            support_cats.update(late.get("top_supported_category_counts") or {})
            suppress_cats.update(late.get("top_suppressed_category_counts") or {})
        out[condition] = {
            "windows": payload,
            "winner_counts": dict(winner_counts),
            "late_top_supported_category_counts": dict(support_cats),
            "late_top_suppressed_category_counts": dict(suppress_cats),
        }
    return out


def _effect_defs(event_kind: str) -> list[tuple[str, str, Callable[[dict], float | None]]]:
    defs = []
    for metric in SOURCE_METRICS:
        defs.extend(
            [
                (f"late_weight_effect:{metric}", metric, _late_weight_effect(event_kind, metric)),
                (f"upstream_context_effect:{metric}", metric, _upstream_effect(event_kind, metric)),
                (f"late_interaction:{metric}", metric, _late_interaction(event_kind, metric)),
                (f"pure_IT_minus_PT:{metric}", metric, _diff(event_kind, metric, "C_it_chat", "A_pt_raw")),
                (f"B_late_minus_A:{metric}", metric, _diff(event_kind, metric, "B_late_raw", "A_pt_raw")),
                (f"C_minus_D_late:{metric}", metric, _diff(event_kind, metric, "C_it_chat", "D_late_ptswap")),
                (f"B_midlate_minus_B_mid:{metric}", metric, _diff(event_kind, metric, "B_midlate_raw", "B_mid_raw")),
                (f"D_midlate_minus_D_mid:{metric}", metric, _diff(event_kind, metric, "D_midlate_ptswap", "D_mid_ptswap")),
            ]
        )
    return defs


def _correlations(records: list[dict], event_kind: str, condition: str, window: str) -> dict[str, float | None]:
    pairs = defaultdict(list)
    for record in records:
        delta = _value(record, event_kind, condition, "delta_cosine_mlp", window)
        neg = _value(record, event_kind, condition, "negative_parallel_norm", window)
        opp = _value(record, event_kind, condition, "opposition_margin_it_vs_pt", window)
        margin = _value(record, event_kind, condition, "margin_writein_it_vs_pt", window)
        target = _value(record, event_kind, condition, "target_vs_alt_margin", window)
        values = {
            "delta_vs_margin": (delta, margin),
            "delta_vs_target_alt": (delta, target),
            "delta_vs_opp_margin": (delta, opp),
            "neg_norm_vs_opp_margin": (neg, opp),
            "neg_norm_vs_margin": (neg, margin),
        }
        for key, (x, y) in values.items():
            if _finite(x) and _finite(y):
                pairs[key].append((float(x), float(y)))
    out = {}
    for key, vals in pairs.items():
        if len(vals) < 3:
            out[key] = None
            continue
        xs = np.array([v[0] for v in vals], dtype=float)
        ys = np.array([v[1] for v in vals], dtype=float)
        if np.std(xs) == 0 or np.std(ys) == 0:
            out[key] = None
        else:
            out[key] = float(np.corrcoef(xs, ys)[0, 1])
    return out


def analyze(root: Path, out_dir: Path, models: list[str], n_boot: int, seed: int) -> dict:
    by_model = {}
    pooled_records = defaultdict(list)
    effect_rows = []

    for mode in PROMPT_MODES:
        for model in models:
            records = _load_records(root, mode, model)
            by_model[f"{mode}/{model}"] = {
                "n_records": len(records),
                "events_present": {
                    kind: sum(1 for record in records if kind in (record.get("events") or {}))
                    for kind in EVENT_KINDS
                },
                "conditions": {
                    kind: _condition_summary(records, kind)
                    for kind in EVENT_KINDS
                    if any(kind in (record.get("events") or {}) for record in records)
                },
            }
            if model in DENSE5:
                pooled_records[mode].extend(records)
            for event_kind in EVENT_KINDS:
                for effect_name, metric, fn in _effect_defs(event_kind):
                    values = _paired(records, fn)
                    effect_rows.append(Effect(mode, model, event_kind, effect_name, metric, values))

    pooled = {}
    correlations = {}
    for mode, records in pooled_records.items():
        pooled[mode] = {
            "n_records": len(records),
            "events_present": {
                kind: sum(1 for record in records if kind in (record.get("events") or {}))
                for kind in EVENT_KINDS
            },
            "conditions": {
                kind: _condition_summary(records, kind)
                for kind in EVENT_KINDS
                if any(kind in (record.get("events") or {}) for record in records)
            },
        }
        correlations[mode] = {
            f"{event_kind}/{condition}/{PRIMARY_WINDOW}": _correlations(records, event_kind, condition, PRIMARY_WINDOW)
            for event_kind in EVENT_KINDS
            for condition in ["A_pt_raw", "B_late_raw", "C_it_chat", "D_late_ptswap"]
        }
        for event_kind in EVENT_KINDS:
            for effect_name, metric, fn in _effect_defs(event_kind):
                values = _paired(records, fn)
                effect_rows.append(Effect(mode, "dense5", event_kind, effect_name, metric, values))

    out_dir.mkdir(parents=True, exist_ok=True)
    effect_payloads = []
    for idx, effect in enumerate(effect_rows):
        stats = _bootstrap(effect.values, n_boot, seed + idx)
        effect_payloads.append(
            {
                "mode": effect.mode,
                "model": effect.model,
                "event_kind": effect.event_kind,
                "effect": effect.effect,
                "metric": effect.metric,
                **stats,
            }
        )
    _write_effects_csv(out_dir / "effects.csv", effect_payloads)
    summary = {
        "root": str(root),
        "models": models,
        "by_model": by_model,
        "pooled": pooled,
        "correlations": correlations,
        "effects": effect_payloads,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _plot_main(summary, out_dir / "productive_opposition_main.png")
    _plot_categories(summary, out_dir / "token_category_support_suppression.png")
    _write_paper_claims(summary, out_dir / "paper_claims_exp21.md")
    return summary


def _write_effects_csv(path: Path, rows: list[dict]) -> None:
    fields = ["mode", "model", "event_kind", "effect", "metric", "n", "mean", "ci_low", "ci_high"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _effect_lookup(summary: dict, *, mode: str, model: str, event_kind: str, effect: str) -> dict | None:
    for row in summary.get("effects", []):
        if row["mode"] == mode and row["model"] == model and row["event_kind"] == event_kind and row["effect"] == effect:
            return row
    return None


def _primary_mode(summary: dict) -> str:
    """Prefer native runs, but make raw-shared-only extension analyses valid."""
    pooled = summary.get("pooled", {})
    for mode in ["native", "raw_shared"]:
        if (pooled.get(mode) or {}).get("n_records", 0):
            return mode
    return "native"


def _condition_window_value(summary: dict, mode: str, event_kind: str, condition: str, window: str, metric: str):
    condition_payload = (
        (summary["pooled"].get(mode, {}).get("conditions", {}).get(event_kind, {}) or {}).get(condition, {}) or {}
    )
    return condition_payload.get("windows", {}).get(window, {}).get(metric)


def _plot_main(summary: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    mode = _primary_mode(summary)
    fig.suptitle(f"Exp21 Productive Opposition: Dense5 Pooled ({mode})")
    event = "first_diff"
    conds = ["A_pt_raw", "B_late_raw", "D_late_ptswap", "C_it_chat"]
    labels = ["PT/PT", "PT/IT late", "IT/PT late", "IT/IT"]

    ax = axes[0, 0]
    vals = [
        _condition_window_value(summary, mode, event, cond, PRIMARY_WINDOW, "delta_cosine_mlp")
        for cond in conds
    ]
    ax.bar(labels, [v if v is not None else 0.0 for v in vals], color=["#5277a3", "#8bb7a2", "#c2905f", "#b95757"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("A. Late MLP delta-cosine")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 1]
    effects = ["late_weight_effect", "upstream_context_effect", "late_interaction"]
    vals = []
    errs = []
    for eff in effects:
        row = _effect_lookup(
            summary,
            mode=mode,
            model="dense5",
            event_kind=event,
            effect=f"{eff}:margin_writein_it_vs_pt",
        )
        mean = (row or {}).get("mean")
        lo = (row or {}).get("ci_low")
        hi = (row or {}).get("ci_high")
        vals.append(mean if mean is not None else 0.0)
        errs.append((0.0 if mean is None or lo is None else mean - lo, 0.0 if mean is None or hi is None else hi - mean))
    ax.bar(["late weights", "upstream", "interaction"], vals, color="#768f5f")
    ax.errorbar(range(len(vals)), vals, yerr=np.array(errs).T, fmt="none", color="black", capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("B. 2x2 source decomposition")

    ax = axes[1, 0]
    windows = ["early", "mid_policy", "late_reconciliation"]
    support_it = [_condition_window_value(summary, mode, event, "C_it_chat", w, "support_it_token") for w in windows]
    support_pt = [_condition_window_value(summary, mode, event, "C_it_chat", w, "support_pt_token") for w in windows]
    x = np.arange(len(windows))
    ax.bar(x - 0.18, [v or 0.0 for v in support_it], width=0.36, label="IT token", color="#b95757")
    ax.bar(x + 0.18, [v or 0.0 for v in support_pt], width=0.36, label="PT token", color="#5277a3")
    ax.set_xticks(x, ["early", "mid", "late"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("C. Pure IT MLP support by window")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    vals = [_condition_window_value(summary, mode, event, "C_it_chat", w, "opposition_margin_it_vs_pt") for w in windows]
    ax.bar(["early", "mid", "late"], [v or 0.0 for v in vals], color="#9b6d9e")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("D. Negative-parallel contribution to IT-PT margin")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_categories(summary: dict, path: Path) -> None:
    mode = _primary_mode(summary)
    event = "first_diff"
    conds = ["A_pt_raw", "C_it_chat", "B_late_raw", "D_late_ptswap"]
    categories = ["FORMAT", "CONTENT", "FUNCTION_OTHER"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in [
        (axes[0], "late_top_supported_category_counts", "Top supported token categories"),
        (axes[1], "late_top_suppressed_category_counts", "Top suppressed token categories"),
    ]:
        bottom = np.zeros(len(conds))
        for cat in categories:
            vals = []
            for cond in conds:
                counts = (
                    summary["pooled"].get(mode, {}).get("conditions", {}).get(event, {}).get(cond, {}).get(key, {})
                )
                total = sum(counts.values()) or 1
                vals.append(counts.get(cat, 0) / total)
            ax.bar(conds, vals, bottom=bottom, label=cat)
            bottom += np.array(vals)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _fmt(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _write_paper_claims(summary: dict, path: Path) -> None:
    mode = _primary_mode(summary)
    native_event = "first_diff"
    late_weight = _effect_lookup(
        summary,
        mode=mode,
        model="dense5",
        event_kind=native_event,
        effect="late_weight_effect:margin_writein_it_vs_pt",
    )
    upstream = _effect_lookup(
        summary,
        mode=mode,
        model="dense5",
        event_kind=native_event,
        effect="upstream_context_effect:margin_writein_it_vs_pt",
    )
    interaction = _effect_lookup(
        summary,
        mode=mode,
        model="dense5",
        event_kind=native_event,
        effect="late_interaction:margin_writein_it_vs_pt",
    )
    b_late_minus_a = _effect_lookup(
        summary,
        mode=mode,
        model="dense5",
        event_kind=native_event,
        effect="B_late_minus_A:margin_writein_it_vs_pt",
    )
    c_minus_d_late = _effect_lookup(
        summary,
        mode=mode,
        model="dense5",
        event_kind=native_event,
        effect="C_minus_D_late:margin_writein_it_vs_pt",
    )
    b_midlate_minus_b_mid = _effect_lookup(
        summary,
        mode=mode,
        model="dense5",
        event_kind=native_event,
        effect="B_midlate_minus_B_mid:margin_writein_it_vs_pt",
    )
    d_midlate_minus_d_mid = _effect_lookup(
        summary,
        mode=mode,
        model="dense5",
        event_kind=native_event,
        effect="D_midlate_minus_D_mid:margin_writein_it_vs_pt",
    )
    c_late_opp = _condition_window_value(
        summary,
        mode,
        native_event,
        "C_it_chat",
        PRIMARY_WINDOW,
        "opposition_margin_it_vs_pt",
    )
    c_delta = _condition_window_value(
        summary,
        mode,
        native_event,
        "C_it_chat",
        PRIMARY_WINDOW,
        "delta_cosine_mlp",
    )
    c_late_margin = _condition_window_value(
        summary,
        mode,
        native_event,
        "C_it_chat",
        PRIMARY_WINDOW,
        "margin_writein_it_vs_pt",
    )
    c_late_target_alt = _condition_window_value(
        summary,
        mode,
        native_event,
        "C_it_chat",
        PRIMARY_WINDOW,
        "target_vs_alt_margin",
    )
    if c_late_opp is not None and c_late_opp > 1e-3:
        opposition_sentence = (
            "The negative-parallel component is productive on the IT-vs-PT token margin proxy in pure IT."
        )
    elif c_late_opp is not None and abs(float(c_late_opp)) <= 1e-3:
        opposition_sentence = (
            "The negative-parallel component is approximately zero on the pure-IT IT-vs-PT margin proxy, "
            "so the paper should not treat raw negative opposition itself as the main mechanism."
        )
    else:
        opposition_sentence = (
            "The negative-parallel component is not positive on the pure-IT IT-vs-PT margin proxy, "
            "so the paper should not claim that raw negative opposition itself is the main mechanism."
        )
    text = f"""# Exp21 Productive Opposition: Paper-Safe Claims

Primary scope: dense5 pooled `{mode}` `first_diff`; DeepSeek is excluded from this dense-family extension.

Key dense5 `{mode}` values:

- Pure IT late `delta_cosine_mlp`: `{_fmt(c_delta)}`.
- Pure IT late IT-minus-PT margin write-in: `{_fmt(c_late_margin)}`.
- Pure IT late pipeline-token-vs-alt margin write-in: `{_fmt(c_late_target_alt)}`.
- Pure IT late negative-parallel contribution to IT-minus-PT margin: `{_fmt(c_late_opp)}`.
- 2x2 late-weight effect on IT-minus-PT margin write-in: `{_fmt((late_weight or {}).get('mean'))}`.
- 2x2 upstream-context effect on IT-minus-PT margin write-in: `{_fmt((upstream or {}).get('mean'))}`.
- 2x2 late interaction on IT-minus-PT margin write-in: `{_fmt((interaction or {}).get('mean'))}`.
- PT host, adding IT late: `{_fmt((b_late_minus_a or {}).get('mean'))}`.
- IT host, removing IT late: `{_fmt((c_minus_d_late or {}).get('mean'))}`.
- PT host, adding IT late on top of IT mid: `{_fmt((b_midlate_minus_b_mid or {}).get('mean'))}`.
- IT host, removing PT mid+late vs PT mid only comparison: `{_fmt((d_midlate_minus_d_mid or {}).get('mean'))}`.

Interpretation rule:

- {opposition_sentence}
- If the late-weight effect is positive, the paper may say that late IT weights add IT-token readout on this proxy.
- If the upstream effect is also large, the paper must phrase the mechanism as cooperation between earlier IT context and late IT readout, not as a late-module-only story.
- If raw `delta_cosine_mlp` is weaker than token-specific write-in metrics, the paper should treat negative residual opposition as a geometric companion rather than the mechanism itself.

Safe wording:

> Exp21 measures MLP-only finite-difference logit effects at the first PT/IT divergent prefix. In `{mode}` dense-family runs, late residual opposition is evaluated by whether its negative-parallel component increases IT-vs-PT token margin. This supports a productive-opposition readout only when the negative component helps the IT token relative to the PT token; otherwise, negative delta-cosine remains only a geometric signature.

Do not claim:

> Negative residuals cause assistant behavior.
"""
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp21 productive-opposition records.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=ALL6)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=21)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.root / "analysis")
    summary = analyze(args.root, out_dir, args.models, args.n_boot, args.seed)
    print(json.dumps({"out_dir": str(out_dir), "models": args.models, "effects": len(summary["effects"])}, indent=2))


if __name__ == "__main__":
    main()
