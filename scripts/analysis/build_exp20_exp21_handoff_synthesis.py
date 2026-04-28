#!/usr/bin/env python3
"""Build paper-facing Exp20/Exp21 handoff synthesis artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


EXP20_ROOT_DEFAULT = Path(
    "results/exp20_divergence_token_counterfactual/"
    "factorial_validation_holdout_fast_20260425_2009_with_early"
)
EXP20_DEFAULT = Path(
    EXP20_ROOT_DEFAULT / "validation_analysis/summary.json"
)
EXP21_ROOT_DEFAULT = Path(
    "results/exp21_productive_opposition/"
    "exp21_full_productive_opposition_clean_20260426_053736"
)
EXP21_DEFAULT = Path(
    EXP21_ROOT_DEFAULT / "analysis/summary.json"
)
OUT_DEFAULT = Path("results/paper_synthesis")
DENSE5 = ["gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b"]
TOLERANCE = 1e-3


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_exp20_records(root: Path, mode: str, models: list[str]) -> dict[str, list[dict]]:
    by_model = {}
    for model in models:
        path = root / mode / model / "exp20_validation_records.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp20 records: {path}")
        by_model[model] = list(_iter_jsonl(path))
    return by_model


def _load_exp21_records(root: Path, mode: str, models: list[str]) -> dict[str, list[dict]]:
    by_model = {}
    for model in models:
        path = root / mode / model / "records.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp21 records: {path}")
        by_model[model] = list(_iter_jsonl(path))
    return by_model


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _stat(
    records_by_model: dict[str, list[dict]],
    fn: Callable[[dict], float | None],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float | int]:
    values_by_model: dict[str, np.ndarray] = {}
    for model, records in records_by_model.items():
        values = [float(value) for record in records if _finite(value := fn(record))]
        if values:
            values_by_model[model] = np.asarray(values, dtype=float)
    if not values_by_model:
        raise ValueError("No finite values for bootstrap statistic.")

    all_values = np.concatenate(list(values_by_model.values()))
    total_n = len(all_values)
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, n_boot, 256):
        size = min(256, n_boot - start)
        sums = np.zeros(size, dtype=float)
        for arr in values_by_model.values():
            idx = rng.integers(0, len(arr), size=(size, len(arr)))
            sums += arr[idx].sum(axis=1)
        chunks.append(sums / total_n)
    boots = np.concatenate(chunks) if chunks else np.asarray([float(all_values.mean())])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(total_n),
        "mean": float(all_values.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def _exp20_cond(exp20: dict, mode: str, condition: str, key: str) -> float:
    return float(exp20["pooled"][mode]["conditions"][condition][key])


def _exp20_frac(exp20: dict, mode: str, condition: str, cls: str) -> float:
    return float(exp20["pooled"][mode]["conditions"][condition]["class_fractions"].get(cls, 0.0))


def _exp20_effect(exp20: dict, mode: str, effect: str) -> dict:
    for row in exp20["effects"]:
        if row["mode"] == mode and row["model"] == "dense5" and row["effect"] == effect:
            return row
    raise KeyError(f"Missing Exp20 effect: {mode}/{effect}")


def _exp21_window(exp21: dict, condition: str, window: str, metric: str, mode: str = "native") -> float:
    return float(
        exp21["pooled"][mode]["conditions"]["first_diff"][condition]["windows"][window][metric]
    )


def _exp21_effect(exp21: dict, effect: str, mode: str = "native") -> dict:
    for row in exp21["effects"]:
        if (
            row["mode"] == mode
            and row["model"] == "dense5"
            and row["event_kind"] == "first_diff"
            and row["effect"] == effect
        ):
            return row
    raise KeyError(f"Missing Exp21 effect: {mode}/{effect}")


def _exp20_readout(record: dict) -> dict:
    payload = (record.get("readouts") or {}).get("first_diff")
    return payload if isinstance(payload, dict) else {}


def _exp20_class(record: dict, condition: str) -> str | None:
    cls = ((_exp20_readout(record).get("condition_token_at_step") or {}).get(condition) or {}).get("class")
    if cls is None or cls == "missing":
        return None
    return str(cls)


def _exp20_class_at(condition: str, target: str) -> Callable[[dict], float | None]:
    def inner(record: dict) -> float | None:
        cls = _exp20_class(record, condition)
        if cls is None:
            return None
        return 1.0 if cls == target else 0.0

    return inner


def _exp20_margin(record: dict, condition: str, window: str = "late_reconciliation") -> float | None:
    condition_payload = ((_exp20_readout(record).get("conditions") or {}).get(condition) or {})
    metric = (
        ((condition_payload.get("windows") or {}).get(window) or {})
        .get("it_minus_pt_margin", {})
        .get("total_delta")
    )
    return float(metric) if _finite(metric) else None


def _exp21_window_record(
    record: dict,
    condition: str,
    window: str,
    metric: str,
    event_kind: str = "first_diff",
) -> float | None:
    try:
        value = record["events"][event_kind]["conditions"][condition]["windows"][window][metric]
    except KeyError:
        return None
    return float(value) if _finite(value) else None


def _diff(a: Callable[[dict], float | None], b: Callable[[dict], float | None]) -> Callable[[dict], float | None]:
    def inner(record: dict) -> float | None:
        av = a(record)
        bv = b(record)
        if av is None or bv is None:
            return None
        return float(av) - float(bv)

    return inner


def _exp21_value(condition: str, window: str, metric: str) -> Callable[[dict], float | None]:
    return lambda record: _exp21_window_record(record, condition, window, metric)


def _exp21_late_weight(metric: str) -> Callable[[dict], float | None]:
    def inner(record: dict) -> float | None:
        b = _exp21_window_record(record, "B_late_raw", "late_reconciliation", metric)
        a = _exp21_window_record(record, "A_pt_raw", "late_reconciliation", metric)
        c = _exp21_window_record(record, "C_it_chat", "late_reconciliation", metric)
        d = _exp21_window_record(record, "D_late_ptswap", "late_reconciliation", metric)
        if any(value is None for value in (b, a, c, d)):
            return None
        return 0.5 * ((float(b) - float(a)) + (float(c) - float(d)))

    return inner


def _exp21_upstream(metric: str) -> Callable[[dict], float | None]:
    def inner(record: dict) -> float | None:
        d = _exp21_window_record(record, "D_late_ptswap", "late_reconciliation", metric)
        a = _exp21_window_record(record, "A_pt_raw", "late_reconciliation", metric)
        c = _exp21_window_record(record, "C_it_chat", "late_reconciliation", metric)
        b = _exp21_window_record(record, "B_late_raw", "late_reconciliation", metric)
        if any(value is None for value in (d, a, c, b)):
            return None
        return 0.5 * ((float(d) - float(a)) + (float(c) - float(b)))

    return inner


def _exp21_interaction(metric: str) -> Callable[[dict], float | None]:
    def inner(record: dict) -> float | None:
        c = _exp21_window_record(record, "C_it_chat", "late_reconciliation", metric)
        d = _exp21_window_record(record, "D_late_ptswap", "late_reconciliation", metric)
        b = _exp21_window_record(record, "B_late_raw", "late_reconciliation", metric)
        a = _exp21_window_record(record, "A_pt_raw", "late_reconciliation", metric)
        if any(value is None for value in (c, d, b, a)):
            return None
        return (float(c) - float(d)) - (float(b) - float(a))

    return inner


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _ci(stat: dict, *, digits: int = 3, pct: bool = False, signed: bool = False) -> str:
    mean = float(stat["mean"])
    lo = float(stat["ci_low"])
    hi = float(stat["ci_high"])
    if pct:
        return f"{_pct(mean)} (95% CI [{_pct(lo)}, {_pct(hi)}])"
    formatter = _signed if signed else _fmt
    return f"{formatter(mean, digits)} (95% CI [{formatter(lo, digits)}, {formatter(hi, digits)}])"


def _plain_ci(stat: dict, *, digits: int = 3, signed: bool = False) -> str:
    formatter = _signed if signed else _fmt
    return f"95% CI [{formatter(float(stat['ci_low']), digits)}, {formatter(float(stat['ci_high']), digits)}]"


def _rows(exp20: dict, exp21: dict, stats: dict[str, dict]) -> list[dict[str, str]]:
    exp20_c = _exp20_cond(exp20, "native", "C_it_chat", "late_margin_mean")
    exp20_d_early = _exp20_cond(exp20, "native", "D_early_ptswap", "late_margin_mean")
    exp20_d_mid = _exp20_cond(exp20, "native", "D_mid_ptswap", "late_margin_mean")
    exp20_d_late = _exp20_cond(exp20, "native", "D_late_ptswap", "late_margin_mean")
    exp20_d_earlymid = _exp20_cond(exp20, "native", "D_earlymid_ptswap", "late_margin_mean")
    exp20_d_midlate = _exp20_cond(exp20, "native", "D_midlate_ptswap", "late_margin_mean")

    pure_it_late_support = _exp21_window(exp21, "C_it_chat", "late_reconciliation", "support_it_token")
    pure_it_mid_support = _exp21_window(exp21, "C_it_chat", "mid_policy", "support_it_token")
    pure_it_early_support = _exp21_window(exp21, "C_it_chat", "early", "support_it_token")
    pt_late_support = _exp21_window(exp21, "A_pt_raw", "late_reconciliation", "support_it_token")
    pt_mid_support = _exp21_window(exp21, "A_pt_raw", "mid_policy", "support_it_token")
    pt_early_support = _exp21_window(exp21, "A_pt_raw", "early", "support_it_token")

    late_weight = _exp21_effect(exp21, "late_weight_effect:margin_writein_it_vs_pt")
    upstream = _exp21_effect(exp21, "upstream_context_effect:margin_writein_it_vs_pt")
    interaction = _exp21_effect(exp21, "late_interaction:margin_writein_it_vs_pt")
    b_late_minus_a = _exp21_effect(exp21, "B_late_minus_A:margin_writein_it_vs_pt")
    c_minus_d_late = _exp21_effect(exp21, "C_minus_D_late:margin_writein_it_vs_pt")
    c_minus_d_early = (
        _exp21_window(exp21, "C_it_chat", "late_reconciliation", "margin_writein_it_vs_pt")
        - _exp21_window(exp21, "D_early_ptswap", "late_reconciliation", "margin_writein_it_vs_pt")
    )
    c_minus_d_mid = (
        _exp21_window(exp21, "C_it_chat", "late_reconciliation", "margin_writein_it_vs_pt")
        - _exp21_window(exp21, "D_mid_ptswap", "late_reconciliation", "margin_writein_it_vs_pt")
    )
    b_midlate_minus_b_mid = _exp21_effect(exp21, "B_midlate_minus_B_mid:margin_writein_it_vs_pt")

    delta_cos_shift = (
        _exp21_window(exp21, "C_it_chat", "late_reconciliation", "delta_cosine_mlp")
        - _exp21_window(exp21, "A_pt_raw", "late_reconciliation", "delta_cosine_mlp")
    )
    negative_parallel = _exp21_window(
        exp21,
        "C_it_chat",
        "late_reconciliation",
        "opposition_margin_it_vs_pt",
    )
    full_update_margin = _exp21_window(
        exp21,
        "C_it_chat",
        "late_reconciliation",
        "margin_writein_it_vs_pt",
    )

    return [
        {
            "evidence": "Exp20 identity",
            "metric": "raw-shared IT-token fraction under PT host",
            "primary_comparison": "PT+IT mid vs PT+IT late",
            "dense5_value": f"{_ci(stats['exp20_raw_b_mid_it'], pct=True)} vs "
            f"{_ci(stats['exp20_raw_b_late_it'], pct=True)}",
            "source_value_numeric": _fmt(
                _exp20_frac(exp20, "raw_shared", "B_mid_raw", "it")
                - _exp20_frac(exp20, "raw_shared", "B_late_raw", "it")
            ),
            "source_ci_low": _fmt(stats["exp20_raw_b_mid_minus_late_it"]["ci_low"]),
            "source_ci_high": _fmt(stats["exp20_raw_b_mid_minus_late_it"]["ci_high"]),
            "interpretation": "Middle grafts transfer opposite-model token identity more than late grafts.",
            "caveat": "Raw-shared is the cleaner identity control but not the native IT deployment prompt.",
        },
        {
            "evidence": "Exp20 mirror identity",
            "metric": "raw-shared PT-token fraction under IT host",
            "primary_comparison": "IT+PT mid vs IT+PT late",
            "dense5_value": f"{_ci(stats['exp20_raw_d_mid_pt'], pct=True)} vs "
            f"{_ci(stats['exp20_raw_d_late_pt'], pct=True)}",
            "source_value_numeric": _fmt(
                _exp20_frac(exp20, "raw_shared", "D_mid_ptswap", "pt")
                - _exp20_frac(exp20, "raw_shared", "D_late_ptswap", "pt")
            ),
            "source_ci_low": _fmt(stats["exp20_raw_d_mid_minus_late_pt"]["ci_low"]),
            "source_ci_high": _fmt(stats["exp20_raw_d_mid_minus_late_pt"]["ci_high"]),
            "interpretation": "Middle swaps transfer opposite-model token identity more than late swaps.",
            "caveat": "This is a raw-shared identity control rather than native IT deployment prompting.",
        },
        {
            "evidence": "Exp20 native readout",
            "metric": "native IT-host margin drop",
            "primary_comparison": "pure IT minus PT early/mid/late swap",
            "dense5_value": f"early {_ci(stats['exp20_native_margin_drop_early'], signed=True)}, "
            f"mid {_ci(stats['exp20_native_margin_drop_mid'], signed=True)}, "
            f"late {_ci(stats['exp20_native_margin_drop_late'], signed=True)}",
            "source_value_numeric": _fmt(exp20_c - exp20_d_late),
            "source_ci_low": _fmt(stats["exp20_native_margin_drop_late"]["ci_low"]),
            "source_ci_high": _fmt(stats["exp20_native_margin_drop_late"]["ci_high"]),
            "interpretation": "Late swap causes the largest single-window loss of native IT-vs-PT token margin.",
            "caveat": "First-divergence token margin is a causal proxy, not direct human-rated behavior.",
        },
        {
            "evidence": "Exp20 native identity retention",
            "metric": "native IT-token fraction under IT host",
            "primary_comparison": "early/mid/late PT swap",
            "dense5_value": f"early {_ci(stats['exp20_native_d_early_it'], pct=True)}, "
            f"mid {_ci(stats['exp20_native_d_mid_it'], pct=True)}, "
            f"late {_ci(stats['exp20_native_d_late_it'], pct=True)}",
            "source_value_numeric": _fmt(_exp20_frac(exp20, "native", "D_late_ptswap", "it")),
            "source_ci_low": _fmt(stats["exp20_native_d_late_it"]["ci_low"]),
            "source_ci_high": _fmt(stats["exp20_native_d_late_it"]["ci_high"]),
            "interpretation": "Late PT swaps preserve IT-token identity more than early or mid swaps.",
            "caveat": "This retention coexists with the largest single-window margin drop.",
        },
        {
            "evidence": "Exp20 combined windows",
            "metric": "native IT-host margin drop",
            "primary_comparison": "pure IT minus PT early+mid / mid+late swap",
            "dense5_value": f"early+mid {_ci(stats['exp20_native_margin_drop_earlymid'], signed=True)}, "
            f"mid+late {_ci(stats['exp20_native_margin_drop_midlate'], signed=True)}",
            "source_value_numeric": _fmt(exp20_c - exp20_d_midlate),
            "source_ci_low": _fmt(stats["exp20_native_margin_drop_midlate"]["ci_low"]),
            "source_ci_high": _fmt(stats["exp20_native_margin_drop_midlate"]["ci_high"]),
            "interpretation": "Multi-window swaps hurt more than any single window, supporting a distributed circuit.",
            "caveat": "The strongest claim is distributed mid-to-late handoff, not a late-only module.",
        },
        {
            "evidence": "Exp21 MLP support",
            "metric": "pure IT support_it_token",
            "primary_comparison": "early vs mid vs late MLP windows",
            "dense5_value": f"early {_ci(stats['exp21_c_support_early'], signed=True)}, "
            f"mid {_ci(stats['exp21_c_support_mid'], signed=True)}, "
            f"late {_ci(stats['exp21_c_support_late'], signed=True)}",
            "source_value_numeric": _fmt(pure_it_late_support),
            "source_ci_low": _fmt(stats["exp21_c_support_late"]["ci_low"]),
            "source_ci_high": _fmt(stats["exp21_c_support_late"]["ci_high"]),
            "interpretation": "Direct IT-token write-out in pure IT is overwhelmingly late.",
            "caveat": "Logit deltas are local MLP finite differences, not probabilities.",
        },
        {
            "evidence": "Exp21 IT shift",
            "metric": "IT-minus-PT support_it_token",
            "primary_comparison": "early vs mid vs late MLP windows",
            "dense5_value": f"early {_ci(stats['exp21_support_shift_early'], signed=True)}, "
            f"mid {_ci(stats['exp21_support_shift_mid'], signed=True)}, "
            f"late {_ci(stats['exp21_support_shift_late'], signed=True)}",
            "source_value_numeric": _fmt(pure_it_late_support - pt_late_support),
            "source_ci_low": _fmt(stats["exp21_support_shift_late"]["ci_low"]),
            "source_ci_high": _fmt(stats["exp21_support_shift_late"]["ci_high"]),
            "interpretation": "Instruction tuning mostly increases late MLP support for the IT divergent token.",
            "caveat": "Absolute support magnitudes are compared within the same fixed-prefix readout design.",
        },
        {
            "evidence": "Exp21 IT-host necessity",
            "metric": "late-window MLP IT-vs-PT margin drop",
            "primary_comparison": "C_it_chat minus D early/mid/late swap",
            "dense5_value": f"early {_ci(stats['exp21_c_minus_d_early_margin'], signed=True)}, "
            f"mid {_ci(stats['exp21_c_minus_d_mid_margin'], signed=True)}, "
            f"late {_ci(stats['exp21_c_minus_d_late_margin'], signed=True)}",
            "source_value_numeric": _fmt(c_minus_d_late["mean"]),
            "source_ci_low": _fmt(stats["exp21_c_minus_d_late_margin"]["ci_low"]),
            "source_ci_high": _fmt(stats["exp21_c_minus_d_late_margin"]["ci_high"]),
            "interpretation": "Removing IT late layers causes the largest single-window MLP margin loss.",
            "caveat": "This is fixed-prefix MLP write-in, not free-running behavioral causality.",
        },
        {
            "evidence": "Exp21 PT-host sufficiency",
            "metric": "late-window MLP IT-vs-PT margin gain",
            "primary_comparison": "B_late - A and B_midlate - B_mid",
            "dense5_value": f"B_late-A {_ci(stats['exp21_b_late_minus_a_margin'], signed=True)}; "
            f"B_midlate-B_mid {_ci(stats['exp21_b_midlate_minus_b_mid_margin'], signed=True)}",
            "source_value_numeric": _fmt(b_late_minus_a["mean"]),
            "source_ci_low": _fmt(stats["exp21_b_late_minus_a_margin"]["ci_low"]),
            "source_ci_high": _fmt(stats["exp21_b_late_minus_a_margin"]["ci_high"]),
            "interpretation": "IT late MLP updates alone are weak in a PT upstream state.",
            "caveat": "Late readout appears context-gated by earlier IT computation.",
        },
        {
            "evidence": "Exp21 source decomposition",
            "metric": "2x2 late-window MLP margin effects",
            "primary_comparison": "late MLP update vs upstream context vs interaction",
            "dense5_value": f"late MLP update {_ci(stats['exp21_late_weight_margin'], signed=True)}, "
            f"upstream {_ci(stats['exp21_upstream_margin'], signed=True)}, "
            f"interaction {_ci(stats['exp21_interaction_margin'], signed=True)}",
            "source_value_numeric": _fmt(upstream["mean"]),
            "source_ci_low": _fmt(stats["exp21_upstream_margin"]["ci_low"]),
            "source_ci_high": _fmt(stats["exp21_upstream_margin"]["ci_high"]),
            "interpretation": "Upstream IT context is larger than the late-MLP main effect, with positive interaction.",
            "caveat": "Late layers are important but not standalone sufficient.",
        },
        {
            "evidence": "Exp21 residual-opposition caveat",
            "metric": "residual-opposing component vs full-update IT-vs-PT margin",
            "primary_comparison": "pure IT late MLP update",
            "dense5_value": f"full update {_ci(stats['exp21_full_margin_late'], signed=True)}; "
            f"residual-opposing component {_ci(stats['exp21_negative_parallel_late'], signed=True)}; "
            f"delta-cosine IT-PT {_ci(stats['exp21_delta_cos_shift_late'], signed=True)}",
            "source_value_numeric": _fmt(negative_parallel),
            "source_ci_low": _fmt(stats["exp21_negative_parallel_late"]["ci_low"]),
            "source_ci_high": _fmt(stats["exp21_negative_parallel_late"]["ci_high"]),
            "interpretation": "Token-specific full-update write-in is the mechanism evidence.",
            "caveat": "Residual-opposing geometry is a marker, not direct IT-token write-in.",
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "evidence",
        "metric",
        "primary_comparison",
        "dense5_value",
        "source_value_numeric",
        "source_ci_low",
        "source_ci_high",
        "interpretation",
        "caveat",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# Exp20/Exp21 Handoff Synthesis",
        "",
        "Paper insertion summary: mid layers are more diagnostic of candidate identity, "
        "late IT MLPs are the strongest tested native readout/write-out stage, and "
        "negative residual opposition is a geometric companion rather than direct write-in evidence.",
        "",
        "| Evidence | Metric | Comparison | Dense5 value with 95% CI | Interpretation | Caveat |",
        "|---|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['evidence']} | {row['metric']} | {row['primary_comparison']} | "
            f"{row['dense5_value']} | {row['interpretation']} | {row['caveat']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _yerr(stats: list[dict]) -> np.ndarray:
    means = np.array([float(stat["mean"]) for stat in stats])
    lows = np.array([float(stat["ci_low"]) for stat in stats])
    highs = np.array([float(stat["ci_high"]) for stat in stats])
    return np.vstack([means - lows, highs - means])


def _plot(path: Path, exp20: dict, exp21: dict, stats: dict[str, dict]) -> None:
    exp20_c = _exp20_cond(exp20, "native", "C_it_chat", "late_margin_mean")
    margin_drops = [
        exp20_c - _exp20_cond(exp20, "native", "D_early_ptswap", "late_margin_mean"),
        exp20_c - _exp20_cond(exp20, "native", "D_mid_ptswap", "late_margin_mean"),
        exp20_c - _exp20_cond(exp20, "native", "D_late_ptswap", "late_margin_mean"),
        exp20_c - _exp20_cond(exp20, "native", "D_earlymid_ptswap", "late_margin_mean"),
        exp20_c - _exp20_cond(exp20, "native", "D_midlate_ptswap", "late_margin_mean"),
    ]
    margin_stats = [
        stats["exp20_native_margin_drop_early"],
        stats["exp20_native_margin_drop_mid"],
        stats["exp20_native_margin_drop_late"],
        stats["exp20_native_margin_drop_earlymid"],
        stats["exp20_native_margin_drop_midlate"],
    ]
    support_stats = [
        stats["exp21_c_support_early"],
        stats["exp21_c_support_mid"],
        stats["exp21_c_support_late"],
    ]
    support_it = [stat["mean"] for stat in support_stats]
    effects = [
        _exp21_effect(exp21, "late_weight_effect:margin_writein_it_vs_pt")["mean"],
        _exp21_effect(exp21, "upstream_context_effect:margin_writein_it_vs_pt")["mean"],
        _exp21_effect(exp21, "late_interaction:margin_writein_it_vs_pt")["mean"],
    ]
    effect_stats = [
        stats["exp21_late_weight_margin"],
        stats["exp21_upstream_margin"],
        stats["exp21_interaction_margin"],
    ]
    full_stats = [
        stats["exp21_full_margin_early"],
        stats["exp21_full_margin_mid"],
        stats["exp21_full_margin_late"],
    ]
    neg_stats = [
        stats["exp21_negative_parallel_early"],
        stats["exp21_negative_parallel_mid"],
        stats["exp21_negative_parallel_late"],
    ]
    full_margin = [stat["mean"] for stat in full_stats]
    neg_margin = [stat["mean"] for stat in neg_stats]

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("First-Divergence Identity/Margin Decomposition (95% CIs)", fontsize=14)

    ax = axes[0, 0]
    labels = ["early", "mid", "late", "early+mid", "mid+late"]
    ax.bar(labels, margin_drops, color=["#8fa7c5", "#d99b6c", "#b95656", "#a9a27a", "#8a6ea8"])
    ax.errorbar(np.arange(len(labels)), margin_drops, yerr=_yerr(margin_stats), fmt="none", color="black", capsize=3)
    ax.set_title("A. Native IT-host margin drop")
    ax.set_ylabel("logits")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[0, 1]
    ax.bar(["early", "mid", "late"], support_it, color=["#8fa7c5", "#d99b6c", "#b95656"])
    ax.errorbar(np.arange(3), support_it, yerr=_yerr(support_stats), fmt="none", color="black", capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("B. Pure IT support for IT token")
    ax.set_ylabel("MLP logit delta")

    ax = axes[1, 0]
    ax.bar(["late MLP", "upstream", "interaction"], effects, color=["#b95656", "#6f9771", "#7d6fa7"])
    ax.errorbar(np.arange(3), effects, yerr=_yerr(effect_stats), fmt="none", color="black", capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("C. MLP 2x2 source decomposition")
    ax.set_ylabel("MLP margin delta")
    ax.tick_params(axis="x", rotation=18)

    ax = axes[1, 1]
    x = np.arange(3)
    ax.bar(x - 0.18, full_margin, width=0.36, label="full MLP update", color="#b95656")
    ax.bar(x + 0.18, neg_margin, width=0.36, label="residual-opposing component", color="#6f7f95")
    ax.errorbar(x - 0.18, full_margin, yerr=_yerr(full_stats), fmt="none", color="black", capsize=3)
    ax.errorbar(x + 0.18, neg_margin, yerr=_yerr(neg_stats), fmt="none", color="black", capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, ["early", "mid", "late"])
    ax.set_title("D. Full write-in vs opposition component")
    ax.set_ylabel("IT-vs-PT margin delta")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _build_stats(exp20_root: Path, exp21_root: Path, n_boot: int, seed: int) -> dict[str, dict]:
    exp20_raw = _load_exp20_records(exp20_root, "raw_shared", DENSE5)
    exp20_native = _load_exp20_records(exp20_root, "native", DENSE5)
    exp21_native = _load_exp21_records(exp21_root, "native", DENSE5)

    stats = {
        "exp20_raw_b_mid_it": _stat(exp20_raw, _exp20_class_at("B_mid_raw", "it"), n_boot=n_boot, seed=seed + 1),
        "exp20_raw_b_late_it": _stat(exp20_raw, _exp20_class_at("B_late_raw", "it"), n_boot=n_boot, seed=seed + 2),
        "exp20_raw_b_mid_minus_late_it": _stat(
            exp20_raw,
            _diff(_exp20_class_at("B_mid_raw", "it"), _exp20_class_at("B_late_raw", "it")),
            n_boot=n_boot,
            seed=seed + 3,
        ),
        "exp20_raw_d_mid_pt": _stat(exp20_raw, _exp20_class_at("D_mid_ptswap", "pt"), n_boot=n_boot, seed=seed + 4),
        "exp20_raw_d_late_pt": _stat(exp20_raw, _exp20_class_at("D_late_ptswap", "pt"), n_boot=n_boot, seed=seed + 5),
        "exp20_raw_d_mid_minus_late_pt": _stat(
            exp20_raw,
            _diff(_exp20_class_at("D_mid_ptswap", "pt"), _exp20_class_at("D_late_ptswap", "pt")),
            n_boot=n_boot,
            seed=seed + 6,
        ),
        "exp20_native_d_early_it": _stat(
            exp20_native, _exp20_class_at("D_early_ptswap", "it"), n_boot=n_boot, seed=seed + 7
        ),
        "exp20_native_d_mid_it": _stat(
            exp20_native, _exp20_class_at("D_mid_ptswap", "it"), n_boot=n_boot, seed=seed + 8
        ),
        "exp20_native_d_late_it": _stat(
            exp20_native, _exp20_class_at("D_late_ptswap", "it"), n_boot=n_boot, seed=seed + 9
        ),
    }
    for offset, (key, condition) in enumerate(
        [
            ("exp20_native_margin_drop_early", "D_early_ptswap"),
            ("exp20_native_margin_drop_mid", "D_mid_ptswap"),
            ("exp20_native_margin_drop_late", "D_late_ptswap"),
            ("exp20_native_margin_drop_earlymid", "D_earlymid_ptswap"),
            ("exp20_native_margin_drop_midlate", "D_midlate_ptswap"),
        ],
        start=10,
    ):
        stats[key] = _stat(
            exp20_native,
            _diff(lambda record: _exp20_margin(record, "C_it_chat"), lambda record, c=condition: _exp20_margin(record, c)),
            n_boot=n_boot,
            seed=seed + offset,
        )

    windows = [("early", "early"), ("mid", "mid_policy"), ("late", "late_reconciliation")]
    for offset, (short, window) in enumerate(windows, start=20):
        stats[f"exp21_c_support_{short}"] = _stat(
            exp21_native,
            _exp21_value("C_it_chat", window, "support_it_token"),
            n_boot=n_boot,
            seed=seed + offset,
        )
        stats[f"exp21_support_shift_{short}"] = _stat(
            exp21_native,
            _diff(
                _exp21_value("C_it_chat", window, "support_it_token"),
                _exp21_value("A_pt_raw", window, "support_it_token"),
            ),
            n_boot=n_boot,
            seed=seed + offset + 10,
        )
        stats[f"exp21_full_margin_{short}"] = _stat(
            exp21_native,
            _exp21_value("C_it_chat", window, "margin_writein_it_vs_pt"),
            n_boot=n_boot,
            seed=seed + offset + 20,
        )
        stats[f"exp21_negative_parallel_{short}"] = _stat(
            exp21_native,
            _exp21_value("C_it_chat", window, "opposition_margin_it_vs_pt"),
            n_boot=n_boot,
            seed=seed + offset + 30,
        )

    margin = "margin_writein_it_vs_pt"
    stats.update(
        {
            "exp21_c_minus_d_early_margin": _stat(
                exp21_native,
                _diff(
                    _exp21_value("C_it_chat", "late_reconciliation", margin),
                    _exp21_value("D_early_ptswap", "late_reconciliation", margin),
                ),
                n_boot=n_boot,
                seed=seed + 60,
            ),
            "exp21_c_minus_d_mid_margin": _stat(
                exp21_native,
                _diff(
                    _exp21_value("C_it_chat", "late_reconciliation", margin),
                    _exp21_value("D_mid_ptswap", "late_reconciliation", margin),
                ),
                n_boot=n_boot,
                seed=seed + 61,
            ),
            "exp21_c_minus_d_late_margin": _stat(
                exp21_native,
                _diff(
                    _exp21_value("C_it_chat", "late_reconciliation", margin),
                    _exp21_value("D_late_ptswap", "late_reconciliation", margin),
                ),
                n_boot=n_boot,
                seed=seed + 62,
            ),
            "exp21_b_late_minus_a_margin": _stat(
                exp21_native,
                _diff(
                    _exp21_value("B_late_raw", "late_reconciliation", margin),
                    _exp21_value("A_pt_raw", "late_reconciliation", margin),
                ),
                n_boot=n_boot,
                seed=seed + 63,
            ),
            "exp21_b_midlate_minus_b_mid_margin": _stat(
                exp21_native,
                _diff(
                    _exp21_value("B_midlate_raw", "late_reconciliation", margin),
                    _exp21_value("B_mid_raw", "late_reconciliation", margin),
                ),
                n_boot=n_boot,
                seed=seed + 64,
            ),
            "exp21_late_weight_margin": _stat(
                exp21_native, _exp21_late_weight(margin), n_boot=n_boot, seed=seed + 65
            ),
            "exp21_upstream_margin": _stat(
                exp21_native, _exp21_upstream(margin), n_boot=n_boot, seed=seed + 66
            ),
            "exp21_interaction_margin": _stat(
                exp21_native, _exp21_interaction(margin), n_boot=n_boot, seed=seed + 67
            ),
            "exp21_delta_cos_shift_late": _stat(
                exp21_native,
                _diff(
                    _exp21_value("C_it_chat", "late_reconciliation", "delta_cosine_mlp"),
                    _exp21_value("A_pt_raw", "late_reconciliation", "delta_cosine_mlp"),
                ),
                n_boot=n_boot,
                seed=seed + 68,
            ),
        }
    )
    return stats


def _check_csv(path: Path, rows: list[dict[str, str]]) -> None:
    expected = {row["evidence"]: float(row["source_value_numeric"]) for row in rows}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            observed = float(row["source_value_numeric"])
            source = expected[row["evidence"]]
            if not math.isclose(observed, source, rel_tol=0.0, abs_tol=TOLERANCE):
                raise AssertionError(f"{row['evidence']} differs from source: {observed} vs {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp20-root", type=Path, default=EXP20_ROOT_DEFAULT)
    parser.add_argument("--exp21-root", type=Path, default=EXP21_ROOT_DEFAULT)
    parser.add_argument("--exp20", type=Path, default=EXP20_DEFAULT)
    parser.add_argument("--exp21", type=Path, default=EXP21_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260426)
    args = parser.parse_args()

    exp20 = _load(args.exp20)
    exp21 = _load(args.exp21)
    stats = _build_stats(args.exp20_root, args.exp21_root, args.n_boot, args.seed)
    rows = _rows(exp20, exp21, stats)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "exp20_exp21_handoff_table.csv"
    md_path = args.out_dir / "exp20_exp21_handoff_note.md"
    fig_path = args.out_dir / "exp20_exp21_handoff_synthesis.png"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    _plot(fig_path, exp20, exp21, stats)
    _check_csv(csv_path, rows)
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "markdown": str(md_path),
                "figure": str(fig_path),
                "rows": len(rows),
                "n_boot": args.n_boot,
                "tolerance": TOLERANCE,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
