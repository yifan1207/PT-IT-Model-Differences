#!/usr/bin/env python3
"""Build paper-facing Exp20/Exp21 handoff synthesis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXP20_DEFAULT = Path(
    "results/exp20_divergence_token_counterfactual/"
    "factorial_validation_holdout_fast_20260425_2009_with_early/"
    "validation_analysis/summary.json"
)
EXP21_DEFAULT = Path(
    "results/exp21_productive_opposition/"
    "exp21_full_productive_opposition_clean_20260426_053736/"
    "analysis/summary.json"
)
OUT_DEFAULT = Path("results/paper_synthesis")
TOLERANCE = 1e-3


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


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


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _rows(exp20: dict, exp21: dict) -> list[dict[str, str]]:
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
            "dense5_value": f"{_pct(_exp20_frac(exp20, 'raw_shared', 'B_mid_raw', 'it'))} vs "
            f"{_pct(_exp20_frac(exp20, 'raw_shared', 'B_late_raw', 'it'))}",
            "source_value_numeric": _fmt(
                _exp20_frac(exp20, "raw_shared", "B_mid_raw", "it")
                - _exp20_frac(exp20, "raw_shared", "B_late_raw", "it")
            ),
            "interpretation": "Middle grafts transfer opposite-model token identity more than late grafts.",
            "caveat": "Raw-shared is the cleaner identity control but not the native IT deployment prompt.",
        },
        {
            "evidence": "Exp20 native readout",
            "metric": "native IT-host margin drop",
            "primary_comparison": "pure IT minus PT early/mid/late swap",
            "dense5_value": f"early {_fmt(exp20_c - exp20_d_early)}, mid {_fmt(exp20_c - exp20_d_mid)}, "
            f"late {_fmt(exp20_c - exp20_d_late)}",
            "source_value_numeric": _fmt(exp20_c - exp20_d_late),
            "interpretation": "Late swap causes the largest single-window loss of native IT-vs-PT token margin.",
            "caveat": "First-divergence token margin is a causal proxy, not direct human-rated behavior.",
        },
        {
            "evidence": "Exp20 combined windows",
            "metric": "native IT-host margin drop",
            "primary_comparison": "pure IT minus PT early+mid / mid+late swap",
            "dense5_value": f"early+mid {_fmt(exp20_c - exp20_d_earlymid)}, "
            f"mid+late {_fmt(exp20_c - exp20_d_midlate)}",
            "source_value_numeric": _fmt(exp20_c - exp20_d_midlate),
            "interpretation": "Multi-window swaps hurt more than any single window, supporting a distributed circuit.",
            "caveat": "The strongest claim is distributed mid-to-late handoff, not a late-only module.",
        },
        {
            "evidence": "Exp21 MLP support",
            "metric": "pure IT support_it_token",
            "primary_comparison": "early vs mid vs late MLP windows",
            "dense5_value": f"early {_fmt(pure_it_early_support)}, mid {_fmt(pure_it_mid_support)}, "
            f"late {_fmt(pure_it_late_support)}",
            "source_value_numeric": _fmt(pure_it_late_support),
            "interpretation": "Direct IT-token write-out in pure IT is overwhelmingly late.",
            "caveat": "Logit deltas are local MLP finite differences, not probabilities.",
        },
        {
            "evidence": "Exp21 IT shift",
            "metric": "IT-minus-PT support_it_token",
            "primary_comparison": "early vs mid vs late MLP windows",
            "dense5_value": f"early {_fmt(pure_it_early_support - pt_early_support)}, "
            f"mid {_fmt(pure_it_mid_support - pt_mid_support)}, "
            f"late {_fmt(pure_it_late_support - pt_late_support)}",
            "source_value_numeric": _fmt(pure_it_late_support - pt_late_support),
            "interpretation": "Instruction tuning mostly increases late MLP support for the IT divergent token.",
            "caveat": "Absolute support magnitudes are compared within the same fixed-prefix readout design.",
        },
        {
            "evidence": "Exp21 IT-host necessity",
            "metric": "late-window MLP IT-vs-PT margin drop",
            "primary_comparison": "C_it_chat minus D early/mid/late swap",
            "dense5_value": f"early {_fmt(c_minus_d_early)}, mid {_fmt(c_minus_d_mid)}, "
            f"late {_fmt(c_minus_d_late['mean'])}",
            "source_value_numeric": _fmt(c_minus_d_late["mean"]),
            "interpretation": "Removing IT late layers causes the largest single-window MLP margin loss.",
            "caveat": "This is fixed-prefix MLP write-in, not free-running behavioral causality.",
        },
        {
            "evidence": "Exp21 PT-host sufficiency",
            "metric": "late-window MLP IT-vs-PT margin gain",
            "primary_comparison": "B_late - A and B_midlate - B_mid",
            "dense5_value": f"B_late-A {_fmt(b_late_minus_a['mean'])}; "
            f"B_midlate-B_mid {_fmt(b_midlate_minus_b_mid['mean'])}",
            "source_value_numeric": _fmt(b_late_minus_a["mean"]),
            "interpretation": "IT late weights alone are weak in a PT upstream state.",
            "caveat": "Late readout appears context-gated by earlier IT computation.",
        },
        {
            "evidence": "Exp21 source decomposition",
            "metric": "2x2 late-window MLP margin effects",
            "primary_comparison": "late weights vs upstream context vs interaction",
            "dense5_value": f"late weights {_fmt(late_weight['mean'])}, upstream {_fmt(upstream['mean'])}, "
            f"interaction {_fmt(interaction['mean'])}",
            "source_value_numeric": _fmt(upstream["mean"]),
            "interpretation": "Upstream IT context is larger than the late-weight main effect, with positive interaction.",
            "caveat": "Late layers are important but not standalone sufficient.",
        },
        {
            "evidence": "Exp21 residual-opposition caveat",
            "metric": "negative-parallel vs full-update IT-vs-PT margin",
            "primary_comparison": "pure IT late MLP update",
            "dense5_value": f"full update {_fmt(full_update_margin)}; negative-parallel {_fmt(negative_parallel)}; "
            f"delta-cosine IT-PT {_fmt(delta_cos_shift)}",
            "source_value_numeric": _fmt(negative_parallel),
            "interpretation": "Token-specific full-update write-in is the mechanism evidence.",
            "caveat": "Negative residual opposition is a geometric marker, not direct IT-token write-in.",
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "evidence",
        "metric",
        "primary_comparison",
        "dense5_value",
        "source_value_numeric",
        "interpretation",
        "caveat",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
        "| Evidence | Metric | Comparison | Dense5 value | Interpretation | Caveat |",
        "|---|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['evidence']} | {row['metric']} | {row['primary_comparison']} | "
            f"{row['dense5_value']} | {row['interpretation']} | {row['caveat']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _plot(path: Path, exp20: dict, exp21: dict) -> None:
    exp20_c = _exp20_cond(exp20, "native", "C_it_chat", "late_margin_mean")
    margin_drops = [
        exp20_c - _exp20_cond(exp20, "native", "D_early_ptswap", "late_margin_mean"),
        exp20_c - _exp20_cond(exp20, "native", "D_mid_ptswap", "late_margin_mean"),
        exp20_c - _exp20_cond(exp20, "native", "D_late_ptswap", "late_margin_mean"),
        exp20_c - _exp20_cond(exp20, "native", "D_earlymid_ptswap", "late_margin_mean"),
        exp20_c - _exp20_cond(exp20, "native", "D_midlate_ptswap", "late_margin_mean"),
    ]
    support_it = [
        _exp21_window(exp21, "C_it_chat", "early", "support_it_token"),
        _exp21_window(exp21, "C_it_chat", "mid_policy", "support_it_token"),
        _exp21_window(exp21, "C_it_chat", "late_reconciliation", "support_it_token"),
    ]
    effects = [
        _exp21_effect(exp21, "late_weight_effect:margin_writein_it_vs_pt")["mean"],
        _exp21_effect(exp21, "upstream_context_effect:margin_writein_it_vs_pt")["mean"],
        _exp21_effect(exp21, "late_interaction:margin_writein_it_vs_pt")["mean"],
    ]
    full_margin = [
        _exp21_window(exp21, "C_it_chat", "early", "margin_writein_it_vs_pt"),
        _exp21_window(exp21, "C_it_chat", "mid_policy", "margin_writein_it_vs_pt"),
        _exp21_window(exp21, "C_it_chat", "late_reconciliation", "margin_writein_it_vs_pt"),
    ]
    neg_margin = [
        _exp21_window(exp21, "C_it_chat", "early", "opposition_margin_it_vs_pt"),
        _exp21_window(exp21, "C_it_chat", "mid_policy", "opposition_margin_it_vs_pt"),
        _exp21_window(exp21, "C_it_chat", "late_reconciliation", "opposition_margin_it_vs_pt"),
    ]

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Exp20/Exp21: Mid Identity, Late Readout, Opposition Caveat", fontsize=14)

    ax = axes[0, 0]
    labels = ["early", "mid", "late", "early+mid", "mid+late"]
    ax.bar(labels, margin_drops, color=["#8fa7c5", "#d99b6c", "#b95656", "#a9a27a", "#8a6ea8"])
    ax.set_title("A. Exp20 native IT-host margin drop")
    ax.set_ylabel("logits")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[0, 1]
    ax.bar(["early", "mid", "late"], support_it, color=["#8fa7c5", "#d99b6c", "#b95656"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("B. Exp21 pure IT support for IT token")
    ax.set_ylabel("MLP logit delta")

    ax = axes[1, 0]
    ax.bar(["late weights", "upstream", "interaction"], effects, color=["#b95656", "#6f9771", "#7d6fa7"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("C. Exp21 2x2 source decomposition")
    ax.set_ylabel("MLP margin delta")
    ax.tick_params(axis="x", rotation=18)

    ax = axes[1, 1]
    x = np.arange(3)
    ax.bar(x - 0.18, full_margin, width=0.36, label="full MLP update", color="#b95656")
    ax.bar(x + 0.18, neg_margin, width=0.36, label="negative-parallel component", color="#6f7f95")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, ["early", "mid", "late"])
    ax.set_title("D. Full write-in vs opposition component")
    ax.set_ylabel("IT-vs-PT margin delta")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


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
    parser.add_argument("--exp20", type=Path, default=EXP20_DEFAULT)
    parser.add_argument("--exp21", type=Path, default=EXP21_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    exp20 = _load(args.exp20)
    exp21 = _load(args.exp21)
    rows = _rows(exp20, exp21)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "exp20_exp21_handoff_table.csv"
    md_path = args.out_dir / "exp20_exp21_handoff_note.md"
    fig_path = args.out_dir / "exp20_exp21_handoff_synthesis.png"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    _plot(fig_path, exp20, exp21)
    _check_csv(csv_path, rows)
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "markdown": str(md_path),
                "figure": str(fig_path),
                "rows": len(rows),
                "tolerance": TOLERANCE,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
