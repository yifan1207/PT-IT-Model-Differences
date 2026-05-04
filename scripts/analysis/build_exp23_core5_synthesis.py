#!/usr/bin/env python3
"""Build the paper-facing Core-5 synthesis for the Exp23 factorial.

Core-5 combines the four smaller dense support families from Exp23 with the
Qwen2.5-32B Exp24 scale check. The script filters the stored support-family
summary to the manuscript scope before combining prompt-bootstrap estimates.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.analysis.build_exp23_dense6_core_synthesis import (
    DEFAULT_DENSE5_POSITION,
    DEFAULT_DENSE5_SUMMARY,
    DEFAULT_QWEN32_POSITION,
    DEFAULT_QWEN32_SUMMARY,
    EFFECTS,
    POSITION_LABELS,
    _combine_family_rows,
    _family_effect_rows,
    _load_json,
    _normalize_position,
    _se_from_ci,
    _write_csv,
)


CORE_SMALL_MODELS = {"llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"}
DEFAULT_OUT_DIR = Path("results/paper_synthesis/exp23_core5")


def _core_family_effect_rows(dense_summary: dict[str, Any], qwen32_summary: dict[str, Any], effect: str, readout: str) -> list[dict[str, Any]]:
    small_rows = [row for row in _family_effect_rows(dense_summary, effect, readout) if row["model"] in CORE_SMALL_MODELS]
    return small_rows + _family_effect_rows(qwen32_summary, effect, readout)


def _effect_table(dense_summary: dict[str, Any], qwen32_summary: dict[str, Any], readout: str) -> list[dict[str, Any]]:
    rows = []
    for effect in EFFECTS:
        families = _core_family_effect_rows(dense_summary, qwen32_summary, effect, readout)
        core5 = _combine_family_rows(families)
        estimates = sorted(float(row["estimate"]) for row in families)
        rows.append(
            {
                "effect": effect,
                "readout": readout,
                "core5_estimate": core5["estimate"],
                "core5_ci95_low": core5["ci95_low"],
                "core5_ci95_high": core5["ci95_high"],
                "core5_n_models": core5["n_models"],
                "family_median": estimates[len(estimates) // 2],
                "trimmed_mean": sum(estimates[1:-1]) / max(1, len(estimates[1:-1])),
                "ci_method": core5["ci_method"],
                "models": ";".join(core5["models"]),
            }
        )
    return rows


def _family_table(dense_summary: dict[str, Any], qwen32_summary: dict[str, Any], readout: str) -> list[dict[str, Any]]:
    by_effect = {
        effect: {row["model"]: row for row in _core_family_effect_rows(dense_summary, qwen32_summary, effect, readout)}
        for effect in ("late_it_given_pt_upstream", "late_it_given_it_upstream", "interaction")
    }
    models = sorted(by_effect["interaction"], key=lambda model: by_effect["interaction"][model]["estimate"])
    rows = []
    for model in models:
        pt = by_effect["late_it_given_pt_upstream"][model]
        it = by_effect["late_it_given_it_upstream"][model]
        ixn = by_effect["interaction"][model]
        rows.append(
            {
                "model": model,
                "readout": readout,
                "n_prompt_clusters": int(ixn["n_prompt_clusters"]),
                "late_it_given_pt_upstream": float(pt["estimate"]),
                "late_it_given_it_upstream": float(it["estimate"]),
                "interaction": float(ixn["estimate"]),
                "interaction_ci95_low": float(ixn["ci95_low"]),
                "interaction_ci95_high": float(ixn["ci95_high"]),
            }
        )
    return rows


def _position_rows(dense5_path: Path, qwen32_path: Path) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    with dense5_path.open() as handle:
        for row in csv.DictReader(handle):
            if row["model"] not in CORE_SMALL_MODELS:
                continue
            label = _normalize_position(row["position_stratum"])
            lo = float(row["ci_low"])
            hi = float(row["ci_high"])
            by_label.setdefault(label, []).append(
                {
                    "model": row["model"],
                    "estimate": float(row["interaction"]),
                    "se": _se_from_ci(lo, hi),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_prompt_clusters": int(row["records"]),
                }
            )
    with qwen32_path.open() as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != "interaction":
                continue
            label = _normalize_position(row["position_label"])
            lo = float(row["ci95_low"])
            hi = float(row["ci95_high"])
            by_label.setdefault(label, []).append(
                {
                    "model": "qwen25_32b",
                    "estimate": float(row["estimate"]),
                    "se": _se_from_ci(lo, hi),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_prompt_clusters": int(row["n_prompt_clusters"]),
                }
            )

    rows = []
    for label in POSITION_LABELS:
        family_rows = by_label.get(label, [])
        if not family_rows:
            continue
        core5 = _combine_family_rows(family_rows)
        rows.append(
            {
                "position_stratum": label,
                "core5_estimate": core5["estimate"],
                "core5_ci95_low": core5["ci95_low"],
                "core5_ci95_high": core5["ci95_high"],
                "core5_n_models": core5["n_models"],
                "core5_n_prompt_clusters": core5["n_prompt_clusters"],
                "ci_method": core5["ci_method"],
                "models": ";".join(core5["models"]),
            }
        )
    return rows


def _plot_core5(path: Path, family_rows: list[dict[str, Any]], effects: list[dict[str, Any]]) -> None:
    common_it = sorted([row for row in family_rows if row["readout"] == "common_it"], key=lambda row: row["interaction"])
    effect_rows = {row["effect"]: row for row in effects if row["readout"] == "common_it"}
    pt = effect_rows["late_it_given_pt_upstream"]
    it = effect_rows["late_it_given_it_upstream"]
    interaction = effect_rows["interaction"]
    labels = {
        "llama31_8b": "Llama 3.1 8B",
        "qwen3_4b": "Qwen3 4B",
        "qwen25_32b": "Qwen2.5 32B",
        "olmo2_7b": "OLMo2 7B",
        "mistral_7b": "Mistral 7B",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.0, 1.35]})
    vals = [pt["core5_estimate"], it["core5_estimate"]]
    lo = [pt["core5_estimate"] - pt["core5_ci95_low"], it["core5_estimate"] - it["core5_ci95_low"]]
    hi = [pt["core5_ci95_high"] - pt["core5_estimate"], it["core5_ci95_high"] - it["core5_estimate"]]
    axes[0].bar([0, 1], vals, color=["#4c78a8", "#f58518"], width=0.62)
    axes[0].errorbar([0, 1], vals, yerr=[lo, hi], fmt="none", color="black", capsize=4, lw=1.1)
    axes[0].set_xticks([0, 1], ["PT upstream", "IT upstream"])
    axes[0].set_ylabel("IT-vs-PT margin shift (logits)")
    axes[0].set_title("IT late-stack effect")
    axes[0].axhline(0, color="#555", lw=0.8)
    axes[0].text(
        0.5,
        max(vals) * 0.88,
        f"interaction {interaction['core5_estimate']:+.2f}\n95% CI "
        f"[{interaction['core5_ci95_low']:+.2f}, {interaction['core5_ci95_high']:+.2f}]",
        ha="center",
        va="center",
        fontsize=9,
    )

    y = list(range(len(common_it)))
    vals = [row["interaction"] for row in common_it]
    xerr = [
        [row["interaction"] - row["interaction_ci95_low"] for row in common_it],
        [row["interaction_ci95_high"] - row["interaction"] for row in common_it],
    ]
    axes[1].barh(y, vals, color="#54a24b", height=0.64)
    axes[1].errorbar(vals, y, xerr=xerr, fmt="none", color="black", capsize=3, lw=1.0)
    axes[1].axvline(0, color="#555", lw=0.8)
    axes[1].set_yticks(y, [labels.get(row["model"], row["model"]) for row in common_it])
    axes[1].set_xlabel("Upstream x late interaction (logits)")
    axes[1].set_title("Core-5 family estimates")
    axes[1].grid(axis="x", color="#dddddd", lw=0.7)
    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_markdown(path: Path, effects: list[dict[str, Any]], positions: list[dict[str, Any]]) -> None:
    by_effect = {row["effect"]: row for row in effects if row["readout"] == "common_it"}
    lines = [
        "# Exp23 Core-5 Synthesis",
        "",
        "Core-5 combines the four smaller dense support families with Qwen2.5-32B.",
        "CIs use an independent normal approximation to stored per-family prompt-bootstrap intervals.",
        "",
        f"- Late IT from PT upstream: `{by_effect['late_it_given_pt_upstream']['core5_estimate']:+.3f}`.",
        f"- Late IT from IT upstream: `{by_effect['late_it_given_it_upstream']['core5_estimate']:+.3f}`.",
        f"- Upstream x late interaction: `{by_effect['interaction']['core5_estimate']:+.3f}`.",
        "",
        "## Position Rows",
        "",
        "| Stratum | Core-5 interaction |",
        "|---|---:|",
    ]
    for row in positions:
        lines.append(
            f"| {row['position_stratum']} | "
            f"`{row['core5_estimate']:+.3f}` "
            f"`[{row['core5_ci95_low']:+.3f}, {row['core5_ci95_high']:+.3f}]` |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense5-summary", type=Path, default=DEFAULT_DENSE5_SUMMARY)
    parser.add_argument("--qwen32-summary", type=Path, default=DEFAULT_QWEN32_SUMMARY)
    parser.add_argument("--dense5-position", type=Path, default=DEFAULT_DENSE5_POSITION)
    parser.add_argument("--qwen32-position", type=Path, default=DEFAULT_QWEN32_POSITION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dense5 = _load_json(args.dense5_summary)
    qwen32 = _load_json(args.qwen32_summary)
    effects = _effect_table(dense5, qwen32, "common_it") + _effect_table(dense5, qwen32, "common_pt")
    families = _family_table(dense5, qwen32, "common_it") + _family_table(dense5, qwen32, "common_pt")
    positions = _position_rows(args.dense5_position, args.qwen32_position)
    _write_csv(args.out_dir / "exp23_core5_core_effects.csv", effects)
    _write_csv(args.out_dir / "exp23_core5_family_effects.csv", families)
    _write_csv(args.out_dir / "exp23_core5_position_sensitivity.csv", positions)
    _plot_core5(args.out_dir / "exp23_core5_interaction.png", families, effects)
    (args.out_dir / "exp23_core5_summary.json").write_text(
        json.dumps(
            {
                "effect_rows": effects,
                "family_rows": families,
                "position_rows": positions,
                "ci_method": "family_bootstrap_normal_approx_from_stored_per_family_cis",
                "inputs": {
                    "dense5_summary": str(args.dense5_summary),
                    "qwen32_summary": str(args.qwen32_summary),
                    "dense5_position": str(args.dense5_position),
                    "qwen32_position": str(args.qwen32_position),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    _write_markdown(args.out_dir / "exp23_core5_summary.md", effects, positions)
    print(f"[core5] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
