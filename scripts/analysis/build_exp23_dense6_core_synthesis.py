#!/usr/bin/env python3
"""Build paper-facing Dense-6 synthesis for the Exp23 core factorial.

This combines the five-family 4B-8B Exp23 summary with the Qwen2.5-32B Exp24
summary. Raw Qwen2.5-32B records are not committed locally, so confidence
intervals are combined from stored per-family prompt-bootstrap intervals using
an independent normal approximation to each family bootstrap distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


Z975 = 1.959963984540054
DEFAULT_DENSE5_SUMMARY = Path(
    "results/exp23_midlate_interaction_suite/"
    "exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/exp23_summary.json"
)
DEFAULT_QWEN32_SUMMARY = Path(
    "results/exp24_32b_external_validity/"
    "exp24_qwen25_32b_full_eval_v21_20260427_194839/"
    "analysis/exp23_midlate_interaction_suite/exp23_summary.json"
)
DEFAULT_DENSE5_POSITION = Path("results/paper_synthesis/exp23_position_sensitivity_per_family.csv")
DEFAULT_QWEN32_POSITION = Path(
    "results/paper_synthesis/exp24_32b_external_validity/exp24_32b_position_sensitivity.csv"
)
DEFAULT_OUT_DIR = Path("results/paper_synthesis/exp23_dense6_core")

EFFECTS = (
    "late_it_given_pt_upstream",
    "late_it_given_it_upstream",
    "late_weight_effect",
    "upstream_context_effect",
    "interaction",
)
POSITION_LABELS = (
    "all positions",
    "positions >=1",
    "positions >=3",
    "position >=5",
    "position >=10",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _se_from_ci(lo: float, hi: float) -> float:
    return (float(hi) - float(lo)) / (2.0 * Z975)


def _combine_family_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    estimates = [float(row["estimate"]) for row in rows]
    ses = [float(row["se"]) for row in rows]
    estimate = sum(estimates) / len(estimates)
    se = math.sqrt(sum(se * se for se in ses)) / len(ses)
    return {
        "estimate": estimate,
        "ci95_low": estimate - Z975 * se,
        "ci95_high": estimate + Z975 * se,
        "n_models": len(rows),
        "n_prompt_clusters": sum(int(row.get("n_prompt_clusters", 0)) for row in rows),
        "models": [row["model"] for row in rows],
        "ci_method": "family_bootstrap_normal_approx_from_stored_per_family_cis",
    }


def _family_effect_rows(summary: dict[str, Any], effect: str, readout: str) -> list[dict[str, Any]]:
    payload = summary["residual_factorial"]["effects"][readout][effect]["model_cis"]
    out = []
    for model, row in payload.items():
        lo = float(row["ci95_low"])
        hi = float(row["ci95_high"])
        out.append(
            {
                "model": model,
                "estimate": float(row["estimate"]),
                "se": _se_from_ci(lo, hi),
                "ci95_low": lo,
                "ci95_high": hi,
                "n_prompt_clusters": int(row.get("n_prompt_clusters") or row.get("n_units") or 0),
            }
        )
    return out


def _effect_table(dense5: dict[str, Any], qwen32: dict[str, Any], readout: str) -> list[dict[str, Any]]:
    rows = []
    for effect in EFFECTS:
        families = _family_effect_rows(dense5, effect, readout) + _family_effect_rows(qwen32, effect, readout)
        dense6 = _combine_family_rows(families)
        no_gemma = _combine_family_rows([row for row in families if row["model"] != "gemma3_4b"])
        estimates = sorted(float(row["estimate"]) for row in families)
        rows.append(
            {
                "effect": effect,
                "readout": readout,
                "dense6_estimate": dense6["estimate"],
                "dense6_ci95_low": dense6["ci95_low"],
                "dense6_ci95_high": dense6["ci95_high"],
                "dense6_n_models": dense6["n_models"],
                "dense6_n_prompt_clusters": dense6["n_prompt_clusters"],
                "gemma_removed_estimate": no_gemma["estimate"],
                "gemma_removed_ci95_low": no_gemma["ci95_low"],
                "gemma_removed_ci95_high": no_gemma["ci95_high"],
                "family_median": (estimates[2] + estimates[3]) / 2.0,
                "trimmed_mean": sum(estimates[1:-1]) / 4.0,
                "ci_method": dense6["ci_method"],
                "models": ";".join(dense6["models"]),
            }
        )
    return rows


def _family_table(dense5: dict[str, Any], qwen32: dict[str, Any], readout: str) -> list[dict[str, Any]]:
    by_effect = {
        effect: {
            row["model"]: row
            for row in (_family_effect_rows(dense5, effect, readout) + _family_effect_rows(qwen32, effect, readout))
        }
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


def _normalize_position(label: str) -> str:
    return label.replace("positions >=5", "position >=5").replace("positions >=10", "position >=10")


def _load_position_rows(dense5_path: Path, qwen32_path: Path) -> dict[str, list[dict[str, Any]]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    with dense5_path.open() as handle:
        for row in csv.DictReader(handle):
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
    return by_label


def _position_table(dense5_path: Path, qwen32_path: Path) -> list[dict[str, Any]]:
    by_label = _load_position_rows(dense5_path, qwen32_path)
    rows = []
    for label in POSITION_LABELS:
        family_rows = by_label.get(label, [])
        if not family_rows:
            continue
        dense6 = _combine_family_rows(family_rows)
        no_gemma_rows = [row for row in family_rows if row["model"] != "gemma3_4b"]
        no_gemma = _combine_family_rows(no_gemma_rows) if len(no_gemma_rows) != len(family_rows) else None
        rows.append(
            {
                "position_stratum": label,
                "dense6_estimate": dense6["estimate"],
                "dense6_ci95_low": dense6["ci95_low"],
                "dense6_ci95_high": dense6["ci95_high"],
                "dense6_n_models": dense6["n_models"],
                "dense6_n_prompt_clusters": dense6["n_prompt_clusters"],
                "gemma_removed_estimate": None if no_gemma is None else no_gemma["estimate"],
                "gemma_removed_ci95_low": None if no_gemma is None else no_gemma["ci95_low"],
                "gemma_removed_ci95_high": None if no_gemma is None else no_gemma["ci95_high"],
                "gemma_removed_n_models": None if no_gemma is None else no_gemma["n_models"],
                "gemma_removed_n_prompt_clusters": None if no_gemma is None else no_gemma["n_prompt_clusters"],
                "ci_method": dense6["ci_method"],
                "models": ";".join(dense6["models"]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, effects: list[dict[str, Any]], positions: list[dict[str, Any]]) -> None:
    by_effect = {row["effect"]: row for row in effects if row["readout"] == "common_it"}
    by_effect_pt = {row["effect"]: row for row in effects if row["readout"] == "common_pt"}
    interaction = by_effect["interaction"]
    pt = by_effect["late_it_given_pt_upstream"]
    it = by_effect["late_it_given_it_upstream"]
    interaction_pt = by_effect_pt["interaction"]
    pt_pt = by_effect_pt["late_it_given_pt_upstream"]
    it_pt = by_effect_pt["late_it_given_it_upstream"]
    lines = [
        "# Exp23 Dense-6 Core Synthesis",
        "",
        "Dense-6 combines the five 4B-8B Exp23 families with the Qwen2.5-32B Exp24 family.",
        "CIs use an independent normal approximation to the stored per-family prompt-bootstrap intervals.",
        "",
        f"- Late IT from PT upstream: `{pt['dense6_estimate']:+.3f}` "
        f"`[{pt['dense6_ci95_low']:+.3f}, {pt['dense6_ci95_high']:+.3f}]`.",
        f"- Late IT from IT upstream: `{it['dense6_estimate']:+.3f}` "
        f"`[{it['dense6_ci95_low']:+.3f}, {it['dense6_ci95_high']:+.3f}]`.",
        f"- Upstream x late interaction: `{interaction['dense6_estimate']:+.3f}` "
        f"`[{interaction['dense6_ci95_low']:+.3f}, {interaction['dense6_ci95_high']:+.3f}]`.",
        f"- Gemma-removed Dense-5 interaction: `{interaction['gemma_removed_estimate']:+.3f}` "
        f"`[{interaction['gemma_removed_ci95_low']:+.3f}, {interaction['gemma_removed_ci95_high']:+.3f}]`.",
        f"- Common-PT cross-check: PT upstream `{pt_pt['dense6_estimate']:+.3f}` "
        f"`[{pt_pt['dense6_ci95_low']:+.3f}, {pt_pt['dense6_ci95_high']:+.3f}]`; "
        f"IT upstream `{it_pt['dense6_estimate']:+.3f}` "
        f"`[{it_pt['dense6_ci95_low']:+.3f}, {it_pt['dense6_ci95_high']:+.3f}]`; "
        f"interaction `{interaction_pt['dense6_estimate']:+.3f}` "
        f"`[{interaction_pt['dense6_ci95_low']:+.3f}, {interaction_pt['dense6_ci95_high']:+.3f}]`.",
        "",
        "## Position Rows",
        "",
        "| Stratum | Dense-6 interaction | Gemma removed |",
        "|---|---:|---:|",
    ]
    for row in positions:
        no_gemma = ""
        if row["gemma_removed_estimate"] is not None:
            no_gemma = (
                f"`{row['gemma_removed_estimate']:+.3f}` "
                f"`[{row['gemma_removed_ci95_low']:+.3f}, {row['gemma_removed_ci95_high']:+.3f}]`"
            )
        lines.append(
            f"| {row['position_stratum']} | "
            f"`{row['dense6_estimate']:+.3f}` "
            f"`[{row['dense6_ci95_low']:+.3f}, {row['dense6_ci95_high']:+.3f}]` | "
            f"{no_gemma} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _plot_dense6(path: Path, family_rows: list[dict[str, Any]], effects: list[dict[str, Any]]) -> None:
    common_it = [row for row in family_rows if row["readout"] == "common_it"]
    common_it = sorted(common_it, key=lambda row: row["interaction"])
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
        "gemma3_4b": "Gemma3 4B",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.0, 1.35]})
    ax = axes[0]
    vals = [pt["dense6_estimate"], it["dense6_estimate"]]
    los = [pt["dense6_estimate"] - pt["dense6_ci95_low"], it["dense6_estimate"] - it["dense6_ci95_low"]]
    his = [pt["dense6_ci95_high"] - pt["dense6_estimate"], it["dense6_ci95_high"] - it["dense6_estimate"]]
    ax.bar([0, 1], vals, color=["#4c78a8", "#f58518"], width=0.62)
    ax.errorbar([0, 1], vals, yerr=[los, his], fmt="none", color="black", capsize=4, lw=1.1)
    ax.set_xticks([0, 1], ["PT upstream", "IT upstream"])
    ax.set_ylabel("IT-vs-PT margin shift (logits)")
    ax.set_title("IT late-stack effect")
    ax.axhline(0, color="#555", lw=0.8)
    ax.text(
        0.5,
        max(vals) * 0.88,
        f"interaction {interaction['dense6_estimate']:+.2f}\n95% CI "
        f"[{interaction['dense6_ci95_low']:+.2f}, {interaction['dense6_ci95_high']:+.2f}]",
        ha="center",
        va="center",
        fontsize=9,
    )

    ax = axes[1]
    y = list(range(len(common_it)))
    vals = [row["interaction"] for row in common_it]
    xerr = [
        [row["interaction"] - row["interaction_ci95_low"] for row in common_it],
        [row["interaction_ci95_high"] - row["interaction"] for row in common_it],
    ]
    ax.barh(y, vals, color="#54a24b", height=0.64)
    ax.errorbar(vals, y, xerr=xerr, fmt="none", color="black", capsize=3, lw=1.0)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_yticks(y, [labels.get(row["model"], row["model"]) for row in common_it])
    ax.set_xlabel("Upstream x late interaction (logits)")
    ax.set_title("Dense-6 family estimates")
    ax.grid(axis="x", color="#dddddd", lw=0.7)
    for spine in ("top", "right"):
        axes[0].spines[spine].set_visible(False)
        axes[1].spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


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
    positions = _position_table(args.dense5_position, args.qwen32_position)
    _write_csv(args.out_dir / "exp23_dense6_core_effects.csv", effects)
    _write_csv(args.out_dir / "exp23_dense6_family_effects.csv", families)
    _write_csv(args.out_dir / "exp23_dense6_position_sensitivity.csv", positions)
    _plot_dense6(args.out_dir / "exp23_dense6_interaction.png", families, effects)
    (args.out_dir / "exp23_dense6_core_summary.json").write_text(
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
    _write_markdown(args.out_dir / "exp23_dense6_core_summary.md", effects, positions)
    print(f"[dense6] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
