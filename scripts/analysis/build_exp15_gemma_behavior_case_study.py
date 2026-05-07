#!/usr/bin/env python3
"""Build the Gemma behavioral case-study artifacts for the convergence paper."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]
SOURCE = (
    "results/exp15_symmetric_behavioral_causality/plots/"
    "exp15_eval_core_600_t512_dense5/exp15_behavior_summary.json"
)
OUT_DIR = Path("results/paper_synthesis")
MODEL = "gemma3_4b"


def load_summary(repo: Path) -> dict[str, Any]:
    with (repo / SOURCE).open() as f:
        return json.load(f)


def build_payload(summary: dict[str, Any]) -> dict[str, Any]:
    model = summary["models"][MODEL]
    pairwise = model["pairwise"]["it_c_vs_dlate"]
    pointwise = model["pointwise"]
    programmatic = model["programmatic"]
    deltas = model["paired_pointwise_deltas"]["it"]["D_late_ptswap"]

    payload = {
        "source_artifact": SOURCE,
        "selection_rule": (
            "Gemma is used as an illustrative case study because it has the largest "
            "bidirectional late-window convergence intervention effects in the dense-family "
            "late-window table."
        ),
        "model": MODEL,
        "n_prompt_records": model["judge_manifest"]["n_pointwise_items"] // len(pointwise),
        "conditions": {
            "baseline": "C_it_chat",
            "early_control": "D_early_ptswap",
            "mid_control": "D_mid_ptswap",
            "late_swap": "D_late_ptswap",
        },
        "assistant_register_pointwise": {
            condition: pointwise[condition]["g2"]
            for condition in ["C_it_chat", "D_early_ptswap", "D_mid_ptswap", "D_late_ptswap"]
        },
        "assistant_register_late_delta": deltas["g2"],
        "pairwise_native_vs_late_swap": {
            "assistant_register_g2": pairwise["pairwise_g2"],
            "safety_format_s2": pairwise["pairwise_s2"],
        },
        "programmatic_checks": {
            "mmlu_forced_choice": {
                condition: programmatic[condition]["mmlu_forced_choice"]
                for condition in ["C_it_chat", "D_late_ptswap"]
            },
            "reasoning_em": {
                condition: programmatic[condition]["exp3_reasoning_em"]
                for condition in ["C_it_chat", "D_late_ptswap"]
            },
            "format_compliance_v2": {
                condition: programmatic[condition]["format_compliance_v2"]
                for condition in ["C_it_chat", "D_late_ptswap"]
            },
        },
    }
    return payload


def write_csv(payload: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, str]] = []

    for condition, values in payload["assistant_register_pointwise"].items():
        rows.append(
            {
                "key": f"gemma_g2_pointwise_{condition}",
                "group": "pointwise",
                "condition": condition,
                "metric": "assistant_register_g2_mean",
                "estimate": str(values["mean"]),
                "ci95_low": str(values["ci95"][0]),
                "ci95_high": str(values["ci95"][1]),
                "n": str(values["n"]),
            }
        )

    delta = payload["assistant_register_late_delta"]
    rows.append(
        {
            "key": "gemma_g2_late_swap_drop",
            "group": "paired_delta",
            "condition": "C_it_chat_minus_D_late_ptswap",
            "metric": "assistant_register_g2_mean_drop",
            "estimate": str(delta["mean"]),
            "ci95_low": str(delta["ci95"][0]),
            "ci95_high": str(delta["ci95"][1]),
            "n": str(delta["n"]),
        }
    )

    for label, values in payload["pairwise_native_vs_late_swap"].items():
        rows.append(
            {
                "key": f"gemma_pairwise_{label}_native_win",
                "group": "pairwise",
                "condition": "C_it_chat_vs_D_late_ptswap",
                "metric": f"{label}_native_win_rate",
                "estimate": str(values["target_win_rate"]),
                "ci95_low": str(values["target_win_ci95"][0]),
                "ci95_high": str(values["target_win_ci95"][1]),
                "n": str(values["n"]),
            }
        )
        rows.append(
            {
                "key": f"gemma_pairwise_{label}_late_swap_win",
                "group": "pairwise",
                "condition": "C_it_chat_vs_D_late_ptswap",
                "metric": f"{label}_late_swap_win_rate",
                "estimate": str(values["other_win_rate"]),
                "ci95_low": str(values["other_win_ci95"][0]),
                "ci95_high": str(values["other_win_ci95"][1]),
                "n": str(values["n"]),
            }
        )

    for metric, values_by_condition in payload["programmatic_checks"].items():
        for condition, values in values_by_condition.items():
            rows.append(
                {
                    "key": f"gemma_programmatic_{metric}_{condition}",
                    "group": "programmatic",
                    "condition": condition,
                    "metric": metric,
                    "estimate": str(values["value"]),
                    "ci95_low": "",
                    "ci95_high": "",
                    "n": str(values["n"]),
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["key", "group", "condition", "metric", "estimate", "ci95_low", "ci95_high", "n"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_note(payload: dict[str, Any], path: Path) -> None:
    g2 = payload["pairwise_native_vs_late_swap"]["assistant_register_g2"]
    s2 = payload["pairwise_native_vs_late_swap"]["safety_format_s2"]
    delta = payload["assistant_register_late_delta"]
    text = f"""# Exp15 Gemma Behavioral Case Study

Selection rule: {payload['selection_rule']}

Primary behavioral readout: compare native Gemma IT (`C_it_chat`) to the same
IT host with a late PT MLP swap (`D_late_ptswap`) under free-running 512-token
generation over the Exp15 600-prompt support.

- Pairwise assistant-register (`G2`) preference for native IT: `{g2['target_win_rate']:.3f}`
  with CI `[{g2['target_win_ci95'][0]:.3f}, {g2['target_win_ci95'][1]:.3f}]`.
- Pairwise safety/format (`S2`) preference for native IT: `{s2['target_win_rate']:.3f}`
  with CI `[{s2['target_win_ci95'][0]:.3f}, {s2['target_win_ci95'][1]:.3f}]`.
- Paired pointwise assistant-register drop under the late PT swap:
  `{delta['mean']:.3f}` with CI `[{delta['ci95'][0]:.3f}, {delta['ci95'][1]:.3f}]`.

This is an illustrative behavioral case study, not a cross-family behavioral
claim. It uses the same matched-prefix intervention family as the convergence
paper's MLP leverage experiments.
"""
    path.write_text(text)


def write_plot(payload: dict[str, Any], path: Path) -> None:
    pointwise = payload["assistant_register_pointwise"]
    pairwise = payload["pairwise_native_vs_late_swap"]

    conditions = ["C_it_chat", "D_early_ptswap", "D_mid_ptswap", "D_late_ptswap"]
    labels = ["IT native", "early PT swap", "mid PT swap", "late PT swap"]
    means = [pointwise[c]["mean"] for c in conditions]
    lows = [pointwise[c]["ci95"][0] for c in conditions]
    highs = [pointwise[c]["ci95"][1] for c in conditions]
    yerr = [[m - l for m, l in zip(means, lows)], [h - m for m, h in zip(means, highs)]]

    pair_labels = ["assistant-register", "safety/format"]
    pair_means = [
        pairwise["assistant_register_g2"]["target_win_rate"],
        pairwise["safety_format_s2"]["target_win_rate"],
    ]
    pair_lows = [
        pairwise["assistant_register_g2"]["target_win_ci95"][0],
        pairwise["safety_format_s2"]["target_win_ci95"][0],
    ]
    pair_highs = [
        pairwise["assistant_register_g2"]["target_win_ci95"][1],
        pairwise["safety_format_s2"]["target_win_ci95"][1],
    ]
    pair_yerr = [[m - l for m, l in zip(pair_means, pair_lows)], [h - m for m, h in zip(pair_means, pair_highs)]]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), dpi=220)
    colors = ["#4f79a7", "#7aa974", "#f0a23a", "#c85d5a"]
    axes[0].bar(labels, means, yerr=yerr, capsize=3, color=colors, edgecolor="none")
    axes[0].set_ylabel("Assistant-register score")
    axes[0].set_ylim(3.5, 4.9)
    axes[0].set_title("Gemma free-running score")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", color="0.88", linewidth=0.8)

    axes[1].bar(pair_labels, pair_means, yerr=pair_yerr, capsize=3, color=["#4f79a7", "#4f79a7"], edgecolor="none")
    axes[1].axhline(0.5, color="0.35", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Native IT preferred over late PT swap")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_title("Blind pairwise preference")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].grid(axis="y", color="0.88", linewidth=0.8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Exp15 Gemma behavioral case study", y=1.04, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def build(repo: Path, out_dir: Path) -> list[Path]:
    payload = build_payload(load_summary(repo))
    out_root = repo / out_dir
    json_path = out_root / "exp15_gemma_behavior_case_study.json"
    csv_path = out_root / "exp15_gemma_behavior_case_study.csv"
    note_path = out_root / "exp15_gemma_behavior_case_study_note.md"
    plot_path = out_root / "exp15_gemma_behavior_case_study.png"
    out_root.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(payload, csv_path)
    write_note(payload, note_path)
    write_plot(payload, plot_path)
    return [json_path, csv_path, note_path, plot_path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    for path in build(args.repo.resolve(), args.out_dir):
        print(path.relative_to(args.repo.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
