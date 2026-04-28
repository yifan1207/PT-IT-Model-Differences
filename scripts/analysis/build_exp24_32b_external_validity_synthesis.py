#!/usr/bin/env python3
"""Build paper-facing Exp24 32B external-validity synthesis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MODELS = ["qwen25_32b", "olmo2_32b"]
SUMMARY_COLUMNS = [
    "model",
    "n_first_divergence_records",
    "pt_upstream_late_effect",
    "pt_upstream_ci_low",
    "pt_upstream_ci_high",
    "it_upstream_late_effect",
    "it_upstream_ci_low",
    "it_upstream_ci_high",
    "interaction",
    "interaction_ci_low",
    "interaction_ci_high",
    "ratio_descriptive",
    "exp20_mid_identity_transfer",
    "exp20_late_identity_transfer",
    "exp21_late_writeout_support",
    "convergence_gap_raw_late_effect",
]


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Any, digits: int = 4) -> str:
    val = _finite(value)
    if val is None:
        return ""
    return f"{val:.{digits}f}"


def _mean(values: list[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not kept:
        return None
    return float(np.mean(kept))


def _exp23_effect(exp23: dict[str, Any], effect: str, readout: str = "common_it") -> dict[str, Any]:
    return (
        ((exp23.get("residual_factorial") or {}).get("effects") or {})
        .get(readout, {})
        .get(effect, {})
    )


def _model_exp23_value(exp23: dict[str, Any], effect: str, model: str) -> float | None:
    return _finite((_exp23_effect(exp23, effect).get("models") or {}).get(model))


def _model_exp20_identity(exp20: dict[str, Any], model: str, condition: str) -> float | None:
    payload = (
        (exp20.get("by_model") or {})
        .get(f"raw_shared/{model}", {})
        .get("conditions", {})
        .get(condition, {})
        .get("class_fractions", {})
        .get("it")
    )
    return _finite(payload)


def _model_exp21_late_support(exp21: dict[str, Any], model: str) -> float | None:
    for mode in ["native", "raw_shared"]:
        value = (
            (exp21.get("by_model") or {})
            .get(f"{mode}/{model}", {})
            .get("conditions", {})
            .get("first_diff", {})
            .get("C_it_chat", {})
            .get("windows", {})
            .get("late_reconciliation", {})
            .get("support_it_token")
        )
        finite = _finite(value)
        if finite is not None:
            return finite
    return None


def _model_bridge_effect(bridge: dict[str, Any], model: str, effect: str = "E_late_pt") -> float | None:
    for row in bridge.get("effects", []):
        if row.get("model") == model and row.get("effect") == effect:
            return _finite(row.get("mean"))
    return None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) < 1e-9:
        return None
    return float(numerator) / float(denominator)


def _model_row(
    *,
    model: str,
    exp23: dict[str, Any],
    exp20: dict[str, Any],
    exp21: dict[str, Any],
    bridge: dict[str, Any],
) -> dict[str, Any]:
    pt_effect = _model_exp23_value(exp23, "late_it_given_pt_upstream", model)
    it_effect = _model_exp23_value(exp23, "late_it_given_it_upstream", model)
    interaction = _model_exp23_value(exp23, "interaction", model)
    n_records = (
        ((exp23.get("residual_factorial") or {}).get("n_units_by_model") or {}).get(model)
    )
    return {
        "model": model,
        "n_first_divergence_records": n_records,
        "pt_upstream_late_effect": pt_effect,
        "pt_upstream_ci_low": None,
        "pt_upstream_ci_high": None,
        "it_upstream_late_effect": it_effect,
        "it_upstream_ci_low": None,
        "it_upstream_ci_high": None,
        "interaction": interaction,
        "interaction_ci_low": None,
        "interaction_ci_high": None,
        "ratio_descriptive": _ratio(it_effect, pt_effect),
        "exp20_mid_identity_transfer": _model_exp20_identity(exp20, model, "B_mid_raw"),
        "exp20_late_identity_transfer": _model_exp20_identity(exp20, model, "B_late_raw"),
        "exp21_late_writeout_support": _model_exp21_late_support(exp21, model),
        "convergence_gap_raw_late_effect": _model_bridge_effect(bridge, model),
    }


def _pooled_row(model_rows: list[dict[str, Any]], exp23: dict[str, Any]) -> dict[str, Any]:
    pt = _exp23_effect(exp23, "late_it_given_pt_upstream")
    it = _exp23_effect(exp23, "late_it_given_it_upstream")
    interaction = _exp23_effect(exp23, "interaction")
    pt_est = _finite(pt.get("estimate"))
    it_est = _finite(it.get("estimate"))
    n_records = sum(int(row.get("n_first_divergence_records") or 0) for row in model_rows)
    return {
        "model": "pooled_32b",
        "n_first_divergence_records": n_records or None,
        "pt_upstream_late_effect": pt_est,
        "pt_upstream_ci_low": _finite(pt.get("ci95_low")),
        "pt_upstream_ci_high": _finite(pt.get("ci95_high")),
        "it_upstream_late_effect": it_est,
        "it_upstream_ci_low": _finite(it.get("ci95_low")),
        "it_upstream_ci_high": _finite(it.get("ci95_high")),
        "interaction": _finite(interaction.get("estimate")),
        "interaction_ci_low": _finite(interaction.get("ci95_low")),
        "interaction_ci_high": _finite(interaction.get("ci95_high")),
        "ratio_descriptive": _ratio(it_est, pt_est),
        "exp20_mid_identity_transfer": _mean([row.get("exp20_mid_identity_transfer") for row in model_rows]),
        "exp20_late_identity_transfer": _mean([row.get("exp20_late_identity_transfer") for row in model_rows]),
        "exp21_late_writeout_support": _mean([row.get("exp21_late_writeout_support") for row in model_rows]),
        "convergence_gap_raw_late_effect": _mean([row.get("convergence_gap_raw_late_effect") for row in model_rows]),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in SUMMARY_COLUMNS})


def _write_md(rows: list[dict[str, Any]], path: Path, run_root: Path, warnings: list[str]) -> None:
    lines = [
        "# Exp24 32B External Validity",
        "",
        f"Run root: `{run_root}`",
        "",
        "| Model | N | PT-upstream late | IT-upstream late | Interaction | Ratio | Exp20 mid IT | Exp20 late IT | Exp21 late support | Raw KL late |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row.get("n_first_divergence_records") or ""),
                    _fmt(row.get("pt_upstream_late_effect")),
                    _fmt(row.get("it_upstream_late_effect")),
                    _fmt(row.get("interaction")),
                    _fmt(row.get("ratio_descriptive"), digits=2),
                    _fmt(row.get("exp20_mid_identity_transfer")),
                    _fmt(row.get("exp20_late_identity_transfer")),
                    _fmt(row.get("exp21_late_writeout_support")),
                    _fmt(row.get("convergence_gap_raw_late_effect")),
                ]
            )
            + " |"
        )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n")


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["model"]) for row in rows if row.get("interaction") is not None]
    values = [_finite(row.get("interaction")) for row in rows if row.get("interaction") is not None]
    if not values:
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        ax.text(0.5, 0.5, "No interaction values available yet", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        return
    lows = [_finite(row.get("interaction_ci_low")) for row in rows if row.get("interaction") is not None]
    highs = [_finite(row.get("interaction_ci_high")) for row in rows if row.get("interaction") is not None]
    yerr = [
        [0.0 if lo is None else max(0.0, val - lo) for val, lo in zip(values, lows, strict=False)],
        [0.0 if hi is None else max(0.0, hi - val) for val, hi in zip(values, highs, strict=False)],
    ]
    x = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(x, values, color=["#4C78A8" if label != "pooled_32b" else "#54A24B" for label in labels])
    ax.errorbar(x, values, yerr=yerr, fmt="none", color="black", capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("Interaction, common IT readout")
    ax.set_title("Exp24 32B residual-state x late-stack interaction")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _paper_sentence_if_success(models: list[str]) -> str:
    model_set = set(models)
    if model_set == {"qwen25_32b", "olmo2_32b"}:
        return (
            "A two-family 32B replication on Qwen2.5-32B and OLMo-2-32B preserves "
            "the positive upstream-state x late-stack interaction, extending the "
            "context-gated late-readout result beyond the original 4B-8B dense-family pool."
        )
    if model_set == {"qwen25_32b"}:
        return (
            "A Qwen2.5-32B external-validity run preserves the positive upstream-state "
            "x late-stack interaction, providing a 32B-scale check of the "
            "context-gated late-readout result."
        )
    if model_set == {"olmo2_32b"}:
        return (
            "An OLMo-2-32B external-validity run preserves the positive upstream-state "
            "x late-stack interaction, providing a 32B-scale check of the "
            "context-gated late-readout result."
        )
    model_list = ", ".join(models) if models else "the requested models"
    return (
        f"The 32B external-validity run over {model_list} preserves the positive "
        "upstream-state x late-stack interaction, providing a scale check of the "
        "context-gated late-readout result."
    )


def build(run_root: Path, out_dir: Path, models: list[str]) -> dict[str, Any]:
    analysis_root = run_root / "analysis"
    exp23 = _load(analysis_root / "exp23_midlate_interaction_suite" / "exp23_summary.json")
    exp20 = _load(analysis_root / "exp20_factorial_validation" / "summary.json")
    exp21 = _load(analysis_root / "exp21_productive_opposition" / "summary.json")
    bridge = _load(analysis_root / "part_a_mlp_kl" / "exp23_midlate_kl_factorial_summary.json")

    warnings: list[str] = []
    for label, payload in [
        ("exp23", exp23),
        ("exp20", exp20),
        ("exp21", exp21),
        ("bridge", bridge),
    ]:
        if not payload:
            warnings.append(f"Missing {label} analysis summary under {analysis_root}.")

    model_rows = [
        _model_row(model=model, exp23=exp23, exp20=exp20, exp21=exp21, bridge=bridge)
        for model in models
    ]
    rows = model_rows + [_pooled_row(model_rows, exp23)]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_dir / "exp24_32b_summary.csv")
    _write_md(rows, out_dir / "exp24_32b_summary.md", run_root, warnings)
    _plot(rows, out_dir / "exp24_32b_interaction.png")

    interactions = [row.get("interaction") for row in model_rows]
    pooled = rows[-1]
    all_positive = all(_finite(value) is not None and float(value) > 0 for value in interactions)
    pooled_ci_low = _finite(pooled.get("interaction_ci_low"))
    claims = {
        "run_root": str(run_root),
        "models": models,
        "all_model_interactions_positive": all_positive if interactions else False,
        "pooled_interaction_ci_excludes_zero": pooled_ci_low is not None and pooled_ci_low > 0,
        "pooled_interaction": pooled.get("interaction"),
        "pooled_interaction_ci_low": pooled.get("interaction_ci_low"),
        "pooled_interaction_ci_high": pooled.get("interaction_ci_high"),
        "warnings": warnings,
        "paper_sentence_if_success": _paper_sentence_if_success(models),
    }
    (out_dir / "exp24_32b_claims.json").write_text(json.dumps(claims, indent=2, sort_keys=True))
    return {"rows": rows, "claims": claims, "out_dir": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_synthesis/exp24_32b_external_validity"))
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()
    payload = build(args.run_root, args.out_dir, list(args.models))
    print(json.dumps({"out_dir": payload["out_dir"], "claims": payload["claims"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
