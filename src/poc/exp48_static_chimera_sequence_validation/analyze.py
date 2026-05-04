"""Analyze Exp48 static chimera and structured-rescue outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.poc.exp48_static_chimera_sequence_validation.config import DEFAULT_MODELS


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[Any]) -> float | None:
    vals = [x for x in (_finite(v) for v in values) if x is not None]
    return float(np.mean(vals)) if vals else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _bootstrap_prompt_mean(rows: list[dict[str, Any]], key: str, *, n_boot: int, seed: int) -> dict[str, Any]:
    by_prompt: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = _finite(row.get(key))
        if val is not None:
            by_prompt[str(row.get("prompt_id"))].append(val)
    vals = np.asarray([np.mean(v) for v in by_prompt.values() if v], dtype=float)
    if vals.size == 0:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "n_prompt_clusters": 0, "n_units": 0}
    estimate = float(vals.mean())
    if n_boot <= 0 or vals.size == 1:
        return {
            "estimate": estimate,
            "ci95_low": estimate,
            "ci95_high": estimate,
            "n_prompt_clusters": int(vals.size),
            "n_units": sum(len(v) for v in by_prompt.values()),
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot = vals[idx].mean(axis=1)
    return {
        "estimate": estimate,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "n_prompt_clusters": int(vals.size),
        "n_units": sum(len(v) for v in by_prompt.values()),
    }


def load_sequence_scores(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "sequence").glob("**/sequence_scores.jsonl.gz")):
        rows.extend(_json_rows(path))
    return rows


def load_rescue_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "rescue_scores").glob("**/rescue_records.jsonl.gz")):
        rows.extend(_json_rows(path))
    return rows


def summarize_sequence(rows: list[dict[str, Any]], *, n_boot: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    matrix: list[dict[str, Any]] = []
    recovery: list[dict[str, Any]] = []
    health: list[dict[str, Any]] = []
    group_fields = ("model", "boundary", "scenario", "component", "interpolation_alpha", "task")
    groups = sorted({tuple(row.get(field) for field in group_fields) for row in rows})
    for group in groups:
        subset = [row for row in rows if tuple(row.get(field) for field in group_fields) == group]
        if not subset:
            continue
        base = dict(zip(group_fields, group, strict=False))
        for cell in ("BB", "BF", "FB", "FF"):
            cell_rows = [row for row in subset if row.get("cell") == cell]
            if not cell_rows:
                continue
            stat = _bootstrap_prompt_mean(cell_rows, "score", n_boot=n_boot, seed=abs(hash(("seq", group, cell))) % (2**32))
            matrix.append({**base, "cell": cell, **stat})
            health.append(
                {
                    **base,
                    "cell": cell,
                    "invalid_output_rate": _mean([row.get("invalid_output") for row in cell_rows]),
                    "loop_rate": _mean([row.get("has_long_loop") for row in cell_rows]),
                    "eos_rate": _mean([row.get("eos_emitted") for row in cell_rows]),
                    "mean_generated_tokens": _mean([row.get("generated_tokens_count") for row in cell_rows]),
                    "mean_step_entropy": _mean([row.get("mean_step_entropy") for row in cell_rows]),
                    "mean_prompt_nll": _mean([row.get("prompt_nll") for row in cell_rows]),
                    "mean_kl_to_base_last": _mean([row.get("kl_to_base_last") for row in cell_rows]),
                    "mean_kl_to_ft_last": _mean([row.get("kl_to_ft_last") for row in cell_rows]),
                    "mean_boundary_rms": _mean([row.get("boundary_last_rms") for row in cell_rows]),
                    "n_rows": len(cell_rows),
                }
            )
        by_prompt: dict[str, dict[str, float]] = defaultdict(dict)
        for row in subset:
            val = _finite(row.get("score"))
            if val is not None:
                by_prompt[str(row.get("prompt_id"))][str(row.get("cell"))] = val
        contrast_rows: list[dict[str, Any]] = []
        for prompt_id, cells in by_prompt.items():
            if not {"BB", "BF", "FF"}.issubset(cells):
                continue
            denom = cells["FF"] - cells["BB"]
            if abs(denom) < 1e-8:
                continue
            contrast_rows.append(
                {
                    "prompt_id": prompt_id,
                    "portable_share": (cells["BF"] - cells["BB"]) / denom,
                    "coadapted_share": (cells["FF"] - cells["BF"]) / denom,
                    "native_gain": denom,
                    "late_only_gain": cells["BF"] - cells["BB"],
                    "ff_minus_bf": cells["FF"] - cells["BF"],
                }
            )
        for metric in ("portable_share", "coadapted_share", "native_gain", "late_only_gain", "ff_minus_bf"):
            stat = _bootstrap_prompt_mean(contrast_rows, metric, n_boot=n_boot, seed=abs(hash(("rec", group, metric))) % (2**32))
            recovery.append({**base, "metric": metric, **stat})
    return matrix, recovery, health


def summarize_rescue(rows: list[dict[str, Any]], *, n_boot: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    group_fields = ("model", "boundary", "condition", "k", "alpha")
    groups = sorted({tuple(row.get(field) for field in group_fields) for row in rows})
    for group in groups:
        subset = [row for row in rows if tuple(row.get(field) for field in group_fields) == group]
        if not subset:
            continue
        base = dict(zip(group_fields, group, strict=False))
        for metric in ("margin_it_minus_pt", "closure_fraction", "native_gap"):
            stat = _bootstrap_prompt_mean(subset, metric, n_boot=n_boot, seed=abs(hash(("rescue", group, metric))) % (2**32))
            out.append({**base, "metric": metric, **stat})
    return out


def token_to_sequence_link(run_root: Path, sequence_recovery: list[dict[str, Any]], exp47_analysis_dir: Path | None) -> list[dict[str, Any]]:
    if exp47_analysis_dir is None:
        return []
    path = exp47_analysis_dir / "portable_coadapted_table.csv"
    if not path.exists():
        return []
    token_rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("scope") == "alias" and row.get("readout") == "common_it" and row.get("slice") in {"full_1400", "instruction_format", "math_domain"}:
                token_rows.append(row)
    out = []
    for seq in sequence_recovery:
        if seq.get("metric") != "portable_share":
            continue
        for tok in token_rows:
            if tok.get("model") != seq.get("model"):
                continue
            out.append(
                {
                    "model": seq.get("model"),
                    "task": seq.get("task"),
                    "boundary": seq.get("boundary"),
                    "scenario": seq.get("scenario"),
                    "component": seq.get("component"),
                    "exp47_slice": tok.get("slice"),
                    "exp47_token_portable_share_P_over_M": tok.get("portable_share_P_over_M"),
                    "exp47_token_interaction_over_native_shift": tok.get("interaction_over_native_shift"),
                    "exp48_sequence_portable_share": seq.get("estimate"),
                    "exp48_ci95_low": seq.get("ci95_low"),
                    "exp48_ci95_high": seq.get("ci95_high"),
                }
            )
    return out


def _plot_sequence(recovery: list[dict[str, Any]], health: list[dict[str, Any]], out_dir: Path) -> None:
    rows = [
        row for row in recovery
        if row.get("metric") == "portable_share"
        and row.get("scenario") == "boundary_sweep"
        and row.get("component") == "blocks_plus_head"
        and row.get("task") in {"ifeval_format", "gsm8k_exact", "governance_structure_proxy", "multiple_choice_exact"}
    ]
    if rows:
        models = [m for m in DEFAULT_MODELS if any(row.get("model") == m for row in rows)]
        boundaries = sorted({int(row.get("boundary")) for row in rows if row.get("boundary") not in {None, ""}})
        fig, ax = plt.subplots(figsize=(10, 5))
        for model in models:
            vals = []
            xs = []
            for b in boundaries:
                vals_here = [
                    _finite(row.get("estimate")) for row in rows
                    if row.get("model") == model and int(row.get("boundary")) == b
                ]
                vals_here = [v for v in vals_here if v is not None]
                if vals_here:
                    xs.append(b)
                    vals.append(float(np.mean(vals_here)))
            if vals:
                ax.plot(xs, vals, marker="o", label=model.replace("llama31_", ""))
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.axhline(1, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Static late-stack boundary")
        ax.set_ylabel("Behavioral portable share")
        ax.set_title("Exp48 Static Chimera Boundary Sweep")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "static_chimera_boundary_sweep.png", dpi=180)
        plt.close(fig)

    health_rows = [row for row in health if row.get("scenario") == "boundary_sweep" and row.get("component") == "blocks_plus_head"]
    if health_rows:
        labels = []
        invalid = []
        loops = []
        for model in DEFAULT_MODELS:
            subset = [row for row in health_rows if row.get("model") == model]
            if not subset:
                continue
            labels.append(model.replace("llama31_", ""))
            invalid.append(_mean([row.get("invalid_output_rate") for row in subset]) or 0.0)
            loops.append(_mean([row.get("loop_rate") for row in subset]) or 0.0)
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(labels))
        ax.bar(x - 0.18, invalid, width=0.36, label="invalid")
        ax.bar(x + 0.18, loops, width=0.36, label="loop")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel("Rate")
        ax.set_title("Chimera Validity Dashboard")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "chimera_validity_dashboard.png", dpi=180)
        plt.close(fig)


def _plot_rescue(rescue_summary: list[dict[str, Any]], out_dir: Path) -> None:
    rows = [
        row for row in rescue_summary
        if row.get("metric") == "closure_fraction" and row.get("condition") == "paired_pca" and row.get("alpha") in {"1.0", 1.0}
    ]
    if not rows:
        return
    models = [m for m in DEFAULT_MODELS if any(row.get("model") == m for row in rows)]
    fig, axes = plt.subplots(1, max(1, len(models)), figsize=(4 * max(1, len(models)), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models, strict=False):
        subset = [row for row in rows if row.get("model") == model]
        boundaries = sorted({int(row.get("boundary")) for row in subset if row.get("boundary") not in {None, ""}})
        for boundary in boundaries:
            b_rows = [row for row in subset if int(row.get("boundary")) == boundary]
            xs = []
            ys = []
            for k in ("1", "4", "16", "64", "256", "full"):
                val = next((_finite(row.get("estimate")) for row in b_rows if str(row.get("k")) == k), None)
                if val is not None:
                    xs.append(k)
                    ys.append(val)
            if ys:
                ax.plot(xs, ys, marker="o", label=f"b{boundary}")
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.axhline(1, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_title(model.replace("llama31_", ""))
        ax.set_xlabel("PCA rank k")
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Closure fraction")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "structured_rescue_closure_curves.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    analysis_dir = args.run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    sequence_rows = load_sequence_scores(args.run_root)
    rescue_rows = load_rescue_rows(args.run_root)
    sequence_matrix, sequence_recovery, health = summarize_sequence(sequence_rows, n_boot=args.n_boot)
    rescue_summary = summarize_rescue(rescue_rows, n_boot=args.n_boot)
    link = token_to_sequence_link(args.run_root, sequence_recovery, args.exp47_analysis_dir)

    _write_csv(analysis_dir / "sequence_four_cell_matrix.csv", sequence_matrix)
    _write_csv(analysis_dir / "sequence_recovery_summary.csv", sequence_recovery)
    _write_csv(analysis_dir / "chimera_validity_dashboard.csv", health)
    _write_csv(analysis_dir / "structured_rescue_summary.csv", rescue_summary)
    _write_csv(analysis_dir / "token_to_sequence_link.csv", link)
    _plot_sequence(sequence_recovery, health, analysis_dir)
    _plot_rescue(rescue_summary, analysis_dir)

    summary = {
        "experiment": "exp48_static_chimera_sequence_validation",
        "run_root": str(args.run_root),
        "n_sequence_score_rows": len(sequence_rows),
        "n_rescue_rows": len(rescue_rows),
        "n_sequence_recovery_rows": len(sequence_recovery),
        "n_rescue_summary_rows": len(rescue_summary),
        "analysis_dir": str(analysis_dir),
    }
    _write_json(analysis_dir / "summary.json", summary)
    report = [
        "# Exp48 Static Chimera Sequence Validation",
        "",
        f"- Run root: `{args.run_root}`",
        f"- Sequence score rows: `{len(sequence_rows)}`",
        f"- Rescue rows: `{len(rescue_rows)}`",
        f"- Recovery summary rows: `{len(sequence_recovery)}`",
        f"- Rescue summary rows: `{len(rescue_summary)}`",
    ]
    (analysis_dir / "exp48_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"ok": True, **summary}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--exp47-analysis-dir", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
