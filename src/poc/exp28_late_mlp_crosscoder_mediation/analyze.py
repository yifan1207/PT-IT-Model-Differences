"""Analyze Exp28 crosscoder mediation records."""

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


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def interaction_from_cells(cells: dict[str, float]) -> float:
    return (cells["U_IT__L_IT"] - cells["U_IT__L_PT"]) - (cells["U_PT__L_IT"] - cells["U_PT__L_PT"])


def _finite(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _mean(values: list[Any]) -> float | None:
    finite = _finite(values)
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _bootstrap_ci(values: list[float], *, n_boot: int, seed: int) -> tuple[float | None, float | None]:
    finite = _finite(values)
    if len(finite) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    arr = np.asarray(finite, dtype=np.float64)
    means = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[idx] = float(sample.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_feature_rows(path: Path, *, limit: int = 100) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    records_path = args.run_root / "mediation" / "records.jsonl.gz"
    if not records_path.exists():
        worker_records = sorted((args.run_root / "mediation").glob("records_w*.jsonl.gz"))
        if worker_records:
            records_path = worker_records[0]
    rows = list(_json_rows(records_path))
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seed = "none" if row.get("control_seed") is None else str(row.get("control_seed"))
        grouped[(str(row.get("feature_set")), int(row.get("k", 0)), seed)].append(row)

    effect_rows: list[dict[str, Any]] = []
    for (feature_set, k, seed), group in sorted(grouped.items()):
        drops = _finite([r.get("interaction_drop") for r in group])
        fracs = _finite([r.get("mediation_fraction") for r in group])
        ablated = _finite([r.get("interaction_ablate") for r in group])
        full = _finite([r.get("interaction_full") for r in group])
        drop_lo, drop_hi = _bootstrap_ci(drops, n_boot=args.n_boot, seed=args.seed)
        frac_lo, frac_hi = _bootstrap_ci(fracs, n_boot=args.n_boot, seed=args.seed + 1)
        pos3 = [r for r in group if r.get("position_ge_3")]
        effect_rows.append(
            {
                "feature_set": feature_set,
                "k": k,
                "control_seed": seed,
                "n": len(group),
                "interaction_full_mean": _mean(full),
                "interaction_ablate_mean": _mean(ablated),
                "interaction_drop_mean": _mean(drops),
                "interaction_drop_ci_lo": drop_lo,
                "interaction_drop_ci_hi": drop_hi,
                "mediation_fraction_mean": _mean(fracs),
                "mediation_fraction_ci_lo": frac_lo,
                "mediation_fraction_ci_hi": frac_hi,
                "position_ge_3_n": len(pos3),
                "position_ge_3_mediation_fraction_mean": _mean([r.get("mediation_fraction") for r in pos3]),
                "mean_delta_norm_frac_it_late": _mean(
                    [
                        ((r.get("hook_summary_it_late") or {}).get("mean_delta_norm_frac"))
                        for r in group
                    ]
                ),
                "mean_active_l0_it_late": _mean(
                    [
                        ((r.get("hook_summary_it_late") or {}).get("mean_active_l0"))
                        for r in group
                    ]
                ),
                "mean_removed_mass_frac_it_late": _mean(
                    [
                        ((r.get("hook_summary_it_late") or {}).get("mean_removed_mass_frac"))
                        for r in group
                    ]
                ),
            }
        )
    out_dir = args.run_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "effects.csv", effect_rows)

    summary = {
        "records_path": str(records_path),
        "n_records": len(rows),
        "n_groups": len(effect_rows),
        "success_gates": _success_gates(effect_rows),
        "effects": effect_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _plot_curve(out_dir / "mediation_curve.png", effect_rows)
    _write_packet(args.run_root, out_dir, summary)
    return summary


def _success_gates(effect_rows: list[dict[str, Any]]) -> dict[str, Any]:
    top_200 = [
        r for r in effect_rows
        if r["feature_set"] == "top_interaction" and int(r["k"]) == 200 and r["control_seed"] == "none"
    ]
    random_200 = [
        r for r in effect_rows
        if r["feature_set"] == "matched_random" and int(r["k"]) == 200
    ]
    recon = [r for r in effect_rows if r["feature_set"] == "full_reconstruction"]
    top_frac = top_200[0].get("mediation_fraction_mean") if top_200 else None
    rand_frac = _mean([r.get("mediation_fraction_mean") for r in random_200])
    recon_frac = recon[0].get("mediation_fraction_mean") if recon else None
    return {
        "top200_mediation_fraction": top_frac,
        "matched_random200_mediation_fraction_mean": rand_frac,
        "full_reconstruction_mediation_fraction": recon_frac,
        "coverage_norm90_mediation_fraction": _coverage_frac(effect_rows, "coverage_norm", 90),
        "coverage_margin_pos90_mediation_fraction": _coverage_frac(effect_rows, "coverage_margin_pos", 90),
        "strong_result": bool(
            top_frac is not None
            and top_frac >= 0.40
            and (rand_frac is None or rand_frac <= 0.10)
            and (recon_frac is None or abs(recon_frac) <= 0.20)
        ),
        "moderate_result": bool(
            top_frac is not None
            and 0.15 <= top_frac < 0.40
            and (rand_frac is None or rand_frac < top_frac)
        ),
    }


def _coverage_frac(effect_rows: list[dict[str, Any]], feature_set: str, k: int) -> float | None:
    rows = [
        r for r in effect_rows
        if r["feature_set"] == feature_set and int(r["k"]) == k and r["control_seed"] == "none"
    ]
    if not rows:
        return None
    return rows[0].get("mediation_fraction_mean")


def _plot_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    styles = {
        "top_interaction": ("#2364aa", "o"),
        "matched_random": ("#999999", "s"),
        "shared": ("#55a868", "^"),
        "pt_biased": ("#c44e52", "v"),
        "shuffle_layers": ("#8172b3", "D"),
        "coverage_norm": ("#dd8452", "P"),
        "coverage_margin_pos": ("#4c72b0", "X"),
        "coverage_margin_abs": ("#64b5cd", "*"),
        "coverage_activation": ("#ccb974", "h"),
    }
    plotted = []
    for feature_set, (color, marker) in styles.items():
        subset = [r for r in rows if r["feature_set"] == feature_set]
        if not subset:
            continue
        by_k: dict[int, list[float]] = defaultdict(list)
        for row in subset:
            val = row.get("mediation_fraction_mean")
            if val is not None and math.isfinite(float(val)):
                by_k[int(row["k"])].append(float(val))
        xs = sorted(by_k)
        ys = [float(np.mean(by_k[x])) for x in xs]
        if xs:
            plotted.append((xs, ys, marker, color, feature_set))
    for ax in axes:
        for xs, ys, marker, color, feature_set in plotted:
            ax.plot(xs, ys, marker=marker, color=color, label=feature_set)
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.set_xscale("symlog", linthresh=50)
        ax.set_xlabel("Global K features / coverage percent")
    axes[0].set_ylabel("Mediation fraction")
    axes[0].set_title("Full range")
    axes[1].set_title("Zoomed main effects")
    axes[1].set_ylim(-0.25, 1.35)
    fig.suptitle("Exp28 Llama Late-MLP Crosscoder Mediation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25, top=0.84)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_packet(run_root: Path, out_dir: Path, summary: dict[str, Any]) -> None:
    top_features = _load_feature_rows(run_root / "feature_stats" / "top_interaction_candidates.csv", limit=50)
    lines = [
        "# Exp28 Crosscoder Feature Packet",
        "",
        f"Records: `{summary['n_records']}`",
        "",
        "## Success Gates",
        "",
        "```json",
        json.dumps(summary["success_gates"], indent=2),
        "```",
        "",
        "## Top Interaction Features",
        "",
        "| layer | latent | score | type | IT/PT act ratio | scaling ratio | local attr |",
        "|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in top_features:
        lines.append(
            "| {layer} | {latent_id} | {interaction_score:.4g} | {feature_type} | "
            "{activation_ratio_it_pt:.3g} | {latent_scaling_ratio:.3g} | {local_margin_attr:.3g} |".format(
                layer=row["layer"],
                latent_id=row["latent_id"],
                interaction_score=float(row["interaction_score"]),
                feature_type=row["feature_type"],
                activation_ratio_it_pt=float(row["activation_ratio_it_pt"]),
                latent_scaling_ratio=float(row["latent_scaling_ratio"]),
                local_margin_attr=float(row["local_margin_attr"]),
            )
        )
    (out_dir / "top_features_packet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
