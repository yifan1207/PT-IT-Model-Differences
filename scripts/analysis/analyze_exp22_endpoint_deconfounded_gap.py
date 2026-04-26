#!/usr/bin/env python
"""Analyze Exp22 endpoint-deconfounded convergence-gap records."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.poc.exp22_endpoint_deconfounded_gap.metrics import as_float, derived_step_metrics

COVARIATES = ["final_entropy", "final_confidence", "final_top1_margin"]
OUTCOMES = [
    "late_kl_mean",
    "prefinal_late_kl_mean",
    "remaining_adj_js",
    "remaining_adj_kl",
    "future_top1_flips",
    "top5_churn",
    "final_top1_stable_top5_entry",
    "late_consensus_stable_top5_entry",
]
PRIMARY_MODELS = ["gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b"]


def _open_jsonl(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def _record_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    paths = sorted(root.glob("**/records.jsonl.gz"))
    paths.extend(sorted(root.glob("**/records.jsonl")))
    return paths


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def flatten_records(root: Path, skip_first_n: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    branch_counts: dict[str, dict[str, int]] = {}
    malformed_examples: list[dict[str, Any]] = []
    paths = _record_paths(root)
    for path in paths:
        with _open_jsonl(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    malformed_examples.append({"path": str(path), "error": repr(exc)})
                    continue
                model = str(rec.get("model", "unknown"))
                variant = str(rec.get("variant", "unknown"))
                branch = f"{model}/{variant}"
                branch_counts.setdefault(branch, {"records": 0, "malformed": 0, "valid": 0})
                branch_counts[branch]["records"] += 1
                if rec.get("malformed") or "probes" not in rec:
                    branch_counts[branch]["malformed"] += 1
                    if len(malformed_examples) < 20:
                        malformed_examples.append(
                            {
                                "path": str(path),
                                "prompt_id": rec.get("prompt_id"),
                                "model": model,
                                "variant": variant,
                                "error": rec.get("error"),
                            }
                        )
                    continue
                branch_counts[branch]["valid"] += 1
                prompt_id = str(rec.get("prompt_id"))
                n_steps = int(rec.get("n_steps", 0))
                generated_ids = rec.get("generated_ids", [])
                for probe_family, payload in rec.get("probes", {}).items():
                    step_count = min(n_steps, len(payload.get("kl_to_final", [])))
                    for step_idx in range(skip_first_n, step_count):
                        try:
                            metrics = derived_step_metrics(payload, step_idx)
                        except Exception:
                            continue
                        if not all(_finite(metrics[cov]) for cov in COVARIATES):
                            continue
                        if not _finite(metrics["late_kl_mean"]):
                            continue
                        row = {
                            "prompt_id": prompt_id,
                            "model": model,
                            "variant": variant,
                            "variant_it": 1 if variant == "it" else 0,
                            "prompt_mode": rec.get("prompt_mode"),
                            "probe_family": str(probe_family),
                            "step_idx": step_idx,
                            "generated_id": int(generated_ids[step_idx]) if step_idx < len(generated_ids) else None,
                        }
                        row.update(metrics)
                        rows.append(row)
    summary = {
        "paths": [str(path) for path in paths],
        "branch_counts": branch_counts,
        "malformed_examples": malformed_examples,
        "n_rows": len(rows),
    }
    return pd.DataFrame(rows), summary


def _decile_bins(series: pd.Series, n_bins: int = 10) -> pd.Series:
    valid = series.astype(float)
    if valid.nunique(dropna=True) <= 1:
        return pd.Series(0, index=series.index, dtype="int64")
    try:
        labels = pd.qcut(valid, q=min(n_bins, valid.nunique(dropna=True)), labels=False, duplicates="drop")
    except ValueError:
        labels = pd.qcut(valid.rank(method="first"), q=min(n_bins, len(valid)), labels=False, duplicates="drop")
    return labels.astype("int64")


def add_endpoint_bins(df: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    out = df.copy()
    for cov in COVARIATES:
        out[f"{cov}_bin"] = -1
    for (_model, _probe), idx in out.groupby(["model", "probe_family"], sort=False).groups.items():
        group = out.loc[idx]
        for cov in COVARIATES:
            out.loc[idx, f"{cov}_bin"] = _decile_bins(group[cov], n_bins=n_bins).to_numpy()
    out["cem_cell"] = list(
        zip(
            out["model"],
            out["probe_family"],
            out["final_entropy_bin"],
            out["final_confidence_bin"],
            out["final_top1_margin_bin"],
            strict=False,
        )
    )
    return out


def compute_cem_weights(df: pd.DataFrame, *, n_bins: int = 10) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = add_endpoint_bins(df, n_bins=n_bins)
    out["cem_weight"] = 0.0
    cell_summaries: list[dict[str, Any]] = []
    for cell, group in out.groupby("cem_cell", sort=False):
        counts = group["variant"].value_counts().to_dict()
        n_pt = int(counts.get("pt", 0))
        n_it = int(counts.get("it", 0))
        if n_pt == 0 or n_it == 0:
            continue
        target = min(n_pt, n_it)
        pt_weight = target / n_pt
        it_weight = target / n_it
        out.loc[group.index[group["variant"] == "pt"], "cem_weight"] = pt_weight
        out.loc[group.index[group["variant"] == "it"], "cem_weight"] = it_weight
        cell_summaries.append({"cell": str(cell), "n_pt": n_pt, "n_it": n_it, "target": target})
    retained: dict[str, float] = {}
    for (model, probe), group in out.groupby(["model", "probe_family"], sort=False):
        retained[f"{model}/{probe}"] = float((group["cem_weight"] > 0).mean())
    return out, {
        "n_cells_matched": len(cell_summaries),
        "retained_fraction": retained,
        "min_retained_fraction": min(retained.values()) if retained else 0.0,
    }


def compute_binned_ipw_weights(df: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    out = add_endpoint_bins(df, n_bins=n_bins)
    out["ipw_weight"] = 0.0
    p_it_overall = float((out["variant"] == "it").mean())
    for _cell, group in out.groupby("cem_cell", sort=False):
        p_it = float((group["variant"] == "it").mean())
        if p_it <= 0.0 or p_it >= 1.0:
            continue
        it_idx = group.index[group["variant"] == "it"]
        pt_idx = group.index[group["variant"] == "pt"]
        out.loc[it_idx, "ipw_weight"] = p_it_overall / p_it
        out.loc[pt_idx, "ipw_weight"] = (1.0 - p_it_overall) / (1.0 - p_it)
    return out


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    vals = values.astype(float).to_numpy()
    w = weights.astype(float).to_numpy()
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(vals[mask], weights=w[mask]))


def weighted_variant_difference(group: pd.DataFrame, outcome: str, weight_col: str) -> float:
    pt = group[group["variant"] == "pt"]
    it = group[group["variant"] == "it"]
    if pt.empty or it.empty:
        return float("nan")
    pt_mean = _weighted_mean(pt[outcome], pt[weight_col])
    it_mean = _weighted_mean(it[outcome], it[weight_col])
    return float(it_mean - pt_mean) if math.isfinite(pt_mean) and math.isfinite(it_mean) else float("nan")


def dense_equal_model_effects(
    df: pd.DataFrame,
    *,
    method: str,
    weight_col: str,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    probes = sorted(df["probe_family"].unique())
    probe_sets: list[tuple[str, pd.DataFrame]] = [(probe, df[df["probe_family"] == probe]) for probe in probes]
    probe_sets.append(("pooled_probe_fe", df))
    for probe_name, probe_df in probe_sets:
        for outcome in OUTCOMES:
            model_diffs: list[float] = []
            model_names: list[str] = []
            for model, model_df in probe_df.groupby("model", sort=True):
                diff = weighted_variant_difference(model_df, outcome, weight_col)
                if math.isfinite(diff):
                    model_diffs.append(diff)
                    model_names.append(str(model))
            estimate = float(np.mean(model_diffs)) if model_diffs else float("nan")
            lo, hi = bootstrap_equal_model_effect(
                probe_df,
                outcome=outcome,
                weight_col=weight_col,
                n_boot=n_boot,
                seed=seed,
            )
            rows.append(
                {
                    "method": method,
                    "probe_family": probe_name,
                    "outcome": outcome,
                    "estimate_it_minus_pt": estimate,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_models": len(model_diffs),
                    "models": ",".join(model_names),
                }
            )
    return rows


def bootstrap_equal_model_effect(
    df: pd.DataFrame,
    *,
    outcome: str,
    weight_col: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    if n_boot <= 0 or df.empty:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed + abs(hash((outcome, weight_col))) % 100_000)
    boot: list[float] = []
    grouped = [(model, group.reset_index(drop=True)) for model, group in df.groupby("model", sort=True)]
    for _ in range(n_boot):
        diffs: list[float] = []
        for _model, group in grouped:
            if group.empty:
                continue
            sample = group.iloc[rng.integers(0, len(group), size=len(group))]
            diff = weighted_variant_difference(sample, outcome, weight_col)
            if math.isfinite(diff):
                diffs.append(diff)
        if diffs:
            boot.append(float(np.mean(diffs)))
    if len(boot) < 10:
        return float("nan"), float("nan")
    lo, hi = np.quantile(np.asarray(boot), [0.025, 0.975])
    return float(lo), float(hi)


def _smd(group: pd.DataFrame, covariate: str, weight_col: str | None = None) -> float:
    pt = group[group["variant"] == "pt"]
    it = group[group["variant"] == "it"]
    if pt.empty or it.empty:
        return float("nan")
    if weight_col is None:
        pt_mean = float(pt[covariate].mean())
        it_mean = float(it[covariate].mean())
        pt_var = float(pt[covariate].var(ddof=0))
        it_var = float(it[covariate].var(ddof=0))
    else:
        pt_mean = _weighted_mean(pt[covariate], pt[weight_col])
        it_mean = _weighted_mean(it[covariate], it[weight_col])
        pt_var = _weighted_var(pt[covariate], pt[weight_col])
        it_var = _weighted_var(it[covariate], it[weight_col])
    denom = math.sqrt(max((pt_var + it_var) / 2.0, 1e-12))
    return float((it_mean - pt_mean) / denom)


def _weighted_var(values: pd.Series, weights: pd.Series) -> float:
    vals = values.astype(float).to_numpy()
    w = weights.astype(float).to_numpy()
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return float("nan")
    mean = np.average(vals[mask], weights=w[mask])
    return float(np.average((vals[mask] - mean) ** 2, weights=w[mask]))


def balance_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, probe), group in df.groupby(["model", "probe_family"], sort=True):
        matched = group[group["cem_weight"] > 0]
        for cov in COVARIATES:
            rows.append(
                {
                    "model": model,
                    "probe_family": probe,
                    "covariate": cov,
                    "smd_before": _smd(group, cov, None),
                    "smd_after": _smd(matched, cov, "cem_weight") if not matched.empty else float("nan"),
                    "matched_retained_fraction": float((group["cem_weight"] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def regression_variant_coefficients(df: pd.DataFrame, *, weight_col: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        needed = [outcome, *COVARIATES, "variant_it", "model", "probe_family"]
        data = df[needed + ([weight_col] if weight_col else [])].dropna().copy()
        if weight_col:
            data = data[data[weight_col] > 0]
        if data.empty:
            continue
        y = data[outcome].astype(float).to_numpy()
        cov = data[COVARIATES].astype(float)
        cov = (cov - cov.mean()) / cov.std(ddof=0).replace(0.0, 1.0)
        columns = [np.ones(len(data)), data["variant_it"].astype(float).to_numpy()]
        columns.extend(cov[col].to_numpy() for col in COVARIATES)
        for col in sorted(data["model"].unique())[1:]:
            columns.append((data["model"] == col).astype(float).to_numpy())
        for col in sorted(data["probe_family"].unique())[1:]:
            columns.append((data["probe_family"] == col).astype(float).to_numpy())
        x = np.column_stack(columns)
        if weight_col:
            w = np.sqrt(data[weight_col].astype(float).to_numpy())
        else:
            w = np.ones(len(data))
        beta, *_ = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)
        rows.append(
            {
                "method": f"regression_{'weighted' if weight_col else 'all'}",
                "outcome": outcome,
                "variant_it_coefficient": float(beta[1]),
                "n_rows": int(len(data)),
            }
        )
    return rows


def plot_summary(effects: pd.DataFrame, balance: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Exp22 endpoint-deconfounded convergence gap", fontsize=14)

    cem = effects[effects["method"] == "cem"]
    late = cem[(cem["outcome"] == "late_kl_mean") & (cem["probe_family"].isin(["raw", "tuned"]))]
    axes[0, 0].bar(late["probe_family"], late["estimate_it_minus_pt"], color=["#4C78A8", "#F58518"])
    for i, row in enumerate(late.itertuples(index=False)):
        if math.isfinite(row.ci95_low) and math.isfinite(row.ci95_high):
            axes[0, 0].errorbar(i, row.estimate_it_minus_pt, yerr=[[row.estimate_it_minus_pt - row.ci95_low], [row.ci95_high - row.estimate_it_minus_pt]], fmt="none", color="black")
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Matched late KL, IT - PT")
    axes[0, 0].set_ylabel("nats")

    endpoint_free = cem[(cem["probe_family"] == "pooled_probe_fe") & (cem["outcome"].isin(["remaining_adj_js", "future_top1_flips", "top5_churn"]))]
    axes[0, 1].bar(endpoint_free["outcome"], endpoint_free["estimate_it_minus_pt"], color="#54A24B")
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Endpoint-free matched checks")
    axes[0, 1].tick_params(axis="x", rotation=20)

    if not balance.empty:
        before = balance.groupby("covariate")["smd_before"].apply(lambda x: np.nanmax(np.abs(x))).reindex(COVARIATES)
        after = balance.groupby("covariate")["smd_after"].apply(lambda x: np.nanmax(np.abs(x))).reindex(COVARIATES)
        x = np.arange(len(COVARIATES))
        axes[1, 0].bar(x - 0.2, before, width=0.4, label="before", color="#B279A2")
        axes[1, 0].bar(x + 0.2, after, width=0.4, label="after", color="#72B7B2")
        axes[1, 0].axhline(0.1, color="black", linestyle="--", linewidth=0.8)
        axes[1, 0].set_xticks(x, COVARIATES, rotation=20)
        axes[1, 0].set_title("Max endpoint covariate SMD")
        axes[1, 0].legend()

    methods = effects[(effects["probe_family"] == "pooled_probe_fe") & (effects["outcome"] == "late_kl_mean")]
    axes[1, 1].bar(methods["method"], methods["estimate_it_minus_pt"], color="#E45756")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Estimator agreement")
    axes[1, 1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_paper_note(summary: dict[str, Any], out_path: Path) -> None:
    controlled = summary.get("controlled_estimates", [])
    lookup = {
        (row["method"], row["probe_family"], row["outcome"]): row
        for row in controlled
    }
    raw = lookup.get(("cem", "raw", "late_kl_mean"), {})
    tuned = lookup.get(("cem", "tuned", "late_kl_mean"), {})
    rem_js = lookup.get(("cem", "pooled_probe_fe", "remaining_adj_js"), {})
    text = f"""# Exp22 Endpoint-Deconfounded Gap Summary

Primary matched estimate, raw probe late KL (IT - PT): {raw.get('estimate_it_minus_pt', float('nan')):.6g}
95% CI: [{raw.get('ci95_low', float('nan')):.6g}, {raw.get('ci95_high', float('nan')):.6g}]

Primary matched estimate, tuned probe late KL (IT - PT): {tuned.get('estimate_it_minus_pt', float('nan')):.6g}
95% CI: [{tuned.get('ci95_low', float('nan')):.6g}, {tuned.get('ci95_high', float('nan')):.6g}]

Endpoint-free remaining adjacent JS, pooled probe fixed-effect summary (IT - PT): {rem_js.get('estimate_it_minus_pt', float('nan')):.6g}

Interpretation rule:
- If raw and tuned controlled late KL remain positive with CIs above zero, call the convergence gap endpoint-deconfounded.
- If controlled KL weakens but endpoint-free churn/path remains positive, describe the result as endpoint-mediated calibration plus persistent path instability.
- If both vanish, reframe the convergence gap as mostly final-confidence/readout calibration.
"""
    out_path.write_text(text)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, load_summary = flatten_records(root, skip_first_n=args.skip_first_n)
    if df.empty:
        raise RuntimeError(f"No valid Exp22 token-step rows found under {root}")
    if args.models:
        df = df[df["model"].isin(args.models)].copy()
    df = df[df["variant"].isin(["pt", "it"])].copy()

    cem_df, matching = compute_cem_weights(df, n_bins=args.n_bins)
    ipw_df = compute_binned_ipw_weights(df, n_bins=args.n_bins)
    cem_df["ipw_weight"] = ipw_df["ipw_weight"].to_numpy()
    bal = balance_table(cem_df)
    effects_rows: list[dict[str, Any]] = []
    effects_rows.extend(dense_equal_model_effects(cem_df, method="cem", weight_col="cem_weight", n_boot=args.n_boot, seed=args.seed))
    effects_rows.extend(dense_equal_model_effects(cem_df, method="binned_ipw", weight_col="ipw_weight", n_boot=args.n_boot, seed=args.seed + 17))
    effects = pd.DataFrame(effects_rows)
    regression = regression_variant_coefficients(cem_df, weight_col=None)
    regression.extend(regression_variant_coefficients(cem_df, weight_col="cem_weight"))

    max_smd_after = float(np.nanmax(np.abs(bal["smd_after"].to_numpy()))) if not bal.empty else float("inf")
    malformed_rates: dict[str, float] = {}
    for branch, counts in load_summary["branch_counts"].items():
        total = max(int(counts.get("records", 0)), 1)
        malformed_rates[branch] = float(counts.get("malformed", 0) / total)
    branch_record_ok = all(rate <= args.max_malformed_rate for rate in malformed_rates.values())
    retained_ok = matching.get("min_retained_fraction", 0.0) >= args.min_retained_fraction
    smd_ok = max_smd_after <= args.max_smd
    nonempty_ok = all(
        not cem_df[(cem_df["model"] == model) & (cem_df["probe_family"] == probe) & (cem_df["cem_weight"] > 0)].empty
        for model in args.models
        for probe in sorted(cem_df["probe_family"].unique())
    )

    def _effect(method: str, probe: str, outcome: str) -> dict[str, Any]:
        rows = effects[(effects["method"] == method) & (effects["probe_family"] == probe) & (effects["outcome"] == outcome)]
        if rows.empty:
            return {}
        return rows.iloc[0].to_dict()

    raw_late = _effect("cem", "raw", "late_kl_mean")
    tuned_late = _effect("cem", "tuned", "late_kl_mean")
    endpoint_free = _effect("cem", "pooled_probe_fe", "remaining_adj_js")
    scientific_acceptance = {
        "raw_late_kl_ci_positive": bool(raw_late and as_float(raw_late.get("ci95_low")) > 0),
        "tuned_late_kl_ci_positive": bool(tuned_late and as_float(tuned_late.get("ci95_low")) > 0),
        "endpoint_free_remaining_adj_js_positive": bool(endpoint_free and as_float(endpoint_free.get("estimate_it_minus_pt")) > 0),
    }
    scientific_acceptance["primary_success"] = bool(
        scientific_acceptance["raw_late_kl_ci_positive"]
        and scientific_acceptance["tuned_late_kl_ci_positive"]
        and scientific_acceptance["endpoint_free_remaining_adj_js_positive"]
    )

    quality = {
        "branch_record_ok": branch_record_ok,
        "retained_ok": retained_ok,
        "smd_ok": smd_ok,
        "nonempty_model_probe_ok": nonempty_ok,
        "max_malformed_rate": max(malformed_rates.values()) if malformed_rates else 0.0,
        "min_retained_fraction": matching.get("min_retained_fraction", 0.0),
        "max_smd_after": max_smd_after,
    }
    quality["ok"] = bool(branch_record_ok and retained_ok and smd_ok and nonempty_ok)

    effects_path = out_dir / "controlled_effects.csv"
    balance_path = out_dir / "balance.csv"
    effects.to_csv(effects_path, index=False)
    bal.to_csv(balance_path, index=False)
    plot_summary(effects, bal, out_dir / "endpoint_deconfounded_gap_main.png")
    summary = {
        "root": str(root),
        "n_rows": int(len(df)),
        "models": args.models,
        "covariates": COVARIATES,
        "outcomes": OUTCOMES,
        "load_summary": load_summary,
        "matching": matching,
        "quality": quality,
        "scientific_acceptance": scientific_acceptance,
        "controlled_estimates": effects.to_dict(orient="records"),
        "regression": regression,
        "malformed_rates": malformed_rates,
        "artifacts": {
            "controlled_effects_csv": str(effects_path),
            "balance_csv": str(balance_path),
            "main_plot": str(out_dir / "endpoint_deconfounded_gap_main.png"),
            "paper_note": str(out_dir / "paper_claims_exp22.md"),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_paper_note(summary, out_dir / "paper_claims_exp22.md")
    if args.fail_on_quality and not quality["ok"]:
        raise SystemExit(f"Exp22 quality gate failed: {quality}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", nargs="+", default=PRIMARY_MODELS)
    parser.add_argument("--skip-first-n", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-malformed-rate", type=float, default=0.01)
    parser.add_argument("--min-retained-fraction", type=float, default=0.25)
    parser.add_argument("--max-smd", type=float, default=0.1)
    parser.add_argument("--fail-on-quality", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps({"quality": summary["quality"], "scientific_acceptance": summary["scientific_acceptance"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

