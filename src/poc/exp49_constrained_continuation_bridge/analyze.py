"""Analyze Exp49 constrained-continuation sequence scores."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.poc.exp49_constrained_continuation_bridge import DEFAULT_HORIZONS, DEFAULT_READOUTS
from src.poc.exp49_constrained_continuation_bridge.common import (
    EXP47_DEFAULT_ROOT,
    EXP49_DEFAULT_ROOT,
    bootstrap_ci,
    cumulative_sum,
    horizon_values,
    json_rows,
    load_exp47_effect_slices,
    safe_float,
    suffix_sum,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CELL_ORDER = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")


def _candidate_valid_for_horizon(candidate: dict[str, Any], horizon: int) -> bool:
    token_ids = candidate.get("token_ids") or []
    if len(token_ids) < int(horizon) + 1:
        return False
    first_eos = candidate.get("first_eos_position")
    if first_eos is not None and int(horizon) > int(first_eos):
        return False
    for pos in candidate.get("non_eos_special_positions") or []:
        if int(pos) <= int(horizon):
            return False
    return True


def _cell_sequence_sum(
    row: dict[str, Any],
    *,
    cell: str,
    readout: str,
    sequence: str,
    horizon: int,
    suffix_only: bool = False,
) -> float | None:
    try:
        values = row["cells"][cell]["readouts"][readout][sequence]["target_logprobs"]
    except KeyError:
        return None
    return suffix_sum(values, horizon) if suffix_only else cumulative_sum(values, horizon)


def _candidate_pair_valid(row: dict[str, Any], lhs: str, rhs: str, horizon: int, *, suffix_only: bool = False) -> bool:
    if suffix_only and horizon <= 0:
        return True
    candidates = row.get("candidates") or {}
    return _candidate_valid_for_horizon(candidates.get(lhs, {}), horizon) and _candidate_valid_for_horizon(
        candidates.get(rhs, {}), horizon
    )


def _decompose(y: dict[str, float | None]) -> dict[str, float | None]:
    if any(y.get(cell) is None for cell in CELL_ORDER):
        return {"P": None, "M": None, "C": None}
    p = float(y["U_PT__L_IT"]) - float(y["U_PT__L_PT"])
    m = float(y["U_IT__L_IT"]) - float(y["U_IT__L_PT"])
    c = m - p
    return {"P": p, "M": m, "C": c}


def row_effects(
    row: dict[str, Any],
    *,
    horizons: list[int],
    readouts: tuple[str, ...],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not row.get("valid") or not row.get("scored"):
        return out
    slices = list(row.get("slices") or ["full_1400"])
    if row.get("position_ge_3"):
        slices.append("position_ge_3")
    if row.get("position_ge_5"):
        slices.append("position_ge_5")

    base_meta = {
        "model": row.get("model"),
        "prompt_id": row.get("prompt_id"),
        "event_kind": row.get("event_kind"),
        "recipe_group": row.get("recipe_group"),
        "category": row.get("category"),
        "source": row.get("source"),
        "token_category": row.get("token_category"),
        "position": row.get("position"),
        "pt_token_id": row.get("pt_token_id"),
        "it_token_id": row.get("it_token_id"),
    }
    for readout in readouts:
        primary_by_h: dict[int, dict[str, float | None]] = {}
        for horizon in horizons:
            valid_primary = _candidate_pair_valid(row, "desc_primary", "base_primary", horizon)
            y_primary: dict[str, float | None] = {}
            if valid_primary:
                for cell in CELL_ORDER:
                    lhs = _cell_sequence_sum(row, cell=cell, readout=readout, sequence="desc_primary", horizon=horizon)
                    rhs = _cell_sequence_sum(row, cell=cell, readout=readout, sequence="base_primary", horizon=horizon)
                    y_primary[cell] = None if lhs is None or rhs is None else lhs - rhs
            else:
                y_primary = {cell: None for cell in CELL_ORDER}
            decomp = _decompose(y_primary)
            primary_by_h[horizon] = decomp

            # Descendant-tail likelihood secondary metrics.
            t_desc: dict[str, float | None] = {}
            for cell in CELL_ORDER:
                val = _cell_sequence_sum(row, cell=cell, readout=readout, sequence="desc_primary", horizon=horizon)
                t_desc[cell] = None if val is None else val / (horizon + 1)
            upstream_tail_gap = (
                None
                if t_desc["U_IT__L_IT"] is None or t_desc["U_PT__L_IT"] is None
                else t_desc["U_IT__L_IT"] - t_desc["U_PT__L_IT"]
            )
            portable_tail_late = (
                None
                if t_desc["U_PT__L_IT"] is None or t_desc["U_PT__L_PT"] is None
                else t_desc["U_PT__L_IT"] - t_desc["U_PT__L_PT"]
            )
            matched_tail_late = (
                None
                if t_desc["U_IT__L_IT"] is None or t_desc["U_IT__L_PT"] is None
                else t_desc["U_IT__L_IT"] - t_desc["U_IT__L_PT"]
            )

            for slice_name in slices:
                out.append(
                    {
                        **base_meta,
                        "slice": slice_name,
                        "readout": readout,
                        "comparison": "primary_desc_vs_base",
                        "horizon": horizon,
                        "valid_horizon": bool(valid_primary and decomp["C"] is not None),
                        **{cell: y_primary.get(cell) for cell in CELL_ORDER},
                        "P": decomp["P"],
                        "M": decomp["M"],
                        "C": decomp["C"],
                        "Ybar_P": None if decomp["P"] is None else decomp["P"] / (horizon + 1),
                        "Ybar_M": None if decomp["M"] is None else decomp["M"] / (horizon + 1),
                        "Ybar_C": None if decomp["C"] is None else decomp["C"] / (horizon + 1),
                        "P_tail": None,
                        "M_tail": None,
                        "C_tail": None,
                        "Pbar_tail": None,
                        "Mbar_tail": None,
                        "Cbar_tail": None,
                        "upstream_tail_gap": upstream_tail_gap,
                        "portable_tail_late": portable_tail_late,
                        "matched_tail_late": matched_tail_late,
                    }
                )

            for comparison, lhs_name, rhs_name in (
                ("same_forced_desc_tail", "desc_primary", "base_forced_desc"),
                ("shuffled_desc_vs_base", "desc_shuffled", "base_primary"),
            ):
                suffix_only = comparison == "same_forced_desc_tail"
                valid_pair = _candidate_pair_valid(row, lhs_name, rhs_name, horizon, suffix_only=suffix_only)
                y_pair: dict[str, float | None] = {}
                if valid_pair:
                    for cell in CELL_ORDER:
                        lhs = _cell_sequence_sum(
                            row,
                            cell=cell,
                            readout=readout,
                            sequence=lhs_name,
                            horizon=horizon,
                            suffix_only=suffix_only,
                        )
                        rhs = _cell_sequence_sum(
                            row,
                            cell=cell,
                            readout=readout,
                            sequence=rhs_name,
                            horizon=horizon,
                            suffix_only=suffix_only,
                        )
                        y_pair[cell] = None if lhs is None or rhs is None else lhs - rhs
                else:
                    y_pair = {cell: None for cell in CELL_ORDER}
                pair_decomp = _decompose(y_pair)
                for slice_name in slices:
                    out.append(
                        {
                            **base_meta,
                            "slice": slice_name,
                            "readout": readout,
                            "comparison": comparison,
                            "horizon": horizon,
                            "valid_horizon": bool(valid_pair and pair_decomp["C"] is not None),
                            **{cell: y_pair.get(cell) for cell in CELL_ORDER},
                            "P": pair_decomp["P"],
                            "M": pair_decomp["M"],
                            "C": pair_decomp["C"],
                            "Ybar_P": None if pair_decomp["P"] is None else pair_decomp["P"] / max(horizon, 1),
                            "Ybar_M": None if pair_decomp["M"] is None else pair_decomp["M"] / max(horizon, 1),
                            "Ybar_C": None if pair_decomp["C"] is None else pair_decomp["C"] / max(horizon, 1),
                            "P_tail": None,
                            "M_tail": None,
                            "C_tail": None,
                            "Pbar_tail": None,
                            "Mbar_tail": None,
                            "Cbar_tail": None,
                            "upstream_tail_gap": None,
                            "portable_tail_late": None,
                            "matched_tail_late": None,
                        }
                    )

        p0 = primary_by_h.get(0, {})
        for record in out:
            if record["comparison"] != "primary_desc_vs_base" or int(record["horizon"]) == 0:
                continue
            horizon = int(record["horizon"])
            if record["P"] is None or p0.get("P") is None:
                continue
            record["P_tail"] = record["P"] - p0["P"]
            record["M_tail"] = record["M"] - p0["M"]
            record["C_tail"] = record["C"] - p0["C"]
            record["Pbar_tail"] = record["P_tail"] / horizon
            record["Mbar_tail"] = record["M_tail"] / horizon
            record["Cbar_tail"] = record["C_tail"] / horizon
    return out


def load_score_rows(scores_dir: Path, models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        paths = sorted((scores_dir / model).glob("sequence_scores_w*.jsonl.gz"))
        if not paths:
            log.warning("[exp49 analyze] no score files for model=%s under %s", model, scores_dir / model)
        for path in paths:
            rows.extend(json_rows(path))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "model",
        "recipe_group",
        "slice",
        "prompt_id",
        "event_kind",
        "readout",
        "comparison",
        "horizon",
        "valid_horizon",
        "P",
        "M",
        "C",
        "P_tail",
        "M_tail",
        "C_tail",
        "Pbar_tail",
        "Mbar_tail",
        "Cbar_tail",
    ]
    fieldnames = preferred + [f for f in fields if f not in preferred]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_effects(df: pd.DataFrame, *, n_boot: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_cols = ["recipe_group", "model", "slice", "readout", "comparison", "horizon"]
    for key, group in df[df["valid_horizon"] == True].groupby(group_cols, dropna=False):  # noqa: E712
        record = dict(zip(group_cols, key))
        for metric in ("P", "M", "C", "P_tail", "M_tail", "C_tail", "Cbar_tail", "upstream_tail_gap"):
            if metric not in group:
                continue
            vals = [safe_float(v) for v in group[metric].tolist()]
            clean_vals = [float(v) for v in vals if v is not None]
            clusters = group.loc[[v is not None for v in vals], "prompt_id"].astype(str).tolist()
            ci = bootstrap_ci(clean_vals, clusters=clusters, n_boot=n_boot, seed=seed)
            for suffix, value in ci.items():
                record[f"{metric}_{suffix}"] = value
        rows.append(record)
    # Cross-model recipe-level view with prompt clustering, keeping model as a column for transparency.
    group_cols = ["recipe_group", "slice", "readout", "comparison", "horizon"]
    for key, group in df[df["valid_horizon"] == True].groupby(group_cols, dropna=False):  # noqa: E712
        record = dict(zip(group_cols, key))
        record["model"] = "ALL"
        for metric in ("P", "M", "C", "P_tail", "M_tail", "C_tail", "Cbar_tail", "upstream_tail_gap"):
            vals = [safe_float(v) for v in group[metric].tolist()]
            clean_vals = [float(v) for v in vals if v is not None]
            clusters = [
                f"{m}:{p}"
                for m, p, keep in zip(group["model"].astype(str), group["prompt_id"].astype(str), [v is not None for v in vals])
                if keep
            ]
            ci = bootstrap_ci(clean_vals, clusters=clusters, n_boot=n_boot, seed=seed)
            for suffix, value in ci.items():
                record[f"{metric}_{suffix}"] = value
        rows.append(record)
    return rows


def reproduction_summary(
    effects: list[dict[str, Any]],
    *,
    exp47_root: Path,
    tolerance: float,
) -> dict[str, Any]:
    exp47 = load_exp47_effect_slices(exp47_root)
    diffs: list[dict[str, Any]] = []
    for row in effects:
        if row.get("comparison") != "primary_desc_vs_base" or int(row.get("horizon", -1)) != 0:
            continue
        if row.get("slice") != "full_1400":
            continue
        key = (str(row.get("model")), str(row.get("prompt_id")), str(row.get("readout")))
        ref = (exp47.get(key) or {}).get("exp47") or {}
        if not ref:
            continue
        diff = {
            "model": row.get("model"),
            "prompt_id": row.get("prompt_id"),
            "readout": row.get("readout"),
        }
        for metric in ("P", "M", "C"):
            if row.get(metric) is None or ref.get(metric) is None:
                continue
            diff[f"{metric}_abs_delta"] = abs(float(row[metric]) - float(ref[metric]))
        diffs.append(diff)
    summary: dict[str, Any] = {"n_compared": len(diffs), "tolerance": float(tolerance)}
    max_deltas: list[float] = []
    for metric in ("P", "M", "C"):
        vals = [d[f"{metric}_abs_delta"] for d in diffs if f"{metric}_abs_delta" in d]
        summary[f"{metric}_mean_abs_delta"] = float(sum(vals) / len(vals)) if vals else None
        summary[f"{metric}_max_abs_delta"] = float(max(vals)) if vals else None
    for diff in diffs:
        vals = [float(diff[f"{metric}_abs_delta"]) for metric in ("P", "M", "C") if f"{metric}_abs_delta" in diff]
        if vals:
            max_deltas.append(max(vals))
    if max_deltas:
        ordered = sorted(max_deltas)

        def quantile(q: float) -> float:
            return float(ordered[int(q * (len(ordered) - 1))])

        summary["max_delta_median"] = quantile(0.5)
        summary["max_delta_q95"] = quantile(0.95)
        summary["max_delta_q99"] = quantile(0.99)
        summary["max_delta_q999"] = quantile(0.999)
        summary["n_over_tolerance"] = int(sum(1 for value in max_deltas if value > float(tolerance)))
        summary["q99_ok"] = bool(summary["max_delta_q99"] <= float(tolerance))
    summary["ok"] = bool(
        diffs
        and all(
            summary.get(f"{metric}_max_abs_delta") is not None
            and float(summary[f"{metric}_max_abs_delta"]) <= float(tolerance)
            for metric in ("P", "M", "C")
        )
    )
    return summary


def unscored_summary(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    unscored = [row for row in score_rows if not row.get("scored")]
    by_reason = Counter(str(row.get("score_reason") or row.get("reason") or "unknown") for row in unscored)
    by_model = Counter(str(row.get("model") or "unknown") for row in unscored)
    missing_event = by_reason.get("candidate_invalid", 0)
    # Candidate-invalid rows currently preserve the original candidate reason in
    # ``reason``. Keep both views because missing first-divergence events are a
    # support-filter outcome, not a scoring/runtime malfunction.
    candidate_reasons = Counter(str(row.get("reason") or "unknown") for row in unscored)
    non_missing_event = [
        row
        for row in unscored
        if str(row.get("reason") or row.get("score_reason") or "") != "missing_event"
    ]
    return {
        "n_unscored": len(unscored),
        "fraction_unscored": len(unscored) / max(len(score_rows), 1),
        "by_score_reason": dict(sorted(by_reason.items())),
        "by_candidate_reason": dict(sorted(candidate_reasons.items())),
        "by_model": dict(sorted(by_model.items())),
        "n_missing_event": int(candidate_reasons.get("missing_event", 0)),
        "missing_event_fraction": int(candidate_reasons.get("missing_event", 0)) / max(len(score_rows), 1),
        "n_runtime_or_malformed": len(non_missing_event),
        "runtime_or_malformed_fraction": len(non_missing_event) / max(len(score_rows), 1),
        "n_candidate_invalid_score_reason": int(missing_event),
    }


def plot_main(agg_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = agg_df[
        (agg_df["model"] == "ALL")
        & (agg_df["comparison"] == "primary_desc_vs_base")
        & (agg_df["readout"] == "common_it")
        & (agg_df["slice"].isin(["instruction_format", "math_domain", "full_1400", "content_fact"]))
    ].copy()
    if subset.empty:
        return
    for metric, ylabel, path_name in (
        ("C_mean", "conditioned interaction C_N", "exp49_cumulative_interaction.png"),
        ("Cbar_tail_mean", "tail-only interaction per token", "exp49_tail_interaction.png"),
    ):
        plt.figure(figsize=(8.5, 4.8))
        for (recipe, slice_name), group in subset.groupby(["recipe_group", "slice"]):
            group = group.sort_values("horizon")
            if metric not in group:
                continue
            plt.plot(group["horizon"], group[metric], marker="o", label=f"{recipe}/{slice_name}")
        plt.axhline(0.0, color="black", lw=0.8)
        plt.xlabel("Continuation horizon N")
        plt.ylabel(ylabel)
        plt.title("Exp49 constrained continuation bridge")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / path_name, dpi=180)
        plt.close()


def analyze(args: argparse.Namespace) -> None:
    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = args.models.split()
    horizons = horizon_values(args.max_tail, [int(x) for x in args.horizons.split()])
    score_rows = load_score_rows(scores_dir, models)
    log.info("[exp49 analyze] loaded %d score rows", len(score_rows))
    effects: list[dict[str, Any]] = []
    for row in score_rows:
        effects.extend(row_effects(row, horizons=horizons, readouts=tuple(args.readouts.split())))
    write_csv(out_dir / "sequence_effects.csv", effects)
    df = pd.DataFrame(effects)
    if df.empty:
        raise RuntimeError("No sequence effects produced")
    agg = aggregate_effects(df, n_boot=int(args.n_boot), seed=int(args.seed))
    write_csv(out_dir / "aggregate_effects.csv", agg)
    agg_df = pd.DataFrame(agg)
    plot_main(agg_df, out_dir / "plots")
    repro = reproduction_summary(
        effects,
        exp47_root=Path(args.exp47_root),
        tolerance=float(args.n0_reproduction_tolerance),
    )
    unscored = unscored_summary(score_rows)
    summary = {
        "models": models,
        "horizons": horizons,
        "n_score_rows": len(score_rows),
        "n_malformed_or_unscored": unscored["n_unscored"],
        "malformed_fraction": unscored["fraction_unscored"],
        "unscored_summary": unscored,
        "n_effect_rows": len(effects),
        "n_aggregate_rows": len(agg),
        "n0_reproduction": repro,
        "readouts": args.readouts.split(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    paper_lines = [
        "# Exp49 Paper-Facing Summary",
        "",
        f"- Score rows: {summary['n_score_rows']}",
        f"- Malformed/unscored fraction: {summary['malformed_fraction']:.4f}",
        f"- Runtime/malformed fraction excluding missing divergence events: {unscored['runtime_or_malformed_fraction']:.4f}",
        f"- N=0 reproduction C max abs delta: {repro.get('C_max_abs_delta')}",
        f"- N=0 reproduction max tolerance-ok: {repro.get('ok')} (tolerance={repro.get('tolerance')}); q99-ok: {repro.get('q99_ok')}",
        "- Use `aggregate_effects.csv` for horizon-level CIs. Primary rows are `comparison=primary_desc_vs_base`, `readout=common_it/common_pt`.",
        "- Tail-only persistence is `Cbar_tail_mean` for horizons N>0; cumulative retention alone includes the forced first token.",
    ]
    (out_dir / "paper_claims_exp49.md").write_text("\n".join(paper_lines) + "\n", encoding="utf-8")
    log.info("[exp49 analyze] wrote %s", out_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exp47-root", default=str(EXP47_DEFAULT_ROOT))
    parser.add_argument(
        "--models",
        default="llama31_meta_instruct llama31_tulu3_sft llama31_tulu3_dpo llama31_tulu3_final llama31_openmath2",
    )
    parser.add_argument("--horizons", default=" ".join(str(x) for x in DEFAULT_HORIZONS))
    parser.add_argument("--readouts", default=" ".join(DEFAULT_READOUTS))
    parser.add_argument("--max-tail", type=int, default=8)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n0-reproduction-tolerance",
        type=float,
        default=0.5,
        help="BF16/runtime tolerance for the N=0 reproduction check against the Exp47 CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    analyze(args)


if __name__ == "__main__":
    main()
