"""Analyze Exp26 residual-opposition mediation records."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from src.poc.exp26_residual_opposition_mediation import DENSE5_MODELS, EXPERIMENT


READOUTS = ["common_it", "common_pt"]
PRIMARY_VARIANTS = [
    "opp_scale_0p5",
    "noopp",
    "flipopp",
    "normpres_noopp",
    "ptlevel_opp",
    "randorth",
]


@dataclass(frozen=True)
class Unit:
    model: str
    prompt_id: str
    event_kind: str
    late_target: str
    variant: str
    seed: int | None
    readout: str
    step: int
    assistant_marker_event: bool
    interaction_full: float
    interaction_variant: float
    drop: float
    mediation_fraction: float | None
    mean_opp_norm_frac: float | None
    mean_update_norm_ratio_after_hook: float | None
    mean_postres_norm_ratio_after_hook: float | None
    seed_std_interaction_variant: float | None = None
    seed_std_drop: float | None = None


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _find_records(root: Path, prompt_mode: str, model: str) -> Path:
    candidates = [
        root / "records" / prompt_mode / model / "records.jsonl.gz",
        root / prompt_mode / model / "records.jsonl.gz",
        root / "residual_opposition" / prompt_mode / model / "records.jsonl.gz",
        root / "residual_factorial" / prompt_mode / model / "records.jsonl.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No records for {prompt_mode}/{model} under {root}")


def _margin(cell: dict[str, Any], readout: str) -> float | None:
    try:
        value = float(cell[readout]["it_vs_pt_margin"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _interaction_from_exp23(event_payload: dict[str, Any], readout: str) -> tuple[float, dict[str, float]] | None:
    cells = event_payload.get("cells") or {}
    vals: dict[str, float] = {}
    for cell in ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT"):
        value = _margin(cells.get(cell, {}), readout)
        if value is None:
            return None
        vals[cell] = value
    interaction = (vals["U_IT__L_IT"] - vals["U_IT__L_PT"]) - (
        vals["U_PT__L_IT"] - vals["U_PT__L_PT"]
    )
    return interaction, vals


def _variant_interaction(
    *,
    exp26_record: dict[str, Any],
    baseline_vals: dict[str, float],
    readout: str,
    late_target: str,
) -> float | None:
    cells = exp26_record.get("cells") or {}
    if late_target == "it":
        u_pt_l_it = _margin(cells.get("U_PT__L_IT_variant", {}), readout)
        u_it_l_it = _margin(cells.get("U_IT__L_IT_variant", {}), readout)
        if u_pt_l_it is None or u_it_l_it is None:
            return None
        return (u_it_l_it - baseline_vals["U_IT__L_PT"]) - (
            u_pt_l_it - baseline_vals["U_PT__L_PT"]
        )
    if late_target == "pt":
        u_pt_l_pt = _margin(cells.get("U_PT__L_PT_variant", {}), readout)
        u_it_l_pt = _margin(cells.get("U_IT__L_PT_variant", {}), readout)
        if u_pt_l_pt is None or u_it_l_pt is None:
            return None
        return (baseline_vals["U_IT__L_IT"] - u_it_l_pt) - (
            baseline_vals["U_PT__L_IT"] - u_pt_l_pt
        )
    raise ValueError(f"unsupported late_target={late_target}")


def _assistant_marker(event: dict[str, Any]) -> bool:
    pt = event.get("pt_token") or {}
    it = event.get("it_token") or {}
    return bool(pt.get("assistant_marker") or it.get("assistant_marker"))


def _safe_fraction(num: float, denom: float) -> float | None:
    if abs(denom) < 1e-12:
        return None
    return num / denom


def load_units(
    *,
    exp26_root: Path,
    exp23_root: Path,
    models: list[str],
    prompt_mode: str,
    event_kind: str,
    late_target: str,
) -> list[Unit]:
    baseline: dict[tuple[str, str, str, str], tuple[float, dict[str, float], dict[str, Any]]] = {}
    for model in models:
        path = _find_records(exp23_root, prompt_mode, model)
        for row in _json_rows(path):
            prompt_id = str(row.get("prompt_id"))
            event_payload = (row.get("events") or {}).get(event_kind) or {}
            if not event_payload.get("valid"):
                continue
            for readout in READOUTS:
                parsed = _interaction_from_exp23(event_payload, readout)
                if parsed is None:
                    continue
                interaction, vals = parsed
                baseline[(model, prompt_id, event_kind, readout)] = (interaction, vals, event_payload)

    units: list[Unit] = []
    for model in models:
        path = _find_records(exp26_root, prompt_mode, model)
        for row in _json_rows(path):
            if not row.get("valid"):
                continue
            prompt_id = str(row.get("prompt_id"))
            if str(row.get("late_target", "it")) != late_target:
                continue
            variant = str(row.get("variant"))
            seed = row.get("seed")
            seed_int = None if seed is None else int(seed)
            diagnostics = row.get("diagnostics") or {}
            for readout in READOUTS:
                base = baseline.get((model, prompt_id, event_kind, readout))
                if base is None:
                    continue
                interaction_full, baseline_vals, event_payload = base
                interaction_variant = _variant_interaction(
                    exp26_record=row,
                    baseline_vals=baseline_vals,
                    readout=readout,
                    late_target=late_target,
                )
                if interaction_variant is None:
                    continue
                drop = interaction_full - interaction_variant
                event = row.get("event") or event_payload.get("event") or {}
                units.append(
                    Unit(
                        model=model,
                        prompt_id=prompt_id,
                        event_kind=str(row.get("event_kind", event_kind)),
                        late_target=late_target,
                        variant=variant,
                        seed=seed_int,
                        readout=readout,
                        step=int(event.get("step", -1)),
                        assistant_marker_event=_assistant_marker(event),
                        interaction_full=float(interaction_full),
                        interaction_variant=float(interaction_variant),
                        drop=float(drop),
                        mediation_fraction=_safe_fraction(float(drop), float(interaction_full)),
                        mean_opp_norm_frac=diagnostics.get("mean_opp_norm_frac"),
                        mean_update_norm_ratio_after_hook=diagnostics.get(
                            "mean_update_norm_ratio_after_hook"
                        ),
                        mean_postres_norm_ratio_after_hook=diagnostics.get(
                            "mean_postres_norm_ratio_after_hook"
                        ),
                    )
                )
    return _add_randorth_aggregate(units)


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not kept:
        return None
    return sum(kept) / len(kept)


def _add_randorth_aggregate(units: list[Unit]) -> list[Unit]:
    out = list(units)
    grouped: dict[tuple[str, str, str, str, str], list[Unit]] = defaultdict(list)
    for unit in units:
        if unit.variant == "randorth" and unit.seed is not None:
            grouped[(unit.model, unit.prompt_id, unit.event_kind, unit.late_target, unit.readout)].append(unit)
    for group in grouped.values():
        if len(group) < 2:
            continue
        first = group[0]
        interaction_variant = _mean([unit.interaction_variant for unit in group])
        drop = _mean([unit.drop for unit in group])
        if interaction_variant is None or drop is None:
            continue
        out.append(
            Unit(
                model=first.model,
                prompt_id=first.prompt_id,
                event_kind=first.event_kind,
                late_target=first.late_target,
                variant="randorth",
                seed=None,
                readout=first.readout,
                step=first.step,
                assistant_marker_event=first.assistant_marker_event,
                interaction_full=first.interaction_full,
                interaction_variant=interaction_variant,
                drop=drop,
                mediation_fraction=_safe_fraction(drop, first.interaction_full),
                mean_opp_norm_frac=_mean([unit.mean_opp_norm_frac for unit in group]),
                mean_update_norm_ratio_after_hook=_mean(
                    [unit.mean_update_norm_ratio_after_hook for unit in group]
                ),
                mean_postres_norm_ratio_after_hook=_mean(
                    [unit.mean_postres_norm_ratio_after_hook for unit in group]
                ),
                seed_std_interaction_variant=float(
                    statistics.pstdev([unit.interaction_variant for unit in group])
                ),
                seed_std_drop=float(statistics.pstdev([unit.drop for unit in group])),
            )
        )
    return out


def _subset(units: list[Unit], subset: str) -> list[Unit]:
    if subset == "all":
        return units
    if subset == "position_ge3":
        return [unit for unit in units if unit.step >= 3]
    if subset == "non_assistant_marker":
        return [unit for unit in units if not unit.assistant_marker_event]
    raise ValueError(subset)


def _scope_models(scope: str, models: list[str]) -> list[str]:
    if scope == "dense5":
        return [model for model in models if model in DENSE5_MODELS]
    if scope == "dense4_no_gemma":
        return [model for model in models if model in DENSE5_MODELS and model != "gemma3_4b"]
    if scope == "family_median":
        return [model for model in models if model in DENSE5_MODELS]
    if scope.startswith("model:"):
        return [scope.split(":", 1)[1]]
    raise ValueError(scope)


def _estimate_once(units: list[Unit], *, scope: str, models: list[str]) -> dict[str, float | None]:
    scope_models = _scope_models(scope, models)
    per_model = []
    for model in scope_models:
        bucket = [unit for unit in units if unit.model == model]
        if not bucket:
            continue
        per_model.append(
            {
                "model": model,
                "n": len(bucket),
                "interaction_full": _mean([unit.interaction_full for unit in bucket]),
                "interaction_variant": _mean([unit.interaction_variant for unit in bucket]),
                "drop": _mean([unit.drop for unit in bucket]),
                "mean_opp_norm_frac": _mean([unit.mean_opp_norm_frac for unit in bucket]),
                "mean_update_norm_ratio_after_hook": _mean(
                    [unit.mean_update_norm_ratio_after_hook for unit in bucket]
                ),
                "mean_postres_norm_ratio_after_hook": _mean(
                    [unit.mean_postres_norm_ratio_after_hook for unit in bucket]
                ),
                "seed_std_drop": _mean([unit.seed_std_drop for unit in bucket]),
            }
        )
    if not per_model:
        return {}
    if scope == "family_median":
        full = float(statistics.median([row["interaction_full"] for row in per_model if row["interaction_full"] is not None]))
        variant = float(statistics.median([row["interaction_variant"] for row in per_model if row["interaction_variant"] is not None]))
        drop = float(statistics.median([row["drop"] for row in per_model if row["drop"] is not None]))
    else:
        full = _mean([row["interaction_full"] for row in per_model])
        variant = _mean([row["interaction_variant"] for row in per_model])
        drop = _mean([row["drop"] for row in per_model])
    return {
        "n_units": int(sum(row["n"] for row in per_model)),
        "n_models": int(len(per_model)),
        "interaction_full": full,
        "interaction_variant": variant,
        "drop": drop,
        "mediation_fraction": None if full is None or drop is None else _safe_fraction(drop, full),
        "mean_opp_norm_frac": _mean([row["mean_opp_norm_frac"] for row in per_model]),
        "mean_update_norm_ratio_after_hook": _mean(
            [row["mean_update_norm_ratio_after_hook"] for row in per_model]
        ),
        "mean_postres_norm_ratio_after_hook": _mean(
            [row["mean_postres_norm_ratio_after_hook"] for row in per_model]
        ),
        "seed_std_drop": _mean([row["seed_std_drop"] for row in per_model]),
    }


def _bootstrap(
    units: list[Unit],
    *,
    scope: str,
    models: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, float | None]:
    est = _estimate_once(units, scope=scope, models=models)
    if not est or n_boot <= 0:
        return {**est, "drop_ci_low": None, "drop_ci_high": None, "mediation_fraction_ci_low": None, "mediation_fraction_ci_high": None}
    rng = np.random.default_rng(seed)
    by_model = {model: [unit for unit in units if unit.model == model] for model in _scope_models(scope, models)}
    drops = []
    fracs = []
    variants = []
    for _ in range(n_boot):
        sampled = []
        for bucket in by_model.values():
            if not bucket:
                continue
            idx = rng.integers(0, len(bucket), size=len(bucket))
            sampled.extend([bucket[int(i)] for i in idx])
        boot_est = _estimate_once(sampled, scope=scope, models=models)
        if boot_est.get("drop") is not None:
            drops.append(float(boot_est["drop"]))
        if boot_est.get("mediation_fraction") is not None:
            fracs.append(float(boot_est["mediation_fraction"]))
        if boot_est.get("interaction_variant") is not None:
            variants.append(float(boot_est["interaction_variant"]))
    return {
        **est,
        "interaction_variant_ci_low": float(np.quantile(variants, 0.025)) if variants else None,
        "interaction_variant_ci_high": float(np.quantile(variants, 0.975)) if variants else None,
        "drop_ci_low": float(np.quantile(drops, 0.025)) if drops else None,
        "drop_ci_high": float(np.quantile(drops, 0.975)) if drops else None,
        "mediation_fraction_ci_low": float(np.quantile(fracs, 0.025)) if fracs else None,
        "mediation_fraction_ci_high": float(np.quantile(fracs, 0.975)) if fracs else None,
    }


def _rows_for_units(units: list[Unit], *, models: list[str], n_boot: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants = sorted({unit.variant for unit in units if unit.seed is None})
    scopes = ["dense5", "dense4_no_gemma", "family_median"] + [f"model:{model}" for model in models]
    subsets = ["all", "position_ge3", "non_assistant_marker"]
    for readout in READOUTS:
        for variant in variants:
            if variant not in PRIMARY_VARIANTS and variant != "randorth":
                continue
            base = [unit for unit in units if unit.readout == readout and unit.variant == variant and unit.seed is None]
            for subset_name in subsets:
                subset_units = _subset(base, subset_name)
                if not subset_units:
                    continue
                for scope in scopes:
                    scope_units = [unit for unit in subset_units if unit.model in _scope_models(scope, models)]
                    if not scope_units:
                        continue
                    est = _bootstrap(scope_units, scope=scope, models=models, n_boot=n_boot, seed=seed)
                    if not est:
                        continue
                    rows.append(
                        {
                            "late_target": base[0].late_target,
                            "readout": readout,
                            "variant": variant,
                            "subset": subset_name,
                            "scope": scope,
                            **est,
                        }
                    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row_lookup(rows: list[dict[str, Any]], *, variant: str, readout: str = "common_it", scope: str = "dense5", subset: str = "all") -> dict[str, Any] | None:
    for row in rows:
        if row["variant"] == variant and row["readout"] == readout and row["scope"] == scope and row["subset"] == subset:
            return row
    return None


def _plot(rows: list[dict[str, Any]], out_path: Path) -> None:
    kept = [
        row
        for row in rows
        if row["readout"] == "common_it" and row["scope"] == "dense5" and row["subset"] == "all"
    ]
    order = [variant for variant in PRIMARY_VARIANTS if any(row["variant"] == variant for row in kept)]
    if not order:
        return
    values = []
    lows = []
    highs = []
    labels = []
    for variant in order:
        row = _row_lookup(kept, variant=variant)
        if row is None or row.get("drop") is None:
            continue
        drop = float(row["drop"])
        values.append(drop)
        lows.append(drop - float(row.get("drop_ci_low") if row.get("drop_ci_low") is not None else drop))
        highs.append(float(row.get("drop_ci_high") if row.get("drop_ci_high") is not None else drop) - drop)
        frac = row.get("mediation_fraction")
        labels.append(f"{variant}\n{frac:.0%}" if isinstance(frac, float) and math.isfinite(frac) else variant)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(values))
    ax.bar(x, values, color="#386fa4")
    ax.errorbar(x, values, yerr=np.array([lows, highs]), fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Interaction drop vs full IT late")
    late_target = kept[0].get("late_target", "it") if kept else "it"
    ax.set_title(f"Exp26 residual-opposition mediation controls ({late_target.upper()} late target)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_report(path: Path, rows: list[dict[str, Any]], units: list[Unit]) -> None:
    def fmt(row: dict[str, Any] | None) -> str:
        if row is None or row.get("drop") is None:
            return "NA"
        ci = ""
        if row.get("drop_ci_low") is not None:
            ci = f" [{row['drop_ci_low']:+.3f}, {row['drop_ci_high']:+.3f}]"
        frac = row.get("mediation_fraction")
        frac_s = "" if frac is None else f", mediation={frac:.1%}"
        return f"{row['drop']:+.3f}{ci}{frac_s}"

    noopp = _row_lookup(rows, variant="noopp")
    half = _row_lookup(rows, variant="opp_scale_0p5")
    flip = _row_lookup(rows, variant="flipopp")
    rand = _row_lookup(rows, variant="randorth")
    norm = _row_lookup(rows, variant="normpres_noopp")
    ptlevel = _row_lookup(rows, variant="ptlevel_opp")
    lines = [
        "# Exp26 Red-Team Report",
        "",
        f"- Records analyzed: `{len(units)}` unit rows after randorth aggregation.",
        f"- Primary noopp drop: `{fmt(noopp)}`.",
        f"- Half-scale drop: `{fmt(half)}`.",
        f"- Flip-opposition drop: `{fmt(flip)}`.",
        f"- Randorth norm-matched drop: `{fmt(rand)}`.",
        f"- Norm-preserving noopp drop: `{fmt(norm)}`.",
        f"- PT-level opposition drop: `{fmt(ptlevel)}`.",
        "",
        "Interpretation rules:",
        "- If `noopp` has a positive CI-excluding-zero drop, residual opposition mediates part of the Exp23 interaction.",
        "- If `normpres_noopp` or `randorth` restores the full interaction, the result is better described as norm/energy mediation rather than residual-opposition direction.",
        "- If only Gemma is positive, keep the result appendix-level and do not promote the paper mechanism.",
    ]
    path.write_text("\n".join(lines) + "\n")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    models = args.models
    late_target = getattr(args, "late_target", "it")
    units = load_units(
        exp26_root=args.exp26_root,
        exp23_root=args.exp23_root,
        models=models,
        prompt_mode=args.prompt_mode,
        event_kind=args.event_kind,
        late_target=late_target,
    )
    if not units:
        raise RuntimeError("No Exp26 units loaded")
    rows = _rows_for_units(units, models=models, n_boot=args.n_boot, seed=args.seed)
    family_rows = [row for row in rows if str(row.get("scope", "")).startswith("model:")]
    position_rows = [row for row in rows if row.get("subset") in {"position_ge3", "non_assistant_marker"}]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "exp26_effects.csv", rows)
    _write_csv(args.out_dir / "exp26_family_table.csv", family_rows)
    _write_csv(args.out_dir / "exp26_position_table.csv", position_rows)
    _plot(rows, args.out_dir / "exp26_mediation_plot.png")
    _write_report(args.out_dir / "exp26_red_team_report.md", rows, units)
    summary = {
        "experiment": EXPERIMENT,
        "exp26_root": str(args.exp26_root),
        "exp23_root": str(args.exp23_root),
        "prompt_mode": args.prompt_mode,
        "event_kind": args.event_kind,
        "late_target": late_target,
        "models": models,
        "n_units": len(units),
        "by_model": {
            model: sum(1 for unit in units if unit.model == model and unit.readout == "common_it" and unit.variant == "noopp")
            for model in models
        },
        "primary": {
            variant: _row_lookup(rows, variant=variant, readout="common_it", scope="dense5", subset="all")
            for variant in PRIMARY_VARIANTS
        },
    }
    (args.out_dir / "exp26_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"out_dir": str(args.out_dir), "n_units": len(units), "rows": len(rows)}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp26 residual-opposition mediation records.")
    parser.add_argument("--exp26-root", type=Path, required=True)
    parser.add_argument("--exp23-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(DENSE5_MODELS))
    parser.add_argument("--prompt-mode", default="raw_shared")
    parser.add_argument("--event-kind", default="first_diff")
    parser.add_argument("--late-target", choices=["it", "pt"], default="it")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
