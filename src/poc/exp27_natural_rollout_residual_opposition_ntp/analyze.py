"""Analyze Exp27 natural-rollout residual-opposition records."""

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

from src.poc.exp27_natural_rollout_residual_opposition_ntp import DENSE5_MODELS, EXPERIMENT, RANDOM_VARIANTS


PRIMARY_VARIANTS = [
    "noopp",
    "normpres_noopp",
    "flipopp",
    "randorth",
    "randremove",
    "randremove_resnorm",
]
BOOT_METRICS = {
    "pt_nll_delta",
    "it_nll_delta",
    "it_minus_pt_nll_delta",
    "pt_true_logit_drop",
    "it_true_logit_drop",
    "it_minus_pt_true_logit_drop",
    "pt_nll_delta_per_opp_frac",
    "it_nll_delta_per_opp_frac",
    "it_minus_pt_nll_delta_per_opp_frac",
    "pt_true_logit_drop_per_opp_frac",
    "it_true_logit_drop_per_opp_frac",
    "it_minus_pt_true_logit_drop_per_opp_frac",
}


@dataclass(frozen=True)
class Unit:
    model: str
    model_variant: str
    prompt_id: str
    category: str
    variant: str
    seed: int | None
    generated_len: int
    n_positions: int
    nll_delta: float
    true_logit_drop: float
    mean_opp_norm_frac: float | None
    nll_delta_per_opp_frac: float | None
    true_logit_drop_per_opp_frac: float | None
    bucket: str = "all"
    seed_std_nll_delta: float | None = None
    seed_std_true_logit_drop: float | None = None


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _find_records(root: Path, model: str) -> Path:
    candidates = [
        root / "records" / model / "records.jsonl.gz",
        root / model / "records.jsonl.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No Exp27 records for {model} under {root}")


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _record_to_units(row: dict[str, Any]) -> list[Unit]:
    if not row.get("valid"):
        return []
    score = row.get("score") or {}
    nll_delta = _finite(score.get("nll_delta"))
    true_logit_drop = _finite(score.get("true_logit_drop"))
    if nll_delta is None or true_logit_drop is None:
        return []
    diagnostics = row.get("diagnostics") or {}
    base = {
        "model": str(row.get("model")),
        "model_variant": str(row.get("model_variant")),
        "prompt_id": str(row.get("prompt_id")),
        "category": str(row.get("category", "unknown")),
        "variant": str(row.get("variant")),
        "seed": None if row.get("seed") is None else int(row.get("seed")),
        "generated_len": int(row.get("generated_len", 0)),
        "mean_opp_norm_frac": _finite(diagnostics.get("mean_opp_norm_frac")),
        "nll_delta_per_opp_frac": _finite(row.get("nll_delta_per_opp_frac")),
        "true_logit_drop_per_opp_frac": _finite(row.get("true_logit_drop_per_opp_frac")),
    }
    units = [
        Unit(
            **base,
            n_positions=int(score.get("n_positions", 0)),
            nll_delta=nll_delta,
            true_logit_drop=true_logit_drop,
            bucket="all",
        )
    ]
    for bucket, payload in (score.get("position_buckets") or {}).items():
        b_nll = _finite(payload.get("nll_delta"))
        b_logit = _finite(payload.get("true_logit_drop"))
        if b_nll is None or b_logit is None:
            continue
        units.append(
            Unit(
                **base,
                n_positions=int(payload.get("n_positions", 0)),
                nll_delta=b_nll,
                true_logit_drop=b_logit,
                bucket=str(bucket),
            )
        )
    return units


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not kept:
        return None
    return sum(kept) / len(kept)


def load_units(root: Path, models: list[str]) -> list[Unit]:
    units: list[Unit] = []
    for model in models:
        path = _find_records(root, model)
        for row in _json_rows(path):
            units.extend(_record_to_units(row))
    return _add_random_aggregates(units)


def _add_random_aggregates(units: list[Unit]) -> list[Unit]:
    out = list(units)
    grouped: dict[tuple[str, str, str, str, str, str], list[Unit]] = defaultdict(list)
    for unit in units:
        if unit.variant in RANDOM_VARIANTS and unit.seed is not None:
            grouped[
                (unit.model, unit.model_variant, unit.prompt_id, unit.category, unit.variant, unit.bucket)
            ].append(unit)
    for group in grouped.values():
        if not group:
            continue
        first = group[0]
        nll_values = [unit.nll_delta for unit in group]
        logit_values = [unit.true_logit_drop for unit in group]
        out.append(
            Unit(
                model=first.model,
                model_variant=first.model_variant,
                prompt_id=first.prompt_id,
                category=first.category,
                variant=first.variant,
                seed=None,
                generated_len=first.generated_len,
                n_positions=first.n_positions,
                nll_delta=float(_mean(nll_values)),
                true_logit_drop=float(_mean(logit_values)),
                mean_opp_norm_frac=_mean([unit.mean_opp_norm_frac for unit in group]),
                nll_delta_per_opp_frac=_mean([unit.nll_delta_per_opp_frac for unit in group]),
                true_logit_drop_per_opp_frac=_mean(
                    [unit.true_logit_drop_per_opp_frac for unit in group]
                ),
                bucket=first.bucket,
                seed_std_nll_delta=float(statistics.pstdev(nll_values)),
                seed_std_true_logit_drop=float(statistics.pstdev(logit_values)),
            )
        )
    return out


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


def _estimate_side(units: list[Unit], *, scope: str, models: list[str]) -> dict[str, float | None]:
    per_model = []
    for model in _scope_models(scope, models):
        bucket = [unit for unit in units if unit.model == model]
        if not bucket:
            continue
        per_model.append(
            {
                "model": model,
                "n": len(bucket),
                "n_positions": sum(unit.n_positions for unit in bucket),
                "generated_len": _mean([unit.generated_len for unit in bucket]),
                "nll_delta": _mean([unit.nll_delta for unit in bucket]),
                "true_logit_drop": _mean([unit.true_logit_drop for unit in bucket]),
                "mean_opp_norm_frac": _mean([unit.mean_opp_norm_frac for unit in bucket]),
                "nll_delta_per_opp_frac": _mean([unit.nll_delta_per_opp_frac for unit in bucket]),
                "true_logit_drop_per_opp_frac": _mean(
                    [unit.true_logit_drop_per_opp_frac for unit in bucket]
                ),
                "seed_std_nll_delta": _mean([unit.seed_std_nll_delta for unit in bucket]),
                "seed_std_true_logit_drop": _mean([unit.seed_std_true_logit_drop for unit in bucket]),
            }
        )
    if not per_model:
        return {}
    reducer = statistics.median if scope == "family_median" else lambda vals: sum(vals) / len(vals)

    def reduce_key(key: str) -> float | None:
        vals = [row[key] for row in per_model if row.get(key) is not None]
        if not vals:
            return None
        return float(reducer([float(value) for value in vals]))

    return {
        "n_units": int(sum(row["n"] for row in per_model)),
        "n_positions": int(sum(row["n_positions"] for row in per_model)),
        "n_models": int(len(per_model)),
        "generated_len": reduce_key("generated_len"),
        "nll_delta": reduce_key("nll_delta"),
        "true_logit_drop": reduce_key("true_logit_drop"),
        "mean_opp_norm_frac": reduce_key("mean_opp_norm_frac"),
        "nll_delta_per_opp_frac": reduce_key("nll_delta_per_opp_frac"),
        "true_logit_drop_per_opp_frac": reduce_key("true_logit_drop_per_opp_frac"),
        "seed_std_nll_delta": reduce_key("seed_std_nll_delta"),
        "seed_std_true_logit_drop": reduce_key("seed_std_true_logit_drop"),
    }


def _estimate_compare(units: list[Unit], *, scope: str, models: list[str]) -> dict[str, float | None]:
    pt = _estimate_side([unit for unit in units if unit.model_variant == "pt"], scope=scope, models=models)
    it = _estimate_side([unit for unit in units if unit.model_variant == "it"], scope=scope, models=models)
    out: dict[str, float | None] = {}
    for prefix, payload in (("pt", pt), ("it", it)):
        for key, value in payload.items():
            out[f"{prefix}_{key}"] = value
    for key in ("nll_delta", "true_logit_drop", "nll_delta_per_opp_frac", "true_logit_drop_per_opp_frac"):
        if pt.get(key) is not None and it.get(key) is not None:
            out[f"it_minus_pt_{key}"] = float(it[key]) - float(pt[key])
        else:
            out[f"it_minus_pt_{key}"] = None
    return out


def _bootstrap_compare(
    units: list[Unit],
    *,
    scope: str,
    models: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, float | None]:
    est = _estimate_compare(units, scope=scope, models=models)
    if n_boot <= 0:
        return est
    rng = np.random.default_rng(seed)
    by_model_prompt: dict[str, dict[str, list[Unit]]] = {}
    for model in _scope_models(scope, models):
        prompt_clusters: dict[str, list[Unit]] = defaultdict(list)
        for unit in units:
            if unit.model == model:
                prompt_clusters[unit.prompt_id].append(unit)
        by_model_prompt[model] = dict(prompt_clusters)
    boot_values: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        sampled = []
        for prompt_clusters in by_model_prompt.values():
            if not prompt_clusters:
                continue
            prompt_ids = list(prompt_clusters)
            picked = rng.choice(prompt_ids, size=len(prompt_ids), replace=True)
            for prompt_id in picked:
                sampled.extend(prompt_clusters[str(prompt_id)])
        boot_est = _estimate_compare(sampled, scope=scope, models=models)
        for key, value in boot_est.items():
            if value is not None and key in BOOT_METRICS:
                boot_values[key].append(float(value))
    for key, values in boot_values.items():
        est[f"{key}_ci_low"] = float(np.quantile(values, 0.025))
        est[f"{key}_ci_high"] = float(np.quantile(values, 0.975))
    return est


def _rows_for_units(units: list[Unit], *, models: list[str], n_boot: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants = sorted({unit.variant for unit in units if unit.seed is None and unit.variant != "full"})
    buckets = sorted({unit.bucket for unit in units})
    categories = ["all"] + sorted({unit.category for unit in units})
    scopes = ["dense5", "dense4_no_gemma", "family_median"] + [f"model:{model}" for model in models]
    for variant in variants:
        if variant not in PRIMARY_VARIANTS:
            continue
        variant_units = [unit for unit in units if unit.variant == variant and unit.seed is None]
        for bucket in buckets:
            bucket_units = [unit for unit in variant_units if unit.bucket == bucket]
            if not bucket_units:
                continue
            for category in categories:
                category_units = (
                    bucket_units if category == "all" else [unit for unit in bucket_units if unit.category == category]
                )
                if not category_units:
                    continue
                for scope in scopes:
                    scope_units = [unit for unit in category_units if unit.model in _scope_models(scope, models)]
                    if not scope_units:
                        continue
                    est = _bootstrap_compare(scope_units, scope=scope, models=models, n_boot=n_boot, seed=seed)
                    rows.append({"variant": variant, "bucket": bucket, "category": category, "scope": scope, **est})
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


def _row_lookup(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    scope: str = "dense5",
    bucket: str = "all",
    category: str = "all",
) -> dict[str, Any] | None:
    for row in rows:
        if row["variant"] == variant and row["scope"] == scope and row["bucket"] == bucket and row["category"] == category:
            return row
    return None


def _plot(rows: list[dict[str, Any]], out_path: Path) -> None:
    kept = [
        row
        for row in rows
        if row["scope"] == "dense5" and row["bucket"] == "all" and row["category"] == "all"
    ]
    order = [variant for variant in PRIMARY_VARIANTS if any(row["variant"] == variant for row in kept)]
    if not order:
        return
    x = np.arange(len(order))
    width = 0.36
    pt_vals = []
    it_vals = []
    pt_err = [[], []]
    it_err = [[], []]
    for variant in order:
        row = _row_lookup(kept, variant=variant)
        if row is None:
            continue
        pt = float(row.get("pt_nll_delta") or 0.0)
        it = float(row.get("it_nll_delta") or 0.0)
        pt_vals.append(pt)
        it_vals.append(it)
        pt_low = row.get("pt_nll_delta_ci_low")
        pt_high = row.get("pt_nll_delta_ci_high")
        it_low = row.get("it_nll_delta_ci_low")
        it_high = row.get("it_nll_delta_ci_high")
        pt_err[0].append(pt - float(pt_low) if pt_low is not None else 0.0)
        pt_err[1].append(float(pt_high) - pt if pt_high is not None else 0.0)
        it_err[0].append(it - float(it_low) if it_low is not None else 0.0)
        it_err[1].append(float(it_high) - it if it_high is not None else 0.0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, pt_vals, width, label="PT", color="#7a869a")
    ax.bar(x + width / 2, it_vals, width, label="IT", color="#386fa4")
    ax.errorbar(x - width / 2, pt_vals, yerr=np.array(pt_err), fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax.errorbar(x + width / 2, it_vals, yerr=np.array(it_err), fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("NLL increase on own greedy continuation")
    ax.set_title("Exp27 natural-rollout residual-opposition ablations")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_report(path: Path, rows: list[dict[str, Any]], units: list[Unit]) -> None:
    def fmt(row: dict[str, Any] | None, key: str) -> str:
        if row is None or row.get(key) is None:
            return "NA"
        ci_low = row.get(f"{key}_ci_low")
        ci_high = row.get(f"{key}_ci_high")
        ci = "" if ci_low is None else f" [{ci_low:+.4f}, {ci_high:+.4f}]"
        return f"{row[key]:+.4f}{ci}"

    noopp = _row_lookup(rows, variant="noopp")
    norm = _row_lookup(rows, variant="normpres_noopp")
    flip = _row_lookup(rows, variant="flipopp")
    randremove = _row_lookup(rows, variant="randremove")
    randres = _row_lookup(rows, variant="randremove_resnorm")
    lines = [
        "# Exp27 Natural-Rollout Report",
        "",
        f"- Unit rows analyzed: `{len(units)}` after random-seed aggregation.",
        f"- `noopp` PT NLL hurt: `{fmt(noopp, 'pt_nll_delta')}`; IT NLL hurt: `{fmt(noopp, 'it_nll_delta')}`; IT-PT: `{fmt(noopp, 'it_minus_pt_nll_delta')}`.",
        f"- `normpres_noopp` IT-PT NLL hurt: `{fmt(norm, 'it_minus_pt_nll_delta')}`.",
        f"- `flipopp` IT-PT NLL hurt: `{fmt(flip, 'it_minus_pt_nll_delta')}`.",
        f"- `randremove` IT-PT NLL hurt: `{fmt(randremove, 'it_minus_pt_nll_delta')}`.",
        f"- `randremove_resnorm` IT-PT NLL hurt: `{fmt(randres, 'it_minus_pt_nll_delta')}`.",
        "",
        "Interpretation:",
        "- Positive NLL hurt means the ablation worsens prediction of the model's own greedy continuation.",
        "- `normpres_noopp` controls MLP-update length; `randremove` controls same-magnitude random removal; `randremove_resnorm` also matches the post-MLP residual norm shift.",
        "- This is a natural-rollout importance measurement, not a same-prefix PT/IT causal comparison.",
    ]
    path.write_text("\n".join(lines) + "\n")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    units = load_units(args.exp27_root, args.models)
    if not units:
        raise RuntimeError("No Exp27 units loaded")
    rows = _rows_for_units(units, models=args.models, n_boot=args.n_boot, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "exp27_effects.csv", rows)
    _write_csv(
        args.out_dir / "exp27_family_table.csv",
        [row for row in rows if str(row.get("scope", "")).startswith("model:")],
    )
    _write_csv(
        args.out_dir / "exp27_category_table.csv",
        [row for row in rows if row.get("category") != "all" and row.get("bucket") == "all"],
    )
    _write_csv(
        args.out_dir / "exp27_position_bucket_table.csv",
        [row for row in rows if row.get("bucket") != "all" and row.get("category") == "all"],
    )
    _plot(rows, args.out_dir / "exp27_natural_rollout_ntp.png")
    _write_report(args.out_dir / "exp27_report.md", rows, units)
    summary = {
        "experiment": EXPERIMENT,
        "exp27_root": str(args.exp27_root),
        "models": args.models,
        "n_units": len(units),
        "primary": {variant: _row_lookup(rows, variant=variant) for variant in PRIMARY_VARIANTS},
    }
    (args.out_dir / "exp27_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"out_dir": str(args.out_dir), "n_units": len(units), "rows": len(rows)}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp27 natural-rollout residual-opposition records.")
    parser.add_argument("--exp27-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(DENSE5_MODELS))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
