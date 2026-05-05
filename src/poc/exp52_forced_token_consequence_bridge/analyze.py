"""Analyze Exp52 forced-token consequence records."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.poc.exp45_behavioral_bridge.metrics import family_balanced_ci
from src.poc.exp52_forced_token_consequence_bridge import DEFAULT_BRANCHES, DEFAULT_MODELS
from src.poc.exp52_forced_token_consequence_bridge.validators import validator_coverage


def _json_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_records(root: Path, models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        path = root / model / "forced_token_records.jsonl.gz"
        if not path.exists():
            paths = sorted((root / model).glob("forced_token_records_w*.jsonl.gz"))
        else:
            paths = [path]
        for one in paths:
            rows.extend(_json_rows(one))
    return rows


def _branch_score_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in records:
        base = {
            "model": row.get("model"),
            "prompt_id": row.get("prompt_id"),
            "event_kind": row.get("event_kind"),
            "category": row.get("category"),
            "source": row.get("source"),
            "position": row.get("position"),
            "position_ge_3": row.get("position_ge_3"),
            "rank_IT_t_PT": (row.get("branch_plausibility") or {}).get("rank_IT_t_PT"),
            "logprob_gap_IT_tIT_minus_tPT": (row.get("branch_plausibility") or {}).get("logprob_gap_IT_tIT_minus_tPT"),
            "plausible_branch": (row.get("branch_plausibility") or {}).get("plausible_rank50_or_gap5"),
            "C_event": (row.get("mechanism") or {}).get("C_event"),
            "P_event": (row.get("mechanism") or {}).get("P_event"),
            "M_event": (row.get("mechanism") or {}).get("M_event"),
        }
        for branch, payload in (row.get("branches") or {}).items():
            if not payload.get("available"):
                continue
            token_meta = (row.get("branch_token_classes") or {}).get(branch) or {}
            for view, view_payload in (payload.get("scores") or {}).items():
                primary = (view_payload or {}).get("primary") or {}
                lexical = (view_payload or {}).get("lexical") or {}
                out.append(
                    {
                        **base,
                        "branch": branch,
                        "view": view,
                        "scoreable": bool(primary.get("scoreable")),
                        "success": _success_float(primary.get("success")),
                        "criteria_type": primary.get("criteria_type"),
                        "expected_behavior": primary.get("expected_behavior"),
                        "forced_token_id": payload.get("forced_token_id"),
                        "forced_token_text": payload.get("forced_token_text"),
                        "branch_rank_IT": token_meta.get("rank_IT"),
                        "branch_logprob_IT": token_meta.get("logprob_IT"),
                        "branch_token_category": token_meta.get("token_category"),
                        "branch_length_bucket": token_meta.get("length_bucket"),
                        "including_token_count": payload.get("including_token_count"),
                        "suffix_token_count": payload.get("suffix_token_count"),
                        "char_len": lexical.get("char_len"),
                        "word_count": lexical.get("word_count"),
                        "refusal_marker": lexical.get("refusal_marker"),
                        "unsafe_marker": lexical.get("unsafe_marker"),
                    }
                )
    return out


def _success_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(bool(value))


def _delta_rows(branch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_event_view: dict[tuple[str, str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in branch_rows:
        key = (
            str(row.get("model")),
            str(row.get("prompt_id")),
            str(row.get("event_kind")),
            str(row.get("view")),
            str(row.get("category")),
        )
        by_event_view.setdefault(key, {})[str(row.get("branch"))] = row
    pairs = {
        "it_minus_pt": ("it_branch", "pt_branch"),
        "it_minus_rank_matched_alt": ("it_branch", "it_rank_matched_alt"),
        "it_minus_token_class_matched_alt": ("it_branch", "token_class_matched_alt"),
        "pt_minus_rank_matched_alt": ("pt_branch", "it_rank_matched_alt"),
    }
    out: list[dict[str, Any]] = []
    for (_model, _prompt, _kind, _view, _category), branches in by_event_view.items():
        for contrast, (lhs, rhs) in pairs.items():
            if lhs not in branches or rhs not in branches:
                continue
            a = branches[lhs]
            b = branches[rhs]
            if not a.get("scoreable") or not b.get("scoreable"):
                continue
            av = _float(a.get("success"))
            bv = _float(b.get("success"))
            if av is None or bv is None:
                continue
            out.append(
                {
                    "model": a.get("model"),
                    "prompt_id": a.get("prompt_id"),
                    "event_kind": a.get("event_kind"),
                    "category": a.get("category"),
                    "source": a.get("source"),
                    "position": a.get("position"),
                    "position_ge_3": a.get("position_ge_3"),
                    "view": a.get("view"),
                    "contrast": contrast,
                    "delta_success": av - bv,
                    "lhs_success": av,
                    "rhs_success": bv,
                    "plausible_branch": a.get("plausible_branch"),
                    "rank_IT_t_PT": a.get("rank_IT_t_PT"),
                    "logprob_gap_IT_tIT_minus_tPT": a.get("logprob_gap_IT_tIT_minus_tPT"),
                    "C_event": a.get("C_event"),
                    "P_event": a.get("P_event"),
                    "M_event": a.get("M_event"),
                    "first_token_sensitive": None,
                }
            )
    return out


def _annotate_first_token_sensitivity(deltas: list[dict[str, Any]]) -> None:
    by_key: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in deltas:
        key = (str(row.get("model")), str(row.get("prompt_id")), str(row.get("event_kind")), str(row.get("contrast")))
        by_key.setdefault(key, {})[str(row.get("view"))] = row
    for views in by_key.values():
        inc = views.get("including_forced_token")
        suffix = views.get("suffix_only")
        if not inc or not suffix:
            continue
        sensitive = float(abs(float(inc["delta_success"]) - float(suffix["delta_success"])) > 0.5)
        inc["first_token_sensitive"] = sensitive
        suffix["first_token_sensitive"] = sensitive


def _aggregate(deltas: list[dict[str, Any]], models: list[str], n_boot: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_specs: list[tuple[str, list[str]]] = [
        ("all", []),
        ("category", ["category"]),
        ("model", ["model"]),
        ("category_model", ["category", "model"]),
        ("category_plausible", ["category", "plausible_branch"]),
        ("category_position_ge3", ["category", "position_ge_3"]),
    ]
    for group_name, cols in group_specs:
        if cols:
            df = pd.DataFrame(deltas)
            if df.empty:
                continue
            groups = df.groupby(cols + ["view", "contrast"], dropna=False)
            iterator = ((key if isinstance(key, tuple) else (key,), group.to_dict("records")) for key, group in groups)
        else:
            keys = sorted({(str(r.get("view")), str(r.get("contrast"))) for r in deltas})
            iterator = (((view, contrast), [r for r in deltas if str(r.get("view")) == view and str(r.get("contrast")) == contrast]) for view, contrast in keys)
        for key, subset in iterator:
            if not subset:
                continue
            if cols:
                label_values = list(key[: len(cols)])
                view = key[len(cols)]
                contrast = key[len(cols) + 1]
            else:
                label_values = []
                view, contrast = key
            ci = family_balanced_ci(subset, "delta_success", models=models, n_boot=n_boot)
            record = {
                "group": group_name,
                "view": view,
                "contrast": contrast,
                "n": len(subset),
                **ci,
            }
            for col, value in zip(cols, label_values):
                record[col] = value
            rows.append(record)
    return rows


def _quartile_rows(deltas: list[dict[str, Any]], models: list[str], n_boot: int) -> list[dict[str, Any]]:
    df = pd.DataFrame([r for r in deltas if _float(r.get("C_event")) is not None])
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (category, view, contrast), group in df.groupby(["category", "view", "contrast"], dropna=False):
        if len(group) < 16:
            continue
        try:
            group = group.copy()
            group["C_event_quartile"] = pd.qcut(group["C_event"].astype(float), 4, labels=False, duplicates="drop")
        except Exception:
            continue
        for quartile, subset_df in group.groupby("C_event_quartile"):
            subset = subset_df.to_dict("records")
            ci = family_balanced_ci(subset, "delta_success", models=models, n_boot=n_boot)
            rows.append(
                {
                    "category": category,
                    "view": view,
                    "contrast": contrast,
                    "C_event_quartile": int(quartile),
                    "n": len(subset),
                    **ci,
                }
            )
    return rows


def _plot_primary(agg: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = agg[
        (agg["group"] == "category")
        & (agg["view"] == "suffix_only")
        & (agg["contrast"].isin(["it_minus_pt", "it_minus_rank_matched_alt", "it_minus_token_class_matched_alt"]))
    ].copy()
    if subset.empty:
        return
    cats = [c for c in ["GOV-FORMAT", "SAFETY", "CONTENT-REASON", "GOV-CONV"] if c in set(subset["category"])]
    contrasts = ["it_minus_pt", "it_minus_rank_matched_alt", "it_minus_token_class_matched_alt"]
    x = np.arange(len(cats), dtype=float)
    width = 0.24
    plt.figure(figsize=(9, 4.8))
    for i, contrast in enumerate(contrasts):
        vals = []
        yerr_lo = []
        yerr_hi = []
        for cat in cats:
            row = subset[(subset["category"] == cat) & (subset["contrast"] == contrast)]
            if row.empty:
                vals.append(0.0)
                yerr_lo.append(0.0)
                yerr_hi.append(0.0)
                continue
            one = row.iloc[0]
            est = float(one.get("estimate") or 0.0)
            lo = float(one.get("ci_low") if one.get("ci_low") is not None else est)
            hi = float(one.get("ci_high") if one.get("ci_high") is not None else est)
            vals.append(est)
            yerr_lo.append(max(est - lo, 0.0))
            yerr_hi.append(max(hi - est, 0.0))
        plt.bar(x + (i - 1) * width, vals, width=width, yerr=[yerr_lo, yerr_hi], capsize=3, label=contrast)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Success-rate difference")
    plt.xlabel("Prompt category")
    plt.xticks(x, cats, rotation=20, ha="right")
    plt.title("Exp52 forced-token consequence bridge (suffix only)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "exp52_suffix_only_objective_deltas.png", dpi=180)
    plt.close()


def analyze(args: argparse.Namespace) -> None:
    models = args.models.split()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(Path(args.records_dir), models)
    branch_rows = _branch_score_rows(records)
    deltas = _delta_rows(branch_rows)
    _annotate_first_token_sensitivity(deltas)
    _write_csv(out_dir / "branch_scores.csv", branch_rows)
    _write_csv(out_dir / "branch_deltas.csv", deltas)
    agg = _aggregate(deltas, models=models, n_boot=int(args.n_boot))
    quartiles = _quartile_rows(deltas, models=models, n_boot=int(args.n_boot))
    _write_csv(out_dir / "aggregate_effects.csv", agg)
    _write_csv(out_dir / "c_event_quartiles.csv", quartiles)
    if agg:
        _plot_primary(pd.DataFrame(agg), out_dir / "plots")

    dataset_records: list[dict[str, Any]] = []
    if args.dataset and Path(args.dataset).exists():
        dataset_records = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    primary = [
        r
        for r in agg
        if r.get("group") == "category"
        and r.get("category") == "GOV-FORMAT"
        and r.get("view") == "suffix_only"
        and r.get("contrast") == "it_minus_pt"
    ]
    malformed = [r for r in records if not r.get("valid", True)]
    first_sensitive = [
        r
        for r in deltas
        if r.get("contrast") == "it_minus_pt"
        and r.get("view") == "suffix_only"
        and _float(r.get("first_token_sensitive")) == 1.0
    ]
    summary = {
        "models": models,
        "n_records": len(records),
        "n_invalid_or_malformed": len(malformed),
        "invalid_fraction": len(malformed) / max(len(records), 1),
        "n_branch_score_rows": len(branch_rows),
        "n_delta_rows": len(deltas),
        "branch_counts": dict(sorted(Counter(str(r.get("branch")) for r in branch_rows).items())),
        "category_counts": dict(sorted(Counter(str(r.get("category")) for r in records).items())),
        "validator_coverage": validator_coverage(dataset_records) if dataset_records else {},
        "primary_gov_format_suffix_it_minus_pt": primary[0] if primary else None,
        "first_token_sensitive_rate_suffix_it_minus_pt": len(first_sensitive)
        / max(
            sum(
                1
                for r in deltas
                if r.get("contrast") == "it_minus_pt" and r.get("view") == "suffix_only"
            ),
            1,
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    report = [
        "# Exp52 Forced-Token Consequence Bridge",
        "",
        f"- Records: {summary['n_records']}",
        f"- Invalid/malformed fraction: {summary['invalid_fraction']:.4f}",
        f"- Branch score rows: {summary['n_branch_score_rows']}",
        f"- Delta rows: {summary['n_delta_rows']}",
        f"- Category counts: {summary['category_counts']}",
        f"- Primary GOV-FORMAT suffix-only IT-minus-PT: {summary['primary_gov_format_suffix_it_minus_pt']}",
        "",
        "Primary paper row is `aggregate_effects.csv` with `group=category`, "
        "`category=GOV-FORMAT`, `view=suffix_only`, `contrast=it_minus_pt`.",
    ]
    (out_dir / "paper_claims_exp52.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--records-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="data/eval_dataset_v2.jsonl")
    parser.add_argument("--models", default=" ".join(DEFAULT_MODELS))
    parser.add_argument("--n-boot", type=int, default=1000)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(description=__doc__)
        add_args(parser)
        args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()

