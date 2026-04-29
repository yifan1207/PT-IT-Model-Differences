#!/usr/bin/env python3
"""Build paper-facing Exp25 OLMo stage-progression artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.cross_model.config import get_spec, model_id_for_variant
from src.poc.exp25_olmo_stage_progression import STAGE_LABELS, STAGE_MODELS


TRANSITIONS = {
    "olmo2_7b_pt_sft": ("PT", "SFT"),
    "olmo2_7b_sft_dpo": ("SFT", "DPO"),
    "olmo2_7b_dpo_rlvr": ("DPO", "RLVR"),
    "olmo2_7b": ("PT", "RLVR"),
}


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


def _load_exp20_records(exp20_root: Path, prompt_mode: str, model: str) -> list[dict[str, Any]]:
    path = exp20_root / prompt_mode / model / "exp20_validation_records.jsonl"
    if not path.exists():
        return []
    return list(_json_rows(path))


def _first_diff(record: dict[str, Any]) -> dict[str, Any]:
    payload = (record.get("readouts") or {}).get("first_diff")
    return payload if isinstance(payload, dict) else {}


def _class_label(record: dict[str, Any], condition: str) -> str | None:
    value = ((_first_diff(record).get("condition_token_at_step") or {}).get(condition) or {}).get("class")
    return str(value) if value is not None else None


def _class_rate(records: list[dict[str, Any]], condition: str, target: str) -> float | None:
    values = []
    for record in records:
        label = _class_label(record, condition)
        if label is not None and label != "missing":
            values.append(1.0 if label == target else 0.0)
    return float(np.mean(values)) if values else None


def _margin(record: dict[str, Any], condition: str, window: str = "late_reconciliation") -> float | None:
    payload = ((_first_diff(record).get("conditions") or {}).get(condition) or {})
    metric = (
        ((payload.get("windows") or {}).get(window) or {})
        .get("it_minus_pt_margin", {})
        .get("total_delta")
    )
    return _finite(metric)


def _margin_mean(records: list[dict[str, Any]], condition: str) -> float | None:
    values = [_margin(record, condition) for record in records]
    kept = [float(value) for value in values if value is not None]
    return float(np.mean(kept)) if kept else None


def _event_count(records: list[dict[str, Any]], kind: str = "first_diff") -> int:
    return sum(1 for record in records if isinstance((record.get("readouts") or {}).get(kind), dict))


def _exp20_summary(exp20_root: Path, prompt_mode: str, model: str) -> dict[str, Any]:
    records = _load_exp20_records(exp20_root, prompt_mode, model)
    return {
        "records": len(records),
        "first_diff_events": _event_count(records, "first_diff"),
        "first_nonformat_events": _event_count(records, "first_nonformat_diff"),
        "first_assistant_marker_events": _event_count(records, "first_assistant_marker_diff"),
        "b_mid_it_match_rate": _class_rate(records, "B_mid_raw", "it"),
        "b_late_it_match_rate": _class_rate(records, "B_late_raw", "it"),
        "b_midlate_it_match_rate": _class_rate(records, "B_midlate_raw", "it"),
        "d_mid_pt_match_rate": _class_rate(records, "D_mid_ptswap", "pt"),
        "d_late_pt_match_rate": _class_rate(records, "D_late_ptswap", "pt"),
        "c_late_margin_mean": _margin_mean(records, "C_it_chat"),
        "b_late_margin_mean": _margin_mean(records, "B_late_raw"),
        "b_midlate_margin_mean": _margin_mean(records, "B_midlate_raw"),
        "d_late_margin_mean": _margin_mean(records, "D_late_ptswap"),
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _effect(summary: dict[str, Any] | None, model: str, name: str, readout: str = "common_it") -> dict[str, Any]:
    if not summary:
        return {}
    payload = (
        ((summary.get("residual_factorial") or {}).get("effects") or {})
        .get(readout, {})
        .get(name, {})
    )
    model_payload = (payload.get("model_cis") or {}).get(model)
    return model_payload if isinstance(model_payload, dict) else {}


def _compatibility_payload(compat_root: Path | None, model: str) -> dict[str, Any]:
    if compat_root is None:
        return {}
    candidates = [
        compat_root / model / "exp23_compatibility_permutation_summary.json",
        compat_root / f"{model}_compatibility_permutation_summary.json",
        compat_root / "exp23_compatibility_permutation_summary.json",
    ]
    for path in candidates:
        payload = _load_json(path)
        if payload is None:
            continue
        models = payload.get("models") or []
        if len(models) == 1 and models[0] != model:
            continue
        return payload
    return {}


def _compat_p_upper(payload: dict[str, Any]) -> float | None:
    return _finite((payload.get("permutation") or {}).get("p_upper"))


def build_summary(
    *,
    exp20_root: Path,
    exp23_summary: Path,
    compatibility_root: Path | None,
    out_dir: Path,
    models: list[str],
    prompt_mode: str,
) -> dict[str, Any]:
    exp23 = _load_json(exp23_summary)
    rows: list[dict[str, Any]] = []
    for model in models:
        spec = get_spec(model)
        earlier, later = TRANSITIONS.get(model, ("pt", "it"))
        exp20 = _exp20_summary(exp20_root, prompt_mode, model)
        interaction = _effect(exp23, model, "interaction")
        late_pt = _effect(exp23, model, "late_it_given_pt_upstream")
        late_it = _effect(exp23, model, "late_it_given_it_upstream")
        late_weight = _effect(exp23, model, "late_weight_effect")
        upstream = _effect(exp23, model, "upstream_context_effect")
        compat = _compatibility_payload(compatibility_root, model)
        row = {
            "model": model,
            "transition": STAGE_LABELS.get(model, model),
            "earlier_stage": earlier,
            "later_stage": later,
            "earlier_repo": model_id_for_variant(spec, "pt"),
            "later_repo": model_id_for_variant(spec, "it"),
            **exp20,
            "interaction": interaction.get("estimate"),
            "interaction_ci_low": interaction.get("ci95_low"),
            "interaction_ci_high": interaction.get("ci95_high"),
            "late_effect_pt_upstream": late_pt.get("estimate"),
            "late_effect_pt_upstream_ci_low": late_pt.get("ci95_low"),
            "late_effect_pt_upstream_ci_high": late_pt.get("ci95_high"),
            "late_effect_later_upstream": late_it.get("estimate"),
            "late_effect_later_upstream_ci_low": late_it.get("ci95_low"),
            "late_effect_later_upstream_ci_high": late_it.get("ci95_high"),
            "late_weight_effect": late_weight.get("estimate"),
            "upstream_context_effect": upstream.get("estimate"),
            "exp23_units": interaction.get("n_units"),
            "exp23_prompt_clusters": interaction.get("n_prompt_clusters"),
            "label_swap_p_upper": _compat_p_upper(compat),
        }
        rows.append(row)

    summary = {
        "experiment": "exp25_olmo_stage_progression",
        "exp20_root": str(exp20_root),
        "exp23_summary": str(exp23_summary),
        "compatibility_root": str(compatibility_root) if compatibility_root else None,
        "prompt_mode": prompt_mode,
        "models": models,
        "interpretation_note": (
            "Adjacent-stage estimates condition on each transition's own first-divergence "
            "events and token pair. They are local transition estimates, not an additive "
            "decomposition of PT->RLVR."
        ),
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "olmo_stage_progression_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_csv(rows, out_dir / "olmo_stage_progression_table.csv")
    _plot(rows, out_dir / "olmo_stage_progression.png")
    return summary


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _as_float(values: list[Any]) -> np.ndarray:
    return np.array([float(v) if v is not None else np.nan for v in values], dtype=float)


def _plot(rows: list[dict[str, Any]], out_path: Path) -> None:
    labels = [row["transition"] for row in rows]
    x = np.arange(len(rows))
    interaction = _as_float([row.get("interaction") for row in rows])
    interaction_lo = _as_float([row.get("interaction_ci_low") for row in rows])
    interaction_hi = _as_float([row.get("interaction_ci_high") for row in rows])
    late_pt = _as_float([row.get("late_effect_pt_upstream") for row in rows])
    late_it = _as_float([row.get("late_effect_later_upstream") for row in rows])
    mid_rate = _as_float([row.get("b_mid_it_match_rate") for row in rows])
    late_rate = _as_float([row.get("b_late_it_match_rate") for row in rows])
    counts = _as_float([row.get("first_diff_events") for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    axes[0].bar(x, interaction, color="#4C78A8")
    yerr = np.vstack([
        np.nan_to_num(interaction - interaction_lo, nan=0.0),
        np.nan_to_num(interaction_hi - interaction, nan=0.0),
    ])
    axes[0].errorbar(x, interaction, yerr=yerr, fmt="none", color="black", capsize=4)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Exp23 upstream x late-stack interaction")
    axes[0].set_ylabel("IT-vs-earlier-stage margin, logits")

    width = 0.36
    axes[1].bar(x - width / 2, late_pt, width=width, label="late effect from earlier upstream", color="#F58518")
    axes[1].bar(x + width / 2, late_it, width=width, label="late effect from later upstream", color="#54A24B")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Context-gated late-stack effect")
    axes[1].set_ylabel("Margin shift, logits")
    axes[1].legend(fontsize=8)

    axes[2].bar(x - width / 2, mid_rate, width=width, label="PT host + later mid", color="#72B7B2")
    axes[2].bar(x + width / 2, late_rate, width=width, label="PT host + later late", color="#B279A2")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Exp20 IT-token transfer proxy")
    axes[2].set_ylabel("P(top-1 matches later-stage token)")
    axes[2].legend(fontsize=8)

    axes[3].bar(x, counts, color="#9D755D")
    axes[3].set_title("First-difference event counts")
    axes[3].set_ylabel("records")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Exp25 OLMo 2 7B stage progression: local transition estimates", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp20-root", type=Path, required=True)
    parser.add_argument("--exp23-summary", type=Path, required=True)
    parser.add_argument("--compatibility-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_synthesis"))
    parser.add_argument("--models", nargs="*", default=STAGE_MODELS)
    parser.add_argument("--prompt-mode", default="raw_shared")
    args = parser.parse_args()
    summary = build_summary(
        exp20_root=args.exp20_root,
        exp23_summary=args.exp23_summary,
        compatibility_root=args.compatibility_root,
        out_dir=args.out_dir,
        models=list(args.models),
        prompt_mode=args.prompt_mode,
    )
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(summary["rows"])}, indent=2))


if __name__ == "__main__":
    main()
