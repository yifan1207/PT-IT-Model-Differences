#!/usr/bin/env python
"""Build paper-facing artifacts for the Exp22 fixed-history template audit."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/paper_synthesis"


def _load_summary(run_root: Path) -> dict[str, Any]:
    path = run_root / "analysis/summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _effect(
    summary: dict[str, Any],
    *,
    teacher_source: str,
    comparison: str,
    method: str,
    probe: str,
    outcome: str,
) -> dict[str, Any]:
    for row in summary["controlled_estimates"]:
        if (
            row.get("teacher_source", "it_native") == teacher_source
            and row["comparison"] == comparison
            and row["method"] == method
            and row["probe_family"] == probe
            and row["outcome"] == outcome
        ):
            return row
    raise KeyError((comparison, method, probe, outcome))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    summary = _load_summary(args.run_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    effect_specs = [
        ("native_fixed_raw_late_kl", "native_fixed_effect", "paired_same_prompt_step", "raw", "late_kl_mean"),
        ("native_fixed_tuned_late_kl", "native_fixed_effect", "paired_same_prompt_step", "tuned", "late_kl_mean"),
        ("raw_fixed_raw_late_kl", "raw_fixed_effect", "paired_same_prompt_step", "raw", "late_kl_mean"),
        ("raw_fixed_tuned_late_kl", "raw_fixed_effect", "paired_same_prompt_step", "tuned", "late_kl_mean"),
        ("template_delta_raw_late_kl", "template_delta", "paired_same_prompt_step", "raw", "late_kl_mean"),
        ("template_delta_tuned_late_kl", "template_delta", "paired_same_prompt_step", "tuned", "late_kl_mean"),
        ("native_fixed_cem_raw_late_kl", "native_fixed_effect", "cem", "raw", "late_kl_mean"),
        ("raw_fixed_cem_raw_late_kl", "raw_fixed_effect", "cem", "raw", "late_kl_mean"),
        ("template_delta_cem_raw_late_kl", "template_delta", "cem", "raw", "late_kl_mean"),
        ("native_fixed_future_top1_flips", "native_fixed_effect", "paired_same_prompt_step", "pooled_probe_fe", "future_top1_flips"),
        ("raw_fixed_future_top1_flips", "raw_fixed_effect", "paired_same_prompt_step", "pooled_probe_fe", "future_top1_flips"),
        ("template_delta_future_top1_flips", "template_delta", "paired_same_prompt_step", "pooled_probe_fe", "future_top1_flips"),
    ]
    teacher_sources = summary.get("teacher_sources") or ["it_native"]
    effect_rows: list[dict[str, Any]] = []
    for teacher_source in teacher_sources:
        for key, comparison, method, probe, outcome in effect_specs:
            try:
                row = _effect(
                    summary,
                    teacher_source=teacher_source,
                    comparison=comparison,
                    method=method,
                    probe=probe,
                    outcome=outcome,
                )
            except KeyError:
                continue
            effect_rows.append(
                {
                    "key": f"{teacher_source}_{key}",
                    "teacher_source": teacher_source,
                    "comparison": comparison,
                    "method": method,
                    "probe_family": probe,
                    "outcome": outcome,
                    "estimate": float(row["estimate"]),
                    "ci95_low": float(row["ci95_low"]),
                    "ci95_high": float(row["ci95_high"]),
                    "models": row["models"],
                    "n_models": int(row["n_models"]),
                }
            )

    effects_csv = args.out_dir / "exp22_fixed_history_template_audit_effects.csv"
    support_csv = args.out_dir / "exp22_fixed_history_template_audit_support.csv"
    _write_csv(effects_csv, effect_rows)
    _write_csv(support_csv, summary.get("support", []))

    plot_src = Path(summary["artifacts"].get("plot", ""))
    plot_dest = args.out_dir / "exp22_fixed_history_template_audit.png"
    if plot_src.exists():
        shutil.copy2(plot_src, plot_dest)

    by_key = {row["key"]: row for row in effect_rows}
    quality = summary["quality"]
    note_blocks = []
    for teacher_source in teacher_sources:
        native = by_key.get(f"{teacher_source}_native_fixed_raw_late_kl", {})
        raw = by_key.get(f"{teacher_source}_raw_fixed_raw_late_kl", {})
        delta = by_key.get(f"{teacher_source}_template_delta_raw_late_kl", {})
        note_blocks.append(
            f"""Teacher source: `{teacher_source}`
- Native fixed effect (`it_native - pt_raw`): `{native.get('estimate', float('nan')):.3f}` nats, 95% CI `[{native.get('ci95_low', float('nan')):.3f}, {native.get('ci95_high', float('nan')):.3f}]`.
- Raw/no-template fixed effect (`it_raw - pt_raw`): `{raw.get('estimate', float('nan')):.3f}` nats, 95% CI `[{raw.get('ci95_low', float('nan')):.3f}, {raw.get('ci95_high', float('nan')):.3f}]`.
- Template delta (`it_native - it_raw`): `{delta.get('estimate', float('nan')):.3f}` nats, 95% CI `[{delta.get('ci95_low', float('nan')):.3f}, {delta.get('ci95_high', float('nan')):.3f}]`."""
        )
    note_block = "\n\n".join(note_blocks)
    note = f"""# Exp22 Fixed-History Template Audit

Run: `{args.run_root.name}`.

This audit generates one greedy teacher continuation per prompt and replays the same token history through PT raw, IT native-chat, and IT raw/no-template cells. Teacher source is explicit so the IT-native and PT-raw mirrors are never pooled unless requested.

Primary paired raw-lens late-KL results:
{note_block}

Quality gates: max malformed rate `{quality['max_malformed_rate']:.4f}`, missing aligned step rows `{quality['missing_aligned_step_rows']}`, minimum CEM retention `{quality['min_retained_fraction']:.3f}`, and maximum post-match SMD `{quality['max_smd_after']:.3f}`.
"""
    note_path = args.out_dir / "exp22_fixed_history_template_audit_note.md"
    note_path.write_text(note)
    payload = {
        "run_root": str(args.run_root),
        "quality": quality,
        "effects": effect_rows,
        "support": summary.get("support", []),
        "artifacts": {
            "effects_csv": str(effects_csv),
            "support_csv": str(support_csv),
            "note": str(note_path),
            "plot": str(plot_dest) if plot_dest.exists() else None,
        },
    }
    json_path = args.out_dir / "exp22_fixed_history_template_audit.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"json": str(json_path), "note": str(note_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
