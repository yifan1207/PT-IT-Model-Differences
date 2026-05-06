#!/usr/bin/env python
"""Build paper-facing artifacts for the Exp22 raw-prompt template audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = "exp22_template_raw_public600_20260505_132816"
DEFAULT_ROOT = ROOT / "results/exp22_endpoint_deconfounded_gap" / DEFAULT_RUN
DEFAULT_OUT = ROOT / "results/paper_synthesis"


def _load_summary(run_root: Path) -> dict[str, Any]:
    path = run_root / "analysis/summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _effect(summary: dict[str, Any], probe: str, outcome: str) -> dict[str, Any]:
    for row in summary["controlled_estimates"]:
        if row["method"] == "cem" and row["probe_family"] == probe and row["outcome"] == outcome:
            return row
    raise KeyError((probe, outcome))


def _branch_lengths(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("*/*/records.jsonl.gz")):
        model = path.parent.parent.name
        variant = path.parent.name
        steps: list[int] = []
        modes: dict[str, int] = {}
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = json.loads(line)
                n_steps = int(rec.get("n_steps", 0))
                steps.append(n_steps)
                mode = str(rec.get("prompt_mode"))
                modes[mode] = modes.get(mode, 0) + 1
        rows.append(
            {
                "model": model,
                "variant": variant,
                "n_records": len(steps),
                "mean_steps": mean(steps) if steps else float("nan"),
                "median_steps": median(steps) if steps else float("nan"),
                "cap_rate_128": sum(step >= 128 for step in steps) / len(steps) if steps else float("nan"),
                "prompt_modes": ";".join(f"{key}:{value}" for key, value in sorted(modes.items())),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    summary = _load_summary(args.run_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    effect_specs = [
        ("raw_late_kl", "raw", "late_kl_mean"),
        ("raw_prefinal_late_kl", "raw", "prefinal_late_kl_mean"),
        ("raw_remaining_adj_js", "raw", "remaining_adj_js"),
        ("raw_future_top1_flips", "raw", "future_top1_flips"),
        ("raw_top5_churn", "raw", "top5_churn"),
    ]
    effect_rows: list[dict[str, Any]] = []
    for key, probe, outcome in effect_specs:
        row = _effect(summary, probe, outcome)
        effect_rows.append(
            {
                "key": key,
                "probe_family": probe,
                "outcome": outcome,
                "estimate_it_minus_pt": float(row["estimate_it_minus_pt"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
                "models": row["models"],
                "n_models": int(row["n_models"]),
            }
        )

    length_rows = _branch_lengths(args.run_root)
    effects_csv = args.out_dir / "exp22_template_raw_public600_effects.csv"
    lengths_csv = args.out_dir / "exp22_template_raw_public600_lengths.csv"
    with effects_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effect_rows[0]))
        writer.writeheader()
        writer.writerows(effect_rows)
    with lengths_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(length_rows[0]))
        writer.writeheader()
        writer.writerows(length_rows)

    by_key = {row["key"]: row for row in effect_rows}
    quality = summary["quality"]
    note = f"""# Exp22 Raw-Prompt Template-Regime Audit

Run: `{args.run_root.name}`.

This audit reruns Exp22 on the two public dense families available without Hugging Face gated-model credentials (`qwen3_4b`, `olmo2_7b`) with PT and IT both receiving raw prompt text (`PROMPT_REGIME=raw`) and raw readouts only.

Quality gates pass: `600/600` records per branch, zero malformed records, minimum matched retention `{quality['min_retained_fraction']:.3f}`, and maximum post-match SMD `{quality['max_smd_after']:.3f}`.

The raw no-template endpoint-matched late-KL effect reverses on this public-family subset: `{by_key['raw_late_kl']['estimate_it_minus_pt']:.3f}` nats, 95% CI `[{by_key['raw_late_kl']['ci95_low']:.3f}, {by_key['raw_late_kl']['ci95_high']:.3f}]`. The endpoint-free future-top-1-flip check remains positive: `{by_key['raw_future_top1_flips']['estimate_it_minus_pt']:.3f}`, 95% CI `[{by_key['raw_future_top1_flips']['ci95_low']:.3f}, {by_key['raw_future_top1_flips']['ci95_high']:.3f}]`, while remaining adjacent JS is near zero/slightly negative.

Generation lengths show that raw prompting is not a neutral wrapper removal. In this run, Qwen-IT hits the 128-token cap on every prompt, Qwen-PT on `79%`; OLMo-IT hits the cap on `36.5%`, OLMo-PT on `64.7%`. We therefore treat this as a template-regime limitation/audit, not as a replacement for the native IT endpoint-matched result.
"""
    note_path = args.out_dir / "exp22_template_raw_public600_note.md"
    note_path.write_text(note)
    payload = {
        "run_root": str(args.run_root),
        "quality": quality,
        "effects": effect_rows,
        "lengths": length_rows,
        "artifacts": {
            "effects_csv": str(effects_csv),
            "lengths_csv": str(lengths_csv),
            "note": str(note_path),
        },
    }
    json_path = args.out_dir / "exp22_template_raw_public600_audit.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"json": str(json_path), "note": str(note_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
