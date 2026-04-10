from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exp11 outputs.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    prompt_rows = _read_jsonl(run_dir / "prompt_summaries.jsonl")
    step_rows = _read_jsonl(run_dir / "step_metrics.jsonl")
    seen = set()
    for row in prompt_rows:
        key = row["prompt_id"]
        if key in seen:
            raise ValueError(f"Duplicate prompt summary row for {key}")
        seen.add(key)

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in prompt_rows:
        by_category[row.get("category", "")].append(row)

    overall = {
        "n_prompts": len(prompt_rows),
        "mean_divergence_step": _mean([row["divergence_step"] for row in prompt_rows if row["divergence_step"] is not None]),
        "pipeline_a_mean_length": _mean([row["pipeline_a"]["generated_text_length"] for row in prompt_rows]),
        "pipeline_b_mean_length": _mean([row["pipeline_b"]["generated_text_length"] for row in prompt_rows]),
        "pipeline_a_mean_structural_ratio_tier12_proxy": _mean(
            [row["pipeline_a"]["structural_token_ratio_tier12_proxy"] for row in prompt_rows]
        ),
        "pipeline_b_mean_structural_ratio_tier12_proxy": _mean(
            [row["pipeline_b"]["structural_token_ratio_tier12_proxy"] for row in prompt_rows]
        ),
    }
    per_category = {}
    for category, rows in by_category.items():
        per_category[category] = {
            "n_prompts": len(rows),
            "mean_divergence_step": _mean([row["divergence_step"] for row in rows if row["divergence_step"] is not None]),
            "pipeline_a_mean_length": _mean([row["pipeline_a"]["generated_text_length"] for row in rows]),
            "pipeline_b_mean_length": _mean([row["pipeline_b"]["generated_text_length"] for row in rows]),
        }

    summary = {"overall": overall, "per_category": per_category}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    layer_metrics = [
        "delta_cosine",
        "kl_to_own_final",
        "structural_mass_prob_tier1",
        "structural_mass_prob_tier12",
        "entropy",
    ]
    cross_metrics = ["cross_kl", "kl_to_pt_final", "residual_cosine", "residual_divergence"]
    trajectory: dict[str, dict[str, list[float | None]]] = {}
    by_pipeline: dict[str, list[dict]] = defaultdict(list)
    for row in step_rows:
        by_pipeline[row["pipeline"]].append(row)

    for pipeline, rows in by_pipeline.items():
        if not rows:
            continue
        metrics = rows[0]["metrics"]
        n_layers = len(metrics["delta_cosine"])
        pipeline_summary: dict[str, list[float | None]] = {}
        for metric_name in layer_metrics + cross_metrics:
            means: list[float | None] = []
            for layer_idx in range(n_layers):
                vals = []
                for row in rows:
                    metric = row["metrics"].get(metric_name)
                    if metric is None:
                        continue
                    vals.append(metric[layer_idx])
                means.append(_mean(vals) if vals else None)
            pipeline_summary[metric_name] = means
        trajectory[pipeline] = pipeline_summary

    (run_dir / "trajectory_summary.json").write_text(json.dumps(trajectory, indent=2))


if __name__ == "__main__":
    main()
