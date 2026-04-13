from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


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


def _mean_trajectory(step_metrics_path: Path, metric_names: list[str]) -> dict[str, dict[str, list[float | None]]]:
    sums: dict[str, dict[str, list[float]]] = {}
    counts: dict[str, dict[str, list[float]]] = {}
    with open(step_metrics_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pipeline = row["pipeline"]
            metrics = row["metrics"]
            pipeline_sums = sums.setdefault(pipeline, {})
            pipeline_counts = counts.setdefault(pipeline, {})
            for metric_name in metric_names:
                values = metrics.get(metric_name)
                if values is None:
                    continue
                metric_sums = pipeline_sums.setdefault(metric_name, [0.0] * len(values))
                metric_counts = pipeline_counts.setdefault(metric_name, [0.0] * len(values))
                for idx, value in enumerate(values):
                    if value is None:
                        continue
                    metric_sums[idx] += float(value)
                    metric_counts[idx] += 1.0

    trajectory: dict[str, dict[str, list[float | None]]] = {}
    for pipeline, pipeline_sums in sums.items():
        pipeline_summary: dict[str, list[float | None]] = {}
        for metric_name, metric_sums in pipeline_sums.items():
            metric_counts = counts[pipeline][metric_name]
            pipeline_summary[metric_name] = [
                (total / count) if count else None
                for total, count in zip(metric_sums, metric_counts, strict=False)
            ]
        trajectory[pipeline] = pipeline_summary
    return trajectory


def _secondary_means(stats: dict) -> dict[str, dict[str, list[float | None]]]:
    result: dict[str, dict[str, list[float | None]]] = {}
    metrics = stats.get("metrics", {})
    for pipeline, pipeline_metrics in metrics.items():
        result[pipeline] = {}
        for metric_name, metric_stats in pipeline_metrics.items():
            sums = metric_stats.get("sum", [])
            counts = metric_stats.get("count", [])
            means: list[float | None] = []
            for total, count in zip(sums, counts, strict=False):
                means.append((total / count) if count else None)
            result[pipeline][metric_name] = means
    return result


def _plot_panel(
    *,
    plot_dir: Path,
    model_name: str,
    primary: dict[str, dict[str, list[float | None]]],
    secondary: dict[str, dict[str, list[float | None]]] | None,
    prompt_summary: dict,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_mass, ax_kl, ax_delta, ax_cross = axes.flat

    primary_colors = [
        ("A", "#1f77b4"),
        ("B", "#d62728"),
        ("C", "#2ca02c"),
        ("A_prime", "#9467bd"),
        ("A_prime_tmpl", "#c5b0d5"),
        ("B2", "#ff7f0e"),
    ]
    secondary_colors = [
        ("A", "#7f7f7f"),
        ("B", "#ff9896"),
        ("C", "#98df8a"),
        ("A_prime", "#c5b0d5"),
        ("A_prime_tmpl", "#ddd0ee"),
        ("B2", "#ffbb78"),
    ]

    def _plot_metric(ax, metric_name: str, title: str, ylabel: str) -> None:
        for pipeline, color in primary_colors:
            values = primary.get(pipeline, {}).get(metric_name)
            if values is not None:
                ax.plot(values, label=f"{pipeline} primary", color=color, linewidth=2)
        if secondary is not None:
            for pipeline, color in secondary_colors:
                values = secondary.get(pipeline, {}).get(metric_name)
                if values is not None:
                    ax.plot(values, label=f"{pipeline} native", color=color, linestyle="--", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

    _plot_metric(ax_mass, "structural_mass_prob_tier1", "Mean Tier 1 Probability Mass", "Probability mass")
    _plot_metric(ax_kl, "kl_to_own_final", "KL To Actual Final Output", "KL")
    _plot_metric(ax_delta, "delta_cosine", "Delta Cosine", "Cosine")
    _plot_metric(ax_cross, "cross_kl", "Cross-Pipeline KL", "KL")

    summary = prompt_summary.get("overall", {})
    title = (
        f"{model_name} | prompts={summary.get('n_prompts', '?')} | "
        f"mean divergence={summary.get('mean_divergence_step', 'n/a')}"
    )
    fig.suptitle(title)
    handles, labels = ax_mass.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(plot_dir / "panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exp11 outputs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--plot-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    plot_dir = Path(args.plot_dir) if args.plot_dir else run_dir / "plots"
    config = _read_json(run_dir / "config.json")
    prompt_rows = _read_jsonl(run_dir / "prompt_summaries.jsonl")
    secondary_stats = _read_json(run_dir / "secondary_trajectory_stats.json")
    seen = set()
    for row in prompt_rows:
        key = row["prompt_id"]
        if key in seen:
            raise ValueError(f"Duplicate prompt summary row for {key}")
        seen.add(key)

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in prompt_rows:
        by_category[row.get("category", "")].append(row)

    teacher_forced = bool(config.get("teacher_forced"))
    configured_pipelines = config.get("pipelines")
    if configured_pipelines:
        pipeline_keys = [f"pipeline_{name.lower()}" for name in configured_pipelines]
    else:
        pipeline_keys = ["pipeline_a", "pipeline_b"]
        if teacher_forced:
            pipeline_keys += ["pipeline_c", "pipeline_a_prime"]

    def _mean_field(rows, pipeline_key, field):
        return _mean(
            [row[pipeline_key][field] for row in rows if pipeline_key in row and field in row[pipeline_key]]
        )

    overall: dict = {
        "n_prompts": len(prompt_rows),
        "mean_divergence_step": _mean([row.get("divergence_step") for row in prompt_rows if row.get("divergence_step") is not None]),
    }
    for key in pipeline_keys:
        overall[f"{key}_mean_length"] = _mean_field(prompt_rows, key, "generated_text_length")
        overall[f"{key}_mean_structural_ratio_tier1_proxy"] = _mean_field(
            prompt_rows, key, "structural_token_ratio_tier1_proxy"
        )
        overall[f"{key}_mean_paragraph_count"] = _mean_field(prompt_rows, key, "paragraph_count")
        overall[f"{key}_mean_header_count"] = _mean_field(prompt_rows, key, "header_count")
        overall[f"{key}_mean_bullet_count"] = _mean_field(prompt_rows, key, "bullet_count")
    if teacher_forced:
        for div_key in (
            "divergence_step_b_vs_c",
            "divergence_step_a_prime_vs_c",
            "divergence_step_a_prime_tmpl_vs_c",
            "divergence_step_b2_vs_c",
        ):
            overall[f"mean_{div_key}"] = _mean(
                [row[div_key] for row in prompt_rows if row.get(div_key) is not None]
            )
    per_category = {}
    for category, rows in by_category.items():
        entry: dict = {
            "n_prompts": len(rows),
            "mean_divergence_step": _mean([row.get("divergence_step") for row in rows if row.get("divergence_step") is not None]),
        }
        for key in pipeline_keys:
            entry[f"{key}_mean_length"] = _mean_field(rows, key, "generated_text_length")
        per_category[category] = entry

    summary = {"overall": overall, "per_category": per_category}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    primary_metric_names = [
        "delta_cosine",
        "kl_to_own_final",
        "structural_mass_prob_tier1",
        "entropy",
        "cross_kl",
        "kl_to_pt_final",
        "residual_cosine",
        "residual_divergence",
    ]
    primary_trajectory = _mean_trajectory(run_dir / "step_metrics.jsonl", primary_metric_names)
    (run_dir / "trajectory_summary.json").write_text(json.dumps(primary_trajectory, indent=2))

    secondary_trajectory = None
    if secondary_stats:
        secondary_trajectory = _secondary_means(secondary_stats)
        (run_dir / "secondary_trajectory_summary.json").write_text(
            json.dumps(secondary_trajectory, indent=2)
        )

    readout_summary = {
        "primary_readout_name": config.get("primary_readout_name"),
        "secondary_readout_name": config.get("secondary_readout_name"),
        "secondary_readout_names_by_pipeline": config.get("secondary_readout_names_by_pipeline"),
        "readout_mode": config.get("readout_mode"),
    }
    (run_dir / "readout_summary.json").write_text(json.dumps(readout_summary, indent=2))

    _plot_panel(
        plot_dir=plot_dir,
        model_name=config.get("model", run_dir.name),
        primary=primary_trajectory,
        secondary=secondary_trajectory,
        prompt_summary=summary,
    )


if __name__ == "__main__":
    main()
