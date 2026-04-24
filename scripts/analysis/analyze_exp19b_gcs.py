#!/usr/bin/env python3
"""Stream Exp19B random-control summaries from GCS and make paper-facing plots.

This intentionally does not download raw ``step_metrics.jsonl`` files.  The
Exp19B run is large enough that local Macs can run out of disk; this script
uses ``gsutil cat`` and extracts only the small layerwise arrays needed for the
specificity-control analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


MODEL_ORDER = ["gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b"]
WINDOWS = ["early", "mid", "late"]
METRICS = ["kl_to_own_final", "delta_cosine", "next_token_rank", "next_token_prob"]
PIPELINE_RE = re.compile(rb'"pipeline"\s*:\s*"([^"]+)"')
PROMPT_RE = re.compile(rb'"prompt_id"\s*:\s*"([^"]+)"')
STEP_RE = re.compile(rb'"step"\s*:\s*([0-9]+)')


@dataclass
class RunningMean:
    total: float = 0.0
    count: int = 0

    def add(self, value: float | None) -> None:
        if value is None or not math.isfinite(value):
            return
        self.total += float(value)
        self.count += 1

    def mean(self) -> float | None:
        return self.total / self.count if self.count else None


@dataclass
class PromptAccumulator:
    values: dict[str, RunningMean] = field(default_factory=lambda: defaultdict(RunningMean))

    def add(self, metric: str, value: float | None) -> None:
        self.values[metric].add(value)

    def means(self) -> dict[str, float]:
        return {metric: mean for metric, acc in self.values.items() if (mean := acc.mean()) is not None}


def _run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True)


def _gsutil_cat(url: str) -> subprocess.Popen:
    return subprocess.Popen(["gsutil", "cat", url], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _list_urls(root: str, suffix: str) -> list[str]:
    pattern = root.rstrip("/") + f"/**/{suffix}"
    return [line.strip() for line in _run_text(["gsutil", "ls", pattern]).splitlines() if line.strip()]


def _cat_json(url: str) -> dict[str, Any]:
    return json.loads(_run_text(["gsutil", "cat", url]))


def _model_from_step_url(url: str) -> str:
    parent = url.rstrip("/").split("/")[-2]
    return parent.split("__", 1)[0]


def _extract_regex(pattern: re.Pattern[bytes], line: bytes) -> bytes | None:
    match = pattern.search(line)
    return match.group(1) if match else None


def _extract_metric_array(line: bytes, key: str) -> list[Any] | None:
    marker = f'"{key}"'.encode()
    idx = line.find(marker)
    if idx < 0:
        return None
    colon = line.find(b":", idx + len(marker))
    if colon < 0:
        return None
    pos = colon + 1
    while pos < len(line) and line[pos] in b" \t":
        pos += 1
    if line.startswith(b"null", pos):
        return None
    if pos >= len(line) or line[pos : pos + 1] != b"[":
        return None
    end = line.find(b"]", pos)
    if end < 0:
        return None
    return json.loads(line[pos : end + 1])


def _mean_slice(values: list[Any] | None, start: int, end: int) -> float | None:
    if values is None:
        return None
    kept: list[float] = []
    for value in values[start:end]:
        if value is None:
            continue
        value_f = float(value)
        if math.isfinite(value_f):
            kept.append(value_f)
    return sum(kept) / len(kept) if kept else None


def _bootstrap_ci(values: list[float], *, seed: int = 0, n_boot: int = 2000) -> dict[str, float | int | None]:
    kept = [float(v) for v in values if math.isfinite(float(v))]
    if not kept:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    n = len(kept)
    means = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += kept[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return {
        "mean": sum(kept) / n,
        "lo": means[int(0.025 * (n_boot - 1))],
        "hi": means[int(0.975 * (n_boot - 1))],
        "n": n,
    }


def _window_for_pipeline(pipeline: str) -> str | None:
    for window in WINDOWS:
        if f"_{window}_" in pipeline:
            return window
    return None


def _control_for_pipeline(pipeline: str) -> str | None:
    if pipeline.endswith("_true"):
        return "true"
    if "_rand_resproj_" in pipeline:
        return "rand_resproj"
    return None


def _seed_for_pipeline(pipeline: str) -> int | None:
    match = re.search(r"_s([0-9]+)$", pipeline)
    return int(match.group(1)) if match else None


def _summarize_model(
    *,
    model: str,
    step_urls: list[str],
    config_by_url: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    prompt_pipeline: dict[str, dict[str, PromptAccumulator]] = defaultdict(lambda: defaultdict(PromptAccumulator))
    baseline_by_step: dict[tuple[str, int], dict[str, float]] = {}
    pending: list[tuple[str, int, str, dict[str, float]]] = []
    seen: set[tuple[str, str, int]] = set()
    duplicate_rows = 0
    parsed_rows = 0
    branch_delta_rows = 0
    baseline_rows = 0
    parse_errors = 0

    for url in step_urls:
        config = config_by_url[url.replace("/step_metrics.jsonl", "/config.json")]
        final_layers = list(config.get("final_region_layers") or [])
        if final_layers:
            final_start = int(final_layers[0])
            final_end = int(final_layers[-1]) + 1
        else:
            n_layers = int(config["n_layers"])
            width = max(1, math.ceil(0.2 * n_layers))
            final_start = n_layers - width
            final_end = n_layers
        proc = _gsutil_cat(url)
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            if not raw_line.strip():
                continue
            try:
                prompt_raw = _extract_regex(PROMPT_RE, raw_line)
                pipeline_raw = _extract_regex(PIPELINE_RE, raw_line)
                step_raw = _extract_regex(STEP_RE, raw_line)
                if prompt_raw is None or pipeline_raw is None or step_raw is None:
                    parse_errors += 1
                    continue
                prompt_id = prompt_raw.decode("utf-8", "replace")
                pipeline = pipeline_raw.decode("utf-8", "replace")
                step = int(step_raw)
                row_key = (pipeline, prompt_id, step)
                if row_key in seen:
                    duplicate_rows += 1
                    continue
                seen.add(row_key)
                parsed_rows += 1
                row_values = {
                    metric: _mean_slice(_extract_metric_array(raw_line, metric), final_start, final_end)
                    for metric in METRICS
                }
                for metric, value in row_values.items():
                    prompt_pipeline[prompt_id][pipeline].add(metric, value)
                if pipeline == "A_prime_raw":
                    baseline_by_step[(prompt_id, step)] = {
                        metric: value for metric, value in row_values.items() if value is not None
                    }
                    baseline_rows += 1
                elif pipeline.startswith("B_"):
                    baseline = baseline_by_step.get((prompt_id, step))
                    if baseline is None:
                        pending.append((prompt_id, step, pipeline, row_values))
                    else:
                        branch_delta_rows += 1
                        for metric, value in row_values.items():
                            if value is not None and baseline.get(metric) is not None:
                                prompt_pipeline[prompt_id][pipeline + "__delta_vs_A"].add(
                                    metric,
                                    float(value) - float(baseline[metric]),
                                )
            except Exception:
                parse_errors += 1
        stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr is not None else ""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"gsutil cat failed for {url}: {stderr[-2000:]}")

    still_pending = 0
    for prompt_id, step, pipeline, row_values in pending:
        baseline = baseline_by_step.get((prompt_id, step))
        if baseline is None:
            still_pending += 1
            continue
        branch_delta_rows += 1
        for metric, value in row_values.items():
            if value is not None and baseline.get(metric) is not None:
                prompt_pipeline[prompt_id][pipeline + "__delta_vs_A"].add(metric, float(value) - float(baseline[metric]))

    prompt_means = {
        prompt_id: {pipeline: acc.means() for pipeline, acc in pipelines.items()}
        for prompt_id, pipelines in prompt_pipeline.items()
    }
    pipeline_metric_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pipelines in prompt_means.values():
        for pipeline, metrics in pipelines.items():
            for metric, value in metrics.items():
                pipeline_metric_values[pipeline][metric].append(float(value))
    pipeline_summary = {
        pipeline: {metric: _bootstrap_ci(values) for metric, values in metrics.items()}
        for pipeline, metrics in sorted(pipeline_metric_values.items())
    }

    contrasts: dict[str, dict[str, Any]] = {}
    for window in WINDOWS:
        true_name = f"B_{window}_true__delta_vs_A"
        rand_names = [
            name
            for name in pipeline_metric_values
            if name.startswith(f"B_{window}_rand_resproj_s") and name.endswith("__delta_vs_A")
        ]
        contrasts[window] = {}
        for metric in METRICS:
            prompt_margins: list[float] = []
            prompt_true: list[float] = []
            prompt_rand: list[float] = []
            for prompt_id, pipelines in prompt_means.items():
                true_value = pipelines.get(true_name, {}).get(metric)
                rand_values = [
                    pipelines.get(rand_name, {}).get(metric)
                    for rand_name in rand_names
                    if pipelines.get(rand_name, {}).get(metric) is not None
                ]
                if true_value is None or not rand_values:
                    continue
                rand_mean = sum(float(v) for v in rand_values) / len(rand_values)
                prompt_true.append(float(true_value))
                prompt_rand.append(rand_mean)
                prompt_margins.append(float(true_value) - rand_mean)
            contrasts[window][metric] = {
                "true_delta": _bootstrap_ci(prompt_true),
                "rand_resproj_delta": _bootstrap_ci(prompt_rand),
                "true_minus_rand_resproj": _bootstrap_ci(prompt_margins),
                "random_pipelines": sorted(rand_names),
            }

    return {
        "model": model,
        "n_prompts": len(prompt_means),
        "parsed_step_rows": parsed_rows,
        "baseline_rows": baseline_rows,
        "branch_delta_rows": branch_delta_rows,
        "pending_branch_rows_unmatched": still_pending,
        "duplicate_rows": duplicate_rows,
        "parse_errors": parse_errors,
        "pipeline_summary": pipeline_summary,
        "contrasts": contrasts,
        "prompt_means": prompt_means,
    }


def _pool_dense(models: dict[str, Any]) -> dict[str, Any]:
    pooled: dict[str, dict[str, Any]] = {window: {} for window in WINDOWS}
    for window in WINDOWS:
        for metric in METRICS:
            margins: list[float] = []
            trues: list[float] = []
            rands: list[float] = []
            per_model = {}
            for model in MODEL_ORDER:
                payload = models.get(model)
                if not payload:
                    continue
                prompt_means = payload["prompt_means"]
                true_name = f"B_{window}_true__delta_vs_A"
                rand_names = [
                    name
                    for name in payload["pipeline_summary"]
                    if name.startswith(f"B_{window}_rand_resproj_s") and name.endswith("__delta_vs_A")
                ]
                model_margins = []
                for pipelines in prompt_means.values():
                    true_value = pipelines.get(true_name, {}).get(metric)
                    rand_values = [
                        pipelines.get(rand_name, {}).get(metric)
                        for rand_name in rand_names
                        if pipelines.get(rand_name, {}).get(metric) is not None
                    ]
                    if true_value is None or not rand_values:
                        continue
                    rand_mean = sum(float(v) for v in rand_values) / len(rand_values)
                    trues.append(float(true_value))
                    rands.append(rand_mean)
                    margins.append(float(true_value) - rand_mean)
                    model_margins.append(float(true_value) - rand_mean)
                per_model[model] = _bootstrap_ci(model_margins)
            pooled[window][metric] = {
                "true_delta": _bootstrap_ci(trues),
                "rand_resproj_delta": _bootstrap_ci(rands),
                "true_minus_rand_resproj": _bootstrap_ci(margins),
                "per_model_true_minus_rand": per_model,
            }
    return pooled


def _plot_kl(summary: dict[str, Any], out_dir: Path) -> None:
    pooled = summary["dense5_pooled"]
    x = range(len(WINDOWS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    true = [pooled[w]["kl_to_own_final"]["true_delta"]["mean"] for w in WINDOWS]
    rand = [pooled[w]["kl_to_own_final"]["rand_resproj_delta"]["mean"] for w in WINDOWS]
    true_err = [
        [
            true[i] - pooled[w]["kl_to_own_final"]["true_delta"]["lo"]
            for i, w in enumerate(WINDOWS)
        ],
        [
            pooled[w]["kl_to_own_final"]["true_delta"]["hi"] - true[i]
            for i, w in enumerate(WINDOWS)
        ],
    ]
    rand_err = [
        [
            rand[i] - pooled[w]["kl_to_own_final"]["rand_resproj_delta"]["lo"]
            for i, w in enumerate(WINDOWS)
        ],
        [
            pooled[w]["kl_to_own_final"]["rand_resproj_delta"]["hi"] - rand[i]
            for i, w in enumerate(WINDOWS)
        ],
    ]
    ax.bar([i - width / 2 for i in x], true, width, yerr=true_err, label="actual IT graft", color="#244a73", capsize=3)
    ax.bar([i + width / 2 for i in x], rand, width, yerr=rand_err, label="matched random control", color="#a9b9c9", capsize=3)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([w.title() for w in WINDOWS])
    ax.set_ylabel("Final-20% Δ KL-to-own-final vs A'")
    ax.set_title("Exp19B: actual IT grafts vs matched random residual-projection controls")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "exp19B_final20_kl_true_vs_random.png", dpi=220)
    fig.savefig(out_dir / "exp19B_final20_kl_true_vs_random.pdf")
    plt.close(fig)


def _plot_margins(summary: dict[str, Any], out_dir: Path) -> None:
    pooled = summary["dense5_pooled"]
    metrics = ["kl_to_own_final", "delta_cosine", "next_token_rank", "next_token_prob"]
    labels = ["KL delay", "δ-cosine", "teacher rank", "teacher prob"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), constrained_layout=True)
    for ax, metric, label in zip(axes, metrics, labels, strict=True):
        values = [pooled[w][metric]["true_minus_rand_resproj"]["mean"] for w in WINDOWS]
        errs = [
            [values[i] - pooled[w][metric]["true_minus_rand_resproj"]["lo"] for i, w in enumerate(WINDOWS)],
            [pooled[w][metric]["true_minus_rand_resproj"]["hi"] - values[i] for i, w in enumerate(WINDOWS)],
        ]
        ax.bar(WINDOWS, values, yerr=errs, color=["#c9d2df", "#8fa8c4", "#244a73"], capsize=3)
        ax.axhline(0, color="black", linewidth=0.85)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Actual IT graft minus matched random")
    fig.suptitle("Exp19B specificity margin: direction-specific effect beyond matched random perturbation")
    fig.savefig(out_dir / "exp19B_specificity_margins.png", dpi=220)
    fig.savefig(out_dir / "exp19B_specificity_margins.pdf")
    plt.close(fig)


def _plot_per_model(summary: dict[str, Any], out_dir: Path) -> None:
    pooled = summary["dense5_pooled"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    offsets = {"early": -0.24, "mid": 0.0, "late": 0.24}
    colors = {"early": "#c9d2df", "mid": "#8fa8c4", "late": "#244a73"}
    for w in WINDOWS:
        per_model = pooled[w]["kl_to_own_final"]["per_model_true_minus_rand"]
        xs = [MODEL_ORDER.index(model) + offsets[w] for model in MODEL_ORDER]
        ys = [per_model[model]["mean"] for model in MODEL_ORDER]
        ax.scatter(xs, ys, s=70, label=w.title(), color=colors[w])
    ax.axhline(0, color="black", linewidth=0.85)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Final-20% KL specificity margin\n(actual IT graft - matched random)")
    ax.set_title("Exp19B per-family specificity margins")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "exp19B_per_model_kl_margins.png", dpi=220)
    fig.savefig(out_dir / "exp19B_per_model_kl_margins.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp19B from GCS without downloading raw JSONL.")
    parser.add_argument("--gcs-root", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    step_urls = _list_urls(args.gcs_root, "step_metrics.jsonl")
    config_urls = _list_urls(args.gcs_root, "config.json")
    config_by_url = {url: _cat_json(url) for url in config_urls}
    step_by_model: dict[str, list[str]] = defaultdict(list)
    for url in step_urls:
        step_by_model[_model_from_step_url(url)].append(url)

    models = {}
    for model in MODEL_ORDER:
        urls = sorted(step_by_model.get(model, []))
        if not urls:
            continue
        print(f"[exp19B] streaming {model}: {len(urls)} step files", flush=True)
        models[model] = _summarize_model(model=model, step_urls=urls, config_by_url=config_by_url)
        print(
            f"[exp19B] {model}: prompts={models[model]['n_prompts']} "
            f"rows={models[model]['parsed_step_rows']} dup={models[model]['duplicate_rows']} "
            f"pending={models[model]['pending_branch_rows_unmatched']} parse_errors={models[model]['parse_errors']}",
            flush=True,
        )

    summary = {
        "analysis": "exp19B_core_random_control",
        "gcs_root": args.gcs_root,
        "models": models,
        "dense5_pooled": _pool_dense(models),
        "notes": {
            "primary_metric": "final-20% mean KL-to-own-final delta vs A_prime_raw",
            "specificity_margin": "actual IT graft delta minus matched random residual-projection control delta",
            "prompt_level": "Rows are first averaged over token steps within each prompt, then bootstrapped over prompts.",
            "dense_pool": "Dense-5 pooled because this Exp19B run excluded DeepSeek by design.",
        },
    }
    summary_path = args.out_dir / "exp19B_summary.json"
    summary_light_path = args.out_dir / "exp19B_summary_light.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    light = dict(summary)
    light["models"] = {
        model: {k: v for k, v in payload.items() if k != "prompt_means"}
        for model, payload in models.items()
    }
    summary_light_path.write_text(json.dumps(light, indent=2), encoding="utf-8")
    _plot_kl(summary, args.out_dir)
    _plot_margins(summary, args.out_dir)
    _plot_per_model(summary, args.out_dir)
    print(f"[exp19B] wrote {summary_path}")
    print(f"[exp19B] wrote {summary_light_path}")
    print(f"[exp19B] plots in {args.out_dir}")


if __name__ == "__main__":
    main()
