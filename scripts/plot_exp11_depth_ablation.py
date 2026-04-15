"""Paper-facing overview plot for exp11.2 depth-specific graft ablation.

Reads per-model exp11.2 run directories containing `depth_ablation_summary.json`
and emits:

  - depth_ablation_metrics.json
  - depth_ablation_main.png
  - depth_ablation_paper_main.png
  - {model}_panel.png copies when available
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


MODEL_DISPLAY = {
    "gemma3_4b": "Gemma3 4B",
    "qwen3_4b": "Qwen3 4B",
    "llama31_8b": "Llama 3.1 8B",
    "mistral_7b": "Mistral 7B",
    "olmo2_7b": "OLMo2 7B",
    "deepseek_v2_lite": "DeepSeek V2 Lite",
}
MODEL_ORDER = [
    "gemma3_4b",
    "qwen3_4b",
    "llama31_8b",
    "mistral_7b",
    "olmo2_7b",
    "deepseek_v2_lite",
]
PIPELINE_ORDER = ["B_early_raw", "B_mid_raw", "B_late_raw"]
PIPELINE_LABEL = {
    "B_early_raw": "Early",
    "B_mid_raw": "Mid",
    "B_late_raw": "Late",
}
PIPELINE_COLOR = {
    "B_early_raw": "#1f77b4",
    "B_mid_raw": "#ff7f0e",
    "B_late_raw": "#d62728",
}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _collect_run_dirs(run_root: Path) -> list[tuple[str, Path]]:
    subdirs = {p.name: p for p in run_root.iterdir() if p.is_dir()}
    rows: list[tuple[str, Path]] = []
    for model in MODEL_ORDER:
        path = None
        if model in subdirs:
            path = subdirs[model]
        else:
            for name, candidate in subdirs.items():
                if name.endswith(f"_{model}") or name.endswith(model):
                    path = candidate
                    break
        if path is not None:
            rows.append((model, path))
    return rows


def _dense_mean(entries: list[dict], pipeline_name: str, region_name: str, metric_name: str) -> float | None:
    values: list[float] = []
    for entry in entries:
        if entry["model"] == "deepseek_v2_lite":
            continue
        value = (
            entry.get("pipelines", {})
            .get(pipeline_name, {})
            .get("regions", {})
            .get(region_name, {})
            .get(metric_name, {})
            .get("delta")
        )
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _plot_main(metrics: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_windows, ax_kl, ax_cross = axes

    y = np.arange(len(metrics))
    ax_windows.set_title("Equal-width graft windows by model")
    ax_windows.set_xlabel("Normalized depth")
    ax_windows.set_yticks(y)
    ax_windows.set_yticklabels([MODEL_DISPLAY.get(entry["model"], entry["model"]) for entry in metrics])
    ax_windows.set_xlim(0, 1)
    ax_windows.grid(axis="x", alpha=0.2)
    for row_idx, entry in enumerate(metrics):
        n_layers = entry.get("n_layers") or 1
        for pipeline_name in PIPELINE_ORDER:
            window = entry.get("pipelines", {}).get(pipeline_name, {}).get("graft_window", {})
            start = float(window.get("start_layer", 0)) / max(n_layers, 1)
            end = float(window.get("end_layer_exclusive", 0)) / max(n_layers, 1)
            ax_windows.barh(
                row_idx,
                max(end - start, 0.0),
                left=start,
                height=0.22,
                color=PIPELINE_COLOR[pipeline_name],
                alpha=0.8,
                label=PIPELINE_LABEL[pipeline_name] if row_idx == 0 else None,
            )
    handles, labels = ax_windows.get_legend_handles_labels()
    if handles:
        ax_windows.legend(frameon=False, fontsize=9, loc="lower right")

    x = np.arange(len(metrics))
    width = 0.22
    for i, pipeline_name in enumerate(PIPELINE_ORDER):
        offsets = x + (i - 1) * width
        kl_values = [
            entry.get("pipelines", {})
            .get(pipeline_name, {})
            .get("regions", {})
            .get("final_20pct", {})
            .get("kl_to_own_final", {})
            .get("delta", 0.0)
            for entry in metrics
        ]
        cross_values = [
            entry.get("pipelines", {})
            .get(pipeline_name, {})
            .get("regions", {})
            .get("final_20pct", {})
            .get("cross_kl", {})
            .get("delta", 0.0)
            for entry in metrics
        ]
        ax_kl.bar(offsets, kl_values, width, color=PIPELINE_COLOR[pipeline_name], label=PIPELINE_LABEL[pipeline_name])
        ax_cross.bar(offsets, cross_values, width, color=PIPELINE_COLOR[pipeline_name], label=PIPELINE_LABEL[pipeline_name])

    labels = [MODEL_DISPLAY.get(entry["model"], entry["model"]) for entry in metrics]
    for ax in (ax_kl, ax_cross):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.grid(axis="y", alpha=0.2)

    ax_kl.set_title("Late-region Δ KL-to-own-final\n(B_window - A'_raw)")
    ax_kl.set_ylabel("Positive = later / less committed")
    ax_cross.set_title("Late-region Δ cross-KL to C\n(B_window - A'_raw)")
    ax_cross.set_ylabel("Negative = closer to C")
    handles, labels_h = ax_kl.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_h, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "exp11.2 matched-prefix depth ablation\n"
        "Equal-width early/mid/late IT-MLP grafts on a PT backbone, teacher-forced to the same IT teacher tokens",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_main(metrics: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_windows, ax_local, ax_late = axes

    y = np.arange(len(metrics))
    ax_windows.set_title("Equal-width graft windows by model")
    ax_windows.set_xlabel("Normalized depth")
    ax_windows.set_yticks(y)
    ax_windows.set_yticklabels([MODEL_DISPLAY.get(entry["model"], entry["model"]) for entry in metrics])
    ax_windows.set_xlim(0, 1)
    ax_windows.grid(axis="x", alpha=0.2)
    for row_idx, entry in enumerate(metrics):
        n_layers = entry.get("n_layers") or 1
        for pipeline_name in PIPELINE_ORDER:
            window = entry.get("pipelines", {}).get(pipeline_name, {}).get("graft_window", {})
            start = float(window.get("start_layer", 0)) / max(n_layers, 1)
            end = float(window.get("end_layer_exclusive", 0)) / max(n_layers, 1)
            ax_windows.barh(
                row_idx,
                max(end - start, 0.0),
                left=start,
                height=0.22,
                color=PIPELINE_COLOR[pipeline_name],
                alpha=0.8,
                label=PIPELINE_LABEL[pipeline_name] if row_idx == 0 else None,
            )
    handles, labels = ax_windows.get_legend_handles_labels()
    if handles:
        ax_windows.legend(frameon=False, fontsize=9, loc="lower right")

    x = np.arange(len(metrics))
    width = 0.22
    for i, pipeline_name in enumerate(PIPELINE_ORDER):
        offsets = x + (i - 1) * width
        local_values = [
            entry.get("pipelines", {})
            .get(pipeline_name, {})
            .get("regions", {})
            .get("graft_window", {})
            .get("kl_to_own_final", {})
            .get("delta", 0.0)
            for entry in metrics
        ]
        late_values = [
            entry.get("pipelines", {})
            .get(pipeline_name, {})
            .get("regions", {})
            .get("final_20pct", {})
            .get("kl_to_own_final", {})
            .get("delta", 0.0)
            for entry in metrics
        ]
        ax_local.bar(offsets, local_values, width, color=PIPELINE_COLOR[pipeline_name], label=PIPELINE_LABEL[pipeline_name])
        ax_late.bar(offsets, late_values, width, color=PIPELINE_COLOR[pipeline_name], label=PIPELINE_LABEL[pipeline_name])

    labels = [MODEL_DISPLAY.get(entry["model"], entry["model"]) for entry in metrics]
    for ax in (ax_local, ax_late):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.grid(axis="y", alpha=0.2)

    ax_local.set_title("Δ KL-to-own-final inside grafted block\n(B_window - A'_raw)")
    ax_local.set_ylabel("Positive = larger local perturbation")
    ax_late.set_title("Δ KL-to-own-final in final 20% of layers\n(B_window - A'_raw)")
    ax_late.set_ylabel("Positive = later / less committed")
    handles, labels_h = ax_late.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_h, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "exp11.2 depth specificity: local perturbation is not enough\n"
        "Only the late graft consistently recreates the late delayed-convergence signature",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[dict] = []
    for model, run_dir in _collect_run_dirs(run_root):
        summary = _read_json(run_dir / "depth_ablation_summary.json")
        if not summary:
            print(f"[skip] no depth_ablation_summary.json for {model}")
            continue
        config = _read_json(run_dir / "config.json")
        summary["model"] = model
        summary["n_layers"] = config.get("n_layers")
        metrics.append(summary)
        panel_src = run_dir / "plots" / "panel.png"
        if panel_src.exists():
            shutil.copy2(panel_src, out_dir / f"{model}_panel.png")

    payload = {
        "models": metrics,
        "dense_family_means": {
            pipeline_name: {
                "final_20pct_delta_kl_to_own_final": _dense_mean(metrics, pipeline_name, "final_20pct", "kl_to_own_final"),
                "final_20pct_delta_cross_kl": _dense_mean(metrics, pipeline_name, "final_20pct", "cross_kl"),
                "final_20pct_delta_delta_cosine": _dense_mean(metrics, pipeline_name, "final_20pct", "delta_cosine"),
            }
            for pipeline_name in PIPELINE_ORDER
        },
    }
    (out_dir / "depth_ablation_metrics.json").write_text(json.dumps(payload, indent=2))
    _plot_main(metrics, out_dir / "depth_ablation_main.png")
    _plot_paper_main(metrics, out_dir / "depth_ablation_paper_main.png")
    print(f"Wrote {out_dir / 'depth_ablation_metrics.json'}")
    print(f"Wrote {out_dir / 'depth_ablation_main.png'}")
    print(f"Wrote {out_dir / 'depth_ablation_paper_main.png'}")


if __name__ == "__main__":
    main()
