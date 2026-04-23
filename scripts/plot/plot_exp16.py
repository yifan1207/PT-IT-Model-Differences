#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = [
    "gemma3_4b",
    "qwen3_4b",
    "llama31_8b",
    "mistral_7b",
    "olmo2_7b",
    "deepseek_v2_lite",
]
MODEL_DISPLAY = {
    "gemma3_4b": "Gemma3 4B",
    "qwen3_4b": "Qwen3 4B",
    "llama31_8b": "Llama 3.1 8B",
    "mistral_7b": "Mistral 7B",
    "olmo2_7b": "OLMo2 7B",
    "deepseek_v2_lite": "DeepSeek V2 Lite",
}
WINDOWS = ["early", "mid", "late"]
WINDOW_COLOR = {
    "early": "#72B7B2",
    "mid": "#F58518",
    "late": "#E45756",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _pair_curve(payload: dict, pair_name: str) -> list[float]:
    curve = payload["pairs"][pair_name]["layer_curve"]
    return [float(v) if v is not None else np.nan for v in curve]


def _resample_curve(values: list[float], out_bins: int) -> list[float]:
    if out_bins <= 0:
        return []
    if not values:
        return [np.nan] * out_bins
    if len(values) == 1:
        return [values[0]] * out_bins
    xs_old = np.linspace(0.0, 1.0, len(values))
    xs_new = np.linspace(0.0, 1.0, out_bins)
    arr = np.asarray(values, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() == 0:
        return [np.nan] * out_bins
    if valid.sum() == 1:
        return [float(arr[valid][0])] * out_bins
    return np.interp(xs_new, xs_old[valid], arr[valid]).tolist()


def _window_delta(bundle: dict, side_key: str, window: str) -> float:
    value = bundle[side_key][window]["final_20pct"]["weighted_delta"]
    return float(value) if value is not None else np.nan


def _plot_main(summary: dict, out_path: Path) -> None:
    dense = summary["dense5"]
    models = [model for model in MODEL_ORDER if model in summary["models"] and model != "deepseek_v2_lite"]
    axis_mode = dense.get("layer_axis_mode", "layer_index")
    pooled_curve = _pair_curve(dense, "JS_AC")
    if axis_mode == "normalized_depth":
        x_layers = np.asarray(dense.get("layer_axis_values", np.linspace(0.0, 1.0, len(pooled_curve))), dtype=float)
        x_label = "Normalized depth"
        curve_title = "Matched-prefix broad-gap curve\nDense-5 pooled JS(A', C) by normalized depth"
        onset_marker = float(dense["corrective_onset_frac"])
        final_region_start = float(dense["final_region_start_frac"])
        dense_curve = pooled_curve
        model_curves = {
            model: _resample_curve(_pair_curve(summary["models"][model], "JS_AC"), len(x_layers))
            for model in models
        }
    else:
        x_layers = np.arange(dense["n_layers"])
        x_label = "Layer"
        curve_title = "Matched-prefix broad-gap curve\nJS(A', C) by layer"
        onset_marker = dense["corrective_onset"] - 0.5
        final_region_start = dense["final_region_start"] - 0.5
        dense_curve = pooled_curve
        model_curves = {model: _pair_curve(summary["models"][model], "JS_AC") for model in models}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax_curve, ax_pt, ax_it = axes

    for model in models:
        curve = model_curves[model]
        ax_curve.plot(x_layers, curve, color="#BDBDBD", alpha=0.5, linewidth=1.0)
    ax_curve.plot(x_layers, dense_curve, color="#1f4b99", linewidth=2.8, label="Dense-5 pooled")
    ax_curve.axvline(onset_marker, color="#444444", linestyle="--", linewidth=1.0)
    ax_curve.axvspan(final_region_start, x_layers[-1], color="#E8EEF9", alpha=0.8)
    ax_curve.set_title(curve_title)
    ax_curve.set_xlabel(x_label)
    ax_curve.set_ylabel("Symmetric JS divergence")
    ax_curve.grid(alpha=0.2)
    ax_curve.legend(frameon=False, loc="upper left")

    x = np.arange(len(WINDOWS))
    pt_vals = [_window_delta(dense, "pt_side_gap_closure", window) for window in WINDOWS]
    it_vals = [_window_delta(dense, "it_side_gap_collapse", window) for window in WINDOWS]
    ax_pt.bar(x, pt_vals, color=[WINDOW_COLOR[window] for window in WINDOWS], width=0.62)
    ax_it.bar(x, it_vals, color=[WINDOW_COLOR[window] for window in WINDOWS], width=0.62)

    for idx, window in enumerate(WINDOWS):
        pt_points = [
            _window_delta(summary["models"][model], "pt_side_gap_closure", window)
            for model in models
        ]
        it_points = [
            _window_delta(summary["models"][model], "it_side_gap_collapse", window)
            for model in models
        ]
        ax_pt.scatter(np.full(len(pt_points), idx), pt_points, color="#222222", s=18, zorder=3)
        ax_it.scatter(np.full(len(it_points), idx), it_points, color="#222222", s=18, zorder=3)

    for ax, vals, title in (
        (ax_pt, pt_vals, "PT-side gap closure\nJS(B_window, C) - JS(A', C)"),
        (ax_it, it_vals, "IT-side gap collapse\nJS(D_window, A) - JS(C, A)"),
    ):
        ax.axhline(0, color="#666666", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([window.title() for window in WINDOWS])
        ax.set_title(title)
        ax.set_ylabel("Final-20% weighted ΔJS")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Exp16 matched-prefix native-JS replay\nDirect same-layer divergence under the frozen exp14 teacher stream",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_model_curves(summary: dict, out_path: Path) -> None:
    present_models = [model for model in MODEL_ORDER if model in summary["models"]]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for ax, model in zip(axes.flat, present_models, strict=False):
        payload = summary["models"][model]
        x_layers = np.arange(payload["n_layers"])
        ax.plot(x_layers, _pair_curve(payload, "JS_AC"), color="#1f4b99", linewidth=2.0)
        ax.axvline(payload["corrective_onset"] - 0.5, color="#444444", linestyle="--", linewidth=0.9)
        ax.axvspan(payload["final_region_start"] - 0.5, payload["n_layers"] - 0.5, color="#E8EEF9", alpha=0.8)
        ax.set_title(MODEL_DISPLAY[model])
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Symmetric JS divergence")
    for ax in axes.flat[len(present_models) :]:
        ax.axis("off")
    fig.suptitle("Appendix: per-model matched-prefix JS(A', C) curves", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_controls(summary: dict, out_path: Path) -> None:
    dense = summary["dense5"]
    deepseek = summary.get("deepseek", {})
    x = np.arange(len(WINDOWS))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax_pt_host, ax_it_host, ax_deepseek = axes

    pt_host = [
        dense["host_local_pt"].get(window, {}).get("final_20pct", {}).get("mean")
        for window in WINDOWS
    ]
    it_host = [
        dense["host_local_it"].get(window, {}).get("final_20pct", {}).get("mean")
        for window in WINDOWS
    ]
    ax_pt_host.bar(x, pt_host, color=[WINDOW_COLOR[window] for window in WINDOWS], width=0.62)
    ax_it_host.bar(x, it_host, color=[WINDOW_COLOR[window] for window in WINDOWS], width=0.62)
    ax_pt_host.set_title("Host-local PT perturbation\nJS(B_window, A')")
    ax_it_host.set_title("Host-local IT perturbation\nJS(D_window, C)")

    if deepseek.get("models"):
        ds_pt = [_window_delta(deepseek, "pt_side_gap_closure", window) for window in WINDOWS]
        ds_it = [_window_delta(deepseek, "it_side_gap_collapse", window) for window in WINDOWS]
        width = 0.34
        ax_deepseek.bar(x - width / 2, ds_pt, width, color="#4C78A8", label="PT side")
        ax_deepseek.bar(x + width / 2, ds_it, width, color="#B279A2", label="IT side")
        ax_deepseek.legend(frameon=False)
        ax_deepseek.set_title("DeepSeek appendix\nFinal-20% target-gap ΔJS")
    else:
        ax_deepseek.axis("off")

    for ax in (ax_pt_host, ax_it_host, ax_deepseek):
        if ax.has_data():
            ax.axhline(0, color="#666666", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([window.title() for window in WINDOWS])
            ax.grid(axis="y", alpha=0.2)
    ax_pt_host.set_ylabel("Weighted mean JS")
    fig.suptitle("Appendix: host-local controls and DeepSeek case", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot exp16 matched-prefix JS replay summaries.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = _read_json(args.summary)
    out_dir = args.out_dir or args.summary.parent
    _plot_main(summary, out_dir / "exp16_js_main.png")
    _plot_model_curves(summary, out_dir / "exp16_js_appendix_models.png")
    _plot_controls(summary, out_dir / "exp16_js_appendix_controls.png")
    print(f"[exp16] wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
