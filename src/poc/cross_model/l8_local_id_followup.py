from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import skdim


MODELS = [
    "gemma3_4b",
    "llama31_8b",
    "qwen3_4b",
    "mistral_7b",
    "olmo2_7b",
    "deepseek_v2_lite",
]

ESTIMATORS = ("twonn", "mle", "lpca")


def _label(model: str) -> str:
    return {
        "gemma3_4b": "Gemma",
        "llama31_8b": "Llama",
        "qwen3_4b": "Qwen",
        "mistral_7b": "Mistral",
        "olmo2_7b": "OLMo",
        "deepseek_v2_lite": "DeepSeek",
    }[model]


def _load_residuals(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with np.load(path) as data:
        residuals = {k: data[k] for k in data.files}
    prompt_ids = list(residuals.keys())
    if not prompt_ids:
        raise ValueError(f"No residuals found in {path}")
    return prompt_ids, residuals


def _stack_layer(residuals: dict[str, np.ndarray], prompt_ids: list[str], layer_idx: int) -> np.ndarray:
    return np.stack([residuals[pid][layer_idx] for pid in prompt_ids], axis=0).astype(np.float64, copy=False)


def _estimate(name: str, X: np.ndarray) -> float:
    if name == "twonn":
        est = skdim.id.TwoNN(discard_fraction=0.1)
    elif name == "mle":
        est = skdim.id.MLE(K=10)
    elif name == "lpca":
        est = skdim.id.lPCA(ver="FO", verbose=False)
    else:
        raise ValueError(name)
    return float(est.fit_transform(X))


def _summarize_curve(values: list[float], late_fraction: float) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    start = max(int(np.floor(len(arr) * (1.0 - late_fraction))), 0)
    return {
        "late_start_layer": int(start),
        "last": float(arr[-1]),
        "late_mean": float(np.nanmean(arr[start:])),
    }


def _run_variant(
    npz_path: str,
    *,
    max_prompts: int | None,
    seed: int,
    late_fraction: float,
) -> dict[str, Any]:
    path = Path(npz_path)
    prompt_ids, residuals = _load_residuals(path)
    if max_prompts is not None and len(prompt_ids) > max_prompts:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(prompt_ids), size=max_prompts, replace=False)
        prompt_ids = [prompt_ids[i] for i in sorted(idx.tolist())]

    n_layers = residuals[prompt_ids[0]].shape[0]
    curves = {name: [] for name in ESTIMATORS}
    for layer_idx in range(n_layers):
        X = _stack_layer(residuals, prompt_ids, layer_idx)
        for name in ESTIMATORS:
            curves[name].append(_estimate(name, X))

    summary = {name: _summarize_curve(curves[name], late_fraction) for name in ESTIMATORS}
    parts = path.parts
    variant = parts[-2]
    model = parts[-3]
    return {
        "model": model,
        "variant": variant,
        "npz_path": str(path),
        "n_prompts": len(prompt_ids),
        "n_layers": n_layers,
        "curves": curves,
        "summary": summary,
    }


def _write_plot(rows: list[dict[str, Any]], out_png: Path) -> None:
    x = np.arange(len(MODELS))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    specs = [
        ("twonn", "Late TwoNN (IT - PT)", "#2563eb"),
        ("mle", "Late MLE (IT - PT)", "#059669"),
        ("lpca", "Late lPCA (IT - PT)", "#d97706"),
    ]
    by_model = {row["model"]: {} for row in rows}
    for row in rows:
        by_model[row["model"]][row["variant"]] = row
    labels = [_label(m) for m in MODELS]
    for ax, (name, title, color) in zip(axes, specs, strict=True):
        vals = [
            by_model[m]["it"]["summary"][name]["late_mean"] - by_model[m]["pt"]["summary"][name]["late_mean"]
            for m in MODELS
        ]
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.bar(x, vals, color=color, alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.suptitle("L8 Local-ID Follow-up Across Estimators", fontsize=14, fontweight="bold")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical local-ID follow-up on merged L8 residuals.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/cross_model/l8_rank_eval_20260414_v2"),
    )
    parser.add_argument("--max-prompts", type=int, default=1400)
    parser.add_argument("--late-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    tasks = [
        str(args.base_dir / model / variant / "L8_residuals.npz")
        for model in MODELS
        for variant in ("pt", "it")
    ]

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                _run_variant,
                task,
                max_prompts=args.max_prompts,
                seed=args.seed,
                late_fraction=args.late_fraction,
            ): task
            for task in tasks
        }
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            print(
                f"[l8-local-id] done model={row['model']} variant={row['variant']} "
                f"n_layers={row['n_layers']} n_prompts={row['n_prompts']}"
            )

    rows.sort(key=lambda r: (MODELS.index(r["model"]), 0 if r["variant"] == "pt" else 1))
    by_model = {model: {} for model in MODELS}
    for row in rows:
        by_model[row["model"]][row["variant"]] = row
    aggregates: dict[str, Any] = {}
    for est in ESTIMATORS:
        deltas = [
            by_model[model]["it"]["summary"][est]["late_mean"]
            - by_model[model]["pt"]["summary"][est]["late_mean"]
            for model in MODELS
        ]
        aggregates[est] = {
            "late_mean_deltas": {model: float(delta) for model, delta in zip(MODELS, deltas, strict=True)},
            "positive": int(sum(delta > 0 for delta in deltas)),
            "negative": int(sum(delta < 0 for delta in deltas)),
            "mean_delta": float(np.mean(deltas)),
        }
    out_json = args.base_dir / "l8_local_id_followup.json"
    out_json.write_text(json.dumps({"rows": rows, "aggregates": aggregates}, indent=2))
    _write_plot(rows, args.base_dir / "l8_local_id_followup.png")
    print(f"[l8-local-id] wrote {out_json}")


if __name__ == "__main__":
    main()
