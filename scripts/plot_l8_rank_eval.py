from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = [
    "gemma3_4b",
    "llama31_8b",
    "qwen3_4b",
    "mistral_7b",
    "olmo2_7b",
    "deepseek_v2_lite",
]


def _label(model: str) -> str:
    return {
        "gemma3_4b": "Gemma",
        "llama31_8b": "Llama",
        "qwen3_4b": "Qwen",
        "mistral_7b": "Mistral",
        "olmo2_7b": "OLMo",
        "deepseek_v2_lite": "DeepSeek",
    }[model]


def _load_pair(base: Path, model: str) -> dict[str, float]:
    out: dict[str, float] = {"model": model}
    for variant in ("pt", "it"):
        rank = json.loads((base / model / variant / "L8_rank_metrics.json").read_text())
        idp = json.loads((base / model / variant / "L8_id_profile.json").read_text())
        start = rank["summary"]["late_start_layer"]
        late_ids = np.asarray(idp["intrinsic_dim"][start:], dtype=np.float64)
        out[f"id_late_mean_{variant}"] = float(np.nanmean(late_ids))
        out[f"pr_late_mean_{variant}"] = float(rank["summary"]["participation_ratio_late_mean"])
        out[f"erank_late_mean_{variant}"] = float(rank["summary"]["effective_rank_late_mean"])
        out[f"pc1_late_mean_{variant}"] = float(rank["summary"]["pc1_ratio_late_mean"])
    out["delta_id_late_mean"] = out["id_late_mean_it"] - out["id_late_mean_pt"]
    out["delta_pr_late_mean"] = out["pr_late_mean_it"] - out["pr_late_mean_pt"]
    out["delta_erank_late_mean"] = out["erank_late_mean_it"] - out["erank_late_mean_pt"]
    out["delta_pc1_late_mean"] = out["pc1_late_mean_it"] - out["pc1_late_mean_pt"]
    return out


def _write_summary(rows: list[dict[str, float]], out_json: Path) -> None:
    payload = {
        "rows": rows,
        "means": {
            k: float(np.mean([row[k] for row in rows]))
            for k in (
                "delta_id_late_mean",
                "delta_pr_late_mean",
                "delta_erank_late_mean",
                "delta_pc1_late_mean",
            )
        },
        "counts": {
            k: {
                "positive": int(sum(row[k] > 0 for row in rows)),
                "negative": int(sum(row[k] < 0 for row in rows)),
            }
            for k in (
                "delta_id_late_mean",
                "delta_pr_late_mean",
                "delta_erank_late_mean",
                "delta_pc1_late_mean",
            )
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))


def _plot(rows: list[dict[str, float]], out_png: Path) -> None:
    labels = [_label(row["model"]) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    specs = [
        ("delta_id_late_mean", "Late TwoNN ID (IT - PT)", "#2563eb"),
        ("delta_pr_late_mean", "Late Participation Ratio (IT - PT)", "#059669"),
        ("delta_erank_late_mean", "Late Effective Rank (IT - PT)", "#d97706"),
        ("delta_pc1_late_mean", "Late PC1 Variance Ratio (IT - PT)", "#dc2626"),
    ]
    for ax, (key, title, color) in zip(axes.flat, specs, strict=True):
        vals = [row[key] for row in rows]
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.bar(x, vals, color=color, alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.suptitle("L8 Covariance-Rank Corroboration Summary", fontsize=14, fontweight="bold")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/cross_model/l8_rank_eval_20260414_v2"),
    )
    args = parser.parse_args()

    base = args.base_dir
    rows = [_load_pair(base, model) for model in MODELS]
    _write_summary(rows, base / "l8_rank_summary_local.json")
    _plot(rows, base / "l8_rank_deltas.png")


if __name__ == "__main__":
    main()
