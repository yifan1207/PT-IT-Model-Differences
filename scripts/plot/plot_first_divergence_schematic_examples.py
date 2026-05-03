#!/usr/bin/env python3
"""Draw the paper-facing first-divergence schematic and examples figure."""

from __future__ import annotations

import gzip
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/paper_synthesis/first_divergence_schematic_examples.png"
DENSE5_RECORD_ROOT = (
    REPO
    / "results/exp23_midlate_interaction_suite/"
    / "exp23_dense5_full_h100x8_20260426_sh4_rw4/residual_factorial/raw_shared"
)
DATASET = REPO / "data/eval_dataset_v2_holdout_0600_1199.jsonl"

EXAMPLES = [
    ("gemma3_4b", "Gemma 3 4B", "v2_GOV-CONV_0832"),
    ("olmo2_7b", "OLMo 2 7B", "v2_GOV-CONV_0792"),
    ("qwen3_4b", "Qwen 3 4B", "v2_GOV-FORMAT_0748"),
    ("llama31_8b", "Llama 3.1 8B", "v2_GOV-CONV_1033"),
]


def load_dataset() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with DATASET.open() as f:
        for line in f:
            row = json.loads(line)
            out[row["id"]] = row
    return out


def prompt_text(row: dict) -> str:
    formats = row.get("formats") or {}
    return formats.get("B") or formats.get("A") or row.get("prompt") or row["id"]


def load_example(model: str, label: str, prompt_id: str, dataset: dict[str, dict]) -> dict:
    path = DENSE5_RECORD_ROOT / model / "records.jsonl.gz"
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            if row["prompt_id"] != prompt_id:
                continue
            event = row["events"]["first_diff"]["event"]
            pt = event["pt_token"]["token_str"]
            it = event["it_token"]["token_str"]
            meta = dataset[prompt_id]
            return {
                "model": label,
                "category": meta["category"],
                "source": meta["source"],
                "position": int(event["step"]),
                "pt": pt,
                "it": it,
                "prompt": prompt_text(meta),
            }
    raise KeyError((model, prompt_id))


def box(ax, xy, text, *, width=0.17, height=0.12, fc="#f7f7f4", ec="#333333", size=9):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.012",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=size)


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.0,
            color="#333333",
        )
    )


def draw_schematic(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.96, "A. Four hybrid passes at the first divergence", fontsize=13, weight="bold")
    ax.text(
        0.0,
        0.88,
        "Same raw prompt and generated prefix. The readout is Y = logit(t_IT) - logit(t_PT).",
        fontsize=9,
        color="#444444",
    )

    box(ax, (0.02, 0.54), "shared\nprefix", width=0.14, height=0.13, fc="#eef2f7")
    box(ax, (0.25, 0.66), "U_PT", width=0.12, height=0.09, fc="#f5efe6")
    box(ax, (0.25, 0.42), "U_IT", width=0.12, height=0.09, fc="#e8f2eb")
    box(ax, (0.47, 0.72), "L_PT", width=0.12, height=0.08, fc="#f5efe6")
    box(ax, (0.47, 0.58), "L_IT", width=0.12, height=0.08, fc="#e8f2eb")
    box(ax, (0.47, 0.48), "L_PT", width=0.12, height=0.08, fc="#f5efe6")
    box(ax, (0.47, 0.34), "L_IT", width=0.12, height=0.08, fc="#e8f2eb")

    for y in [0.705, 0.61]:
        arrow(ax, (0.16, 0.605), (0.25, y))
    for y in [0.465, 0.38]:
        arrow(ax, (0.16, 0.605), (0.25, y))
    arrow(ax, (0.37, 0.705), (0.47, 0.76))
    arrow(ax, (0.37, 0.705), (0.47, 0.62))
    arrow(ax, (0.37, 0.465), (0.47, 0.52))
    arrow(ax, (0.37, 0.465), (0.47, 0.38))

    outputs = [
        (0.70, 0.72, "Y(U_PT,L_PT)"),
        (0.70, 0.58, "Y(U_PT,L_IT)"),
        (0.70, 0.48, "Y(U_IT,L_PT)"),
        (0.70, 0.34, "Y(U_IT,L_IT)"),
    ]
    for x, y, label in outputs:
        box(ax, (x, y), label, width=0.19, height=0.08, fc="#ffffff", size=8)
        arrow(ax, (0.59, y + 0.04), (x, y + 0.04))

    ax.text(
        0.02,
        0.13,
        "interaction = [Y(U_IT,L_IT) - Y(U_IT,L_PT)] - [Y(U_PT,L_IT) - Y(U_PT,L_PT)]",
        fontsize=9,
        family="monospace",
        color="#222222",
    )
    ax.text(
        0.02,
        0.06,
        "Interpretation: how much more the IT late stack helps when it reads an IT-shaped upstream state.",
        fontsize=9,
        color="#444444",
    )


def draw_examples(ax, examples: list[dict]):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.96, "B. Concrete divergent-token examples", fontsize=13, weight="bold")

    cols = [0.00, 0.17, 0.67, 0.76, 0.89]
    headers = ["Model", "Prompt fragment", "pos.", "PT token", "IT token"]
    for x, h in zip(cols, headers):
        ax.text(x, 0.87, h, fontsize=9, weight="bold")
    ax.plot([0, 0.98], [0.84, 0.84], color="#333333", linewidth=0.9)

    y = 0.75
    for ex in examples:
        prompt = textwrap.shorten(ex["prompt"].replace("\n", " "), width=50, placeholder="...")
        prompt = textwrap.fill(prompt, width=34)
        ax.text(cols[0], y, ex["model"], fontsize=8.2, va="top")
        ax.text(cols[1], y, prompt, fontsize=8.2, va="top")
        ax.text(cols[2], y, str(ex["position"]), fontsize=8.2, va="top", family="monospace")
        ax.text(cols[3], y, repr(ex["pt"]), fontsize=8.2, va="top", family="monospace", color="#8a4b08")
        ax.text(cols[4], y, repr(ex["it"]), fontsize=8.2, va="top", family="monospace", color="#126336")
        y -= 0.18

    ax.text(
        0.0,
        0.05,
        "Examples are illustrative; all statistics use the full first-divergence support.",
        fontsize=8.5,
        color="#555555",
    )


def main() -> None:
    dataset = load_dataset()
    examples = [load_example(*item, dataset) for item in EXAMPLES]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.4), gridspec_kw={"width_ratios": [1.0, 1.35]})
    draw_schematic(axes[0])
    draw_examples(axes[1], examples)
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=1.4)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
