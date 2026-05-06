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
OUT_PDF = REPO / "results/paper_synthesis/first_divergence_schematic_examples.pdf"
DENSE5_RECORD_ROOT = (
    REPO
    / "results/exp23_midlate_interaction_suite/"
    / "exp23_dense5_full_h100x8_20260426_sh4_rw4/residual_factorial/raw_shared"
)
DATASET = REPO / "data/eval_dataset_v2_holdout_0600_1199.jsonl"

EXAMPLES = [
    ("mistral_7b", "Mistral 7B", "v2_GOV-CONV_0832"),
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


def box(
    ax,
    xy,
    text,
    *,
    width=0.17,
    height=0.12,
    fc="#f7f7f4",
    ec="#333333",
    size=9,
    weight="normal",
):
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
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        weight=weight,
    )


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


def row_arrow(ax, x0, x1, y, *, gap=0.012):
    """Draw a horizontal row arrow that stops before adjacent boxes.

    FancyArrowPatch places the arrow head at the endpoint, so using exact box
    edges makes the head overlap the next box. The small explicit gap keeps the
    schematic readable after downscaling in the PDF.
    """
    arrow(ax, (x0 + gap, y), (x1 - gap, y))


def draw_schematic(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.96, "A. Four hybrid passes at the first divergence", fontsize=14, weight="bold")
    ax.text(
        0.0,
        0.88,
        r"Same raw prompt and generated prefix. Readout: $Y=\mathrm{logit}(t_{\mathrm{IT}})-\mathrm{logit}(t_{\mathrm{PT}})$.",
        fontsize=9.5,
        color="#444444",
    )

    prefix_x, prefix_y, prefix_w, prefix_h = 0.02, 0.31, 0.13, 0.45
    box(
        ax,
        (prefix_x, prefix_y),
        "shared\nprefix",
        width=prefix_w,
        height=prefix_h,
        fc="#eef2f7",
        size=9,
        weight="bold",
    )

    rows = [
        (0.69, r"$U_{\mathrm{PT}}$", "#f5efe6", r"$L_{\mathrm{PT}}$", "#f5efe6", r"$Y(U_{\mathrm{PT}},L_{\mathrm{PT}})$"),
        (0.56, r"$U_{\mathrm{PT}}$", "#f5efe6", r"$L_{\mathrm{IT}}$", "#e8f2eb", r"$Y(U_{\mathrm{PT}},L_{\mathrm{IT}})$"),
        (0.43, r"$U_{\mathrm{IT}}$", "#e8f2eb", r"$L_{\mathrm{PT}}$", "#f5efe6", r"$Y(U_{\mathrm{IT}},L_{\mathrm{PT}})$"),
        (0.30, r"$U_{\mathrm{IT}}$", "#e8f2eb", r"$L_{\mathrm{IT}}$", "#e8f2eb", r"$Y(U_{\mathrm{IT}},L_{\mathrm{IT}})$"),
    ]
    u_x, l_x, y_x = 0.23, 0.43, 0.65
    w_u, w_l, w_y, h = 0.14, 0.14, 0.29, 0.075
    for y, u_label, u_fc, l_label, l_fc, out_label in rows:
        center = y + h / 2
        box(ax, (u_x, y), u_label, width=w_u, height=h, fc=u_fc, size=10)
        box(ax, (l_x, y), l_label, width=w_l, height=h, fc=l_fc, size=10)
        box(ax, (y_x, y), out_label, width=w_y, height=h, fc="#ffffff", size=8.4)
        row_arrow(ax, prefix_x + prefix_w, u_x, center)
        row_arrow(ax, u_x + w_u, l_x, center)
        row_arrow(ax, l_x + w_l, y_x, center)

    ax.text(0.30, 0.79, "upstream state", fontsize=9, color="#555555", ha="center")
    ax.text(0.50, 0.79, "late stack", fontsize=9, color="#555555", ha="center")
    ax.text(0.795, 0.79, "scored margin", fontsize=9, color="#555555", ha="center")

    ax.text(
        0.02,
        0.13,
        r"interaction = [$Y(U_{\mathrm{IT}},L_{\mathrm{IT}})-Y(U_{\mathrm{IT}},L_{\mathrm{PT}})$]"
        r" - [$Y(U_{\mathrm{PT}},L_{\mathrm{IT}})-Y(U_{\mathrm{PT}},L_{\mathrm{PT}})$]",
        fontsize=8.8,
        color="#222222",
    )
    ax.text(
        0.02,
        0.06,
        "Interpretation: how much more the IT late stack helps when it reads an IT-shaped upstream state.",
        fontsize=9.3,
        color="#444444",
    )


def draw_examples(ax, examples: list[dict]):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.96, "B. Concrete divergent-token examples", fontsize=13, weight="bold")

    cols = [0.00, 0.20, 0.70, 0.78, 0.90]
    headers = ["Model", "Prompt fragment", "pos.", "PT token", "IT token"]
    for x, h in zip(cols, headers):
        ax.text(x, 0.87, h, fontsize=9.5, weight="bold")
    ax.plot([0, 0.98], [0.84, 0.84], color="#333333", linewidth=0.9)

    y = 0.75
    for ex in examples:
        prompt = textwrap.shorten(ex["prompt"].replace("\n", " "), width=58, placeholder="...")
        prompt = textwrap.fill(prompt, width=44)
        ax.text(cols[0], y, ex["model"], fontsize=9, va="top")
        ax.text(cols[1], y, prompt, fontsize=9, va="top")
        ax.text(cols[2], y, str(ex["position"]), fontsize=9, va="top", family="monospace")
        ax.text(cols[3], y, repr(ex["pt"]), fontsize=9, va="top", family="monospace", color="#8a4b08")
        ax.text(cols[4], y, repr(ex["it"]), fontsize=9, va="top", family="monospace", color="#126336")
        y -= 0.18

    ax.text(
        0.0,
        0.05,
        "Examples are illustrative; all statistics use the full first-divergence support.",
        fontsize=9,
        color="#555555",
    )


def main() -> None:
    dataset = load_dataset()
    examples = [load_example(*item, dataset) for item in EXAMPLES]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.4), gridspec_kw={"height_ratios": [1.45, 1.0]})
    draw_schematic(axes[0])
    draw_examples(axes[1], examples)
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=1.4)
    fig.savefig(OUT, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(OUT)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
