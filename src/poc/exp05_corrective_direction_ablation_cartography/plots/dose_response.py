from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_dose_response(rows: list[dict], x_key: str, output_path: str | Path, title: str) -> None:
    if not rows:
        return

    # Group by benchmark+metric so each measure gets its own line.
    # Mixing all benchmarks into one line conflates different scales and tasks.
    groups: dict[str, list[dict]] = {}
    for r in rows:
        key = f"{r.get('benchmark', '?')}/{r.get('metric', '?')}"
        groups.setdefault(key, []).append(r)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group_rows in sorted(groups.items()):
        group_rows = sorted(group_rows, key=lambda r: r[x_key])
        ax.plot([r[x_key] for r in group_rows], [r["value"] for r in group_rows], marker="o", label=label)
    ax.set_xlabel(x_key.replace("_", " "))
    ax.set_ylabel("score")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

