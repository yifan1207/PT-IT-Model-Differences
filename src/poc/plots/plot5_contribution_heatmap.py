"""
Plot 5: Contribution heatmap — one per prompt.

For one prompt, selects the top-20 features by attribution and shows their
token-level contribution profiles side by side.

  rows    = top-20 features sorted by attribution (highest at top)
  columns = top-50 tokens (selected by max |c_value| across all 20 features)
  color   = contribution value  (red = +promote, blue = −suppress, white = 0)

The correct target token column is annotated with a box.

Each feature's contribution to token t is:
  c(f, t) = activation(f) × (W_dec[f] @ W_U)[t]

These are stored in top50_contributions for each feature. Tokens not in a feature's
stored top-50 are treated as 0 (their contribution is small by construction).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def make_plot(results: list[dict], output_dir: str) -> None:
    """Generate one heatmap per prompt. output_dir is a directory, not a file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for r in results:
        _make_single_heatmap(r, output_dir)


def _make_single_heatmap(prompt_result: dict, output_dir: str) -> None:
    prompt_id = prompt_result["prompt_id"]
    correct_token = prompt_result["correct_token"].strip()
    features = prompt_result["features"]

    if not features:
        return

    # Select top-20 features by attribution
    top_features = sorted(features, key=lambda f: f["attribution"], reverse=True)[:20]

    # Build column set: union of each feature's top-50 token_ids,
    # then keep the 50 with highest max |c_value| across all selected features
    token_cvals: dict[int, dict] = {}  # token_id → {"token_str": str, "max_abs": float}
    for f in top_features:
        for entry in f["top50_contributions"]:
            tid = entry["token_id"]
            if tid not in token_cvals or abs(entry["c_value"]) > token_cvals[tid]["max_abs"]:
                token_cvals[tid] = {
                    "token_str": entry["token_str"],
                    "max_abs": abs(entry["c_value"]),
                }

    # Sort by max absolute contribution, keep top-50
    sorted_tokens = sorted(token_cvals.items(), key=lambda kv: kv[1]["max_abs"], reverse=True)
    col_token_ids = [tid for tid, _ in sorted_tokens[:50]]
    col_labels = [token_cvals[tid]["token_str"] for tid in col_token_ids]

    # Build contribution matrix: shape [n_rows, n_cols]
    # For cells outside a feature's stored top-50, contribution is ~0
    n_rows = len(top_features)
    n_cols = len(col_token_ids)
    matrix = np.zeros((n_rows, n_cols))

    col_index = {tid: j for j, tid in enumerate(col_token_ids)}
    for i, f in enumerate(top_features):
        for entry in f["top50_contributions"]:
            j = col_index.get(entry["token_id"])
            if j is not None:
                matrix[i, j] = entry["c_value"]

    # Row labels: "L{layer} F{feat_idx} [{prompt_id}]"
    row_labels = [f"L{f['layer']:02d} F{f['feature_idx']} a={f['activation']:+.2f}" for f in top_features]

    # Find column of the correct target token (exact match on stripped token_str)
    target_col = next(
        (j for j, lbl in enumerate(col_labels) if lbl.strip() == correct_token),
        None,
    )

    # Plot
    vmax = max(abs(matrix).max(), 1e-6)
    fig_w = max(14, n_cols * 0.28)
    fig_h = max(6, n_rows * 0.38)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="contribution  c(f,t) = activation × logit_vec[t]",
                 fraction=0.02, pad=0.02)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([_clean_label(lbl) for lbl in col_labels],
                       rotation=90, fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Highlight target token column
    if target_col is not None:
        for row in range(n_rows):
            rect = mpatches.FancyBboxPatch(
                (target_col - 0.5, row - 0.5), 1, 1,
                boxstyle="square,pad=0", linewidth=1.5,
                edgecolor="gold", facecolor="none",
            )
            ax.add_patch(rect)
        ax.get_xticklabels()[target_col].set_color("goldenrod")
        ax.get_xticklabels()[target_col].set_fontweight("bold")

    ax.set_title(
        f"Plot 5 [{prompt_id}]: '{prompt_result['prompt'][:60]}'  →  '{correct_token}'\n"
        f"Top-{n_rows} features by attribution × top-{n_cols} tokens by |contribution|",
        fontsize=10,
    )

    fig.tight_layout()
    out_path = Path(output_dir) / f"plot5_heatmap_{prompt_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 5 [{prompt_id}] saved → {out_path}")


def _clean_label(s: str) -> str:
    """Replace non-printable chars and whitespace for readability in axis labels."""
    return repr(s)[1:-1][:12]  # repr strips quotes, limit to 12 chars
