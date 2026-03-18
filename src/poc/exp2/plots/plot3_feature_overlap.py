"""
Plot 3: Feature set overlap (Jaccard similarity) between categories per layer.

For each layer, computes the Jaccard similarity between the sets of features
that were active in any prompt from each pair of categories:

  J(A, B)[layer] = |active_A[layer] ∩ active_B[layer]| / |active_A[layer] ∪ active_B[layer]|

where active_C[layer] = union of all feature indices that fired in any prompt in category C.

High Jaccard → categories recruit overlapping feature sets → similar mechanism.
Low Jaccard → categories use distinct features → mechanistically different.

Requires the .npz file produced by collect_all / save_results.

  X-axis : transformer layer
  Y-axis : Jaccard similarity (0–1)
  Lines  : IC∩OOC, IC∩R, OOC∩R
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


PAIR_STYLE = {
    ("in_context", "out_of_context"): {"color": "#9C27B0", "label": "IC ∩ OOC"},
    ("in_context", "reasoning"):      {"color": "#F44336", "label": "IC ∩ R"},
    ("out_of_context", "reasoning"):  {"color": "#795548", "label": "OOC ∩ R"},
}


def _load_active_sets(npz_path: Path) -> dict[str, dict[int, set]]:
    """Load .npz and build category → layer → set of active feature indices.

    npz keys are prompt_ids. Each value is a np.ndarray of shape [n_steps, N_LAYERS]
    with dtype=object (each cell is an int32 array of active feature indices).

    We need to cross-reference with results to know each prompt's category.
    Returns None if npz_path doesn't exist.
    """
    if not npz_path.exists():
        return None
    data = np.load(str(npz_path), allow_pickle=True)
    return data  # caller does cross-ref with results


def _build_per_category_active_sets(results: list[dict], npz_path: Path
                                    ) -> dict[str, dict[int, set]] | None:
    """Returns category → layer → set[feature_idx] or None if .npz missing."""
    npz = _load_active_sets(npz_path)
    if npz is None:
        return None

    pid_to_cat = {r["prompt_id"]: r["category"] for r in results}
    n_layers = len(results[0]["l0"][0]) if results else 34

    # category → layer → set
    cat_sets: dict[str, dict[int, set]] = defaultdict(lambda: defaultdict(set))

    for pid in npz.files:
        cat = pid_to_cat.get(pid)
        if cat is None:
            continue
        arr = npz[pid]  # [n_steps, n_layers] object array
        n_steps, n_layers_arr = arr.shape
        for step in range(n_steps):
            for layer in range(n_layers_arr):
                idxs = arr[step, layer]
                if idxs is not None and len(idxs) > 0:
                    cat_sets[cat][layer].update(idxs.tolist())

    return dict(cat_sets)


def make_plot(results: list[dict], output_dir: str, npz_path: str | None = None) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if npz_path is None:
        # Infer from output_dir (sibling of plots/)
        npz_path = Path(output_dir).parent / "exp2_results.npz"
    else:
        npz_path = Path(npz_path)

    cat_sets = _build_per_category_active_sets(results, npz_path)

    if cat_sets is None:
        print(f"  Plot 3 skipped — active feature data not found: {npz_path}")
        return

    n_layers = len(results[0]["l0"][0]) if results else 34
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(11, 5))

    for (cat_a, cat_b), style in PAIR_STYLE.items():
        sets_a = cat_sets.get(cat_a, {})
        sets_b = cat_sets.get(cat_b, {})
        jaccards = []
        for layer in range(n_layers):
            a = sets_a.get(layer, set())
            b = sets_b.get(layer, set())
            union = len(a | b)
            if union == 0:
                jaccards.append(np.nan)
            else:
                jaccards.append(len(a & b) / union)
        ax.plot(layers, jaccards, color=style["color"], label=style["label"], lw=2)

    ax.set_xlabel("Transformer layer", fontsize=11)
    ax.set_ylabel("Jaccard similarity", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Plot 3: Feature set overlap (Jaccard) between categories per layer\n"
        "High = categories share features; Low = mechanistically distinct",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, n_layers - 1)
    fig.tight_layout()

    out_path = Path(output_dir) / "plot3_feature_overlap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 3 saved → {out_path}")
