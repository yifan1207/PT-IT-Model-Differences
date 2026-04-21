"""
Plot 5: Contribution heatmap — one per prompt.

For one prompt, selects the top-20 features by attribution and shows their
token-level contribution profiles side by side.

  rows    = top-20 features sorted by attribution (highest at top)
  columns = top-50 real tokens (selected by max |c_value| across all 20 features)
  color   = contribution value  (red = +promote, blue = −suppress, white = 0)

Row labels are enriched with the top-2 tokens each feature most promotes,
derived from top50_contributions. This gives a quick semantic read of each feature.

The correct target token column is annotated with a box.

Each feature's contribution to token t is:
  c(f, t) = activation(f) × (W_dec[f] @ W_U)[t]

These are stored in top50_contributions for each feature. Tokens not in a feature's
stored top-50 are treated as 0 (their contribution is small by construction).

Note: <unusedXXXX> entries are filtered at plot time so that heatmaps generated
before the real-token mask fix in attribution.py are still rendered correctly.
"""
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from pathlib import Path

# Build a font fallback list for full Unicode coverage.
# matplotlib 3.6+ supports per-glyph font fallback: each character uses the
# first font in the list that has a glyph for it.
#
#   Noto Sans CJK SC    – best-quality CJK (Chinese, Japanese, Korean)
#   Noto Sans Cuneiform – ancient Cuneiform (U+12000–U+123FF, SMP)
#                         Unifont does NOT cover Cuneiform; this font does.
#   Unifont             – GNU Unifont BMP: covers ALL of U+0000–U+FFFF
#                         including Bengali, Arabic, Devanagari, Thai, …
#   Unifont Upper       – GNU Unifont Upper: all SMP/TIP planes
#   DejaVu Sans         – always-available baseline for Latin/Greek/Cyrillic
#
# Download once with:
#   curl -L https://ftp.gnu.org/gnu/unifont/unifont-17.0.04/unifont-17.0.04.otf \
#        -o ~/.local/share/fonts/unifont.otf
#   curl -L https://ftp.gnu.org/gnu/unifont/unifont-17.0.04/unifont_upper-17.0.04.otf \
#        -o ~/.local/share/fonts/unifont_upper.otf
#   curl -L "https://fonts.gstatic.com/s/notosanscuneiform/v18/bMrrmTWK7YY-MF22aHGGd7H8PhJtvBDWgb8.ttf" \
#        -o ~/.local/share/fonts/NotoSansCuneiform-Regular.ttf
_FONT_PATHS = [
    ("Noto Sans CJK SC",    Path.home() / ".local/share/fonts/NotoSansCJK-Regular.otf"),
    ("Noto Sans Cuneiform", Path.home() / ".local/share/fonts/NotoSansCuneiform-Regular.ttf"),
    ("Unifont",             Path.home() / ".local/share/fonts/unifont.otf"),
    ("Unifont Upper",       Path.home() / ".local/share/fonts/unifont_upper.otf"),
]
_FONT_FAMILY: list[str] = []
for name, path in _FONT_PATHS:
    if path.exists():
        fm.fontManager.addfont(str(path))
        _FONT_FAMILY.append(name)
_FONT_FAMILY.append("DejaVu Sans")

_UNUSED_RE = re.compile(r"^<unused\d+>$")


def _is_unused(token_str: str) -> bool:
    return bool(_UNUSED_RE.match(token_str or ""))


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
    # filtered to exclude <unusedXXXX> tokens (handles data from before mask fix).
    # Then keep the 50 columns with highest max |c_value| across all selected features.
    token_cvals: dict[int, dict] = {}  # token_id → {"token_str": str, "max_abs": float}
    for f in top_features:
        for entry in f["top50_contributions"]:
            if _is_unused(entry["token_str"]):
                continue
            tid = entry["token_id"]
            if tid not in token_cvals or abs(entry["c_value"]) > token_cvals[tid]["max_abs"]:
                token_cvals[tid] = {
                    "token_str": entry["token_str"],
                    "max_abs": abs(entry["c_value"]),
                }

    if not token_cvals:
        print(f"  Plot 5 [{prompt_id}] skipped — no real tokens in top50_contributions")
        return

    # Sort by max absolute contribution across any of the top-20 features, keep top-50.
    # These are the tokens that the circuit-selected features push hardest —
    # not necessarily the model's top predicted tokens (those emerge from all features summed).
    sorted_tokens = sorted(token_cvals.items(), key=lambda kv: kv[1]["max_abs"], reverse=True)
    col_token_ids = [tid for tid, _ in sorted_tokens[:50]]
    col_raw_strs = [token_cvals[tid]["token_str"] for tid in col_token_ids]
    col_labels = [_fmt_token(s) for s in col_raw_strs]

    # Build contribution matrix: shape [n_rows, n_cols]
    n_rows = len(top_features)
    n_cols = len(col_token_ids)
    matrix = np.zeros((n_rows, n_cols))

    col_index = {tid: j for j, tid in enumerate(col_token_ids)}
    for i, f in enumerate(top_features):
        for entry in f["top50_contributions"]:
            j = col_index.get(entry["token_id"])
            if j is not None:
                matrix[i, j] = entry["c_value"]

    # Row labels: layer/feature + top-2 promoted real tokens for quick semantic read
    row_labels = [_make_row_label(f) for f in top_features]

    # Find column of the correct target token — match on raw string (before escaping/annotation)
    target_col = next(
        (j for j, raw in enumerate(col_raw_strs)
         if raw.strip().lower() == correct_token.lower()),
        None,
    )

    # Plot — extra height per column for two-line labels (token + prob)
    vmax = max(abs(matrix).max(), 1e-6)
    fig_w = max(18, n_cols * 0.40)
    fig_h = max(8, n_rows * 0.50)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="contribution  c(f,t) = activation × logit_vec[t]",
                 fraction=0.02, pad=0.02)

    # Apply font fallback list so each glyph picks its first capable font.
    # This handles Latin, CJK, Cuneiform, and other scripts without □ boxes.
    plt.rcParams["font.family"] = _FONT_FAMILY

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, ha="center", fontsize=7.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7.5)

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
        f"Rows: top-{n_rows} features by |activation × W_dec[f]·W_U[target]|, strongest at top\n"
        f"Cols: top-{n_cols} tokens by max |c(f,t)| across those features  ·  "
        f"Color = c(f,t) = activation × W_dec[f]·W_U[t]  ·  red=promotes  blue=suppresses  gold=correct answer",
        fontsize=8.5,
    )

    # Suppress benign "missing from DejaVu Sans" glyph warnings.
    # When a warning lists ONLY DejaVu Sans, the glyph was found in an earlier
    # font (e.g. Unifont) and rendered correctly; DejaVu is just the last resort
    # that also lacks it. Warnings listing ALL fonts indicate truly missing glyphs
    # (PUA chars are already filtered out in _fmt_token so these shouldn't occur).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Glyph .* missing from font", category=UserWarning)
        warnings.filterwarnings("ignore", message="Matplotlib currently does not support", category=UserWarning)
        fig.tight_layout()
        out_path = Path(output_dir) / f"plot5_heatmap_{prompt_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 5 [{prompt_id}] saved → {out_path}")


def _make_row_label(f: dict) -> str:
    """Build a rich row label: layer/feature + top-2 real tokens this feature most promotes."""
    base = f"L{f['layer']:02d} F{f['feature_idx']} a={f['activation']:+.2f}"

    # Pull top-2 real tokens by positive c_value (what this feature most promotes)
    pos_entries = sorted(
        (e for e in f["top50_contributions"]
         if e["c_value"] > 0 and not _is_unused(e["token_str"])),
        key=lambda e: e["c_value"],
        reverse=True,
    )[:2]

    if pos_entries:
        token_tags = ", ".join(_fmt_token(e["token_str"]) for e in pos_entries)
        return f"{base}  [{token_tags}]"
    return base


def _fmt_token(s: str) -> str:
    """Format a token string for axis display.

    - Strips SentencePiece leading space and marks it as '·' (word-initial marker).
    - Removes Private Use Area chars (U+E000–U+F8FF, U+F0000+): these are
      tokenizer encoding artifacts with no visual glyph in any standard font.
    - All other scripts (CJK, Bengali, Arabic, Cuneiform, …) are kept as-is
      and rendered by the font fallback stack (Unifont covers the BMP entirely).
    - Truncates to 14 chars.
    """
    if not s:
        return "[empty]"
    if s.startswith(" "):
        s = "·" + s[1:]
    s = s.replace("\n", "↵").replace("\t", "→")
    # Strip Private Use Area codepoints — no standard font has these
    s = "".join(
        ch for ch in s
        if not (0xE000 <= ord(ch) <= 0xF8FF or 0xF0000 <= ord(ch))
    )
    return (s or "[pua]")[:14]
