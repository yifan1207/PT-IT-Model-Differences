"""
Summarize top corrective-layer transcoder features using Claude Opus 4.6 (extended thinking)
via OpenRouter.

Usage:
    OPENROUTER_API_KEY=sk-or-... uv run python -m src.poc.exp03_corrective_stage_characterization.summarize_features
    uv run python -m src.poc.exp03_corrective_stage_characterization.summarize_features --variant it --top-k 100
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


# ── filtering ────────────────────────────────────────────────────────────────

def _is_english_interpretable(label: str) -> bool:
    """Keep if the label is dominated by ASCII-Latin tokens and has at least one real English word."""
    tokens = [t.strip() for t in label.split(",") if t.strip()]
    if not tokens:
        return False
    # Each token: count what fraction of its chars are pure ASCII printable
    ascii_token_count = 0
    for tok in tokens:
        ascii_frac = sum(1 for c in tok if ord(c) < 128 and c.isprintable()) / max(len(tok), 1)
        if ascii_frac >= 0.85:
            ascii_token_count += 1
    # Need at least half the tokens to be ASCII-dominant
    if ascii_token_count < max(1, len(tokens) // 2):
        return False
    # Need at least one token that's a real word (≥3 ASCII letters, not pure punctuation/symbols)
    has_real_word = any(
        sum(1 for c in tok if c.isascii() and c.isalpha()) >= 3
        for tok in tokens
    )
    return has_real_word


# ── data loading ─────────────────────────────────────────────────────────────

def _load_summary(path: Path) -> dict[int, dict[str, np.ndarray]]:
    raw = np.load(path, allow_pickle=True)
    out: dict[int, dict[str, np.ndarray]] = {}
    for key in raw.files:
        prefix, _, layer_str = key.partition("_l")
        if not layer_str:
            continue
        layer = int(layer_str)
        out.setdefault(layer, {})[prefix] = raw[key]
    return out


def _rank_and_filter(
    summary: dict[int, dict[str, np.ndarray]],
    variant: str,
    layer_start: int,
    top_k: int,
) -> list[dict]:
    """Return top_k interpretable English features sorted by total activation."""
    from src.poc.shared.feature_labels import get_feature_labels

    rows = []
    for layer_i in range(layer_start, 34):
        if layer_i not in summary:
            continue
        d = summary[layer_i]
        if "sum" not in d or "count" not in d:
            continue
        sums   = d["sum"].astype(float)
        counts = d["count"].astype(int)
        labels = get_feature_labels(variant, layer_i)
        for feat_idx in np.flatnonzero(sums > 0):
            fi    = int(feat_idx)
            total = float(sums[fi])
            cnt   = int(counts[fi])
            mean  = total / cnt if cnt > 0 else 0.0
            lbl   = labels.get(fi, f"F{fi}")
            rows.append({"layer": layer_i, "feat": fi, "total": total, "count": cnt, "mean": mean, "label": lbl})

    rows.sort(key=lambda x: x["total"], reverse=True)

    # Take top_k * 3 by activation first, then filter for English (avoids scanning all 16k*14 features)
    candidates = rows[: top_k * 4]
    kept = [r for r in candidates if _is_english_interpretable(r["label"])]
    return kept[:top_k]


# ── prompt ────────────────────────────────────────────────────────────────────

def _feature_table(rows: list[dict]) -> str:
    grand_total = sum(r["total"] for r in rows)
    cumulative  = 0.0
    lines = []
    for i, r in enumerate(rows, 1):
        cumulative += r["total"]
        pct = cumulative / grand_total * 100
        lines.append(
            f"{i:3d}. [L{r['layer']}:F{r['feat']:5d}]  {r['label']:<50s}"
            f"  total={r['total']:10.0f}  mean={r['mean']:.3f}  count={r['count']:6d}  (cum {pct:.1f}%)"
        )
    return "\n".join(lines)


def _build_prompt(it_rows: list[dict], pt_rows: list[dict], layer_start: int) -> str:
    lines = [
        "You are analyzing internal transcoder features of two variants of Gemma 3 4B:",
        "  - IT = instruction-tuned (RLHF/alignment fine-tuned)",
        "  - PT = pretrained base model (no alignment)",
        "",
        f"These features come from the corrective processing stage (layers {layer_start}–33 of 34 total).",
        "They were collected across 2,936 generation steps on math, factual, and safety prompts.",
        "Each feature label shows the output tokens that feature most strongly BOOSTS when it fires.",
        "Format: `Rank. [L<layer>:F<idx>] <top boosted tokens>  | total=<activation_sum>  mean=<per_step>  count=<times_fired>`",
        "",
        "Higher `total` = more globally influential. Weight top-ranked features more in your analysis.",
        "",
        "YOUR TASK:",
        "1. Identify 5–8 thematic groups that appear in EITHER or BOTH variants.",
        "2. For each group:",
        "   - Short name",
        "   - Which features belong (cite rank numbers from IT and/or PT lists)",
        "   - Is this group IT-only, PT-only, or shared? What does that tell us?",
        "   - 2–3 sentences on what computation they represent",
        "   - Layer distribution (early corrective 20–25 vs late 26–33)",
        "3. Highlight the most striking IT vs PT differences — features present in one but absent/weak in the other.",
        "4. Call out anomalous or hard-to-categorize features.",
        "5. Write a 4–5 sentence conclusion: what does comparing these two sets reveal about what alignment/RLHF adds to the corrective stage?",
        "",
        "Be direct and specific. This is for a mechanistic interpretability research paper. No hedging filler.",
        "",
        f"═══ IT TOP {len(it_rows)} FEATURES (instruction-tuned, corrective layers {layer_start}–33) ═══",
        "",
        _feature_table(it_rows),
        "",
        f"═══ PT TOP {len(pt_rows)} FEATURES (pretrained base, corrective layers {layer_start}–33) ═══",
        "",
        _feature_table(pt_rows),
    ]
    return "\n".join(lines)


# ── API call ──────────────────────────────────────────────────────────────────

def _call_openrouter(prompt: str, api_key: str) -> str:
    import urllib.request, json as _json

    payload = _json.dumps({
        "model": "anthropic/claude-opus-4.6",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000,
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/structural-semantic-features",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = _json.loads(resp.read())

    return data["choices"][0]["message"]["content"]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-start", default=20, type=int)
    parser.add_argument("--top-k",       default=100, type=int)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY before running.")

    from src.poc.exp03_corrective_stage_characterization.config import Exp3Config

    it_path = Path(Exp3Config(model_variant="it").run_dir) / "feature_importance_summary.npz"
    pt_path = Path(Exp3Config(model_variant="pt").run_dir) / "feature_importance_summary.npz"
    for p in (it_path, pt_path):
        if not p.exists():
            sys.exit(f"Summary not found: {p}")

    print(f"Loading IT summary from {it_path} ...")
    it_summary = _load_summary(it_path)
    print(f"Loading PT summary from {pt_path} ...")
    pt_summary = _load_summary(pt_path)

    print(f"Ranking top {args.top_k} English features per variant ...")
    it_rows = _rank_and_filter(it_summary, "it", args.layer_start, args.top_k)
    pt_rows = _rank_and_filter(pt_summary, "pt", args.layer_start, args.top_k)
    print(f"  IT → {len(it_rows)} features,  PT → {len(pt_rows)} features")

    prompt = _build_prompt(it_rows, pt_rows, args.layer_start)
    out_dir = Path("results/exp3")
    (out_dir / "feature_contrast_prompt.txt").write_text(prompt)
    print(f"\nSending IT+PT contrast to Claude Opus 4.6 via OpenRouter ...")

    answer = _call_openrouter(prompt, api_key)

    md_path = out_dir / "feature_contrast_llm.md"
    md_path.write_text(
        f"# IT vs PT Feature Contrast — corrective layers {args.layer_start}–33\n\n"
        f"_Claude Opus 4.6 via OpenRouter. Top {args.top_k} English features per variant._\n\n"
        + answer
    )
    print(f"\nDone → {md_path}")


if __name__ == "__main__":
    main()
