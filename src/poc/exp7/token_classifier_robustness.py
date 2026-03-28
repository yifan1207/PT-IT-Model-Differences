"""Token classifier specification and robustness check (Exp7 0E).

Documents the structural-token classifier used in exp3/exp6 and tests
sensitivity to boundary shifts. Two perturbation tests:
  1. STRUCTURAL ↔ DISCOURSE boundary: reclassify 500 boundary-adjacent tokens
  2. FUNCTION ↔ CONTENT boundary: reclassify 500 short ambiguous words

Usage:
  uv run python -m src.poc.exp7.token_classifier_robustness \\
      --sample-outputs results/exp6/merged_A1_it_v4/sample_outputs.jsonl \\
      --output-dir results/exp7/0E/

  # Quick test
  uv run python -m src.poc.exp7.token_classifier_robustness \\
      --sample-outputs results/exp6/merged_A1_it_v4/sample_outputs.jsonl \\
      --output-dir results/exp7/0E/ --n-outputs 200

Outputs:
  results/exp7/0E/classifier_documentation.json
  results/exp7/0E/classifier_robustness.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


# ── Import classifier ─────────────────────────────────────────────────────────

def _get_classifier():
    from src.poc.exp3.analysis.word_categories import (
        classify_generated_tokens_by_word,
        STRUCTURAL_PATTERN,
        PUNCTUATION_PATTERN,
        FUNCTION_WORDS,
        DISCOURSE_WORDS,
        DISCOURSE_PHRASES,
        WordCategory,
    )
    return classify_generated_tokens_by_word, {
        "structural_pattern": STRUCTURAL_PATTERN.pattern,
        "punctuation_pattern": PUNCTUATION_PATTERN.pattern,
        "function_words": sorted(FUNCTION_WORDS),
        "discourse_words": sorted(DISCOURSE_WORDS),
        "discourse_phrases": sorted(DISCOURSE_PHRASES),
    }


def _score_str(text: str) -> float:
    try:
        from src.poc.exp3.analysis.word_categories import classify_generated_tokens_by_word
        words = text.split()
        if not words:
            return 0.0
        cats = classify_generated_tokens_by_word([{"token_str": w + " "} for w in words])
        gov = {"STRUCTURAL", "DISCOURSE", "PUNCTUATION"}
        return sum(1 for c in cats if c in gov) / max(len(cats), 1)
    except Exception:
        return float("nan")


# ── Documentation ─────────────────────────────────────────────────────────────

def document_classifier(texts: list[str], n_examples: int = 20) -> dict:
    """Count token occurrences per category and sample examples."""
    classify, patterns = _get_classifier()

    category_counts: Counter[str] = Counter()
    category_examples: dict[str, list[str]] = {}

    for text in texts[:2000]:  # cap for speed
        words = text.split()
        if not words:
            continue
        cats = classify([{"token_str": w + " "} for w in words])
        for word, cat in zip(words, cats):
            category_counts[cat] += 1
            if cat not in category_examples:
                category_examples[cat] = []
            if len(category_examples[cat]) < n_examples:
                clean = word.strip()
                if clean and clean not in category_examples[cat]:
                    category_examples[cat].append(clean)

    return {
        "category_counts": dict(category_counts),
        "category_examples": category_examples,
        "classifier_patterns": patterns,
        "n_texts_analyzed": min(len(texts), 2000),
        "categories": ["STRUCTURAL", "DISCOURSE", "PUNCTUATION", "FUNCTION", "CONTENT"],
    }


# ── Perturbation tests ────────────────────────────────────────────────────────

def _collect_boundary_tokens(
    texts: list[str],
    boundary: str,
    n_perturb: int = 500,
) -> list[str]:
    """Find tokens near a classification boundary.

    boundary='structural_discourse': discourse single-word tokens that could
      be reclassified as STRUCTURAL (short uppercase/markdown-like patterns)
    boundary='function_content': short content words (3-4 chars) near function
      word list (common but not in the hardcoded set)
    """
    from src.poc.exp3.analysis.word_categories import (
        classify_generated_tokens_by_word, DISCOURSE_WORDS, FUNCTION_WORDS,
    )

    candidate_counts: Counter[str] = Counter()

    for text in texts:
        words = text.split()
        if not words:
            continue
        cats = classify([{"token_str": w + " "} for w in words])
        for word, cat in zip(words, cats):
            w = word.strip().lower()
            if boundary == "structural_discourse":
                # Discourse tokens that are short or single-char (borderline structural)
                if cat == "DISCOURSE" and len(w) <= 7:
                    candidate_counts[w] += 1
            elif boundary == "function_content":
                # Short content words (3-4 chars) not in function set
                if cat == "CONTENT" and 2 <= len(w) <= 4:
                    candidate_counts[w] += 1

    # Return top-N most frequent candidates
    return [tok for tok, _ in candidate_counts.most_common(n_perturb)]


def _str_with_override(
    texts: list[str],
    override_tokens: set[str],
    override_category: str,
) -> list[float]:
    """Compute STR with certain tokens reclassified."""
    from src.poc.exp3.analysis.word_categories import (
        classify_generated_tokens_by_word, WordCategory,
    )
    gov = {"STRUCTURAL", "DISCOURSE", "PUNCTUATION"}
    scores = []
    for text in texts:
        words = text.split()
        if not words:
            continue
        cats = classify([{"token_str": w + " "} for w in words])
        gov_count = 0
        total = len(cats)
        for word, cat in zip(words, cats):
            effective_cat = override_category if word.strip().lower() in override_tokens else cat
            if effective_cat in gov:
                gov_count += 1
        scores.append(gov_count / max(total, 1))
    return scores


def perturbation_test(
    texts_by_condition: dict[str, list[str]],
    n_perturb: int = 500,
) -> dict:
    """Run both boundary perturbation tests.

    Returns delta STR statistics for each perturbation.
    """
    # Use baseline condition texts for finding boundary tokens
    baseline_texts = texts_by_condition.get(
        "A1_baseline",
        next(iter(texts_by_condition.values())) if texts_by_condition else [],
    )

    results = {}

    for boundary, target_cat in [
        ("structural_discourse", "STRUCTURAL"),  # reclassify discourse → structural
        ("structural_discourse_reverse", "CONTENT"),  # reclassify discourse → content
        ("function_content", "FUNCTION"),  # reclassify content → function
        ("function_content_reverse", "CONTENT"),  # reclassify function → content
    ]:
        base_boundary = boundary.replace("_reverse", "")
        override_tokens = set(_collect_boundary_tokens(baseline_texts, base_boundary, n_perturb))
        if not override_tokens:
            continue

        cond_deltas = []
        for cond, texts in texts_by_condition.items():
            # Baseline STR
            baseline_scores = [_score_str(t) for t in texts]
            baseline_scores = [s for s in baseline_scores if not np.isnan(s)]

            # Perturbed STR
            perturbed_scores = _str_with_override(texts, override_tokens, target_cat)

            if baseline_scores and perturbed_scores:
                delta = float(np.mean(perturbed_scores)) - float(np.mean(baseline_scores))
                cond_deltas.append(delta)

        if cond_deltas:
            results[f"{boundary}_to_{target_cat}"] = {
                "n_tokens_reclassified": len(override_tokens),
                "target_category": target_cat,
                "delta_str_mean": float(np.mean(cond_deltas)),
                "delta_str_std": float(np.std(cond_deltas)),
                "delta_str_max_abs": float(np.max(np.abs(cond_deltas))),
                "n_conditions_tested": len(cond_deltas),
            }

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Token classifier robustness (Exp7 0E)")
    p.add_argument("--sample-outputs", required=True,
                   help="Path to sample_outputs.jsonl from merged A1 run")
    p.add_argument("--output-dir", default="results/exp7/0E/")
    p.add_argument("--n-outputs", type=int, default=None,
                   help="Cap number of output records (for testing)")
    p.add_argument("--n-perturb", type=int, default=500)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load generated texts grouped by condition
    print(f"[0E] Loading {args.sample_outputs}...", flush=True)
    texts_by_condition: dict[str, list[str]] = {}
    n_loaded = 0
    with open(args.sample_outputs) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if args.n_outputs and n_loaded >= args.n_outputs:
                break
            cond = row["condition"]
            text = row.get("generated_text", "")
            if text:
                texts_by_condition.setdefault(cond, []).append(text)
                n_loaded += 1

    all_texts = [t for texts in texts_by_condition.values() for t in texts]
    print(f"[0E] Loaded {len(all_texts)} texts across {len(texts_by_condition)} conditions.", flush=True)

    # ── Documentation ──────────────────────────────────────────────────────────
    print("[0E] Documenting classifier...", flush=True)
    doc = document_classifier(all_texts)
    doc_path = output_dir / "classifier_documentation.json"
    with open(doc_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"[0E] Classifier documentation → {doc_path}", flush=True)

    # Print category stats
    for cat, count in sorted(doc["category_counts"].items()):
        examples = doc["category_examples"].get(cat, [])[:5]
        print(f"  {cat:15s}: {count:8d} tokens  (e.g. {examples})", flush=True)

    # ── Perturbation test ──────────────────────────────────────────────────────
    print(f"[0E] Running perturbation tests (n_perturb={args.n_perturb})...", flush=True)
    perturb_results = perturbation_test(texts_by_condition, args.n_perturb)

    robustness = {
        "perturbation_tests": perturb_results,
        "max_delta_str": max(
            (r["delta_str_max_abs"] for r in perturb_results.values()), default=float("nan")
        ),
        "n_conditions": len(texts_by_condition),
        "n_total_texts": len(all_texts),
        "n_perturb": args.n_perturb,
    }

    rob_path = output_dir / "classifier_robustness.json"
    with open(rob_path, "w") as f:
        json.dump(robustness, f, indent=2)

    print(
        f"\n[0E] Robustness results:\n"
        f"  Max |Δ STR| across all perturbations: {robustness['max_delta_str']:.4f}",
        flush=True,
    )
    for ptype, r in perturb_results.items():
        print(
            f"  {ptype:45s}: mean Δ={r['delta_str_mean']:+.4f}  "
            f"max |Δ|={r['delta_str_max_abs']:.4f}  (n_tokens={r['n_tokens_reclassified']})",
            flush=True,
        )
    print(f"[0E] saved → {rob_path}", flush=True)

    if robustness["max_delta_str"] < 0.01:
        print("[0E] ✓ PASS: max |Δ STR| < 0.01 — classifier is robust", flush=True)
    else:
        print(f"[0E] ⚠ WARNING: max |Δ STR| = {robustness['max_delta_str']:.4f} > 0.01", flush=True)


if __name__ == "__main__":
    main()
