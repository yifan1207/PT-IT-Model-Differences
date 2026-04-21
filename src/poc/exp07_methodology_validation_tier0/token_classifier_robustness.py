"""Token classifier specification and robustness check (Exp7 0E).

Documents the structural-token classifier used in exp3/exp6 and tests
sensitivity to boundary shifts. Two perturbation tests:
  1. STRUCTURAL <-> DISCOURSE boundary: reclassify 500 boundary-adjacent tokens
  2. FUNCTION <-> CONTENT boundary: reclassify 500 short ambiguous words

Additional analyses:
  - LLM judge validation: score 200 tokens with Claude, compute Cohen's kappa
  - Human judgment CSV template: export for manual spot-check
  - Baseline category rates + enrichment ratio (IT/PT token category divergence)

Usage:
  uv run python -m src.poc.exp07_methodology_validation_tier0.token_classifier_robustness \\
      --sample-outputs results/exp6/merged_A1_it_v4/sample_outputs.jsonl \\
      --output-dir results/exp7/0E/

  # Quick test
  uv run python -m src.poc.exp07_methodology_validation_tier0.token_classifier_robustness \\
      --sample-outputs results/exp6/merged_A1_it_v4/sample_outputs.jsonl \\
      --output-dir results/exp7/0E/ --n-outputs 200

  # LLM judge validation (requires API key)
  uv run python -m src.poc.exp07_methodology_validation_tier0.token_classifier_robustness \\
      --sample-outputs results/exp6/merged_A1_it_v4/sample_outputs.jsonl \\
      --output-dir results/exp7/0E/ --llm-judge

Outputs:
  results/exp7/0E/classifier_documentation.json
  results/exp7/0E/classifier_robustness.json
  results/exp7/0E/human_judgment_template.csv
  results/exp7/0E/category_enrichment.json
  results/exp7/0E/llm_judge_validation.json   (if --llm-judge)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


# ── Import classifier ─────────────────────────────────────────────────────────

def _get_classifier():
    from src.poc.exp03_corrective_stage_characterization.analysis.word_categories import (
        classify_generated_tokens_by_word,
        _STRUCTURAL_TOKEN_RE as STRUCTURAL_PATTERN,
        _PUNCT_ONLY_RE as PUNCTUATION_PATTERN,
        _FUNCTION_WORDS as FUNCTION_WORDS,
        _DISCOURSE_SINGLE as DISCOURSE_WORDS,
        _DISCOURSE_PHRASES as DISCOURSE_PHRASES,
        WordCategory,
    )
    return classify_generated_tokens_by_word, {
        "structural_pattern": STRUCTURAL_PATTERN.pattern,
        "punctuation_pattern": PUNCTUATION_PATTERN.pattern,
        "function_words": sorted(FUNCTION_WORDS),
        "discourse_words": sorted(DISCOURSE_WORDS),
        "discourse_phrases": sorted([" ".join(p) for p in DISCOURSE_PHRASES]),
    }


def _classify_tokens(words: list[str]):
    """Classify a list of word strings."""
    from src.poc.exp03_corrective_stage_characterization.analysis.word_categories import classify_generated_tokens_by_word
    return classify_generated_tokens_by_word([{"token_str": w + " "} for w in words])


def _score_str(text: str) -> float:
    try:
        words = text.split()
        if not words:
            return 0.0
        cats = _classify_tokens(words)
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

    for text in texts[:2000]:
        words = text.split()
        if not words:
            continue
        cats = _classify_tokens(words)
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


# ── Baseline category rates + enrichment ratio ──────────────────────────────

def compute_category_enrichment(
    texts_by_condition: dict[str, list[str]],
) -> dict:
    """Compute per-category token rates per condition and enrichment ratios.

    Enrichment ratio = rate_in_condition / rate_in_baseline.
    For IT vs PT comparisons, high enrichment in governance categories
    (STRUCTURAL, DISCOURSE, PUNCTUATION) supports the STR metric.
    """
    categories = ["STRUCTURAL", "DISCOURSE", "PUNCTUATION", "FUNCTION", "CONTENT"]
    results: dict[str, dict] = {}

    for cond, texts in texts_by_condition.items():
        counts: Counter[str] = Counter()
        total_tokens = 0

        for text in texts[:2000]:
            words = text.split()
            if not words:
                continue
            cats = _classify_tokens(words)
            for cat in cats:
                counts[cat] += 1
                total_tokens += 1

        if total_tokens == 0:
            continue

        rates = {cat: counts.get(cat, 0) / total_tokens for cat in categories}
        results[cond] = {
            "rates": rates,
            "total_tokens": total_tokens,
            "n_texts": min(len(texts), 2000),
        }

    # Compute enrichment ratios relative to baseline
    baseline_key = None
    for cond in results:
        if cond.endswith("_baseline") or cond == "A1_baseline":
            baseline_key = cond
            break

    if baseline_key and baseline_key in results:
        baseline_rates = results[baseline_key]["rates"]
        for cond in results:
            enrichment = {}
            for cat in categories:
                br = baseline_rates.get(cat, 0)
                cr = results[cond]["rates"].get(cat, 0)
                enrichment[cat] = cr / br if br > 1e-8 else float("nan")
            results[cond]["enrichment_vs_baseline"] = enrichment

    return results


# ── Human judgment CSV template ──────────────────────────────────────────────

def generate_human_judgment_template(
    texts: list[str],
    output_path: Path,
    n_tokens: int = 200,
    seed: int = 42,
) -> None:
    """Export a CSV template for human annotation of token categories.

    Randomly samples tokens from generated texts, pre-fills the classifier's
    label, and leaves a column for human judgment.

    Columns: token, context (5 words around token), classifier_label, human_label, agree
    """
    rng = np.random.default_rng(seed)
    token_entries: list[dict] = []

    for text in texts[:1000]:
        words = text.split()
        if len(words) < 3:
            continue
        cats = _classify_tokens(words)
        for i, (word, cat) in enumerate(zip(words, cats)):
            context_start = max(0, i - 2)
            context_end = min(len(words), i + 3)
            context = " ".join(words[context_start:context_end])
            token_entries.append({
                "token": word.strip(),
                "context": context,
                "classifier_label": cat,
            })

    if not token_entries:
        return

    # Stratified sample: proportional to category distribution, but ensure
    # all categories are represented
    by_cat: dict[str, list[dict]] = {}
    for entry in token_entries:
        by_cat.setdefault(entry["classifier_label"], []).append(entry)

    sampled: list[dict] = []
    per_cat = max(n_tokens // max(len(by_cat), 1), 10)
    for cat, entries in by_cat.items():
        k = min(per_cat, len(entries))
        idxs = rng.choice(len(entries), k, replace=False)
        for idx in idxs:
            sampled.append(entries[idx])

    # Fill remaining slots randomly
    remaining = n_tokens - len(sampled)
    if remaining > 0:
        extra_idxs = rng.choice(len(token_entries), min(remaining, len(token_entries)), replace=False)
        for idx in extra_idxs:
            if token_entries[idx] not in sampled:
                sampled.append(token_entries[idx])

    sampled = sampled[:n_tokens]
    rng.shuffle(sampled)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["token", "context", "classifier_label", "human_label", "agree"])
        writer.writeheader()
        for entry in sampled:
            writer.writerow({
                "token": entry["token"],
                "context": entry["context"],
                "classifier_label": entry["classifier_label"],
                "human_label": "",  # to be filled by annotator
                "agree": "",  # to be filled: Y/N
            })

    print(f"[0E] Human judgment template ({len(sampled)} tokens) -> {output_path}", flush=True)


# ── LLM judge validation ────────────────────────────────────────────────────

def llm_judge_validation(
    texts: list[str],
    output_path: Path,
    n_tokens: int = 200,
    seed: int = 42,
) -> dict:
    """Score 200 tokens with an LLM judge and compute Cohen's kappa vs classifier.

    Uses Claude via OpenRouter to classify tokens into the 5 categories,
    then computes inter-rater agreement (Cohen's weighted kappa).
    """
    import os
    rng = np.random.default_rng(seed)

    # Collect tokens with context
    token_entries: list[dict] = []
    for text in texts[:500]:
        words = text.split()
        if len(words) < 3:
            continue
        cats = _classify_tokens(words)
        for i, (word, cat) in enumerate(zip(words, cats)):
            context_start = max(0, i - 3)
            context_end = min(len(words), i + 4)
            context = " ".join(words[context_start:context_end])
            token_entries.append({
                "token": word.strip(),
                "context": context,
                "classifier_label": cat,
            })

    if len(token_entries) < n_tokens:
        print(f"[0E] Warning: only {len(token_entries)} tokens available for LLM judge", flush=True)
        n_tokens = len(token_entries)

    idxs = rng.choice(len(token_entries), n_tokens, replace=False)
    sample = [token_entries[i] for i in idxs]

    SYSTEM_PROMPT = """You are classifying tokens from language model output into 5 categories:

STRUCTURAL: Markdown formatting tokens (headers ##, bullets *, -, numbered lists 1., bold **, code blocks ```, etc.)
DISCOURSE: Discourse markers and transition words (however, therefore, additionally, furthermore, etc.)
PUNCTUATION: Standalone punctuation tokens that serve formatting purposes
FUNCTION: Function words (articles, prepositions, conjunctions, pronouns: the, a, of, in, and, etc.)
CONTENT: Content/semantic words (nouns, verbs, adjectives, adverbs with meaning)

For each token, classify it based on its role in the context. Return ONLY a JSON array of objects:
[{"token": "...", "category": "STRUCTURAL|DISCOURSE|PUNCTUATION|FUNCTION|CONTENT"}]"""

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("[0E] OPENROUTER_API_KEY not set, skipping LLM judge validation", flush=True)
        return {"error": "OPENROUTER_API_KEY not set", "n_tokens": 0}

    try:
        import httpx
    except ImportError:
        print("[0E] httpx not installed, skipping LLM judge validation", flush=True)
        return {"error": "httpx not installed", "n_tokens": 0}

    # Batch tokens for API efficiency (20 per request)
    batch_size = 20
    llm_labels: list[str | None] = [None] * len(sample)

    for batch_start in range(0, len(sample), batch_size):
        batch = sample[batch_start:batch_start + batch_size]
        user_msg = "Classify these tokens:\n\n"
        for i, entry in enumerate(batch):
            user_msg += f'{i+1}. Token: "{entry["token"]}" | Context: "...{entry["context"]}..."\n'

        try:
            resp = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "anthropic/claude-sonnet-4-6",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 2000,
                },
                timeout=30,
            )
            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON from response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                for j, item in enumerate(parsed):
                    idx = batch_start + j
                    if idx < len(llm_labels):
                        cat = item.get("category", "").upper()
                        if cat in {"STRUCTURAL", "DISCOURSE", "PUNCTUATION", "FUNCTION", "CONTENT"}:
                            llm_labels[idx] = cat
        except Exception as e:
            print(f"[0E] LLM judge batch {batch_start} failed: {e}", flush=True)

    # Compute Cohen's kappa
    valid_pairs = [
        (sample[i]["classifier_label"], llm_labels[i])
        for i in range(len(sample))
        if llm_labels[i] is not None
    ]

    if len(valid_pairs) < 10:
        return {"error": "Too few valid LLM responses", "n_valid": len(valid_pairs)}

    categories = ["STRUCTURAL", "DISCOURSE", "PUNCTUATION", "FUNCTION", "CONTENT"]
    classifier_labels = [p[0] for p in valid_pairs]
    llm_labels_clean = [p[1] for p in valid_pairs]

    # Confusion matrix
    confusion: dict[str, dict[str, int]] = {c: {l: 0 for l in categories} for c in categories}
    agree = 0
    for cl, ll in valid_pairs:
        confusion[cl][ll] += 1
        if cl == ll:
            agree += 1

    # Cohen's kappa
    n = len(valid_pairs)
    p_o = agree / n

    # Expected agreement
    classifier_counts = Counter(classifier_labels)
    llm_counts = Counter(llm_labels_clean)
    p_e = sum(
        (classifier_counts.get(c, 0) / n) * (llm_counts.get(c, 0) / n)
        for c in categories
    )

    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 1e-8 else 1.0

    result = {
        "n_tokens_scored": len(valid_pairs),
        "n_tokens_attempted": n_tokens,
        "observed_agreement": float(p_o),
        "expected_agreement": float(p_e),
        "cohens_kappa": float(kappa),
        "kappa_interpretation": (
            "substantial" if kappa >= 0.61 else
            "moderate" if kappa >= 0.41 else
            "fair" if kappa >= 0.21 else
            "slight"
        ),
        "confusion_matrix": confusion,
        "per_category_agreement": {
            c: confusion[c][c] / max(sum(confusion[c].values()), 1) for c in categories
        },
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"[0E] LLM judge validation: kappa={kappa:.3f} ({result['kappa_interpretation']}), "
        f"agreement={p_o:.1%} on {len(valid_pairs)} tokens -> {output_path}",
        flush=True,
    )

    return result


# ── Perturbation tests ────────────────────────────────────────────────────────

def _collect_boundary_tokens(
    texts: list[str],
    boundary: str,
    n_perturb: int = 500,
) -> list[str]:
    """Find tokens near a classification boundary."""
    candidate_counts: Counter[str] = Counter()

    for text in texts:
        words = text.split()
        if not words:
            continue
        cats = _classify_tokens(words)
        for word, cat in zip(words, cats):
            w = word.strip().lower()
            if boundary == "structural_discourse":
                if cat == "DISCOURSE" and len(w) <= 7:
                    candidate_counts[w] += 1
            elif boundary == "function_content":
                if cat == "CONTENT" and 2 <= len(w) <= 4:
                    candidate_counts[w] += 1

    return [tok for tok, _ in candidate_counts.most_common(n_perturb)]


def _str_with_override(
    texts: list[str],
    override_tokens: set[str],
    override_category: str,
) -> list[float]:
    """Compute STR with certain tokens reclassified."""
    gov = {"STRUCTURAL", "DISCOURSE", "PUNCTUATION"}
    scores = []
    for text in texts:
        words = text.split()
        if not words:
            continue
        cats = _classify_tokens(words)
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
    """Run both boundary perturbation tests."""
    baseline_texts = texts_by_condition.get(
        "A1_baseline",
        next(iter(texts_by_condition.values())) if texts_by_condition else [],
    )

    results = {}

    for boundary, target_cat in [
        ("structural_discourse", "STRUCTURAL"),
        ("structural_discourse_reverse", "CONTENT"),
        ("function_content", "FUNCTION"),
        ("function_content_reverse", "CONTENT"),
    ]:
        base_boundary = boundary.replace("_reverse", "")
        override_tokens = set(_collect_boundary_tokens(baseline_texts, base_boundary, n_perturb))
        if not override_tokens:
            continue

        cond_deltas = []
        for cond, texts in texts_by_condition.items():
            baseline_scores = [_score_str(t) for t in texts]
            baseline_scores = [s for s in baseline_scores if not np.isnan(s)]
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
    p.add_argument("--llm-judge", action="store_true",
                   help="Run LLM judge validation (requires OPENROUTER_API_KEY)")
    p.add_argument("--n-judge-tokens", type=int, default=200,
                   help="Number of tokens to score with LLM judge")
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
    print(f"[0E] Classifier documentation -> {doc_path}", flush=True)

    for cat, count in sorted(doc["category_counts"].items()):
        examples = doc["category_examples"].get(cat, [])[:5]
        print(f"  {cat:15s}: {count:8d} tokens  (e.g. {examples})", flush=True)

    # ── Baseline category rates + enrichment ──────────────────────────────────
    print("[0E] Computing category enrichment ratios...", flush=True)
    enrichment = compute_category_enrichment(texts_by_condition)
    enrich_path = output_dir / "category_enrichment.json"
    with open(enrich_path, "w") as f:
        json.dump(enrichment, f, indent=2)
    print(f"[0E] Category enrichment -> {enrich_path}", flush=True)

    # Print summary for baseline
    baseline_key = None
    for cond in enrichment:
        if cond.endswith("_baseline") or cond == "A1_baseline":
            baseline_key = cond
            break
    if baseline_key:
        rates = enrichment[baseline_key]["rates"]
        print(f"  Baseline ({baseline_key}) category rates:", flush=True)
        for cat in ["STRUCTURAL", "DISCOURSE", "PUNCTUATION", "FUNCTION", "CONTENT"]:
            print(f"    {cat:15s}: {rates.get(cat, 0):.4f}", flush=True)

    # ── Human judgment CSV template ───────────────────────────────────────────
    print("[0E] Generating human judgment template...", flush=True)
    csv_path = output_dir / "human_judgment_template.csv"
    generate_human_judgment_template(all_texts, csv_path)

    # ── LLM judge validation ─────────────────────────────────────────────────
    if args.llm_judge:
        print(f"[0E] Running LLM judge validation ({args.n_judge_tokens} tokens)...", flush=True)
        judge_path = output_dir / "llm_judge_validation.json"
        llm_judge_validation(all_texts, judge_path, n_tokens=args.n_judge_tokens)

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
        f"  Max |delta STR| across all perturbations: {robustness['max_delta_str']:.4f}",
        flush=True,
    )
    for ptype, r in perturb_results.items():
        print(
            f"  {ptype:45s}: mean delta={r['delta_str_mean']:+.4f}  "
            f"max |delta|={r['delta_str_max_abs']:.4f}  (n_tokens={r['n_tokens_reclassified']})",
            flush=True,
        )
    print(f"[0E] saved -> {rob_path}", flush=True)

    if robustness["max_delta_str"] < 0.01:
        print("[0E] PASS: max |delta STR| < 0.01 -- classifier is robust", flush=True)
    else:
        print(f"[0E] WARNING: max |delta STR| = {robustness['max_delta_str']:.4f} > 0.01", flush=True)


if __name__ == "__main__":
    main()
