#!/usr/bin/env python3
"""B0 Step 1: Classify Gemma Scope feature labels into GOVERNANCE/SAFETY/CONTENT/OTHER.

Uses Claude Sonnet via OpenRouter (OpenAI-compatible API) to classify all 16,384 features
per corrective layer (L20-L33, 14 layers = 229,376 total labels) in batches.

Input:  cache/feature_labels/it/layer_{20..33}.json
Output: results/exp6/feature_classifications/layer_{20..33}.json
        {feature_idx_str: "GOVERNANCE"|"SAFETY"|"CONTENT"|"OTHER", ...}

Cost estimate: ~1,150 API calls × ~300 tokens each ≈ $3-5 at Sonnet rates.
Run time: ~25 minutes.

Usage:
    uv run python scripts/classify_governance_features.py
    uv run python scripts/classify_governance_features.py --layers 20,21 --batch-size 100

Requires OPENROUTER_API_KEY in .env or environment.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


# ── .env loader (no external dependency) ────────────────────────────────────

def _load_dotenv(path: str = ".env") -> None:
    """Parse .env and set env vars (does not overwrite already-set vars)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


# ── Classification prompt ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a neural network feature classifier for a language model interpretability project.

You will be given labels for features extracted from Gemma 3 (a language model). Each label is derived from the top-activated tokens for that feature.

Classify each feature label into exactly one category:

GOVERNANCE: Features related to output formatting, structure, and register. Examples:
  - Structural tokens: "Answer:", "Question:", numbered lists, bullet points, markdown headers
  - Discourse markers: "however", "therefore", "in summary", "first", "second"
  - Turn-taking patterns: "User:", "Assistant:", conversation formatting
  - Punctuation patterns: colons after labels, em dashes, ellipses in structured contexts
  - Response structure: introductory phrases, conclusion phrases, paragraph breaks
  - Register markers: formal/informal register cues, hedging language ("I think", "it seems")

SAFETY: Features related to harmful content avoidance, ethics, or refusal behaviors.
  - Refusal phrases: "I cannot", "I'm sorry", "I must decline"
  - Harm-related tokens: violence, illegal activities, dangerous instructions
  - Warning phrases: "caution", "disclaimer", "warning"

CONTENT: Features related to factual knowledge, entities, or semantic topics.
  - Named entities: person names, place names, organizations
  - Domain-specific terms: medical, legal, scientific, technical vocabulary
  - Factual concepts: historical events, scientific concepts, definitions
  - Language-specific tokens (multilingual content)

OTHER: Everything else.
  - Unknown or uninterpretable features
  - Code/programming syntax (unless clearly about structure)
  - Very short or ambiguous labels that don't fit the above

Respond with ONLY a JSON object mapping each feature index to its category.
Example: {"0": "GOVERNANCE", "1": "CONTENT", "2": "OTHER", "3": "SAFETY"}"""

_USER_TEMPLATE = """Classify these neural network feature labels:

{labels_block}

Respond with ONLY a JSON object: {{"feature_idx": "CATEGORY", ...}}"""


def _make_labels_block(batch: list[tuple[str, str]]) -> str:
    return "\n".join(f"{idx}: {label}" for idx, label in batch)


# ── OpenRouter API call with retry ────────────────────────────────────────────

def _classify_batch(
    client,
    batch: list[tuple[str, str]],
    model: str,
    max_retries: int = 3,
) -> dict[str, str]:
    """Send one batch to OpenRouter/Claude and parse the JSON response."""
    labels_block = _make_labels_block(batch)
    user_msg = _USER_TEMPLATE.format(labels_block=labels_block)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            text = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            result = json.loads(text)
            # Validate all indices are present
            missing = [idx for idx, _ in batch if str(idx) not in result]
            if missing:
                print(f"    Warning: {len(missing)} indices missing from response, defaulting to OTHER", flush=True)
                for m in missing:
                    result[str(m)] = "OTHER"
            return result
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    Parse error (attempt {attempt+1}): {e}. Retrying in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"    Failed after {max_retries} attempts. Defaulting all to OTHER.", flush=True)
                return {str(idx): "OTHER" for idx, _ in batch}
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(60, 2 ** attempt * 5)
                print(f"    API error (attempt {attempt+1}): {e}. Retrying in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"    API failed: {e}. Defaulting all to OTHER.", flush=True)
                return {str(idx): "OTHER" for idx, _ in batch}
    return {str(idx): "OTHER" for idx, _ in batch}


# ── Per-layer classifier ──────────────────────────────────────────────────────

def classify_layer(
    client,
    layer_idx: int,
    labels_path: Path,
    out_path: Path,
    batch_size: int,
    model: str,
    rate_limit_sleep: float,
) -> None:
    """Classify all features for one layer, with resume support."""
    print(f"\nLayer {layer_idx}: loading labels from {labels_path}", flush=True)

    with open(labels_path) as f:
        labels: dict[str, str] = json.load(f)

    n_features = len(labels)
    print(f"  {n_features} features to classify", flush=True)

    # Resume: load existing results
    existing: dict[str, str] = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        print(f"  Resuming: {len(existing)} already classified", flush=True)

    to_classify = [
        (idx, label) for idx, label in labels.items()
        if idx not in existing
    ]
    print(f"  {len(to_classify)} remaining to classify", flush=True)

    if not to_classify:
        print(f"  Layer {layer_idx} already complete.", flush=True)
        return

    n_batches = (len(to_classify) + batch_size - 1) // batch_size
    for batch_i in range(n_batches):
        batch = to_classify[batch_i * batch_size: (batch_i + 1) * batch_size]
        print(f"  Batch {batch_i + 1}/{n_batches} ({len(batch)} features)...", flush=True)

        result = _classify_batch(client, batch, model)
        existing.update(result)

        # Save after each batch (crash-safe)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(existing, f)

        if batch_i < n_batches - 1:
            time.sleep(rate_limit_sleep)

    from collections import Counter
    counts = Counter(existing.values())
    print(f"  Done: {counts}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _load_dotenv()

    p = argparse.ArgumentParser()
    p.add_argument("--layers", default="20,21,22,23,24,25,26,27,28,29,30,31,32,33")
    p.add_argument("--labels-dir", default="cache/feature_labels/it")
    p.add_argument("--out-dir", default="results/exp6/feature_classifications")
    p.add_argument("--batch-size", type=int, default=200,
                   help="Labels per API call.")
    p.add_argument("--model", default="anthropic/claude-sonnet-4-6",
                   help="OpenRouter model ID.")
    p.add_argument("--rate-limit-sleep", type=float, default=1.0,
                   help="Seconds to sleep between batches.")
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Add it to .env or set it in the environment."
        )

    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    layers = [int(x) for x in args.layers.split(",")]
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)

    for layer_idx in layers:
        labels_path = labels_dir / f"layer_{layer_idx}.json"
        if not labels_path.exists():
            print(f"Warning: labels not found for layer {layer_idx}: {labels_path}", flush=True)
            continue
        out_path = out_dir / f"layer_{layer_idx}.json"
        classify_layer(
            client, layer_idx, labels_path, out_path,
            batch_size=args.batch_size,
            model=args.model,
            rate_limit_sleep=args.rate_limit_sleep,
        )

    print("\n=== Feature classification complete ===")
    from collections import Counter
    total_counts: Counter = Counter()
    for layer_idx in layers:
        out_path = out_dir / f"layer_{layer_idx}.json"
        if out_path.exists():
            with open(out_path) as f:
                data = json.load(f)
            total_counts.update(data.values())
    print(f"Total across all layers: {dict(total_counts)}")
    total = sum(total_counts.values())
    for cat, cnt in sorted(total_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt} ({100*cnt/total:.1f}%)")


if __name__ == "__main__":
    main()
