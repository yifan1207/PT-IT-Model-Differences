"""
Neuronpedia feature label fetching and category analysis (Experiment E1a).

Tests prediction P5: Feature Category Transition.
"Pre-dip exclusive features are predominantly lexical/syntactic (>60%).
Post-dip exclusive features are predominantly semantic/entity-level (>60%).
In IT specifically, post-dip features include format/instruction features
absent from PT's post-dip population."

Method
------
1. Identify three feature populations from exp4 active_features .npz:
     pre_dip_exclusive  : active at layer (dip-1) but NOT layer (dip+1)
     post_dip_exclusive : active at layer (dip+1) but NOT layer (dip-1)
     surviving          : active at BOTH layers (dip-1) and (dip+1)

2. For each population, fetch feature labels from Neuronpedia API:
     GET https://www.neuronpedia.org/api/feature/{model_id}/{source}/{index}
     Response: .explanations[0].description (human-readable label)

3. Categorise labels using keyword matching:
     LEXICAL    : token patterns, morphology, syntax, POS, bigrams
     SEMANTIC   : concepts, entities, topics, relations
     FORMAT     : instruction-following, discourse markers, format, output
     AMBIGUOUS  : unclear or unlabelled

4. Compute category distribution per population; compare PT vs IT.

Neuronpedia model IDs for Gemma Scope 2 transcoders
----------------------------------------------------
Model IDs and source strings are inferred from the Gemma Scope 1 naming
convention applied to Gemma Scope 2.  Verify live before bulk fetching:

    PT:  model_id = "gemma-3-4b"   source = "{layer}-gemmascope-2-transcoder-16k"
    IT:  model_id = "gemma-3-4b-it" source = "{layer}-gemmascope-2-transcoder-16k"

NOTE: As of March 2026, Gemma Scope 2 support on Neuronpedia is being actively
      added.  Test with a single feature (e.g. layer 10, index 0) before running
      a bulk fetch.  A 404 means labels are not yet indexed for that model/layer.
"""
import time
import json
import requests
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional


# ── Neuronpedia configuration ──────────────────────────────────────────────────

_NP_BASE_URL = "https://www.neuronpedia.org/api/feature"

# Model IDs on Neuronpedia — verify these before bulk-fetching
_NP_MODEL_IDS = {
    "pt": "gemma-3-4b",
    "it": "gemma-3-4b-it",
}

# Source string pattern for Gemma Scope 2 16k transcoders
def _np_source(layer: int, variant_width: str = "16k") -> str:
    return f"{layer}-gemmascope-2-transcoder-{variant_width}"


# ── Feature population extraction ─────────────────────────────────────────────

def extract_feature_populations(
    npz_path: str,
    pre_dip_layer: int = 10,
    post_dip_layer: int = 12,
    min_frequency: float = 0.05,
    max_features_per_pop: int = 200,
) -> dict:
    """Identify pre-dip exclusive, post-dip exclusive, and surviving features.

    Parameters
    ----------
    npz_path          : path to exp4_features.npz (or exp3)
    pre_dip_layer     : layer index just before the dip (default 10 = dip_layer - 1)
    post_dip_layer    : layer index just after the dip  (default 12 = dip_layer + 1)
    min_frequency     : minimum fraction of prompts where a feature must appear
                        to be included.  Filters out single-occurrence noise.
    max_features_per_pop : cap per population for Neuronpedia API cost control

    Returns
    -------
    dict with keys:
        "pre_dip_exclusive"  : list[int] — feature indices
        "post_dip_exclusive" : list[int]
        "surviving"          : list[int]
        "n_prompts"          : int
        "pre_dip_layer"      : int
        "post_dip_layer"     : int
    """
    data = np.load(npz_path, allow_pickle=True)
    prompt_ids = data.files
    n_prompts  = len(prompt_ids)

    # Count how often each feature appears at each layer across prompts
    count_pre  = defaultdict(int)
    count_post = defaultdict(int)

    for pid in prompt_ids:
        af = data[pid]  # [n_layers] object array

        def _get_features(layer: int) -> set:
            if layer >= len(af):
                return set()
            arr = af[layer]
            return set(arr.tolist()) if len(arr) > 0 else set()

        pre_set  = _get_features(pre_dip_layer)
        post_set = _get_features(post_dip_layer)

        for feat in pre_set:
            count_pre[feat] += 1
        for feat in post_set:
            count_post[feat] += 1

    min_count = max(1, int(min_frequency * n_prompts))

    frequent_pre  = set(f for f, c in count_pre.items()  if c >= min_count)
    frequent_post = set(f for f, c in count_post.items() if c >= min_count)

    pre_exclusive  = sorted(frequent_pre  - frequent_post)
    post_exclusive = sorted(frequent_post - frequent_pre)
    surviving      = sorted(frequent_pre  & frequent_post)

    # Sort by frequency (most frequent first) before capping
    pre_exclusive  = sorted(pre_exclusive,  key=lambda f: count_pre[f],  reverse=True)
    post_exclusive = sorted(post_exclusive, key=lambda f: count_post[f], reverse=True)
    surviving      = sorted(surviving,
                            key=lambda f: count_pre[f] + count_post[f], reverse=True)

    return {
        "pre_dip_exclusive":  pre_exclusive[:max_features_per_pop],
        "post_dip_exclusive": post_exclusive[:max_features_per_pop],
        "surviving":          surviving[:max_features_per_pop],
        "n_prompts":          n_prompts,
        "pre_dip_layer":      pre_dip_layer,
        "post_dip_layer":     post_dip_layer,
        "count_pre":          dict(count_pre),
        "count_post":         dict(count_post),
    }


# ── Neuronpedia API ────────────────────────────────────────────────────────────

def fetch_feature_label(
    model_variant: str,
    layer: int,
    feature_idx: int,
    timeout: float = 10.0,
) -> Optional[str]:
    """Fetch a single feature label from Neuronpedia.

    Returns the first explanation description, or None if not found.
    """
    model_id = _NP_MODEL_IDS.get(model_variant)
    if model_id is None:
        return None

    source = _np_source(layer)
    url    = f"{_NP_BASE_URL}/{model_id}/{source}/{feature_idx}"

    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data         = r.json()
        explanations = data.get("explanations", [])
        if explanations:
            return explanations[0].get("description")
        return None
    except Exception:
        return None


def bulk_fetch_labels(
    model_variant: str,
    layer: int,
    feature_indices: list[int],
    delay_between: float = 0.1,
    cache_path: Optional[str] = None,
) -> dict[int, Optional[str]]:
    """Fetch labels for multiple features with optional disk caching.

    Parameters
    ----------
    model_variant   : "pt" or "it"
    layer           : transcoder layer index
    feature_indices : list of feature indices to fetch
    delay_between   : sleep between API calls to avoid rate limiting (seconds)
    cache_path      : if given, load/save a JSON cache at this path

    Returns
    -------
    dict: feature_index → label string (or None if not found)
    """
    cache: dict[str, Optional[str]] = {}
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached labels from {cache_path}")

    results = {}
    to_fetch = []

    for idx in feature_indices:
        key = str(idx)
        if key in cache:
            results[idx] = cache[key]
        else:
            to_fetch.append(idx)

    print(f"  Fetching {len(to_fetch)} labels from Neuronpedia "
          f"(layer {layer}, model_variant={model_variant}) ...")

    for i, idx in enumerate(to_fetch):
        label = fetch_feature_label(model_variant, layer, idx)
        results[idx] = label
        cache[str(idx)] = label
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(to_fetch)} ...")
        if delay_between > 0:
            time.sleep(delay_between)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"  Saved cache ({len(cache)} entries) → {cache_path}")

    return results


def fetch_population_labels(
    populations: dict,
    model_variant: str,
    pre_dip_layer: int = 10,
    post_dip_layer: int = 12,
    cache_dir: Optional[str] = None,
    delay_between: float = 0.1,
) -> dict:
    """Fetch Neuronpedia labels for all three feature populations.

    Returns
    -------
    dict with keys:
        "pre_dip_labels"  : dict[int, Optional[str]]
        "post_dip_labels" : dict[int, Optional[str]]
        "surviving_labels": dict[int, Optional[str]]
        "layer_labels"    : dict[int, dict[int, Optional[str]]]  keyed by layer
    """
    def _cache(layer: int) -> Optional[str]:
        if cache_dir is None:
            return None
        return str(Path(cache_dir) / model_variant / f"layer_{layer}_labels.json")

    print("\n[feature_labels] Fetching pre-dip exclusive labels ...")
    pre_labels = bulk_fetch_labels(
        model_variant, pre_dip_layer,
        populations["pre_dip_exclusive"],
        delay_between=delay_between,
        cache_path=_cache(pre_dip_layer),
    )

    print("\n[feature_labels] Fetching post-dip exclusive labels ...")
    post_labels = bulk_fetch_labels(
        model_variant, post_dip_layer,
        populations["post_dip_exclusive"],
        delay_between=delay_between,
        cache_path=_cache(post_dip_layer),
    )

    print("\n[feature_labels] Fetching surviving labels ...")
    surviving_labels = bulk_fetch_labels(
        model_variant, pre_dip_layer,
        populations["surviving"],
        delay_between=delay_between,
        cache_path=_cache(pre_dip_layer),
    )

    return {
        "pre_dip_labels":   pre_labels,
        "post_dip_labels":  post_labels,
        "surviving_labels": surviving_labels,
        "layer_labels":     {pre_dip_layer: pre_labels, post_dip_layer: post_labels},
    }


# ── Label categorisation ───────────────────────────────────────────────────────

# Keyword sets for category classification — order matters (first match wins)
_CATEGORY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("FORMAT", [
        "instruction", "format", "discourse", "template", "prompt",
        "response", "assistant", "safety", "output", "style", "refusal",
        "chat", "user", "system",
    ]),
    ("SEMANTIC", [
        "concept", "entity", "topic", "relation", "meaning", "semantic",
        "abstract", "object", "event", "person", "place", "time", "action",
        "category", "class", "domain", "subject",
    ]),
    ("LEXICAL", [
        "token", "word", "morpholog", "syntax", "grammar", "pos ", "part-of-speech",
        "suffix", "prefix", "bigram", "trigram", "n-gram", "punctuat",
        "article", "preposition", "conjunction", "verb", "noun", "adjective",
        "tense", "plural", "singular",
    ]),
]


def categorise_label(label: Optional[str]) -> str:
    """Classify a Neuronpedia label into LEXICAL, SEMANTIC, FORMAT, or AMBIGUOUS.

    Uses keyword matching on the lowercased label string.
    Returns AMBIGUOUS if no keywords match or label is None.
    """
    if not label:
        return "UNLABELLED"
    label_lower = label.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(kw in label_lower for kw in keywords):
            return category
    return "AMBIGUOUS"


def compute_category_distribution(
    labels: dict[int, Optional[str]],
) -> dict[str, float]:
    """Compute fraction of features in each category.

    Returns dict: category → fraction (sums to 1.0, NaN if empty).
    """
    categories = ["LEXICAL", "SEMANTIC", "FORMAT", "AMBIGUOUS", "UNLABELLED"]
    counts = defaultdict(int)

    for idx, label in labels.items():
        cat = categorise_label(label)
        counts[cat] += 1

    total = sum(counts.values())
    if total == 0:
        return {cat: float("nan") for cat in categories}

    return {cat: counts[cat] / total for cat in categories}


def summarise_population_categories(
    label_data: dict,
    populations: dict,
) -> dict:
    """Compute category distributions for all three populations.

    Returns dict with keys:
        "pre_dip"    : dict[category, fraction]
        "post_dip"   : dict[category, fraction]
        "surviving"  : dict[category, fraction]
        "n_labelled" : dict[population, int]
    """
    pre_dist  = compute_category_distribution(label_data["pre_dip_labels"])
    post_dist = compute_category_distribution(label_data["post_dip_labels"])
    surv_dist = compute_category_distribution(label_data["surviving_labels"])

    def n_labelled(labels: dict) -> int:
        return sum(1 for v in labels.values() if v is not None)

    return {
        "pre_dip":   pre_dist,
        "post_dip":  post_dist,
        "surviving": surv_dist,
        "n_labelled": {
            "pre_dip":   n_labelled(label_data["pre_dip_labels"]),
            "post_dip":  n_labelled(label_data["post_dip_labels"]),
            "surviving": n_labelled(label_data["surviving_labels"]),
        },
        "n_total": {
            "pre_dip":   len(populations["pre_dip_exclusive"]),
            "post_dip":  len(populations["post_dip_exclusive"]),
            "surviving": len(populations["surviving"]),
        },
    }
