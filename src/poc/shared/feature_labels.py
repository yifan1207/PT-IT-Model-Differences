"""
Feature label lookup for Gemma Scope 2 4B transcoders.

Labels are derived from the top_tokens field in examples.safetensors:
each feature's label = the output tokens its activation most boosts,
decoded with the Gemma tokenizer.

Results are cached to disk at CACHE_DIR so subsequent calls are instant.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_REPO_IT = "google/gemma-scope-2-4b-it"
_REPO_PT = "google/gemma-scope-2-4b-pt"
_VARIANT_MAP = {"it": _REPO_IT, "pt": _REPO_PT}

CACHE_DIR = Path("cache/feature_labels")

# How many top-boosted tokens to include in the label string
_N_LABEL_TOKENS = 4


def _repo_for_variant(variant: str) -> str:
    return _VARIANT_MAP.get(variant, _REPO_IT)


def _cache_path(variant: str, layer: int) -> Path:
    return CACHE_DIR / variant / f"layer_{layer}.json"


def _load_from_hub(variant: str, layer: int) -> dict[int, str]:
    """Download examples.safetensors for one layer, return {feat_idx: label_str}."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    repo = _repo_for_variant(variant)
    hf_path = f"transcoder_all/layer_{layer}_width_16k_l0_big_affine/examples.safetensors"
    local = hf_hub_download(repo, hf_path)

    with safe_open(local, framework="pt") as f:
        top_tokens = f.get_tensor("top_tokens").numpy()  # [n_features, 10]

    model_name = "google/gemma-3-4b-it" if variant == "it" else "google/gemma-3-4b-pt"
    tok = AutoTokenizer.from_pretrained(model_name)

    labels: dict[int, str] = {}
    for feat_idx in range(top_tokens.shape[0]):
        token_ids = top_tokens[feat_idx, :_N_LABEL_TOKENS].tolist()
        decoded = [tok.decode([tid]).strip() for tid in token_ids]
        # Remove empty strings, deduplicate preserving order
        seen: set[str] = set()
        clean: list[str] = []
        for t in decoded:
            if t and t not in seen:
                seen.add(t)
                clean.append(t)
        labels[feat_idx] = ", ".join(clean) if clean else f"F{feat_idx}"
    return labels


def get_feature_labels(variant: str, layer: int) -> dict[int, str]:
    """Return {feat_idx: label_str} for the given variant and layer, with disk cache."""
    cache = _cache_path(variant, layer)
    if cache.exists():
        with open(cache) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    print(f"    [feature_labels] fetching layer {layer} for {variant} from HuggingFace ...")
    labels = _load_from_hub(variant, layer)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, ensure_ascii=False)
    return labels


def label_for(variant: str, layer: int, feat_idx: int, fallback: str | None = None) -> str:
    """Convenience: single feature label. Falls back gracefully if unavailable."""
    try:
        labels = get_feature_labels(variant, layer)
        return labels.get(feat_idx, fallback or f"F{feat_idx}")
    except Exception:
        return fallback or f"F{feat_idx}"


def make_label_str(variant: str, layer: int, feat_idx: int) -> str:
    """Full label string: 'L{layer}:F{idx} → token1, token2, ...' """
    desc = label_for(variant, layer, feat_idx)
    return f"L{layer}:F{feat_idx} \u2192 {desc}"
