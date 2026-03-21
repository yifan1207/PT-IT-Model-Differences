"""
Alignment-based feature continuity for dip analyses.

Raw transcoder feature indices are only meaningful within a single layer of a
single model. This module replaces invalid raw-id overlap logic with alignment
based on activation profiles over prompts and generation steps.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse


EventMode = Literal["step", "prompt"]
EventKey = tuple[str, int] | str


@dataclass(frozen=True)
class FeatureMatch:
    feature_a: int
    feature_b: int
    score: float


def _prompt_ids_in_order(features_dict: dict[str, np.ndarray]) -> list[str]:
    return list(features_dict.keys())


def _event_keys_single_source(
    features_dict: dict[str, np.ndarray],
    layer_a: int,
    layer_b: int,
    event_mode: EventMode,
) -> list[EventKey]:
    keys: list[EventKey] = []
    for prompt_id in _prompt_ids_in_order(features_dict):
        af = features_dict[prompt_id]
        n_steps, n_layers = af.shape
        if layer_a >= n_layers or layer_b >= n_layers:
            continue
        if event_mode == "prompt":
            keys.append(prompt_id)
        else:
            keys.extend((prompt_id, step) for step in range(n_steps))
    return keys


def _event_keys_two_sources(
    features_dict_a: dict[str, np.ndarray],
    features_dict_b: dict[str, np.ndarray],
    layer_a: int,
    layer_b: int,
    event_mode: EventMode,
) -> list[EventKey]:
    keys: list[EventKey] = []
    common_ids = [pid for pid in _prompt_ids_in_order(features_dict_a) if pid in features_dict_b]
    for prompt_id in common_ids:
        af_a = features_dict_a[prompt_id]
        af_b = features_dict_b[prompt_id]
        n_steps_a, n_layers_a = af_a.shape
        n_steps_b, n_layers_b = af_b.shape
        if layer_a >= n_layers_a or layer_b >= n_layers_b:
            continue
        if event_mode == "prompt":
            keys.append(prompt_id)
        else:
            n_steps = min(n_steps_a, n_steps_b)
            keys.extend((prompt_id, step) for step in range(n_steps))
    return keys


def _features_for_event(af: np.ndarray, layer: int, event: EventKey) -> set[int]:
    if layer >= af.shape[1]:
        return set()
    if isinstance(event, tuple):
        _, step = event
        if step >= af.shape[0]:
            return set()
        arr = af[step, layer]
    else:
        arrs = [af[step, layer] for step in range(af.shape[0])]
        feats: set[int] = set()
        for arr in arrs:
            if arr is not None and len(arr) > 0:
                feats.update(int(x) for x in arr.tolist())
        return feats
    if arr is None or len(arr) == 0:
        return set()
    return {int(x) for x in arr.tolist()}


def build_feature_event_sets(
    features_dict: dict[str, np.ndarray],
    layer: int,
    event_mode: EventMode = "step",
) -> tuple[dict[int, set[EventKey]], list[EventKey]]:
    event_keys = _event_keys_single_source(features_dict, layer, layer, event_mode)
    feat_events: dict[int, set[EventKey]] = {}
    for event in event_keys:
        prompt_id = event[0] if isinstance(event, tuple) else event
        feats = _features_for_event(features_dict[prompt_id], layer, event)
        for feat in feats:
            feat_events.setdefault(feat, set()).add(event)
    return feat_events, event_keys


def _build_reverse_event_lookup(feat_events: dict[int, set[EventKey]]) -> dict[EventKey, set[int]]:
    lookup: dict[EventKey, set[int]] = {}
    for feat, events in feat_events.items():
        for event in events:
            lookup.setdefault(event, set()).add(feat)
    return lookup


def mutual_nearest_feature_matches(
    feat_events_a: dict[int, set[EventKey]],
    feat_events_b: dict[int, set[EventKey]],
    event_keys: list[EventKey],
    min_score: float = 0.20,
) -> list[FeatureMatch]:
    if not feat_events_a or not feat_events_b or not event_keys:
        return []

    event_index = {event: idx for idx, event in enumerate(event_keys)}
    feats_a = sorted(feat_events_a)
    feats_b = sorted(feat_events_b)
    feat_a_index = {feat: idx for idx, feat in enumerate(feats_a)}
    feat_b_index = {feat: idx for idx, feat in enumerate(feats_b)}

    rows_a: list[int] = []
    cols_a: list[int] = []
    for feat, events in feat_events_a.items():
        col = feat_a_index[feat]
        for event in events:
            row = event_index.get(event)
            if row is not None:
                rows_a.append(row)
                cols_a.append(col)

    rows_b: list[int] = []
    cols_b: list[int] = []
    for feat, events in feat_events_b.items():
        col = feat_b_index[feat]
        for event in events:
            row = event_index.get(event)
            if row is not None:
                rows_b.append(row)
                cols_b.append(col)

    if not rows_a or not rows_b:
        return []

    data_a = np.ones(len(rows_a), dtype=np.float32)
    data_b = np.ones(len(rows_b), dtype=np.float32)
    mat_a = sparse.csr_matrix((data_a, (rows_a, cols_a)), shape=(len(event_keys), len(feats_a)))
    mat_b = sparse.csr_matrix((data_b, (rows_b, cols_b)), shape=(len(event_keys), len(feats_b)))

    overlap = (mat_a.T @ mat_b).tocoo()
    if overlap.nnz == 0:
        return []

    norms_a = np.sqrt(np.asarray(mat_a.sum(axis=0)).ravel())
    norms_b = np.sqrt(np.asarray(mat_b.sum(axis=0)).ravel())

    candidate_scores: list[FeatureMatch] = []
    for row, col, val in zip(overlap.row, overlap.col, overlap.data, strict=False):
        norm_a = float(norms_a[row])
        norm_b = float(norms_b[col])
        if norm_a == 0.0 or norm_b == 0.0:
            continue
        score = float(val / (norm_a * norm_b))
        if score >= min_score:
            candidate_scores.append(FeatureMatch(feats_a[row], feats_b[col], score))

    candidate_scores.sort(key=lambda m: (-m.score, m.feature_a, m.feature_b))
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[FeatureMatch] = []
    for match in candidate_scores:
        if match.feature_a in used_a or match.feature_b in used_b:
            continue
        used_a.add(match.feature_a)
        used_b.add(match.feature_b)
        matches.append(match)
    return matches


def continuity_for_sets(
    feats_a: set[int],
    feats_b: set[int],
    matches: list[FeatureMatch],
) -> float:
    if not feats_a and not feats_b:
        return float("nan")
    matched_a = {m.feature_a for m in matches if m.feature_a in feats_a and m.feature_b in feats_b}
    union = len(feats_a) + len(feats_b) - len(matched_a)
    if union == 0:
        return float("nan")
    return len(matched_a) / union


def build_population_partition(
    features_dict: dict[str, np.ndarray],
    pre_layer: int,
    post_layer: int,
    min_frequency: float = 0.05,
    event_mode: EventMode = "prompt",
    min_match_score: float = 0.20,
    top_k_for_matching: int | None = None,
) -> dict:
    feat_events_pre, _ = build_feature_event_sets(features_dict, pre_layer, event_mode=event_mode)
    feat_events_post, _ = build_feature_event_sets(features_dict, post_layer, event_mode=event_mode)

    n_prompts = len(_prompt_ids_in_order(features_dict))
    min_count = max(1, int(min_frequency * n_prompts))

    feat_events_pre = {
        feat: events for feat, events in feat_events_pre.items()
        if len({e[0] if isinstance(e, tuple) else e for e in events}) >= min_count
    }
    feat_events_post = {
        feat: events for feat, events in feat_events_post.items()
        if len({e[0] if isinstance(e, tuple) else e for e in events}) >= min_count
    }

    match_events_pre = feat_events_pre
    match_events_post = feat_events_post
    if top_k_for_matching is not None:
        match_events_pre = dict(sorted(
            feat_events_pre.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )[:top_k_for_matching])
        match_events_post = dict(sorted(
            feat_events_post.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )[:top_k_for_matching])

    event_keys = _event_keys_single_source(features_dict, pre_layer, post_layer, event_mode)
    matches = mutual_nearest_feature_matches(
        match_events_pre,
        match_events_post,
        event_keys,
        min_score=min_match_score,
    )
    matched_pre = {m.feature_a for m in matches}
    matched_post = {m.feature_b for m in matches}

    count_pre = {feat: len(events) for feat, events in feat_events_pre.items()}
    count_post = {feat: len(events) for feat, events in feat_events_post.items()}

    return {
        "pre_exclusive": {feat for feat in feat_events_pre if feat not in matched_pre},
        "surviving_pre": matched_pre,
        "surviving_post": matched_post,
        "surviving_pairs": {(m.feature_a, m.feature_b) for m in matches},
        "post_exclusive": {feat for feat in feat_events_post if feat not in matched_post},
        "count_pre": count_pre,
        "count_post": count_post,
        "matches": matches,
        "n_prompts": n_prompts,
        "n_events": len(event_keys),
        "top_k_for_matching": top_k_for_matching,
        "matching_pre_size": len(match_events_pre),
        "matching_post_size": len(match_events_post),
    }
