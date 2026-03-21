import numpy as np

from src.poc.exp3.analysis.word_categories import classify_generated_tokens_by_word
from src.poc.exp4.analysis.feature_alignment import (
    build_population_partition,
    continuity_for_sets,
    mutual_nearest_feature_matches,
)


def test_word_level_discourse_phrase_marks_all_tokens():
    tokens = [
        {"token_str": "Of"},
        {"token_str": " "},
        {"token_str": "course"},
        {"token_str": ","},
        {"token_str": " the"},
        {"token_str": " answer"},
    ]
    categories = classify_generated_tokens_by_word(tokens)
    assert categories[:3] == ["DISCOURSE", "PUNCTUATION", "DISCOURSE"]
    assert categories[4:] == ["FUNCTION", "CONTENT"]


def test_word_level_function_and_hyphenated_content():
    tokens = [
        {"token_str": "the"},
        {"token_str": " state-of-the-art"},
        {"token_str": " system"},
    ]
    categories = classify_generated_tokens_by_word(tokens)
    assert categories == ["FUNCTION", "CONTENT", "CONTENT"]


def test_word_level_structural_and_punctuation():
    tokens = [
        {"token_str": "###"},
        {"token_str": "\n"},
        {"token_str": "Results"},
        {"token_str": ":"},
    ]
    categories = classify_generated_tokens_by_word(tokens)
    assert categories == ["STRUCTURAL", "PUNCTUATION", "CONTENT", "PUNCTUATION"]


def _features_dict(layer_features: dict[int, list[int]], n_layers: int = 4) -> dict[str, np.ndarray]:
    af = np.empty((1, n_layers), dtype=object)
    for layer in range(n_layers):
        af[0, layer] = np.array(layer_features.get(layer, []), dtype=np.int32)
    return {"prompt_0": af}


def test_mutual_nearest_alignment_matches_profile_equivalents():
    event_keys = ["prompt_0", "prompt_1"]
    feat_events_a = {10: {"prompt_0", "prompt_1"}}
    feat_events_b = {77: {"prompt_0", "prompt_1"}}
    matches = mutual_nearest_feature_matches(feat_events_a, feat_events_b, event_keys, min_score=0.0)
    assert len(matches) == 1
    assert matches[0].feature_a == 10
    assert matches[0].feature_b == 77


def test_population_partition_uses_alignment_not_raw_id_overlap():
    af_a = np.empty((2, 3), dtype=object)
    af_a[0, 0] = np.array([1], dtype=np.int32)
    af_a[1, 0] = np.array([1], dtype=np.int32)
    af_a[0, 2] = np.array([9], dtype=np.int32)
    af_a[1, 2] = np.array([9], dtype=np.int32)
    af_a[0, 1] = af_a[1, 1] = np.array([], dtype=np.int32)

    partition = build_population_partition(
        {"prompt_0": af_a},
        pre_layer=0,
        post_layer=2,
        min_frequency=0.0,
        event_mode="step",
        min_match_score=0.0,
    )
    assert partition["pre_exclusive"] == set()
    assert partition["post_exclusive"] == set()
    assert partition["surviving_pairs"] == {(1, 9)}


def test_continuity_for_sets_uses_matches():
    matches = []
    assert continuity_for_sets({1, 2}, {3, 4}, matches) == 0.0
