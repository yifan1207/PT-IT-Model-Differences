from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp37_random_prefix_baseline.analyze_random_prefix_baselines import (
    _overlap_matched,
    interaction_from_margins,
    Unit,
)
from src.poc.exp37_random_prefix_baseline.collect_prefix_manifests import make_manifest_record


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return " ".join(str(x) for x in ids)


def test_interaction_from_margins() -> None:
    effects = interaction_from_margins(
        {
            "U_PT__L_PT": 0.0,
            "U_PT__L_IT": 1.0,
            "U_IT__L_PT": 2.0,
            "U_IT__L_IT": 6.0,
        }
    )
    assert effects is not None
    assert effects["late_effect_from_pt_upstream"] == 1.0
    assert effects["late_effect_from_it_upstream"] == 4.0
    assert effects["interaction"] == 3.0


def test_manifest_record_uses_exp23_prefix_shim() -> None:
    event = {
        "kind": "first_diff",
        "step": 3,
        "pt_token": {"token_id": 10},
        "it_token": {"token_id": 11},
    }
    record = make_manifest_record(
        prompt_id="p0",
        model_name="toy",
        max_new_tokens=64,
        prefix_ids_for_exp23=[7, 8, 9],
        event=event,
        tokenizer=TinyTokenizer(),
        source_prefix_ids=[1, 2, 3],
    )
    assert record["free_runs"]["A_pt_raw"]["generated_token_ids"][: event["step"]] == [7, 8, 9]
    assert record["free_runs"]["C_it_chat"]["generated_token_ids"] == [1, 2, 3]
    assert record["divergence_events"]["first_diff"]["pt_token"]["token_id"] == 10


def test_overlap_matched_keeps_shared_bins() -> None:
    margins = {
        "U_PT__L_PT": 0.0,
        "U_PT__L_IT": 1.0,
        "U_IT__L_PT": 2.0,
        "U_IT__L_IT": 6.0,
    }
    meta = {
        "sampled_prefix_step": 3,
        "pt_entropy": 1.0,
        "it_entropy": 1.2,
        "pt_confidence": 0.7,
        "it_confidence": 0.6,
        "pt_top1_top2_margin": 0.4,
        "it_top1_top2_margin": 0.5,
    }
    ref = [Unit("first", "m", "p0", "common_it", margins, meta)]
    base = [Unit("base", "m", "p1", "common_it", margins, dict(meta))]
    ref_kept, base_kept, info = _overlap_matched(ref, base)
    assert len(ref_kept) == 1
    assert len(base_kept) == 1
    assert info["n_overlap_bins"] == 1
