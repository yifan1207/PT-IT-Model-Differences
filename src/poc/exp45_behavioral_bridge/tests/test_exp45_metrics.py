from __future__ import annotations

from src.poc.exp45_behavioral_bridge.analyze import _one_step_effect_rows, _rollout_effect_rows
from src.poc.exp45_behavioral_bridge.metrics import lexical_metrics


def test_lexical_metrics_has_first_token_removed_view() -> None:
    out = lexical_metrics("Sure.\n- One\n- Two", post_first_text="\n- One\n- Two")
    assert out["full_answer_opening"] is True
    assert out["full_bullet_count"] == 2
    assert out["post_first_bullet_count"] == 2
    assert out["full_lexical_it_like_score"] > out["post_first_lexical_it_like_score"]


def test_one_step_contrasts_use_hybrid_portability() -> None:
    rows = []
    vals = {
        "U_PT__L_PT": 0.0,
        "U_PT__L_IT": 0.2,
        "U_IT__L_PT": 0.4,
        "U_IT__L_IT": 0.9,
    }
    for cell, val in vals.items():
        rows.append(
            {
                "model": "m",
                "prompt_id": "p",
                "event_kind": "first_diff",
                "cell": cell,
                "margin_it_minus_pt": val,
                "pairwise_it_win": float(val > 0.5),
                "top5_it": float(val > 0.1),
                "top1_it": float(val > 0.8),
                "it_rank": 10 - val,
                "it_gap_to_top1": val - 1.0,
            }
        )
    effects = _one_step_effect_rows(rows, models=["m"], n_boot=0)
    effect = next(r for r in effects if r["effect"] == "pairwise_portability_gap" and r["metric"] == "margin_it_minus_pt")
    assert abs(effect["estimate"] - 0.7) < 1e-9
    interaction = next(r for r in effects if r["effect"] == "factorial_interaction" and r["metric"] == "margin_it_minus_pt")
    assert abs(interaction["estimate"] - 0.3) < 1e-9


def test_rollout_contrasts_include_post_first_score() -> None:
    rows = []
    vals = {
        "U_PT__L_PT": 0.0,
        "U_PT__L_IT": 0.5,
        "U_IT__L_PT": 0.2,
        "U_IT__L_IT": 1.1,
    }
    for cell, val in vals.items():
        rows.append(
            {
                "model": "m",
                "prompt_id": "p",
                "event_kind": "first_diff",
                "cell": cell,
                "full_lexical_it_like_score": val,
                "post_first_lexical_it_like_score": val / 2,
                "full_word_count": 10 + val,
                "post_first_word_count": 9 + val,
                "first_generated_is_t_it": cell == "U_IT__L_IT",
                "first_generated_is_t_pt": cell == "U_PT__L_PT",
            }
        )
    effects = _rollout_effect_rows(rows, models=["m"], n_boot=0)
    gap = next(
        r
        for r in effects
        if r["effect"] == "behavioral_portability_gap" and r["metric"] == "post_first_lexical_it_like_score"
    )
    assert abs(gap["estimate"] - 0.3) < 1e-9

