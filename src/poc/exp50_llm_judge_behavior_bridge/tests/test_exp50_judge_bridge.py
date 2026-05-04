from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp50_llm_judge_behavior_bridge.analyze import analyze
from src.poc.exp50_llm_judge_behavior_bridge.judge_requests import build_judge_requests, load_rollout_records


CELLS = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")


def _toy_rollout(cell: str, *, model: str = "toy_model", prompt_id: str = "p0") -> dict:
    first_token = {
        "U_PT__L_PT": " base",
        "U_PT__L_IT": " maybe",
        "U_IT__L_PT": " draft",
        "U_IT__L_IT": " Sure",
    }[cell]
    return {
        "source_run": "toy_run",
        "model": model,
        "prompt_id": prompt_id,
        "event_kind": "first_diff",
        "position": 3,
        "position_ge_3": True,
        "category": "GOV-FORMAT",
        "boundary_layer": 19,
        "cell": cell,
        "raw_prompt": "Write a short structured answer.",
        "shared_prefix_text": "Answer:",
        "generated_tokens": [first_token, " response", "."],
        "generated_tokens_count": 3,
        "continuation_text": first_token + " response.",
    }


def _write_rollouts(root: Path, rows: list[dict]) -> None:
    out = root / "raw" / "toy_model" / "rollout_records.jsonl.gz"
    out.parent.mkdir(parents=True)
    with gzip.open(out, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_requests_are_order_balanced() -> None:
    rows = [_toy_rollout(cell) for cell in CELLS]
    requests = build_judge_requests(rows, include_same_text_controls=False)
    primary = [row for row in requests if row["comparison"] == "native_it_necessity"]
    assert len(primary) == 2
    assert {row["order_index"] for row in primary} == {0, 1}
    assert primary[0]["a_cell"] == primary[1]["b_cell"]
    assert primary[0]["b_cell"] == primary[1]["a_cell"]
    assert primary[0]["target_is_a"] != primary[1]["target_is_a"]
    assert primary[0]["response_format"]["json_schema"]["strict"] is True


def test_load_rollout_records_from_exp45_style_root(tmp_path: Path) -> None:
    rows = [_toy_rollout(cell) for cell in CELLS]
    _write_rollouts(tmp_path, rows)
    loaded = load_rollout_records([tmp_path])
    assert len(loaded) == 4
    assert {row["cell"] for row in loaded} == set(CELLS)
    assert all(row["source_run"] == tmp_path.name for row in loaded)


def test_analyze_order_balanced_interaction(tmp_path: Path) -> None:
    rows = [_toy_rollout(cell) for cell in CELLS]
    requests = build_judge_requests(rows, include_same_text_controls=True, control_fraction=1.0)
    response_path = tmp_path / "judge_responses.jsonl.gz"
    winners = {
        "native_it_necessity": "target",
        "pt_upstream_sufficiency": "other",
        "positive_control_full_pt_to_it": "target",
        "upstream_vs_late_diagnostic": "target",
    }
    with gzip.open(response_path, "wt", encoding="utf-8") as handle:
        for row in requests:
            winner_mode = winners.get(row["comparison"])
            if row["comparison"] == "same_text_tie_control":
                winner = "TIE"
            elif winner_mode == "target":
                winner = "A" if row["target_is_a"] else "B"
            else:
                winner = "B" if row["target_is_a"] else "A"
            out = {
                **{k: v for k, v in row.items() if k not in {"messages", "response_format"}},
                "judge_provider": "openai",
                "judge_model": "toy",
                "result": {
                    "winner": winner,
                    "confidence": 0.9,
                    "margin": "large" if winner != "TIE" else "tie_or_unclear",
                    "primary_reason_tag": "instruction_following",
                    "tie_reason": "equally_good" if winner == "TIE" else "not_tie",
                    "length_bias_flag": False,
                    "short_rationale": "toy",
                },
            }
            handle.write(json.dumps(out) + "\n")
    summary = analyze(responses=response_path, out_dir=tmp_path / "analysis", models=["toy_model"], n_boot=0)
    assert summary["n_invalid"] == 0
    interaction = [
        row
        for row in summary["primary_interactions"]
        if row["group"] == "all" and row["metric"] == "behavioral_interaction"
    ][0]
    assert interaction["estimate"] == 1.0
    assert summary["paper_grade_checks"]["same_text_control_reported"] is True
