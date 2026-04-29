from __future__ import annotations

import json
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYZER_PATH = REPO_ROOT / "scripts" / "analysis" / "analyze_olmo_stage_progression.py"
spec = importlib.util.spec_from_file_location("analyze_olmo_stage_progression", ANALYZER_PATH)
assert spec is not None and spec.loader is not None
_analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_analyzer)
build_summary = _analyzer.build_summary


def _exp20_record(prompt_id: str) -> dict:
    def condition(cls: str, margin: float) -> tuple[dict, dict]:
        return (
            {"class": cls, "token_id": 1, "decoded": "x"},
            {"windows": {"late_reconciliation": {"it_minus_pt_margin": {"total_delta": margin}}}},
        )

    token_at_step = {}
    conditions = {}
    for name, cls, margin in [
        ("A_pt_raw", "pt", -1.0),
        ("B_mid_raw", "it", 0.5),
        ("B_late_raw", "other", 0.25),
        ("B_midlate_raw", "it", 0.75),
        ("C_it_chat", "it", 1.0),
        ("D_mid_ptswap", "pt", -0.5),
        ("D_late_ptswap", "pt", -0.75),
    ]:
        token_at_step[name], conditions[name] = condition(cls, margin)
    return {
        "prompt_id": prompt_id,
        "readouts": {
            "first_diff": {
                "condition_token_at_step": token_at_step,
                "conditions": conditions,
            }
        },
    }


def test_build_summary_writes_stage_table(tmp_path: Path) -> None:
    model = "olmo2_7b_pt_sft"
    exp20_root = tmp_path / "exp20"
    exp20_model_dir = exp20_root / "raw_shared" / model
    exp20_model_dir.mkdir(parents=True)
    with (exp20_model_dir / "exp20_validation_records.jsonl").open("w") as handle:
        handle.write(json.dumps(_exp20_record("p0")) + "\n")
        handle.write(json.dumps(_exp20_record("p1")) + "\n")

    exp23_summary = tmp_path / "exp23_summary.json"
    exp23_summary.write_text(json.dumps({
        "residual_factorial": {
            "effects": {
                "common_it": {
                    "interaction": {"model_cis": {model: {"estimate": 2.0, "ci95_low": 1.5, "ci95_high": 2.5, "n_units": 2, "n_prompt_clusters": 2}}},
                    "late_it_given_pt_upstream": {"model_cis": {model: {"estimate": 0.2, "ci95_low": 0.1, "ci95_high": 0.3}}},
                    "late_it_given_it_upstream": {"model_cis": {model: {"estimate": 2.2, "ci95_low": 2.0, "ci95_high": 2.4}}},
                    "late_weight_effect": {"model_cis": {model: {"estimate": 1.2}}},
                    "upstream_context_effect": {"model_cis": {model: {"estimate": 3.1}}},
                }
            }
        }
    }))

    compat_root = tmp_path / "compat"
    compat_model = compat_root / model
    compat_model.mkdir(parents=True)
    (compat_model / "exp23_compatibility_permutation_summary.json").write_text(json.dumps({
        "models": [model],
        "permutation": {"p_upper": 0.001},
    }))

    out_dir = tmp_path / "out"
    summary = build_summary(
        exp20_root=exp20_root,
        exp23_summary=exp23_summary,
        compatibility_root=compat_root,
        out_dir=out_dir,
        models=[model],
        prompt_mode="raw_shared",
    )

    row = summary["rows"][0]
    assert row["transition"] == "PT->SFT"
    assert row["first_diff_events"] == 2
    assert row["b_mid_it_match_rate"] == 1.0
    assert row["b_late_it_match_rate"] == 0.0
    assert row["interaction"] == 2.0
    assert row["label_swap_p_upper"] == 0.001
    assert (out_dir / "olmo_stage_progression_table.csv").exists()
    assert (out_dir / "olmo_stage_progression_summary.json").exists()
    assert (out_dir / "olmo_stage_progression.png").exists()
