from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.analyze_exp23_midlate_kl_factorial import analyze, write_outputs


CONDITION_VALUES = {
    "A_prime_raw": 1.0,
    "B_mid_raw": 1.2,
    "B_late_raw": 1.5,
    "B_midlate_raw": 1.9,
    "C_it_chat": 2.0,
    "D_mid_ptswap": 1.8,
    "D_late_ptswap": 1.4,
    "D_midlate_ptswap": 1.0,
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _mock_model_dir(root: Path, model: str) -> None:
    model_dir = root / "merged" / model
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model": model,
                "n_layers": 10,
                "pipeline_prompt_modes": {
                    "A_prime_raw": "raw_format_b",
                    "B_mid_raw": "raw_format_b",
                    "B_late_raw": "raw_format_b",
                    "B_midlate_raw": "raw_format_b",
                    "C_it_chat": "it_chat_template",
                    "D_mid_ptswap": "it_chat_template",
                    "D_late_ptswap": "it_chat_template",
                    "D_midlate_ptswap": "it_chat_template",
                },
            }
        )
    )

    rows = []
    for prompt_id in ["p0", "p1"]:
        for condition, value in CONDITION_VALUES.items():
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "pipeline": condition,
                    "step": 0,
                    "metrics": {
                        "kl_to_own_final": [
                            99.0,
                            99.0,
                            99.0,
                            99.0,
                            99.0,
                            99.0,
                            99.0,
                            99.0,
                            value,
                            value,
                        ]
                    },
                }
            )
    _write_jsonl(model_dir / "step_metrics.jsonl", rows)


def test_exp23_analysis_recovers_factorial_effects(tmp_path: Path) -> None:
    run_root = tmp_path / "exp23"
    for model in ["gemma3_4b", "qwen3_4b"]:
        _mock_model_dir(run_root, model)

    summary = analyze(run_root, ["gemma3_4b", "qwen3_4b"], n_boot=20, seed=0)
    out_dir = run_root / "analysis"
    write_outputs(summary, out_dir)

    assert (out_dir / "exp23_midlate_kl_factorial_summary.json").exists()
    assert (out_dir / "exp23_midlate_kl_factorial_effects.csv").exists()
    assert (out_dir / "exp23_midlate_kl_factorial_report.md").exists()

    import matplotlib

    matplotlib.use("Agg")
    from scripts.plot.plot_exp23_midlate_kl_factorial import main as plot_main

    old_argv = sys.argv
    try:
        sys.argv = [
            "plot_exp23_midlate_kl_factorial.py",
            "--summary",
            str(out_dir / "exp23_midlate_kl_factorial_summary.json"),
            "--out",
            str(out_dir / "exp23_midlate_kl_factorial.png"),
        ]
        plot_main()
    finally:
        sys.argv = old_argv
    assert (out_dir / "exp23_midlate_kl_factorial.png").exists()

    dense_rows = {
        row["effect"]: row
        for row in summary["effects"]
        if row["model"] == "dense5"
    }
    assert math.isclose(dense_rows["E_mid_pt"]["mean"], 0.2)
    assert math.isclose(dense_rows["E_late_pt"]["mean"], 0.5)
    assert math.isclose(dense_rows["E_midlate_pt"]["mean"], 0.9)
    assert math.isclose(dense_rows["I_pt"]["mean"], 0.2)
    assert math.isclose(dense_rows["L_given_M_pt"]["mean"], 0.7)
    assert math.isclose(dense_rows["L_alone_pt"]["mean"], 0.5)

    assert math.isclose(dense_rows["E_mid_it"]["mean"], -0.2)
    assert math.isclose(dense_rows["E_late_it"]["mean"], -0.6)
    assert math.isclose(dense_rows["E_midlate_it"]["mean"], -1.0)
    assert math.isclose(dense_rows["I_it"]["mean"], -0.2)
    assert math.isclose(dense_rows["I_collapse_it"]["mean"], 0.2)

    for payload in summary["models"].values():
        validation = payload["validation"]
        assert validation["missing_conditions"] == []
        assert validation["pt_common_prompt_count"] == 2
        assert validation["it_common_prompt_count"] == 2
        assert validation["prompt_mode_mismatches"] == {}
