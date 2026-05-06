from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.poc.cross_model.config import get_spec
from src.poc.exp11_matched_prefix_mlp_graft.run import (
    _late_window_sweep_pipeline_specs,
    _parse_late_window_sweep_specs,
)
from scripts.analysis.analyze_exp55_late_window_robustness import analyze


def test_default_late_window_sweep_specs_for_llama() -> None:
    spec = get_spec("llama31_8b")
    windows = _parse_late_window_sweep_specs(
        "llama31_8b",
        spec,
        ["prelate_half", "late_full", "late_center_half", "late_terminal_half", "terminal_quarter"],
    )
    assert windows["prelate_half"] == (12, 19)
    assert windows["late_full"] == (19, 32)
    assert windows["late_center_half"] == (22, 29)
    assert windows["late_terminal_half"] == (25, 32)
    assert windows["terminal_quarter"] == (28, 32)


def test_custom_late_window_sweep_spec_and_pipeline_names() -> None:
    spec = get_spec("gemma3_4b")
    windows = _parse_late_window_sweep_specs("gemma3_4b", spec, ["tiny_terminal:30-34"])
    assert windows == {"tiny_terminal": (30, 34)}
    pipelines = _late_window_sweep_pipeline_specs(
        model="gemma3_4b",
        spec=spec,
        window_specs=["tiny_terminal:30-34"],
    )
    assert pipelines["B_lws_tiny_terminal_raw"]["host_variant"] == "pt"
    assert pipelines["B_lws_tiny_terminal_raw"]["donor_variant"] == "it"
    assert pipelines["D_lws_tiny_terminal_ptswap"]["host_variant"] == "it"
    assert pipelines["D_lws_tiny_terminal_ptswap"]["donor_variant"] == "pt"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_exp55_analyzer_computes_paired_branch_minus_baseline(tmp_path: Path) -> None:
    model_dir = tmp_path / "merged" / "gemma3_4b"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model": "gemma3_4b",
                "n_layers": 4,
                "final_region_layers": [3],
                "graft_windows_by_pipeline": {
                    "B_lws_late_full_raw": {
                        "start_layer": 2,
                        "end_layer_exclusive": 4,
                        "display_range": "2-3",
                        "host_variant": "pt",
                        "donor_variant": "it",
                        "window_name": "late_full",
                    },
                    "D_lws_late_full_ptswap": {
                        "start_layer": 2,
                        "end_layer_exclusive": 4,
                        "display_range": "2-3",
                        "host_variant": "it",
                        "donor_variant": "pt",
                        "window_name": "late_full",
                    },
                },
            }
        )
    )
    rows = []
    for prompt_id in ["p0", "p1"]:
        for step in [0, 1]:
            rows.extend(
                [
                    {
                        "prompt_id": prompt_id,
                        "pipeline": "A_prime_raw",
                        "step": step,
                        "metrics": {
                            "kl_to_own_final": [0, 0, 1, 1],
                            "delta_cosine": [0, 0, 0, 0],
                            "cross_kl": [0, 0, 1, 1],
                        },
                    },
                    {
                        "prompt_id": prompt_id,
                        "pipeline": "B_lws_late_full_raw",
                        "step": step,
                        "metrics": {
                            "kl_to_own_final": [0, 0, 2, 2],
                            "delta_cosine": [0, 0, 0, 0],
                            "cross_kl": [0, 0, 2, 2],
                        },
                    },
                    {
                        "prompt_id": prompt_id,
                        "pipeline": "C_it_chat",
                        "step": step,
                        "metrics": {
                            "kl_to_own_final": [0, 0, 3, 3],
                            "delta_cosine": [0, 0, 0, 0],
                            "cross_kl": [0, 0, 3, 3],
                        },
                    },
                    {
                        "prompt_id": prompt_id,
                        "pipeline": "D_lws_late_full_ptswap",
                        "step": step,
                        "metrics": {
                            "kl_to_own_final": [0, 0, 2, 2],
                            "delta_cosine": [0, 0, 0, 0],
                            "cross_kl": [0, 0, 2, 2],
                        },
                    },
                ]
            )
    _write_jsonl(model_dir / "step_metrics.jsonl", rows)

    summary = analyze(tmp_path, tmp_path / "analysis", models=["gemma3_4b"], n_boot=20, seed=0)
    effects = {
        (row["level"], row["side"], row["region"], row["metric"]): row
        for row in summary["effects"]
        if row["window_name"] == "late_full"
    }
    assert effects[("model", "it_graft_into_pt", "final_20pct", "kl_to_own_final")]["estimate"] == 1.0
    assert effects[("model", "pt_swap_into_it", "final_20pct", "kl_to_own_final")]["estimate"] == -1.0
    assert effects[("model", "it_graft_into_pt", "graft_window", "kl_to_own_final")]["estimate"] == 1.0
    assert effects[("dense5_model_mean", "pt_swap_into_it", "final_20pct", "kl_to_own_final")]["n_models"] == 1
