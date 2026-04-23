from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp18_midlate_token_handoff.handoff_analysis import run_analysis


def _metrics(n_layers: int, *, target_rank: int, top1_token: int):
    top20 = []
    for layer in range(n_layers):
        if layer >= 12:
            ids = [top1_token, 101, 102, 103, 104, 42]
        else:
            ids = [top1_token, 101, 102, 103, 104, 105]
        top20.append(ids + list(range(200, 214)))
    return {
        "top1_token": [top1_token] * n_layers,
        "top20_ids": top20,
        "next_token_rank": [target_rank] * n_layers,
    }


def test_handoff_analysis_reads_tiny_depth_trace(tmp_path):
    n_layers = 34
    root = tmp_path / "depth"
    model_dir = root / "gemma3_4b"
    model_dir.mkdir(parents=True)
    rows = [
        ("C_it_chat", 42, " answer", _metrics(n_layers, target_rank=1, top1_token=42)),
        ("A_prime_raw", 42, " answer", _metrics(n_layers, target_rank=50, top1_token=7)),
        ("B_early_raw", 42, " answer", _metrics(n_layers, target_rank=45, top1_token=8)),
        ("B_mid_raw", 42, " answer", _metrics(n_layers, target_rank=30, top1_token=9)),
        ("B_late_raw", 42, " answer", _metrics(n_layers, target_rank=5, top1_token=42)),
    ]
    with (model_dir / "step_metrics.jsonl").open("w") as handle:
        for pipeline, token_id, token_str, metrics in rows:
            handle.write(
                json.dumps(
                    {
                        "prompt_id": "p0",
                        "pipeline": pipeline,
                        "step": 0,
                        "token_id": token_id,
                        "token_str": token_str,
                        "metrics": metrics,
                    }
                )
                + "\n"
            )

    summary = run_analysis(
        models=["gemma3_4b"],
        depth_roots=[root],
        out_dir=tmp_path / "out",
        max_prompts=None,
        allow_missing=False,
    )
    model = summary["models"]["gemma3_4b"]
    assert model["n_prompts"] == 1
    assert model["n_steps"] == 1
    assert "mid_policy" in model["windows"]
    assert "late_reconciliation" in model["windows"]
    assert model["handoff"]["by_token_category"]["CONTENT"]["count"] >= 1
