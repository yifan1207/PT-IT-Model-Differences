from __future__ import annotations

import gzip
import json
from pathlib import Path

from src.poc.exp51_native_history_crosspatch.analyze import analyze_run


def _event(horizon: int, base: float) -> dict:
    return {
        "valid": True,
        "event": {"horizon": horizon},
        "cells": {
            "U_PT__L_PT": {"common_it": {"it_vs_pt_margin": base}},
            "U_PT__L_IT": {"common_it": {"it_vs_pt_margin": base + 0.5}},
            "U_IT__L_PT": {"common_it": {"it_vs_pt_margin": base + 1.0}},
            "U_IT__L_IT": {"common_it": {"it_vs_pt_margin": base + 2.0}},
        },
        "noop_patch_checks": {
            "noop": {"common_it_margin_abs_delta": 0.0},
        },
    }


def _row(model: str, prompt_id: str) -> dict:
    return {
        "model": model,
        "prompt_id": prompt_id,
        "prompt_mode": "raw_shared",
        "history_source": "it",
        "category": "GOV-FORMAT",
        "valid": True,
        "native_history": {
            "horizon_status": {
                "4": {
                    "prefix_not_terminated": True,
                    "local_disagreement": True,
                    "valid_after_real_token_mask": True,
                },
                "8": {
                    "prefix_not_terminated": True,
                    "local_disagreement": False,
                    "valid_after_real_token_mask": False,
                    "reason": "pt_it_agree",
                },
            }
        },
        "events": {"it_h4": _event(4, 0.0)},
    }


def test_exp51_analysis_interaction_and_support(tmp_path: Path) -> None:
    root = tmp_path / "run"
    out = root / "it" / "raw_shared" / "llama31_8b"
    out.mkdir(parents=True)
    with gzip.open(out / "records.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(_row("llama31_8b", "p0")) + "\n")
        handle.write(json.dumps(_row("llama31_8b", "p1")) + "\n")

    summary = analyze_run(
        run_root=root,
        models=["llama31_8b"],
        history_sources=["it"],
        prompt_mode="raw_shared",
        readouts=["common_it"],
        primary_horizons=[4, 8, 16],
        n_boot=100,
        seed=0,
    )

    h4 = summary["effects"]["it"]["common_it"]["by_horizon"]["4"]["interaction"]
    assert h4["estimate"] == 0.5
    assert h4["n_units"] == 2
    support = summary["support"]["by_model_history_horizon"]["llama31_8b::it::h4"]
    assert support["n_valid_after_real_token_mask"] == 2
    support_h8 = summary["support"]["by_model_history_horizon"]["llama31_8b::it::h8"]
    assert support_h8["n_local_disagreements"] == 0
