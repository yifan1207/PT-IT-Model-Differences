from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp26_residual_opposition_mediation.analyze import analyze
from src.poc.exp26_residual_opposition_mediation.variants import (
    apply_variant,
    decompose_negative_parallel,
)


def test_negative_parallel_decomposition_preserves_update() -> None:
    residual = torch.tensor([1.0, 0.0])
    update = torch.tensor([-3.0, 4.0])
    opposing, remainder, coeff = decompose_negative_parallel(update, residual)
    assert torch.allclose(opposing, torch.tensor([-3.0, 0.0]))
    assert torch.allclose(remainder, torch.tensor([0.0, 4.0]))
    assert torch.allclose(opposing + remainder, update)
    assert coeff.item() < 0


def test_noopp_removes_negative_projection() -> None:
    residual = torch.tensor([1.0, 0.0])
    update = torch.tensor([-3.0, 4.0])
    result = apply_variant(update, residual, variant="noopp")
    assert torch.dot(result.update.float(), residual) == 0


def test_flipopp_changes_projection_sign() -> None:
    residual = torch.tensor([1.0, 0.0])
    update = torch.tensor([-3.0, 4.0])
    result = apply_variant(update, residual, variant="flipopp")
    assert torch.dot(result.update.float(), residual) > 0


def test_randorth_is_orthogonal_and_norm_matched() -> None:
    residual = torch.tensor([1.0, 0.0, 0.0])
    update = torch.tensor([-3.0, 4.0, 0.0])
    opposing, remainder, _ = decompose_negative_parallel(update, residual)
    result = apply_variant(update, residual, variant="randorth", rand_seed=123)
    replacement = result.update.float() - remainder
    assert torch.allclose(torch.dot(replacement, residual), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(replacement.norm(), opposing.norm(), atol=1e-5)


def test_normpres_noopp_matches_original_update_norm() -> None:
    residual = torch.tensor([1.0, 0.0])
    update = torch.tensor([-3.0, 4.0])
    result = apply_variant(update, residual, variant="normpres_noopp")
    assert torch.allclose(result.update.float().norm(), update.norm(), atol=1e-5)


def test_ptlevel_opp_scales_opposition() -> None:
    residual = torch.tensor([1.0, 0.0])
    update = torch.tensor([-4.0, 2.0])
    result = apply_variant(update, residual, variant="ptlevel_opp", alpha=0.25)
    assert torch.allclose(result.update.float(), torch.tensor([-1.0, 2.0]), atol=1e-5)


def _write_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _cell(margin: float) -> dict:
    return {
        "common_it": {"it_vs_pt_margin": margin},
        "common_pt": {"it_vs_pt_margin": margin},
    }


def test_analyzer_joins_exp26_variants_to_exp23_baseline(tmp_path: Path) -> None:
    exp23 = tmp_path / "exp23"
    exp26 = tmp_path / "exp26"
    model = "gemma3_4b"
    event = {
        "step": 4,
        "pt_token": {"token_id": 1, "assistant_marker": False},
        "it_token": {"token_id": 2, "assistant_marker": False},
    }
    _write_gz(
        exp23 / "residual_factorial" / "raw_shared" / model / "records.jsonl.gz",
        [
            {
                "model": model,
                "prompt_id": "p0",
                "events": {
                    "first_diff": {
                        "valid": True,
                        "event": event,
                        "cells": {
                            "U_PT__L_PT": _cell(0.0),
                            "U_PT__L_IT": _cell(1.0),
                            "U_IT__L_PT": _cell(0.5),
                            "U_IT__L_IT": _cell(3.0),
                        },
                    }
                },
            }
        ],
    )
    _write_gz(
        exp26 / "records" / "raw_shared" / model / "records.jsonl.gz",
        [
            {
                "model": model,
                "prompt_id": "p0",
                "event_kind": "first_diff",
                "variant": "noopp",
                "seed": None,
                "valid": True,
                "event": event,
                "cells": {
                    "U_PT__L_IT_variant": _cell(0.8),
                    "U_IT__L_IT_variant": _cell(2.0),
                },
                "diagnostics": {"mean_opp_norm_frac": 0.2},
            }
        ],
    )
    out = tmp_path / "analysis"
    analyze(
        argparse.Namespace(
            exp26_root=exp26,
            exp23_root=exp23,
            out_dir=out,
            models=[model],
            prompt_mode="raw_shared",
            event_kind="first_diff",
            n_boot=10,
            seed=0,
        )
    )
    summary = json.loads((out / "exp26_summary.json").read_text())
    primary = summary["primary"]["noopp"]
    assert primary["interaction_full"] == 1.5
    assert primary["interaction_variant"] == 0.7
    assert primary["drop"] == 0.8
    assert (out / "exp26_mediation_plot.png").exists()
