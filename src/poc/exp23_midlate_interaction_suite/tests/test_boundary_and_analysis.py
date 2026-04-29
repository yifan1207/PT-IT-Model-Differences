from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.poc.exp23_midlate_interaction_suite.boundary import BoundaryStateCapture, BoundaryStatePatch

_ANALYZER_PATH = _REPO_ROOT / "scripts" / "analysis" / "analyze_exp23_midlate_interaction_suite.py"
_SPEC = importlib.util.spec_from_file_location("analyze_exp23_midlate_interaction_suite", _ANALYZER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ANALYZER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ANALYZER
_SPEC.loader.exec_module(_ANALYZER)
_summarize_residual = _ANALYZER._summarize_residual


class AddLayer(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.full((3,), value), requires_grad=False)

    def forward(self, hidden):
        return hidden + self.bias


def test_boundary_capture_and_full_prefix_patch() -> None:
    layer = AddLayer(1.0)
    capture = BoundaryStateCapture(layer)
    x = torch.arange(18, dtype=torch.float32).view(2, 3, 3)
    out = layer(x)
    captured = capture.snapshot()
    capture.close()
    assert torch.equal(captured, x)
    assert torch.equal(out, x + 1.0)

    donor = torch.zeros_like(x)
    patch = BoundaryStatePatch(layer, donor)
    patched_out = layer(x)
    patch.close()
    assert torch.equal(patched_out, donor + 1.0)
    assert patch.n_patches == 1
    assert patch.last_max_abs_input_delta == float(x.abs().max().item())


def test_boundary_patch_noop_identity() -> None:
    """When donor == host the patch must be a numerical no-op."""
    layer = AddLayer(1.0)
    capture = BoundaryStateCapture(layer)
    x = torch.arange(18, dtype=torch.float32).view(2, 3, 3)
    _ = layer(x)
    captured = capture.snapshot()
    capture.close()

    # donor = the captured host hidden state, so the patch is an identity op.
    patch = BoundaryStatePatch(layer, captured, noop_tol=1e-6)
    patched_out = layer(x)
    patch.close()
    # Output must equal the unpatched output to bitwise tolerance.
    assert torch.equal(patched_out, x + 1.0)
    assert patch.last_max_abs_input_delta is not None
    assert patch.last_max_abs_input_delta <= 1e-6


def test_boundary_patch_noop_tol_violation_raises() -> None:
    """A donor that differs from host by more than ``noop_tol`` must error."""
    import pytest

    layer = AddLayer(1.0)
    x = torch.arange(18, dtype=torch.float32).view(2, 3, 3)
    donor = x + 0.01  # differs by 0.01 per coordinate
    patch = BoundaryStatePatch(layer, donor, noop_tol=1e-6)
    with pytest.raises(RuntimeError, match="no-op identity check failed"):
        _ = layer(x)
    patch.close()


def _write_record(path: Path, model: str, prompt_id: str, base: float) -> None:
    cells = {
        "U_PT__L_PT": base,
        "U_PT__L_IT": base + 1.0,
        "U_IT__L_PT": base + 2.0,
        "U_IT__L_IT": base + 5.0,
    }
    payload = {
        "prompt_id": prompt_id,
        "model": model,
        "events": {
            "first_diff": {
                "valid": True,
                "cells": {
                    cell: {
                        "common_it": {
                            "it_vs_pt_margin": margin,
                            "token_choice_class": "it" if cell.endswith("L_IT") else "pt",
                            "trajectory": {"late_kl_mean": margin / 10},
                        },
                        "common_pt": {
                            "it_vs_pt_margin": margin - 0.5,
                            "token_choice_class": "other",
                            "trajectory": {"late_kl_mean": margin / 20},
                        },
                    }
                    for cell, margin in cells.items()
                },
                "noop_patch_checks": {
                    "noop": {"common_it_margin_abs_delta": 1e-6},
                },
            }
        },
    }
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def test_residual_factorial_interaction_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    for model in ("gemma3_4b", "qwen3_4b"):
        out_dir = run_root / "residual_factorial" / "raw_shared" / model
        out_dir.mkdir(parents=True)
        path = out_dir / "records.jsonl.gz"
        _write_record(path, model, f"{model}_0", 0.0)
        _write_record(path, model, f"{model}_1", 0.5)

    summary = _summarize_residual(
        run_root=run_root,
        models=["gemma3_4b", "qwen3_4b"],
        prompt_mode="raw_shared",
        readouts=["common_it"],
        n_boot=50,
        seed=1,
    )
    effects = summary["effects"]["common_it"]
    assert effects["late_it_given_pt_upstream"]["estimate"] == 1.0
    assert effects["late_it_given_it_upstream"]["estimate"] == 3.0
    assert effects["interaction"]["estimate"] == 2.0
    assert summary["quality"]["noop_margin_abs_delta_max"] == 1e-6
