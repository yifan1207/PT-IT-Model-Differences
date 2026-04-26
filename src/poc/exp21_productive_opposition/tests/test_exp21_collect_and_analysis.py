from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.analyze_exp21_productive_opposition import analyze
from src.poc.exp21_productive_opposition.collect import _layer_metrics


def test_layer_metrics_finite_difference_support_signs() -> None:
    final_norm = torch.nn.Identity()
    lm_head = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(
            torch.tensor(
                [
                    [0.0, 1.0],  # IT token
                    [1.0, 0.0],  # PT token
                    [1.0, 1.0],  # alternative
                ]
            )
        )
    row = _layer_metrics(
        pre=torch.tensor([1.0, 0.0]),
        update=torch.tensor([-1.0, 2.0]),
        final_norm=final_norm,
        lm_head=lm_head,
        real_token_mask=torch.tensor([True, True, True]),
        tokenizer=_TinyTokenizer(),
        y_it=0,
        y_pt=1,
        pipeline_token=0,
        top_k=2,
    )
    assert row["delta_cosine_mlp"] < 0
    assert row["support_it_token"] == 2.0
    assert row["support_pt_token"] == -1.0
    assert row["margin_writein_it_vs_pt"] == 3.0
    assert row["productive_opposition"] is True
    assert row["opposition_margin_it_vs_pt"] > 0


def test_analyzer_smoke_on_mock_record(tmp_path: Path) -> None:
    root = tmp_path / "exp21"
    model_dir = root / "native" / "gemma3_4b"
    model_dir.mkdir(parents=True)
    record = _mock_record()
    with gzip.open(model_dir / "records.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    out = root / "analysis"
    summary = analyze(root, out, ["gemma3_4b"], n_boot=5, seed=0)
    assert (out / "summary.json").exists()
    assert (out / "effects.csv").exists()
    assert (out / "productive_opposition_main.png").exists()
    assert summary["pooled"]["native"]["events_present"]["first_diff"] == 1


class _TinyTokenizer:
    def decode(self, token_ids, **_kwargs):
        return {0: " Sure", 1: " The", 2: " answer"}.get(int(token_ids[0]), " token")


def _mock_condition(value: float) -> dict:
    windows = {}
    for window in ["early", "mid_policy", "late_reconciliation", "exp11_early", "exp11_mid", "exp11_late"]:
        windows[window] = {
            "delta_cosine_mlp": -0.1 * value,
            "productive_opposition_rate": 1.0,
            "margin_writein_it_vs_pt": value,
            "opposition_margin_it_vs_pt": value / 2,
            "target_vs_alt_margin": value / 3,
            "top_supported_category_counts": {"FORMAT": 1},
            "top_suppressed_category_counts": {"FUNCTION_OTHER": 1},
        }
    return {"winner_class": "it", "windows": windows}


def _mock_record() -> dict:
    conditions = {
        "A_pt_raw": _mock_condition(1.0),
        "B_mid_raw": _mock_condition(1.5),
        "B_late_raw": _mock_condition(2.0),
        "B_midlate_raw": _mock_condition(2.5),
        "C_it_chat": _mock_condition(3.0),
        "D_mid_ptswap": _mock_condition(2.0),
        "D_late_ptswap": _mock_condition(1.5),
        "D_midlate_ptswap": _mock_condition(1.0),
        "B_late_identity": _mock_condition(1.0),
        "D_late_identity": _mock_condition(3.0),
        "B_late_rand_resproj_s0": _mock_condition(1.2),
        "D_late_rand_resproj_s0": _mock_condition(1.8),
    }
    return {
        "prompt_id": "mock",
        "model": "gemma3_4b",
        "prompt_mode": "native",
        "events": {
            "first_diff": {
                "event": {
                    "kind": "first_diff",
                    "step": 0,
                    "pt_token": {"token_id": 1},
                    "it_token": {"token_id": 0},
                },
                "conditions": conditions,
            }
        },
    }
