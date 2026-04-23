from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.analyze_exp20 import analyze_run


def _summary_window(total: float) -> dict:
    return {
        "y_it_logit": {"mean_step_delta": total / 2, "total_delta": total, "start_value": 0.0, "end_value": total},
        "y_pt_logit": {"mean_step_delta": 0.0, "total_delta": 0.0, "start_value": 0.0, "end_value": 0.0},
        "it_minus_pt_margin": {
            "mean_step_delta": total / 2,
            "total_delta": total,
            "start_value": 0.0,
            "end_value": total,
        },
    }


def test_analyze_exp20_tiny_run(tmp_path: Path):
    record = {
        "prompt_id": "p0",
        "model": "gemma3_4b",
        "prompt_mode": "native",
        "max_new_tokens": 128,
        "free_runs": {
            "A_pt_raw": {"n_steps": 128},
            "C_it_chat": {"n_steps": 128},
        },
        "pairwise_agreement": {
            "A_pt_raw__C_it_chat": {
                "compared": 128,
                "same": 3,
                "agreement_fraction": 3 / 128,
                "first_divergence_step": 3,
            }
        },
        "cluster_summary": {
            "mean_cluster_entropy": 0.4,
            "mean_unique_token_count": 2.0,
            "mean_majority_size": 5.0,
            "leaves_majority_step": {"A_pt_raw": 3},
            "unique_token_step": {"C_it_chat": 4},
        },
        "divergence_events": {
            "first_diff": {
                "kind": "first_diff",
                "step": 3,
                "shared_prefix_clean": True,
                "pt_token": {
                    "token_id": 10,
                    "token_str": " cat",
                    "token_category": "CONTENT",
                    "token_category_collapsed": "CONTENT",
                    "assistant_marker": False,
                },
                "it_token": {
                    "token_id": 20,
                    "token_str": " Sure",
                    "token_category": "DISCOURSE",
                    "token_category_collapsed": "FORMAT",
                    "assistant_marker": True,
                },
            }
        },
        "readouts": {
            "first_diff": {
                "condition_token_at_step": {
                    "A_pt_raw": {"token_id": 10, "class": "pt"},
                    "C_it_chat": {"token_id": 20, "class": "it"},
                },
                "conditions": {
                    "A_pt_raw": {
                        "winner": "pt",
                        "layerwise": {"y_it_first_top20_layer": 4},
                        "windows": {"late_reconciliation": _summary_window(1.25)},
                    },
                    "C_it_chat": {
                        "winner": "it",
                        "layerwise": {"y_it_first_top20_layer": 2},
                        "windows": {"late_reconciliation": _summary_window(3.5)},
                    },
                },
            }
        },
    }
    out_dir = tmp_path / "gemma3_4b"
    out_dir.mkdir()
    (out_dir / "exp20_records.jsonl").write_text(json.dumps(record) + "\n")
    summary = analyze_run(tmp_path)
    assert summary["quality"]["ok"]
    assert summary["by_model"]["gemma3_4b"]["n_records"] == 1
    readout = summary["pooled"]["dense5"]["readouts"]["first_diff"]["by_condition"]["C_it_chat"]
    assert readout["winner"]["it"]["fraction"] == 1.0
    metric = readout["windows"]["late_reconciliation"]["it_minus_pt_margin.total_delta"]
    assert metric["mean"] == 3.5
