from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.build_exp24_32b_external_validity_synthesis import build
from src.poc.cross_model.adapters import get_adapter
from src.poc.cross_model.config import get_spec
from src.poc.exp11_matched_prefix_mlp_graft.run import _causal_windows_for_model
from src.poc.exp20_divergence_token_counterfactual.collect import DEPTH_ABLATION_WINDOWS


def test_32b_specs_adapters_and_windows() -> None:
    for model in ["qwen25_32b", "olmo2_32b"]:
        spec = get_spec(model)
        assert spec.n_layers == 64
        assert spec.d_model == 5120
        assert spec.n_heads == 40
        assert spec.n_kv_heads == 8
        assert spec.multi_gpu is True
        assert len(spec.global_attn_layers) == 64
        assert get_adapter(model) is not None
        assert DEPTH_ABLATION_WINDOWS[model] == {
            "early": (0, 26),
            "mid": (19, 45),
            "late": (38, 64),
        }
        causal = _causal_windows_for_model(model, spec, include_midlate=True)
        assert causal["B_midlate_raw"] == (19, 64)
        assert causal["D_midlate_ptswap"] == (19, 64)


def test_exp24_synthesis_writes_expected_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    analysis = run_root / "analysis"
    (analysis / "exp23_midlate_interaction_suite").mkdir(parents=True)
    (analysis / "exp20_factorial_validation").mkdir(parents=True)
    (analysis / "exp21_productive_opposition").mkdir(parents=True)
    (analysis / "part_a_mlp_kl").mkdir(parents=True)

    models = ["qwen25_32b", "olmo2_32b"]
    (analysis / "exp23_midlate_interaction_suite" / "exp23_summary.json").write_text(
        json.dumps(
            {
                "residual_factorial": {
                    "n_units_by_model": {"qwen25_32b": 10, "olmo2_32b": 12},
                    "effects": {
                        "common_it": {
                            "late_it_given_pt_upstream": {
                                "estimate": 0.5,
                                "ci95_low": 0.4,
                                "ci95_high": 0.6,
                                "models": {"qwen25_32b": 0.4, "olmo2_32b": 0.6},
                            },
                            "late_it_given_it_upstream": {
                                "estimate": 2.0,
                                "ci95_low": 1.8,
                                "ci95_high": 2.2,
                                "models": {"qwen25_32b": 1.5, "olmo2_32b": 2.5},
                            },
                            "interaction": {
                                "estimate": 1.5,
                                "ci95_low": 1.2,
                                "ci95_high": 1.8,
                                "models": {"qwen25_32b": 1.1, "olmo2_32b": 1.9},
                            },
                        }
                    },
                }
            }
        )
    )
    (analysis / "exp20_factorial_validation" / "summary.json").write_text(
        json.dumps(
            {
                "by_model": {
                    f"raw_shared/{model}": {
                        "conditions": {
                            "B_mid_raw": {"class_fractions": {"it": 0.3}},
                            "B_late_raw": {"class_fractions": {"it": 0.2}},
                        }
                    }
                    for model in models
                }
            }
        )
    )
    (analysis / "exp21_productive_opposition" / "summary.json").write_text(
        json.dumps(
            {
                "by_model": {
                    f"native/{model}": {
                        "conditions": {
                            "first_diff": {
                                "C_it_chat": {
                                    "windows": {
                                        "late_reconciliation": {"support_it_token": 0.7}
                                    }
                                }
                            }
                        }
                    }
                    for model in models
                }
            }
        )
    )
    (analysis / "part_a_mlp_kl" / "exp23_midlate_kl_factorial_summary.json").write_text(
        json.dumps(
            {
                "effects": [
                    {"model": "qwen25_32b", "effect": "E_late_pt", "mean": 0.11},
                    {"model": "olmo2_32b", "effect": "E_late_pt", "mean": 0.22},
                ]
            }
        )
    )

    out_dir = tmp_path / "synthesis"
    payload = build(run_root, out_dir, models)
    assert (out_dir / "exp24_32b_summary.csv").exists()
    assert (out_dir / "exp24_32b_summary.md").exists()
    assert (out_dir / "exp24_32b_interaction.png").exists()
    assert (out_dir / "exp24_32b_claims.json").exists()
    assert payload["claims"]["all_model_interactions_positive"] is True
    assert payload["claims"]["pooled_interaction_ci_excludes_zero"] is True
