from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.analyze_exp22_endpoint_deconfounded_gap import (
    balance_table,
    bootstrap_equal_model_effect,
    compute_cem_weights,
    dense_equal_model_effects,
    flatten_records,
)


def _toy_df() -> pd.DataFrame:
    rows = []
    for model in ["gemma3_4b", "qwen3_4b"]:
        for probe in ["raw", "tuned"]:
            for i in range(20):
                for variant in ["pt", "it"]:
                    rows.append(
                        {
                            "prompt_id": f"{model}-{probe}-{i}-{variant}",
                            "model": model,
                            "probe_family": probe,
                            "variant": variant,
                            "variant_it": int(variant == "it"),
                            "final_entropy": float(i % 10),
                            "final_confidence": float((i % 10) / 10.0),
                            "final_top1_margin": float((i % 10) / 5.0),
                            "late_kl_mean": 1.0 + int(variant == "it"),
                            "prefinal_late_kl_mean": 1.0 + int(variant == "it"),
                            "remaining_adj_js": 0.1 + 0.1 * int(variant == "it"),
                            "remaining_adj_kl": 0.2 + 0.1 * int(variant == "it"),
                            "future_top1_flips": 1.0 + int(variant == "it"),
                            "top5_churn": 0.2 + 0.1 * int(variant == "it"),
                            "final_top1_stable_top5_entry": 0.5,
                            "late_consensus_stable_top5_entry": 0.5,
                        }
                    )
    return pd.DataFrame(rows)


def test_cem_weights_balance_and_effect_direction() -> None:
    df, matching = compute_cem_weights(_toy_df(), n_bins=5)
    assert matching["min_retained_fraction"] == 1.0
    bal = balance_table(df)
    assert float(bal["smd_after"].abs().max()) < 1e-9
    effects = dense_equal_model_effects(df, method="cem", weight_col="cem_weight", n_boot=0, seed=0)
    late = [
        row
        for row in effects
        if row["probe_family"] == "raw" and row["outcome"] == "late_kl_mean"
    ][0]
    assert math.isclose(late["estimate_it_minus_pt"], 1.0)


def test_prompt_cluster_bootstrap_returns_finite_interval() -> None:
    df, _matching = compute_cem_weights(_toy_df(), n_bins=5)
    lo, hi = bootstrap_equal_model_effect(
        df[df["probe_family"] == "raw"],
        outcome="late_kl_mean",
        weight_col="cem_weight",
        n_boot=50,
        seed=0,
    )
    assert math.isfinite(lo)
    assert math.isfinite(hi)
    assert lo <= 1.0 <= hi


def _mock_probe_payload(n_steps: int = 7, n_layers: int = 10) -> dict:
    kl = [[0.3 - 0.02 * layer for layer in range(n_layers)] for _ in range(n_steps)]
    for row in kl:
        row[-1] = 0.0
    entropy = [[1.0 + 0.01 * layer for layer in range(n_layers)] for _ in range(n_steps)]
    confidence = [[0.2 + 0.01 * layer for layer in range(n_layers)] for _ in range(n_steps)]
    margin = [[0.1 + 0.01 * layer for layer in range(n_layers)] for _ in range(n_steps)]
    top1_ids = [[[layer % 3] for layer in range(n_layers)] for _ in range(n_steps)]
    top5_ids = [[[layer % 3, 10, 11, 12, 13] for layer in range(n_layers)] for _ in range(n_steps)]
    top5_logprobs = [[[-0.1, -0.2, -0.3, -0.4, -0.5] for _ in range(n_layers)] for _ in range(n_steps)]
    adjacent = [[0.01 for _ in range(n_layers - 1)] + [None] for _ in range(n_steps)]
    return {
        "kl_to_final": kl,
        "entropy": entropy,
        "confidence": confidence,
        "top1_margin": margin,
        "top1_ids": top1_ids,
        "top5_ids": top5_ids,
        "top5_logprobs": top5_logprobs,
        "adjacent_kl": adjacent,
        "adjacent_js": adjacent,
    }


def test_flatten_records_on_mocked_jsonl(tmp_path: Path) -> None:
    out_dir = tmp_path / "gemma3_4b" / "pt"
    out_dir.mkdir(parents=True)
    path = out_dir / "records.jsonl.gz"
    rec = {
        "prompt_id": "p0",
        "model": "gemma3_4b",
        "variant": "pt",
        "prompt_mode": "raw",
        "n_layers": 10,
        "n_steps": 7,
        "generated_ids": list(range(7)),
        "generated_text": "mock",
        "probes": {"raw": _mock_probe_payload()},
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(rec) + "\n")
    df, summary = flatten_records(tmp_path, skip_first_n=5)
    assert summary["branch_counts"]["gemma3_4b/pt"]["valid"] == 1
    assert len(df) == 2
    assert set(df["probe_family"]) == {"raw"}
