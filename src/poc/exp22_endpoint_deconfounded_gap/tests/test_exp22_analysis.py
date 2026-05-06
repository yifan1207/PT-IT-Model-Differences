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
from scripts.analysis.analyze_exp22_fixed_history_template_audit import (
    cem_effects as fixed_cem_effects,
    flatten_records as flatten_fixed_records,
    paired_effects as fixed_paired_effects,
)
from src.poc.exp22_endpoint_deconfounded_gap.fixed_history import _cell_prompt, _prompt_for_regime, _target_stats


class _FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs) -> str:
        return f"<chat>{messages[0]['content']}</chat><assistant>"


def test_prompt_regime_metadata_for_template_ablation() -> None:
    record = {"formats": {"B": "What is 2+2?\nAnswer:"}}
    native_prompt, native_mode = _prompt_for_regime(
        record,
        variant="it",
        tokenizer=_FakeTokenizer(),
        prompt_regime="native",
    )
    raw_it_prompt, raw_it_mode = _prompt_for_regime(
        record,
        variant="it",
        tokenizer=_FakeTokenizer(),
        prompt_regime="raw",
    )
    raw_pt_prompt, raw_pt_mode = _prompt_for_regime(
        record,
        variant="pt",
        tokenizer=_FakeTokenizer(),
        prompt_regime="native",
    )

    assert native_prompt.startswith("<chat>")
    assert native_mode == "native_chat"
    assert raw_it_prompt == "What is 2+2?\nAnswer:"
    assert raw_it_mode == "raw_no_template"
    assert raw_pt_prompt == raw_it_prompt
    assert raw_pt_mode == "raw"


def test_fixed_history_cell_prompt_metadata() -> None:
    record = {"formats": {"B": "What is 2+2?\nAnswer:"}}
    native_prompt, native_mode = _cell_prompt(record, cell="it_native", tokenizer=_FakeTokenizer())
    raw_prompt, raw_mode = _cell_prompt(record, cell="it_raw", tokenizer=_FakeTokenizer())
    pt_prompt, pt_mode = _cell_prompt(record, cell="pt_raw", tokenizer=_FakeTokenizer())

    assert native_prompt.startswith("<chat>")
    assert native_mode == "native_chat"
    assert raw_prompt == "What is 2+2?\nAnswer:"
    assert raw_mode == "raw_no_template"
    assert pt_prompt == raw_prompt
    assert pt_mode == "raw"


def test_fixed_history_target_stats_preserve_forced_token() -> None:
    import torch

    logits = torch.tensor([0.0, 5.0, 1.0])
    stats = _target_stats(logits, token_id=2)
    assert stats["final_top1_id"] == 1
    assert stats["target_rank"] == 2
    assert stats["top1_matches_forced"] is False
    assert stats["target_logprob"] is not None


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


def _mock_fixed_probe_payload(offset: float, n_steps: int = 7, n_layers: int = 10) -> dict:
    payload = _mock_probe_payload(n_steps=n_steps, n_layers=n_layers)
    payload["kl_to_final"] = [
        [float(value + offset) for value in row]
        for row in payload["kl_to_final"]
    ]
    return payload


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


def test_fixed_history_flatten_and_effects_on_mocked_jsonl(tmp_path: Path) -> None:
    forced_ids = list(range(7))
    for model in ["gemma3_4b", "qwen3_4b"]:
        for cell, offset in [("pt_raw", 0.0), ("it_native", 1.0), ("it_raw", 0.25)]:
            out_dir = tmp_path / model / cell
            out_dir.mkdir(parents=True)
            path = out_dir / "records.jsonl.gz"
            rec = {
                "prompt_id": f"{model}-p0",
                "model": model,
                "variant": "pt" if cell == "pt_raw" else "it",
                "cell": cell,
                "teacher_source": "it_native",
                "prompt_mode": "raw" if cell != "it_native" else "native_chat",
                "n_layers": 10,
                "n_steps": 7,
                "forced_ids": forced_ids,
                "generated_ids": forced_ids,
                "teacher_generated_text": "mock",
                "target_logprob": [-1.0] * 7,
                "target_rank": [2] * 7,
                "top1_matches_forced": [False] * 7,
                "probes": {"raw": _mock_fixed_probe_payload(offset)},
            }
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(rec) + "\n")

    df, support, summary = flatten_fixed_records(tmp_path, skip_first_n=5)
    assert summary["branch_counts"]["gemma3_4b/it_native/pt_raw"]["valid"] == 1
    assert len(df) == 12
    assert len(support) == 12

    paired = fixed_paired_effects(
        df,
        comparison_name="native_fixed_effect",
        control_cell="pt_raw",
        treatment_cell="it_native",
        n_boot=20,
        seed=0,
        teacher_source="it_native",
    )
    late = [
        row
        for row in paired
        if row["method"] == "paired_same_prompt_step"
        and row["probe_family"] == "raw"
        and row["outcome"] == "late_kl_mean"
    ][0]
    assert math.isclose(late["estimate"], 1.0)

    cem_rows, _balance, matching = fixed_cem_effects(
        df,
        comparison_name="raw_fixed_effect",
        control_cell="pt_raw",
        treatment_cell="it_raw",
        n_bins=2,
        n_boot=0,
        seed=0,
        teacher_source="it_native",
    )
    late_cem = [
        row
        for row in cem_rows
        if row["comparison"] == "raw_fixed_effect"
        and row["probe_family"] == "raw"
        and row["outcome"] == "late_kl_mean"
    ][0]
    assert math.isclose(late_cem["estimate"], 0.25)
    assert matching["min_retained_fraction"] == 1.0


def test_fixed_history_teacher_sources_do_not_cross_pair(tmp_path: Path) -> None:
    forced_ids = list(range(7))
    source_offsets = {
        "it_native": {"pt_raw": 0.0, "it_native": 1.0, "it_raw": 0.25},
        "pt_raw": {"pt_raw": 0.0, "it_native": 2.0, "it_raw": 0.5},
    }
    model = "gemma3_4b"
    for cell in ["pt_raw", "it_native", "it_raw"]:
        out_dir = tmp_path / model / cell
        out_dir.mkdir(parents=True)
        path = out_dir / "records.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for teacher_source, offsets in source_offsets.items():
                rec = {
                    "prompt_id": "shared-p0",
                    "model": model,
                    "variant": "pt" if cell == "pt_raw" else "it",
                    "cell": cell,
                    "teacher_source": teacher_source,
                    "prompt_mode": "raw" if cell != "it_native" else "native_chat",
                    "n_layers": 10,
                    "n_steps": 7,
                    "forced_ids": forced_ids,
                    "generated_ids": forced_ids,
                    "teacher_generated_text": "mock",
                    "target_logprob": [-1.0] * 7,
                    "target_rank": [2] * 7,
                    "top1_matches_forced": [False] * 7,
                    "probes": {"raw": _mock_fixed_probe_payload(offsets[cell])},
                }
                handle.write(json.dumps(rec) + "\n")

    df, _support, summary = flatten_fixed_records(tmp_path, skip_first_n=5)
    assert summary["branch_counts"]["gemma3_4b/it_native/pt_raw"]["valid"] == 1
    assert summary["branch_counts"]["gemma3_4b/pt_raw/pt_raw"]["valid"] == 1
    paired_it = fixed_paired_effects(
        df,
        comparison_name="native_fixed_effect",
        control_cell="pt_raw",
        treatment_cell="it_native",
        n_boot=0,
        seed=0,
        teacher_source="it_native",
    )
    paired_pt = fixed_paired_effects(
        df,
        comparison_name="native_fixed_effect",
        control_cell="pt_raw",
        treatment_cell="it_native",
        n_boot=0,
        seed=0,
        teacher_source="pt_raw",
    )
    late_it = [
        row
        for row in paired_it
        if row["probe_family"] == "raw" and row["outcome"] == "late_kl_mean"
    ][0]
    late_pt = [
        row
        for row in paired_pt
        if row["probe_family"] == "raw" and row["outcome"] == "late_kl_mean"
    ][0]
    assert math.isclose(late_it["estimate"], 1.0)
    assert math.isclose(late_pt["estimate"], 2.0)
