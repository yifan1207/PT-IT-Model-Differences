"""Main orchestrator for Exp6 steering experiments.

Usage (Approach A):
    python src/poc/exp6/run.py \\
        --experiment A1 --variant it --device cuda:0 \\
        --corrective-direction-path results/exp5/precompute_v2/precompute/corrective_directions.npz \\
        --worker-index 0 --n-workers 8

Usage (Approach B):
    python src/poc/exp6/run.py \\
        --experiment B1 --variant it --device cuda:0 \\
        --governance-features-path results/exp6/governance_feature_sets.json \\
        --mean-feature-acts-path results/exp6/precompute/mean_feature_acts_it \\
        --feature-set method12_top100 --gamma 5.0 \\
        --worker-index 0 --n-workers 8
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when run as a script (src/poc/exp6/run.py → ../../..)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import torch

from src.poc.collect import load_dataset_records
from src.poc.exp6.benchmarks.governance import evaluate_governance_benchmark
from src.poc.exp6.benchmarks.governance_v2 import score_format_compliance_v2, score_mmlu_forced_choice
from src.poc.exp6.config import Exp6Config
from src.poc.exp6.interventions import build_intervention
from src.poc.exp6.runtime import GeneratedSample6, generate_record_A, generate_record_B, generate_records_A_batch, generate_records_B_batch
from src.poc.exp5.benchmarks.custom import evaluate_custom_benchmark
from src.poc.exp5.utils import ensure_dir, sanitise_json, save_json
from src.poc.shared.model import load_model

_DISK_WARN_GB = 5.0
_DISK_STOP_GB = 2.0
# Active v2 governance benchmark (shared-generation pass, all records)
_GOV_BENCHMARKS = {"structural_token_ratio"}
# Legacy v1 benchmarks still dispatchable but not in any default config list
_LEGACY_GOV_BENCHMARKS = {"turn_structure", "format_compliance", "mmlu_accuracy", "coherent_assistant_rate"}
_MMLU_FC_BENCHMARK = "mmlu_forced_choice"        # special forced-choice generation
_EXP5_BENCHMARKS = {"factual_em": "exp3_factual_em", "reasoning_em": "exp3_reasoning_em",
                     "alignment_behavior": "exp3_alignment_behavior"}


def _free_gb() -> float:
    return shutil.disk_usage(".").free / 1e9


def _check_disk() -> None:
    free = _free_gb()
    if free < _DISK_STOP_GB:
        raise RuntimeError(f"[disk] only {free:.1f} GB free — stopping to prevent data loss.")
    if free < _DISK_WARN_GB:
        print(f"[disk] WARNING: {free:.1f} GB free", flush=True)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(sanitise_json(row), ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Condition generators ──────────────────────────────────────────────────────

def _A1_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """A1: Remove corrective direction from IT — α sweep + random-direction control.

    α sweep: 1.0=identity, 0=full removal, <0=amplification, >1=over-suppression.
    Random control: same perturbation magnitude but random unit vector at corrective
    layers — shows the governance effect is direction-specific, not just noise injection.
    """
    ALPHA_VALUES = [5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0, -2.0, -3.0, -5.0]
    CTRL_BETAS   = [0.0, 1.0, 3.0]   # injection magnitudes for direction-specificity controls
    CORR_LAYERS  = list(range(cfg.proposal_boundary, cfg.n_layers))   # layers 20-33

    specs = [("A1_baseline", replace(cfg, method="none", directional_alpha=1.0))]

    # Main α sweep: remove corrective direction at corrective layers
    for alpha in ALPHA_VALUES:
        specs.append((f"A1_alpha_{alpha:g}", replace(cfg,
            method="directional_remove",
            ablation_layers=CORR_LAYERS,
            directional_alpha=alpha,
        )))

    # Control: random unit vector injection — same layers, same magnitude, random direction
    # Expected: no governance effect → confirms the effect is direction-specific
    for beta in CTRL_BETAS:
        specs.append((f"A1_ctrl_random_b{beta:g}", replace(cfg,
            method="directional_random",
            ablation_layers=CORR_LAYERS,
            directional_beta=beta,
        )))

    return specs  # 1 + 14 + 3 = 18 conditions


def _A1_notmpl_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """A1_notmpl: Identical α-sweep as A1 but with apply_chat_template=False.

    Tests whether the governance dose-response is weight-intrinsic or template-gated.
    If the same dose-response appears without the chat template, the effect is driven
    by the model weights at corrective layers, not by template-induced processing modes.

    Expected:
    - Weight-intrinsic: similar dose-response to A1 → strong mechanistic claim
    - Template-gated:   flat metrics across α → framing adjustment needed
    """
    ALPHA_VALUES = [5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0, -2.0, -3.0, -5.0]
    CORR_LAYERS  = list(range(cfg.proposal_boundary, cfg.n_layers))   # layers 20-33
    specs = [("A1notmpl_baseline", replace(cfg, method="none", directional_alpha=1.0,
                                           apply_chat_template=False))]
    for alpha in ALPHA_VALUES:
        specs.append((f"A1notmpl_alpha_{alpha:g}", replace(cfg,
            method="directional_remove",
            ablation_layers=CORR_LAYERS,
            directional_alpha=alpha,
            apply_chat_template=False,
        )))
    return specs  # 1 + 14 = 15 conditions


def _A2_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """A2: Inject corrective direction into PT — β sweep + controls."""
    BETA_VALUES = [-5.0, -3.0, -2.0, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    CTRL_BETAS = [0.0, 1.0, 3.0]

    specs = [("A2_baseline_pt", replace(cfg, method="none", directional_beta=0.0))]

    # Main β sweep with corrective direction
    for beta in BETA_VALUES:
        name = f"A2_beta_{beta:g}"
        specs.append((name, replace(cfg,
            method="directional_add",
            ablation_layers=list(range(cfg.proposal_boundary, cfg.n_layers)),
            directional_beta=beta,
        )))

    # Control: random unit vector
    for beta in CTRL_BETAS:
        name = f"A2_ctrl_random_b{beta:g}"
        specs.append((name, replace(cfg,
            method="directional_random",
            ablation_layers=list(range(cfg.proposal_boundary, cfg.n_layers)),
            directional_beta=beta,
        )))

    # Control: rotated direction (orthogonal to corrective)
    for beta in CTRL_BETAS:
        name = f"A2_ctrl_rotated_b{beta:g}"
        specs.append((name, replace(cfg,
            method="directional_rotated",
            ablation_layers=list(range(cfg.proposal_boundary, cfg.n_layers)),
            directional_beta=beta,
        )))

    # Control: content-layer direction injected at corrective layers
    for beta in CTRL_BETAS:
        name = f"A2_ctrl_content_b{beta:g}"
        specs.append((name, replace(cfg,
            method="content_direction",
            ablation_layers=list(range(cfg.proposal_boundary, cfg.n_layers)),
            directional_beta=beta,
        )))

    return specs  # 1 + 15 + 9 = 25 conditions


def _B1_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """B1: Feature clamping — γ sweep × feature set sweep."""
    GAMMA_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    # Method 1+2 combined feature sets
    M12_SETS = ["method12_top10", "method12_top50", "method12_top100", "method12_top500", "method12_all"]
    # Method 3: IT-amplified top-100
    M3_SETS = ["method3_it_amplified_top100"]

    specs = [("B1_baseline", replace(cfg, method="none", gamma=1.0))]
    for fset in M12_SETS + M3_SETS:
        for gamma in GAMMA_VALUES:
            name = f"B1_{fset}_g{gamma:g}"
            specs.append((name, replace(cfg, method="feature_clamp", gamma=gamma, feature_set=fset)))
    return specs


def _B2_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """B2: W_dec governance direction injection."""
    BETA_VALUES = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    specs = [("B2_baseline", replace(cfg, method="none"))]
    for fset in ["method12_top100", "method3_it_amplified_top100"]:
        for beta in BETA_VALUES:
            # B2a: inject into PT
            specs.append((f"B2a_{fset}_b{beta:g}", replace(cfg,
                model_variant="pt",
                method="wdec_inject",
                directional_beta=beta,
                feature_set=fset,
            )))
        for beta in BETA_VALUES:
            # B2b: subtract from IT
            specs.append((f"B2b_{fset}_b{-beta:g}", replace(cfg,
                model_variant="it",
                method="wdec_inject",
                directional_beta=-beta,
                feature_set=fset,
            )))
    return specs


def _B3_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """B3: Control feature sets."""
    GAMMA_VALUES = [0.0, 1.0, 5.0]
    CTRL_SETS = ["random_100", "method12_content_top100", "method3_it_suppressed_top100"]
    specs = []
    for fset in CTRL_SETS:
        for gamma in GAMMA_VALUES:
            specs.append((f"B3_{fset}_g{gamma:g}", replace(cfg,
                method="feature_clamp", gamma=gamma, feature_set=fset,
            )))
    return specs


def _B4_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """B4: Layer specificity — early/mid/late corrective sub-ranges."""
    GAMMA_VALUES = [0.0, 5.0]
    specs = []
    for layer_range in ["early_20_25", "mid_26_29", "late_30_33"]:
        for gamma in GAMMA_VALUES:
            specs.append((f"B4_{layer_range}_g{gamma:g}", replace(cfg,
                method="feature_clamp", gamma=gamma,
                feature_set="method12_top100",
                feature_layer_range=layer_range,
            )))
    return specs


def _B5_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """B5: Full γ sweep × layer-range — controlled layer-specificity study.

    Extends B4 from 2 γ values to a full sweep including negative γ (suppression below
    mean) so we can characterise the dose-response separately for each sub-range of the
    corrective stage, matching the A1_early/A1_mid ablation design on the B side.

    Layer ranges:
      all        — layers 20-33 (same as B1, reference)
      early_20_25 — first 6 corrective layers
      mid_26_29   — middle 4 corrective layers
      late_30_33  — final 4 corrective layers

    Gamma sweep includes negative values (γ < 0 → feature activation set below zero,
    effectively suppressing; γ = 0 → full suppression to 0; γ = 1 → neutral/mean).
    """
    GAMMA_VALUES = [-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    LAYER_RANGES = ["all", "early_20_25", "mid_26_29", "late_30_33"]

    specs = [("B5_baseline", replace(cfg, method="none", gamma=1.0))]
    for layer_range in LAYER_RANGES:
        for gamma in GAMMA_VALUES:
            specs.append((f"B5_{layer_range}_g{gamma:g}", replace(cfg,
                method="feature_clamp", gamma=gamma,
                feature_set="method12_top100",
                feature_layer_range=layer_range,
            )))
    return specs  # 1 + 4 × 10 = 41 conditions


def _A1_early_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """A1_early: Remove corrective direction from IT at early layers (1–11) — α sweep.

    Layer-specificity ablation: same direction and α sweep as A1, but applied to
    content/encoding layers instead of the corrective stage (20–33).
    Expected: little or no effect on governance metrics → confirms the corrective stage
    is the mechanistic locus of structural steering.
    """
    ALPHA_VALUES = [5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0, -2.0, -3.0, -5.0]
    EARLY_LAYERS = list(range(1, 12))  # layers 1–11
    specs = [("A1early_baseline", replace(cfg, method="none", directional_alpha=1.0))]
    for alpha in ALPHA_VALUES:
        specs.append((f"A1early_alpha_{alpha:g}", replace(cfg,
            method="directional_remove",
            ablation_layers=EARLY_LAYERS,
            directional_alpha=alpha,
        )))
    return specs  # 15 conditions


def _A1_mid_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """A1_mid: Remove corrective direction from IT at mid layers (12–19) — α sweep.

    Layer-specificity ablation: same as A1_early but applied to the pre-corrective
    middle layers.  Together with A1_early and A1 (20–33), sweeps the full depth of
    the network to isolate where the direction has causal governance leverage.
    """
    ALPHA_VALUES = [5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0, -2.0, -3.0, -5.0]
    MID_LAYERS = list(range(12, 20))   # layers 12–19
    specs = [("A1mid_baseline", replace(cfg, method="none", directional_alpha=1.0))]
    for alpha in ALPHA_VALUES:
        specs.append((f"A1mid_alpha_{alpha:g}", replace(cfg,
            method="directional_remove",
            ablation_layers=MID_LAYERS,
            directional_alpha=alpha,
        )))
    return specs  # 15 conditions


def _A5a_early_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """A5a_early: Progressive skip at early layers (1–11).

    Layer-specificity control for A5a: same progressive-skip method but applied
    to the early encoding range (layers 1–11) instead of the corrective range (20–33).
    Skip starts from end of range (layer 11) and progressively includes more layers
    going backward to layer 1.

    Expected: content/reasoning metrics degrade as more early layers are skipped,
    but governance metrics should NOT show the format-before-coherence dissociation
    seen in A5a — confirming corrective-layer specificity of governance steering.

    11 conditions + 1 baseline = 12 total.
    """
    specs = [("A5aearly_baseline", replace(cfg, method="none"))]
    for start in range(11, 0, -1):   # 11, 10, ..., 1
        skip_layers = list(range(start, 12))   # [start..11]
        specs.append((f"A5aearly_skip_from_{start}", replace(cfg,
            method="progressive_skip",
            ablation_layers=skip_layers,
        )))
    return specs  # 1 + 11 = 12 conditions


def _A5a_mid_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """A5a_mid: Progressive skip at mid layers (12–19).

    Layer-specificity control for A5a: same progressive-skip method applied to
    the mid pre-corrective range (layers 12–19).
    Skip starts from end of range (layer 19) and progressively includes more layers
    going backward to layer 12.

    Expected: similar to A5a_early — degradation in content but not the selective
    governance drop seen in corrective-layer skipping (A5a).

    8 conditions + 1 baseline = 9 total.
    """
    specs = [("A5amid_baseline", replace(cfg, method="none"))]
    for start in range(19, 11, -1):   # 19, 18, ..., 12
        skip_layers = list(range(start, 20))   # [start..19]
        specs.append((f"A5amid_skip_from_{start}", replace(cfg,
            method="progressive_skip",
            ablation_layers=skip_layers,
        )))
    return specs  # 1 + 8 = 9 conditions


def _A5a_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """Exp5A rerun (progressive skip): zero MLP+attention at layers [start..33].

    8 conditions: skip_from_33, skip_from_32, ..., skip_from_20.
    No direction vectors used — raw layer removal identical to Exp5's 'skip' method.
    Rerun with eval_dataset_v2 + v2 benchmark suite for comparable numbers.
    Applied to IT model — measures governance degradation as more corrective layers removed.
    """
    specs = [("A5a_baseline", replace(cfg, method="none"))]
    for start in range(33, 19, -1):   # 33, 32, ..., 20  (8 conditions)
        skip_layers = list(range(start, cfg.n_layers))
        specs.append((f"A5a_skip_from_{start}", replace(cfg,
            method="progressive_skip",
            ablation_layers=skip_layers,
        )))
    return specs  # 1 + 8 = 9 conditions


def _A5a_notmpl_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """A5a_notmpl: Progressive skip at corrective layers (20-33), apply_chat_template=False.

    Paired with A5a to test whether corrective-layer specialization persists without
    the chat template. If layers 31-33 still show the format-before-coherence
    dissociation without template, that is strong evidence for weight-intrinsic
    layer specialization independent of template-induced processing modes.
    """
    specs = [("A5anotmpl_baseline", replace(cfg, method="none", apply_chat_template=False))]
    for start in range(33, 19, -1):   # 33, 32, ..., 20  (14 conditions)
        skip_layers = list(range(start, cfg.n_layers))
        specs.append((f"A5anotmpl_skip_from_{start}", replace(cfg,
            method="progressive_skip",
            ablation_layers=skip_layers,
            apply_chat_template=False,
        )))
    return specs  # 1 + 14 = 15 conditions


def _A5b_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """Exp5B rerun (α-sweep): remove corrective direction at layers 20-33, α ∈ {-1..2}.

    7 conditions matching the original Exp5B α values.
    Rerun with v2 precomputed direction + eval_dataset_v2 + v2 benchmarks.
    Applied to IT model — bridges Exp5B and the denser A1 sweep.
    """
    ALPHA_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    CORR_LAYERS  = list(range(cfg.proposal_boundary, cfg.n_layers))   # layers 20-33
    specs = [("A5b_baseline", replace(cfg, method="none", directional_alpha=1.0))]
    for alpha in ALPHA_VALUES:
        specs.append((f"A5b_alpha_{alpha:g}", replace(cfg,
            method="directional_remove",
            ablation_layers=CORR_LAYERS,
            directional_alpha=alpha,
        )))
    return specs  # 1 + 7 = 8 conditions


def _A1_rand_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """A1_rand: Identical α-sweep as A1 but with random unit vectors at corrective layers.

    Specificity control for reviewer R4: if the dose-response in A1 reflects the
    *content* of the IT-PT corrective direction (not just any perturbation at layers
    20-33), random directions should produce flat metrics across all α values.

    Direction source: results/exp6/precompute/random_directions.npz  (seed=42, fixed).
    Pass this file via --corrective-direction-path when launching.

    Expected:  flat across α  →  governance effect is direction-specific ✓
    Concern:   dose-response present  →  any perturbation at these layers causes effect
               (weakens mechanistic claim, requires reframing)
    """
    ALPHA_VALUES = [5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0, -2.0, -3.0, -5.0]
    CORR_LAYERS  = list(range(cfg.proposal_boundary, cfg.n_layers))   # layers 20-33

    specs = [("A1rand_baseline", replace(cfg, method="none", directional_alpha=1.0))]
    for alpha in ALPHA_VALUES:
        specs.append((f"A1rand_alpha_{alpha:g}", replace(cfg,
            method="directional_remove",
            ablation_layers=CORR_LAYERS,
            directional_alpha=alpha,
        )))
    return specs  # 1 + 14 = 15 conditions


def _A1_rand_matched_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """A1_rand_matched: Projection-magnitude-matched random direction control (Exp7 0C).

    Addresses the projection-magnitude confound in A1_rand: a random unit vector in d=2560
    has expected |⟨h, d̂_rand⟩| ≈ ‖h‖/√d ≈ 50× smaller than the corrective direction.
    This function rescales the random projection to match the corrective projection
    magnitude per token, giving a fair like-for-like comparison.

    Uses method="directional_random_matched" which calls _project_remove_magnitude_matched()
    in interventions.py. Same α-sweep as A1. Pass the canonical corrective_directions.npz
    as --corrective-direction-path (used to compute magnitude for scaling).

    Expected: flat metrics across all α → confirms governance effect is direction-specific,
    not just a large-perturbation artifact.
    """
    ALPHA_VALUES = [5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0, -2.0, -3.0, -5.0]
    CORR_LAYERS  = list(range(cfg.proposal_boundary, cfg.n_layers))   # layers 20-33

    specs = [("A1randmatched_baseline", replace(cfg, method="none", directional_alpha=1.0))]
    for alpha in ALPHA_VALUES:
        specs.append((f"A1randmatched_alpha_{alpha:g}", replace(cfg,
            method="directional_random_matched",
            ablation_layers=CORR_LAYERS,
            directional_alpha=alpha,
        )))
    return specs  # 1 + 14 = 15 conditions


def _A1_rand_matched_multiseed_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """A1_rand_matched_multiseed: Multi-seed random direction control (Exp7 0C robustness).

    Runs the magnitude-matched random control with 5 different random seeds at a reduced
    alpha sweep (alpha=0 is the strongest test). Reports mean +/- std across seeds.
    """
    SEEDS = [42, 137, 271, 503, 719]
    ALPHA_VALUES = [2.0, 1.0, 0.5, 0.0, -1.0]  # reduced sweep
    CORR_LAYERS = list(range(cfg.proposal_boundary, cfg.n_layers))

    specs = [("A1randms_baseline", replace(cfg, method="none", directional_alpha=1.0))]
    for seed in SEEDS:
        for alpha in ALPHA_VALUES:
            specs.append((f"A1randms_s{seed}_alpha_{alpha:g}", replace(cfg,
                method="directional_random_matched",
                ablation_layers=CORR_LAYERS,
                directional_alpha=alpha,
                random_direction_seed=seed,
            )))
    return specs  # 1 + 5*5 = 26 conditions


def _A1_formula_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """A1_formula: Intervention formula sensitivity analysis (Exp7 0I).

    Tests 4 intervention formula variants at the same α values as the canonical A1
    sweep to confirm the governance-content dissociation is robust to the choice of
    hook point and formula:
      1. mlp_proj_remove      — current canonical (directional_remove on MLP output)
      2. mlp_additive         — additive injection (directional_add, already exists)
      3. residual_proj_remove — projection-removal on full residual stream
      4. attn_proj_remove     — projection-removal on self_attn output only

    Full α sweep matching A1 (14 values including negatives). 4 × 15 = 60 conditions.
    """
    ALPHAS = [5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0, -2.0, -3.0, -5.0]
    CORR_LAYERS = list(range(cfg.proposal_boundary, cfg.n_layers))  # layers 20-33

    METHODS: list[tuple[str, str]] = [
        ("mlp_proj_remove",      "directional_remove"),           # canonical baseline
        ("mlp_additive",         "directional_add"),               # additive (already exists)
        ("residual_proj_remove", "directional_remove_residual"),   # full residual stream
        ("attn_proj_remove",     "directional_remove_attn"),       # self_attn output only
    ]

    specs = []
    for method_name, method_type in METHODS:
        specs.append((f"A1formula_{method_name}_baseline", replace(cfg, method="none")))
        for alpha in ALPHAS:
            specs.append((f"A1formula_{method_name}_alpha_{alpha:g}", replace(cfg,
                method=method_type,
                ablation_layers=CORR_LAYERS,
                directional_alpha=alpha,
            )))
    return specs  # 4 × 15 = 60 conditions


def _A1_single_layer_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """A1_single_layer: Per-layer importance sweep (Exp7 0F).

    Tests each corrective layer individually at alpha=0 (full removal) to identify
    which specific layers carry the most governance signal. This connects to 0J
    (tuned-lens commitment onset): the most important single layers should cluster
    near the commitment onset boundary.

    Produces 1 baseline + 14 single-layer conditions (layers 20-33).
    """
    CORR_LAYERS = list(range(cfg.proposal_boundary, cfg.n_layers))  # layers 20-33

    specs = [("A1single_baseline", replace(cfg, method="none", directional_alpha=1.0))]
    for layer in CORR_LAYERS:
        specs.append((f"A1single_layer_{layer}", replace(cfg,
            method="directional_remove",
            ablation_layers=[layer],
            directional_alpha=0.0,  # full removal
        )))
    return specs  # 1 + 14 = 15 conditions


def _condition_specs(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    match cfg.experiment:
        case "A1": return _A1_conditions(cfg)
        case "A1_notmpl": return _A1_notmpl_conditions(cfg)
        case "A1_early": return _A1_early_conditions(cfg)
        case "A1_mid": return _A1_mid_conditions(cfg)
        case "A1_rand": return _A1_rand_conditions(cfg)
        case "A1_rand_matched": return _A1_rand_matched_conditions(cfg)
        case "A1_rand_matched_multiseed": return _A1_rand_matched_multiseed_conditions(cfg)
        case "A1_formula": return _A1_formula_conditions(cfg)
        case "A1_single_layer": return _A1_single_layer_conditions(cfg)
        case "A2": return _A2_conditions(cfg)
        case "A5a": return _A5a_conditions(cfg)
        case "A5a_early": return _A5a_early_conditions(cfg)
        case "A5a_mid": return _A5a_mid_conditions(cfg)
        case "A5a_notmpl": return _A5a_notmpl_conditions(cfg)
        case "A5b": return _A5b_conditions(cfg)
        case "B1": return _B1_conditions(cfg)
        case "B2": return _B2_conditions(cfg)
        case "B3": return _B3_conditions(cfg)
        case "B4": return _B4_conditions(cfg)
        case "B5": return _B5_conditions(cfg)
        case _: raise ValueError(f"Unknown experiment: {cfg.experiment!r}")


# ── Record selection ──────────────────────────────────────────────────────────

def _select_records(records: list[dict], benchmark: str) -> list[dict]:
    """Filter records to those relevant for the given benchmark."""
    if benchmark in ("factual_em", "mmlu_accuracy", "mmlu_forced_choice"):
        return [r for r in records if r.get("category") == "CONTENT-FACT"]
    if benchmark in ("reasoning_em",):
        return [r for r in records if r.get("category") == "CONTENT-REASON"]
    if benchmark == "alignment_behavior":
        return [r for r in records if r.get("category") == "SAFETY"]
    if benchmark in ("format_compliance", "format_compliance_v2"):
        return [r for r in records if r.get("category") == "GOV-FORMAT"]
    # structural_token_ratio, turn_structure, coherent_assistant_rate: use all records
    return records


# ── Scoring dispatch ──────────────────────────────────────────────────────────

def _score_benchmark(
    benchmark: str,
    records: list[dict],
    outputs: list[GeneratedSample6],
) -> Any:
    # v2 governance scorers
    if benchmark == "format_compliance_v2":
        return score_format_compliance_v2(records, outputs)
    if benchmark == "mmlu_forced_choice":
        return score_mmlu_forced_choice(records, outputs)
    # v1/v2 governance scorers dispatched through evaluate_governance_benchmark
    if benchmark in _GOV_BENCHMARKS or benchmark in _LEGACY_GOV_BENCHMARKS:
        return evaluate_governance_benchmark(benchmark, records, outputs)
    # Map exp6 benchmark names to exp5 names
    exp5_name = _EXP5_BENCHMARKS.get(benchmark, benchmark)
    # exp5 scorers expect GeneratedSample objects — adapt GeneratedSample6
    from src.poc.exp5.runtime import GeneratedSample as GS5
    adapted = [GS5(
        record_id=o.record_id, prompt=o.prompt,
        generated_text=o.generated_text, generated_tokens=o.generated_tokens,
        hidden_states={}, logit_lens_entropy=[], top1_token_per_layer=[],
    ) for o in outputs]
    return evaluate_custom_benchmark(exp5_name, records, adapted)


# ── Per-condition runner ──────────────────────────────────────────────────────

def _run_condition_A(
    name: str,
    cfg: Exp6Config,
    records: list[dict],
    loaded: Any,
    scores_jsonl: Path,
    samples_jsonl: Path,
    done_benchmarks: set[str],
    adapter: Any = None,
) -> list[dict]:
    """Run one A-experiment condition (nnsight trace).

    Benchmarks that use all records (structural_token_ratio) share a single
    generation pass to avoid redundant inference.
    mmlu_forced_choice uses a separate short generation with format C and max_new_tokens=3.
    """
    intervention = build_intervention(cfg)
    score_rows: list[dict] = []

    # Pre-generate outputs for all-records benchmarks once
    _ALL_RECORDS_BENCHMARKS = _GOV_BENCHMARKS | _LEGACY_GOV_BENCHMARKS  # all use full-records shared generation
    all_records_benchmarks_todo = [b for b in cfg.benchmarks
                                   if b in _ALL_RECORDS_BENCHMARKS and b not in done_benchmarks]
    cached_all_outputs: list[GeneratedSample6] | None = None

    BATCH_SIZE = 8

    if all_records_benchmarks_todo:
        print(f"[exp6] {name}: generating all-records outputs (batch={BATCH_SIZE}, shared for {all_records_benchmarks_todo})", flush=True)
        cached_all_outputs = []
        for batch_start in range(0, len(records), BATCH_SIZE):
            batch = records[batch_start: batch_start + BATCH_SIZE]
            cached_all_outputs.extend(generate_records_A_batch(batch, loaded, cfg, intervention, adapter=adapter, batch_size=BATCH_SIZE))
            done_so_far = min(batch_start + BATCH_SIZE, len(records))
            if done_so_far % 100 < BATCH_SIZE or done_so_far == len(records):
                _check_disk()
                print(f"[exp6] {name}/all_records: {done_so_far}/{len(records)} done", flush=True)

        # Save logit-lens top-1 data if collected (Step 4: commitment delay analysis)
        if cfg.collect_logit_lens and cached_all_outputs and cached_all_outputs[0].logit_lens_top1 is not None:
            ll_path = samples_jsonl.parent / f"logit_lens_top1_{name}.npz"
            ll_payload = {}
            for o in cached_all_outputs:
                if o.logit_lens_top1 is not None:
                    ll_payload[o.record_id] = np.array(o.logit_lens_top1, dtype=np.int16)
            np.savez_compressed(ll_path, **ll_payload)
            print(f"[exp6] {name}: saved logit-lens top-1 → {ll_path} ({len(ll_payload)} records)", flush=True)

    for benchmark in cfg.benchmarks:
        if benchmark in done_benchmarks:
            print(f"[exp6] {name}/{benchmark}: already done, skipping", flush=True)
            continue

        if benchmark in _ALL_RECORDS_BENCHMARKS and cached_all_outputs is not None:
            bench_records = records
            outputs = cached_all_outputs
        elif benchmark == _MMLU_FC_BENCHMARK:
            # Forced-choice: short generation with format C
            bench_records = _select_records(records, benchmark)
            # Only v2 records with format C are valid
            bench_records = [r for r in bench_records if "C" in r.get("formats", {})]
            fc_cfg = replace(cfg, prompt_format="C", max_gen_tokens=3, apply_chat_template=False)
            outputs = []
            for batch_start in range(0, len(bench_records), BATCH_SIZE):
                batch = bench_records[batch_start: batch_start + BATCH_SIZE]
                outputs.extend(generate_records_A_batch(batch, loaded, fc_cfg, intervention, adapter=adapter, batch_size=BATCH_SIZE))
                done_so_far = min(batch_start + BATCH_SIZE, len(bench_records))
                if done_so_far % 100 < BATCH_SIZE or done_so_far == len(bench_records):
                    _check_disk()
                    print(f"[exp6] {name}/{benchmark}: {done_so_far}/{len(bench_records)} done", flush=True)
        else:
            bench_records = _select_records(records, benchmark)
            outputs = []
            for batch_start in range(0, len(bench_records), BATCH_SIZE):
                batch = bench_records[batch_start: batch_start + BATCH_SIZE]
                outputs.extend(generate_records_A_batch(batch, loaded, cfg, intervention, adapter=adapter, batch_size=BATCH_SIZE))
                done_so_far = min(batch_start + BATCH_SIZE, len(bench_records))
                if done_so_far % 100 < BATCH_SIZE or done_so_far == len(bench_records):
                    _check_disk()
                    print(f"[exp6] {name}/{benchmark}: {done_so_far}/{len(bench_records)} done", flush=True)

        result = _score_benchmark(benchmark, bench_records, outputs)
        row = {
            "condition": name,
            "benchmark": result.benchmark,
            "metric": result.metric,
            "value": result.value,
            "n": result.n,
            "experiment": cfg.experiment,
            "method": cfg.method,
            "alpha": cfg.directional_alpha,
            "beta": cfg.directional_beta,
            "layers": cfg.ablation_layers,
        }
        _append_jsonl(scores_jsonl, [row])
        _append_jsonl(samples_jsonl, [{
            "condition": name, "benchmark": benchmark,
            "record_id": o.record_id, "prompt": o.prompt,
            "generated_text": o.generated_text, "category": o.category,
        } for o in outputs])
        score_rows.append(row)
        print(f"[exp6] {name}/{benchmark}: value={result.value:.4f} n={result.n}", flush=True)

    return score_rows


def _get_raw_model(loaded: Any) -> Any:
    """Extract the underlying HuggingFace nn.Module from the nnsight/circuit-tracer wrapper."""
    # nnsight stores the wrapped nn.Module as ._model on both LanguageModel and ReplacementModel.
    # Verify the attribute exists to catch version changes early.
    if not hasattr(loaded.model, "_model"):
        raise AttributeError(
            "loaded.model has no ._model attribute — nnsight API may have changed. "
            "Inspect loaded.model.__dict__ and update _get_raw_model()."
        )
    return loaded.model._model


BATCH_SIZE_B = 8


def _run_condition_B(
    name: str,
    cfg: Exp6Config,
    records: list[dict],
    loaded: Any,
    scores_jsonl: Path,
    samples_jsonl: Path,
    done_benchmarks: set[str],
    hooks_config: dict,
) -> list[dict]:
    """Run one B-experiment condition (batched forward hooks, partial-save for crash resilience)."""
    model_raw = _get_raw_model(loaded)
    tokenizer = loaded.tokenizer
    real_token_mask = loaded.real_token_mask
    score_rows: list[dict] = []

    _ALL_RECORDS_BENCHMARKS = _GOV_BENCHMARKS | _LEGACY_GOV_BENCHMARKS  # all use full-records shared generation
    all_records_benchmarks_todo = [b for b in cfg.benchmarks
                                   if b in _ALL_RECORDS_BENCHMARKS and b not in done_benchmarks]
    cached_all_outputs: list[GeneratedSample6] | None = None

    if all_records_benchmarks_todo:
        # Partial-save path — resume across crashes
        partial_path = scores_jsonl.parent / f"_partial_{name}_all_records.jsonl"
        already_done: dict[str, GeneratedSample6] = {}
        if partial_path.exists():
            for line in partial_path.read_text().splitlines():
                try:
                    d = json.loads(line)
                    already_done[d["record_id"]] = GeneratedSample6(
                        record_id=d["record_id"], prompt=d["prompt"],
                        generated_text=d["generated_text"],
                        generated_tokens=d.get("generated_tokens", []),
                        category=d.get("category", ""),
                    )
                except Exception:
                    pass

        remaining = [r for r in records if r["id"] not in already_done]
        print(
            f"[exp6] {name}: generating all-records outputs (batch={BATCH_SIZE_B}, "
            f"shared for {all_records_benchmarks_todo}, "
            f"{len(already_done)} resumed, {len(remaining)} to go)",
            flush=True,
        )

        for batch_start in range(0, len(remaining), BATCH_SIZE_B):
            batch = remaining[batch_start: batch_start + BATCH_SIZE_B]
            new_outputs = generate_records_B_batch(
                batch, model_raw, tokenizer, real_token_mask, cfg, hooks_config,
                batch_size=BATCH_SIZE_B,
            )
            for o in new_outputs:
                already_done[o.record_id] = o
            # Append to partial file immediately
            with open(partial_path, "a") as pf:
                for o in new_outputs:
                    pf.write(json.dumps({
                        "record_id": o.record_id, "prompt": o.prompt,
                        "generated_text": o.generated_text,
                        "generated_tokens": o.generated_tokens,
                        "category": o.category,
                    }) + "\n")
            done_so_far = min(batch_start + BATCH_SIZE_B, len(remaining)) + len(records) - len(remaining)
            if done_so_far % 100 < BATCH_SIZE_B or done_so_far >= len(records):
                _check_disk()
                print(f"[exp6] {name}/all_records: {done_so_far}/{len(records)} done", flush=True)

        # Reconstruct in original record order
        cached_all_outputs = [already_done[r["id"]] for r in records if r["id"] in already_done]
        # Clean up partial file once fully done
        if partial_path.exists():
            partial_path.unlink()

    for benchmark in cfg.benchmarks:
        if benchmark in done_benchmarks:
            print(f"[exp6] {name}/{benchmark}: already done, skipping", flush=True)
            continue

        if benchmark in _ALL_RECORDS_BENCHMARKS and cached_all_outputs is not None:
            bench_records = records
            outputs = cached_all_outputs
        elif benchmark == _MMLU_FC_BENCHMARK:
            # Forced-choice: short generation with format C, no chat template
            bench_records = _select_records(records, benchmark)
            bench_records = [r for r in bench_records if "C" in r.get("formats", {})]
            fc_cfg = replace(cfg, prompt_format="C", max_gen_tokens=3, apply_chat_template=False)
            outputs = generate_records_B_batch(
                bench_records, model_raw, tokenizer, real_token_mask, fc_cfg, hooks_config,
                batch_size=BATCH_SIZE_B,
            )
            print(f"[exp6] {name}/{benchmark}: {len(outputs)}/{len(bench_records)} done", flush=True)
        else:
            bench_records = _select_records(records, benchmark)
            outputs = generate_records_B_batch(
                bench_records, model_raw, tokenizer, real_token_mask, cfg, hooks_config,
                batch_size=BATCH_SIZE_B,
            )
            print(f"[exp6] {name}/{benchmark}: {len(outputs)}/{len(bench_records)} done", flush=True)

        result = _score_benchmark(benchmark, bench_records, outputs)
        row = {
            "condition": name,
            "benchmark": result.benchmark,
            "metric": result.metric,
            "value": result.value,
            "n": result.n,
            "experiment": cfg.experiment,
            "method": cfg.method,
            "gamma": cfg.gamma,
            "feature_set": cfg.feature_set,
            "feature_layer_range": cfg.feature_layer_range,
        }
        _append_jsonl(scores_jsonl, [row])
        _append_jsonl(samples_jsonl, [{
            "condition": name, "benchmark": benchmark,
            "record_id": o.record_id, "prompt": o.prompt,
            "generated_text": o.generated_text, "category": o.category,
        } for o in outputs])
        score_rows.append(row)
        print(f"[exp6] {name}/{benchmark}: value={result.value:.4f} n={result.n}", flush=True)

    return score_rows


# ── B-experiment hooks config loader ─────────────────────────────────────────

def _load_B_hooks_config(cfg: Exp6Config, loaded: Any) -> dict:
    """Load all feature-steering artifacts into a hooks_config dict."""
    config: dict = {}

    # Determine needed methods from experiment type, not base cfg.method (which may be "none")
    _FEATURE_CLAMP_EXPERIMENTS = {"B1", "B3", "B4", "B5"}
    _WDEC_INJECT_EXPERIMENTS = {"B2"}
    needs_feature_clamp = cfg.experiment in _FEATURE_CLAMP_EXPERIMENTS
    needs_wdec = cfg.experiment in _WDEC_INJECT_EXPERIMENTS

    if needs_feature_clamp or cfg.method == "feature_clamp":
        # Transcoders: loaded.transcoder_list is list[transcoder] indexed by layer
        config["transcoders"] = loaded.transcoder_list

        # Governance feature sets
        feat_path = Path(cfg.governance_features_path)
        if not feat_path.exists():
            raise FileNotFoundError(f"Governance features not found: {feat_path}")
        with open(feat_path) as f:
            all_sets = json.load(f)
        # Build {layer_idx: [feature_indices]} from the chosen feature set key
        config["governance_features"] = {
            k: all_sets.get(k, {}).get(cfg.feature_set, [])
            for k in all_sets
        }

        # Mean feature activations ā[f]
        mean_acts_dir = Path(cfg.mean_feature_acts_path)
        mean_acts: dict[int, np.ndarray] = {}
        for l_idx in cfg.corrective_layers:
            npy_path = mean_acts_dir / f"layer_{l_idx}.npy"
            if npy_path.exists():
                mean_acts[l_idx] = np.load(str(npy_path))
        config["mean_acts"] = mean_acts

    if needs_wdec or cfg.method == "wdec_inject":
        import torch
        gov_dir_path = Path(cfg.governance_direction_path)
        if not gov_dir_path.exists():
            raise FileNotFoundError(f"Governance direction not found: {gov_dir_path}")
        gov_dirs: dict[int, torch.Tensor] = {}
        with np.load(str(gov_dir_path)) as d:
            for k in d.files:
                if k.startswith("layer_"):
                    l_idx = int(k.split("_", 1)[1])
                    gov_dirs[l_idx] = torch.tensor(d[k], dtype=torch.float32).to(cfg.device)
        config["governance_direction"] = gov_dirs

    return config


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Run exp6 steering experiments.")
    p.add_argument("--experiment", choices=["A1", "A1_notmpl", "A1_early", "A1_mid", "A1_rand", "A1_rand_matched", "A1_rand_matched_multiseed", "A1_formula", "A1_single_layer", "A2", "A5a", "A5a_early", "A5a_mid", "A5a_notmpl", "A5b", "B1", "B2", "B3", "B4", "B5"], required=True)
    p.add_argument("--variant", choices=["pt", "it"], default="it")
    p.add_argument("--model-name", default="",
                   help="Model family from cross_model registry (e.g. llama31_8b). "
                        "When set, derives model_id, n_layers, d_model, proposal_boundary "
                        "from MODEL_REGISTRY. Empty = legacy Gemma path.")
    p.add_argument("--no-chat-template", action="store_true", default=False,
                   help="Force apply_chat_template=False for all variants (raw format B input). "
                        "Required for Phase 0 multi-model consistency.")
    p.add_argument("--dataset", default="data/eval_dataset_v2.jsonl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-gen-tokens", type=int, default=200)
    p.add_argument("--n-eval-examples", type=int, default=1400)
    p.add_argument("--run-name", default="")
    p.add_argument("--output-base", default="results/exp6",
                   help="Base directory for run outputs (default: results/exp6). "
                        "Use results/exp7/0C etc. for exp7 experiments.")
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)

    # A-experiment paths
    p.add_argument("--corrective-direction-path", default="results/exp5/precompute_v2/precompute/corrective_directions.npz")
    p.add_argument("--content-direction-path", default="results/exp6/precompute/content_direction_aggregate.npz")

    # B-experiment paths
    p.add_argument("--governance-features-path", default="results/exp6/governance_feature_sets.json")
    p.add_argument("--mean-feature-acts-path", default="results/exp6/precompute/mean_feature_acts_it")
    p.add_argument("--governance-direction-path", default="results/exp6/precompute/governance_directions.npz")
    p.add_argument("--feature-set", default="method12_top100")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--feature-layer-range",
                   choices=["all", "early_20_25", "mid_26_29", "late_30_33"], default="all")
    p.add_argument("--collect-logit-lens", action="store_true", default=False,
                   help="Capture top-1 predicted token per decoder layer per generated step "
                        "(Step 4: commitment delay analysis). Saved to logit_lens_top1.npz. ~20%% overhead.")
    # Layer range overrides (used by Exp7 0F layer-range sensitivity)
    p.add_argument("--proposal-boundary", type=int, default=None,
                   help="Override cfg.proposal_boundary (default 20). Sets the first corrective layer "
                        "for A1/A2/A5a experiments. E.g. --proposal-boundary 18 for layers 18-33.")
    p.add_argument("--n-layers", type=int, default=None,
                   help="Override cfg.n_layers (default 34). Sets the last layer index+1. "
                        "E.g. --n-layers 32 for layers 20-31 (proposal_boundary=20).")
    p.add_argument("--eval-record-ids", default=None,
                   help="Path to JSON file with list of record IDs to evaluate on. "
                        "Only records whose 'record_id' is in this list will be used. "
                        "Used by 0H to restrict evaluation to held-out-800 records.")

    args = p.parse_args()

    worker_suffix = f"_w{args.worker_index}" if args.n_workers > 1 else ""
    run_name_base = args.run_name or f"{args.experiment}_{args.variant}"
    run_name = run_name_base + worker_suffix

    is_B = args.experiment.startswith("B")

    # Build config — model_family triggers MODEL_REGISTRY lookup for architecture params
    cfg_kwargs: dict = {}
    if args.model_name:
        cfg_kwargs["model_family"] = args.model_name
    if args.proposal_boundary is not None:
        cfg_kwargs["proposal_boundary"] = args.proposal_boundary
    if args.n_layers is not None:
        cfg_kwargs["n_layers"] = args.n_layers

    cfg = Exp6Config(
        experiment=args.experiment,
        model_variant=args.variant,
        dataset_path=args.dataset,
        device=args.device,
        max_gen_tokens=args.max_gen_tokens,
        n_eval_examples=args.n_eval_examples,
        run_name=run_name,
        output_base=args.output_base,
        corrective_direction_path=args.corrective_direction_path,
        content_direction_path=args.content_direction_path,
        governance_features_path=args.governance_features_path,
        mean_feature_acts_path=args.mean_feature_acts_path,
        governance_direction_path=args.governance_direction_path,
        feature_set=args.feature_set,
        gamma=args.gamma,
        directional_beta=args.beta,
        feature_layer_range=args.feature_layer_range,
        skip_transcoders=(not is_B),
        collect_logit_lens=args.collect_logit_lens,
        **cfg_kwargs,
    )

    # Force no-chat-template if requested (Phase 0 multi-model consistency)
    if args.no_chat_template:
        object.__setattr__(cfg, "apply_chat_template", False)

    # Build steering adapter for multi-model support
    steering_adapter = None
    if args.model_name:
        from src.poc.exp6.model_adapter import get_steering_adapter
        steering_adapter = get_steering_adapter(args.model_name)
        print(f"[exp6] model={args.model_name} n_layers={cfg.n_layers} d_model={cfg.d_model} "
              f"proposal_boundary={cfg.proposal_boundary} corrective_layers={cfg.corrective_layers[0]}-{cfg.corrective_layers[-1]}",
              flush=True)

    ensure_dir(cfg.run_dir)
    ensure_dir(cfg.plots_dir)

    # Augment run_config with reproducibility metadata
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"
    run_config_dict = {
        **cfg.to_dict(),
        "git_hash": git_hash,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "worker_index": args.worker_index,
        "n_workers": args.n_workers,
    }
    save_json(cfg.run_dir / "run_config.json", run_config_dict)

    scores_jsonl = cfg.run_dir / "scores.jsonl"
    samples_jsonl = cfg.run_dir / "sample_outputs.jsonl"

    all_scores = _read_jsonl(scores_jsonl)
    done_pairs: set[tuple[str, str]] = {(r["condition"], r["benchmark"]) for r in all_scores}
    if done_pairs:
        print(f"[exp6] resuming — {len(done_pairs)} (condition, benchmark) pairs already done", flush=True)

    _check_disk()
    records = load_dataset_records(cfg.dataset_path, prompt_format=cfg.prompt_format)

    # Filter to specific record IDs if provided (used by 0H held-out evaluation)
    if args.eval_record_ids:
        allowed_ids = set(json.loads(Path(args.eval_record_ids).read_text()))
        records = [r for r in records if r.get("record_id", r.get("id")) in allowed_ids]
        print(f"[exp6] filtered to {len(records)} records from {args.eval_record_ids}", flush=True)

    if cfg.n_eval_examples and len(records) > cfg.n_eval_examples:
        records = records[:cfg.n_eval_examples]
    loaded = load_model(cfg)

    # Override real_token_mask with adapter's generalized version when multi-model
    if steering_adapter is not None:
        device = torch.device(cfg.device)
        loaded.real_token_mask = steering_adapter.real_token_mask(loaded.tokenizer, device)

    condition_specs = _condition_specs(cfg)

    if args.n_workers > 1:
        condition_specs = [
            (name, cond_cfg)
            for i, (name, cond_cfg) in enumerate(condition_specs)
            if i % args.n_workers == args.worker_index
        ]
        print(
            f"[exp6] worker {args.worker_index}/{args.n_workers} — "
            f"{len(condition_specs)} conditions: {[n for n, _ in condition_specs]}",
            flush=True,
        )

    # Load B-experiment hooks config once (same for all B conditions)
    hooks_config: dict = {}
    if is_B:
        hooks_config = _load_B_hooks_config(cfg, loaded)

    for condition_name, condition_cfg in condition_specs:
        # B2 mixes IT and PT conditions — skip any that don't match the loaded model variant.
        if condition_cfg.model_variant != args.variant:
            print(
                f"[exp6] skip {condition_name} "
                f"(variant={condition_cfg.model_variant}, loaded={args.variant})",
                flush=True,
            )
            continue

        done_benchmarks = {b for (c, b) in done_pairs if c == condition_name}
        if done_benchmarks >= set(condition_cfg.benchmarks):
            print(f"[exp6] skip {condition_name} (all done)", flush=True)
            continue

        print(f"[exp6] running {condition_name}", flush=True)

        if is_B:
            # Update hooks_config for condition-specific gamma / feature_set
            condition_hooks = dict(hooks_config)
            if condition_cfg.method == "feature_clamp" and "governance_features" in hooks_config:
                # Re-slice feature sets for this condition
                feat_path = Path(condition_cfg.governance_features_path)
                with open(feat_path) as f:
                    all_sets = json.load(f)
                condition_hooks["governance_features"] = {
                    k: all_sets.get(k, {}).get(condition_cfg.feature_set, [])
                    for k in all_sets
                }
            _run_condition_B(
                condition_name, condition_cfg, records, loaded,
                scores_jsonl, samples_jsonl, done_benchmarks, condition_hooks,
            )
        else:
            _run_condition_A(
                condition_name, condition_cfg, records, loaded,
                scores_jsonl, samples_jsonl, done_benchmarks,
                adapter=steering_adapter,
            )

        print(f"[exp6] {condition_name} done", flush=True)

    # Write final summary
    all_scores = _read_jsonl(scores_jsonl)
    save_json(cfg.run_dir / "scores.json", all_scores)
    print(f"[exp6] done → {cfg.run_dir}")


if __name__ == "__main__":
    main()
