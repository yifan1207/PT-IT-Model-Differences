# CLAUDE.md — Project Context for Claude Code

## Core directive
Always implement 100% as planned. Overcome engineering difficulties — don't bypass by cutting down the plan or fool yourself that a shortcut is still rigorous.

## Project overview
Mechanistic interpretability study on instruction-tuned LLMs. We find a "corrective direction" in MLP outputs (layers ~20-33) that instruction tuning introduces. Removing this direction from generation degrades governance/structural formatting while preserving content quality — evidence of a distinct structural control mechanism.

## Key experiments

### Exp6 (core intervention framework)
- `src/poc/exp6/run.py` — Main intervention runner. Supports multi-worker parallelism.
- **Worker suffix logic (line 979-981)**: `run_name = run_name_base + f"_w{worker_index}"` only when `n_workers > 1`. Scripts must NOT add `_w{i}` themselves (double-suffix bug was fixed in 0C/0F/0H/0I scripts).
- **Resume logic (line ~1079)**: Conditions with all benchmarks done are skipped (`[exp6] skip CONDITION (all done)`), enabling safe parallel/resume runs.
- Merging: `scripts/merge_exp6_workers.py` combines per-worker scores.jsonl into merged dir.

### Exp7 Tier 0 (methodology validation — 10 experiments)
All plot code: `scripts/plot_exp7_tier0.py`
All plot data: `results/exp7/plots/data/`

| ID | Name | Status | Notes |
|----|------|--------|-------|
| 0A | Direction stability (bootstrap) | DONE | 0A/bootstrap_results.json |
| 0B | Matched-token direction | DONE | 0B/matched_cosines_*.json |
| 0C | Random direction control | DONE | 5-seed multiseed + single-seed, merged |
| 0D | Bootstrap 95% CIs | DONE | ci_A1_programmatic.json, effect_sizes.json |
| 0E | Classifier robustness | DONE | classifier_robustness.json |
| 0F | Layer range sensitivity | DONE | layer_range_sensitivity_table.csv |
| 0G | Tuned-lens commitment | **NEEDS RERUN** | Training killed at step 20-50/250. See below. |
| 0H | Calibration split | DONE | governance-selected vs random-600 vs bottom-600 |
| 0I | Formula comparison | **NEEDS RERUN** | Old run has 5 alphas. Updated script has 14 alphas (60 conditions). |
| 0J | Onset threshold sensitivity | DONE | onset_sensitivity.png |

All plots have 95% CI bands or ±1σ error bars where data supports it (from 0D bootstrap).

## IMMEDIATE NEXT STEPS

### 0. Phase 0: Multi-model steering (HIGHEST PRIORITY)
Extends causal steering from Gemma-only to all 6 models. Code changes DONE.

**New files:**
- `src/poc/exp6/model_adapter.py` — SteeringAdapter bridging cross_model adapters to exp6
- `scripts/precompute_directions_multimodel.py` — Multi-model direction extraction (4-phase pipeline)
- `scripts/run_phase0_multimodel.sh` — Orchestration script

**Modified files:**
- `src/poc/exp6/config.py` — Added `model_family` field, derives arch params from MODEL_REGISTRY
- `src/poc/exp6/interventions.py` — `register_hooks()` accepts optional `adapter` param
- `src/poc/exp6/runtime.py` — Generalized EOS tokens, logit-lens hooks, real_token_mask
- `src/poc/exp6/run.py` — Added `--model-name` CLI arg (IT uses chat template by default; `--no-chat-template` available for ablation)

**Run order:**
```bash
# Step 1: Precompute directions for 5 new models (~6h, GPUs 2-7)
bash scripts/run_phase0_multimodel.sh --step precompute

# Step 2: Validate (3-point sanity check, ~2h)
bash scripts/run_phase0_multimodel.sh --step validate

# Step 3: Full A1 alpha-sweep + logit lens, all 6 models (~17h, GPUs 2-7)
bash scripts/run_phase0_multimodel.sh --step steer

# Step 4: LLM judge (~4h, API-bound, can overlap with steps 5-7)
bash scripts/run_phase0_multimodel.sh --step judge

# Step 5: 1A PCA of IT-PT direction (~1.5h, 6 models in parallel)
bash scripts/run_phase0_multimodel.sh --step pca

# Step 6: 1C ID under steering (TwoNN, ~30min, 6 models in parallel)
bash scripts/run_phase0_multimodel.sh --step id-steering

# Step 7: 1B Commitment vs alpha (CPU post-processing, ~30min)
bash scripts/run_phase0_multimodel.sh --step commitment
```

**Piggybacked experiments (1A/1B/1C):**
- **1A PCA**: `src/poc/exp8/pca_rank1.py` — Is PC1 > 60%? Rank-1 justified?
- **1B Commitment**: `src/poc/exp8/commitment_vs_alpha.py` — Commitment layer vs α (THE causal link)
- **1C ID under steering**: `src/poc/exp8/id_under_steering.py` — TwoNN ID at α=1.0, 0.0, -1.0

**Key design decisions:**
- ALL experiments use `apply_chat_template=True` for IT models (their native trained distribution); PT models get raw text
- Previous no-template runs are kept as ablation evidence (template-free results confirm corrective stage is weight-encoded)
- Gemma A1_it_v4 (with chat template) is the canonical Gemma steering result
- Cross-model directions, steering, and commitment delay all being rerun with chat template
- Output: `results/cross_model/{model}/directions/` and `results/cross_model/{model}/exp6/`
- GPUs 0-1 reserved for other users; GPUs 2-7 available

### 1. Rerun 0I (expanded alpha values)
Script: `scripts/run_exp7_0I.sh` — already updated with 14 alpha values, 4 methods, 60 conditions.
```bash
# 4 workers, ~2-3 hours
bash scripts/run_exp7_0I.sh
```

### 2. Rerun 0G (tuned-lens training from scratch)
The 0G training script was at `/tmp/run_0G_tuned_lens.sh` which may not survive reboot. Recreate or run directly:

```bash
# Train tuned-lens for ALL 6 models × 2 variants = 12 runs
# Uses all available GPUs, batches of NGPUS at a time
# Joint training mode (~34× faster than sequential)
# Estimated: ~4-6 hours on 8 GPUs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
mkdir -p logs/exp7/0G

MODELS=(gemma3_4b llama31_8b qwen3_4b mistral_7b deepseek_v2_lite olmo2_7b)
for model in "${MODELS[@]}"; do
    for variant in pt it; do
        # Find a free GPU and run:
        uv run python -m src.poc.cross_model.tuned_lens \
            --model "$model" --variant "$variant" \
            --device "cuda:N" \
            > "logs/exp7/0G/${model}_${variant}.log" 2>&1
    done
done

# After training, eval:
uv run python -m src.poc.cross_model.tuned_lens \
    --model "$model" --variant "$variant" \
    --device "cuda:N" --eval-only --n-eval-examples 200

# After eval, regenerate cross-model CIs (per-token method):
uv run python -m src.poc.exp7.bootstrap_ci \
    --merged-dir results/exp6/merged_A1_it_v4 \
    --cross-model-dir results/cross_model

# Copy CI to plots data:
cp results/exp7/0D/ci_cross_model.json results/exp7/plots/data/

# Generate commitment plots:
uv run python scripts/plot_tuned_lens_commitment.py
```

### 3. Regenerate all Tier 0 plots after reruns
```bash
uv run python scripts/plot_exp7_tier0.py
```

## Tuned-lens recipe (Belrose et al. 2023 — exact)
- SGD Nesterov: lr=1.0, momentum=0.9, weight_decay=0
- 250 steps, 262,144 tokens/step (65.5M total token-activations, 70M unique tokens loaded)
- Linear LR decay to 0 (no warmup)
- 2048-token chunks, identity init + zero bias
- KL divergence loss (sum reduction / batch_size)
- Gradient clipping to norm 1.0
- `model.requires_grad_(False)` before training (saves ~3 GB on lm_head gradients)
- Joint all-layer training: all probes trained simultaneously per forward pass

Code: `src/poc/cross_model/tuned_lens.py`

## Model registry (6 models)
Config: `src/poc/cross_model/config.py`

| Model | HF ID (PT) | Layers | d_model | Notes |
|-------|-----------|--------|---------|-------|
| gemma3_4b | google/gemma-3-4b-pt | 34 | 2560 | Global attn every 6th layer |
| llama31_8b | meta-llama/Llama-3.1-8B | 32 | 4096 | |
| qwen3_4b | Qwen/Qwen3-4B | 36 | 2560 | |
| mistral_7b | mistralai/Mistral-7B-v0.3 | 32 | 4096 | Sliding window 4096 |
| deepseek_v2_lite | deepseek-ai/DeepSeek-V2-Lite | 27 | 2048 | MoE, max_new_tokens=64 |
| olmo2_7b | allenai/OLMo-2-1124-7B | 32 | 4096 | |

None use multi_gpu (all fit on 1×80GB A100).

## Known bugs (FIXED — don't reintroduce)

1. **Double-suffix merge bug**: Scripts must NOT pass `--run-name "${NAME}_w${i}"` — `exp6/run.py` already appends `_w{worker_index}`. Fixed in 0C/0F/0H/0I scripts.

2. **record_id vs id**: `eval_dataset_v2.jsonl` uses field `"id"`, not `"record_id"`. Filter code in exp6/run.py was fixed.

3. **Benchmark name aliases**: Data uses `exp3_alignment_behavior` and `exp3_reasoning_em` but some code looked for `alignment_behavior` and `reasoning_em`. Both variants must be checked. Fixed in `layer_range_analysis.py` and `plot_exp7_tier0.py`.

4. **model.requires_grad_(False)**: Must be called before tuned-lens training. Without it, backward() stores ~3 GB of useless gradients on lm_head/final_norm, causing OOM in joint training.

5. **Per-token vs per-prompt commitment CI**: Cross-model commitment CIs must use per-token values (SKIP_FIRST_N=5, 50k token subsample, BCa bootstrap), not per-prompt averaging which is biased by generation length. Fixed in `src/poc/exp7/bootstrap_ci.py`.

## Key file paths

```
# Phase 0 (multi-model steering)
src/poc/exp6/model_adapter.py       — SteeringAdapter for multi-model hook resolution
scripts/precompute_directions_multimodel.py — Multi-model direction extraction
scripts/run_phase0_multimodel.sh    — Phase 0 orchestration (precompute/validate/steer/judge)

# Exp6 core
src/poc/exp6/run.py                 — Core intervention runner (--model-name for multi-model)
src/poc/exp6/config.py              — Exp6Config (model_family field for multi-model)
src/poc/exp6/interventions.py       — Hook registration (adapter param for multi-model)
src/poc/exp6/runtime.py             — Generation (adapter param for multi-model)

# Exp7 / Cross-model
scripts/plot_exp7_tier0.py          — All Tier 0 plots (0A-0J)
scripts/plot_tuned_lens_commitment.py — 0G commitment plots
scripts/merge_exp6_workers.py       — Merge multi-worker results
scripts/run_exp7_0*.sh              — Per-experiment run scripts
src/poc/cross_model/tuned_lens.py   — Tuned-lens training/eval
src/poc/cross_model/config.py       — MODEL_REGISTRY
src/poc/cross_model/adapters/       — Per-model architecture adapters
src/poc/exp7/bootstrap_ci.py        — Bootstrap CIs + effect sizes

# Results
results/exp6/merged_A1_it_v4/       — Canonical A1 intervention results (Gemma, with chat template)
results/cross_model/{model}/directions/ — Per-model corrective directions
results/cross_model/{model}/exp6/   — Per-model steering results (Phase 0)
results/exp7/0*/                    — Per-experiment Tier 0 results
results/exp7/plots/                 — All PNG plots
results/exp7/plots/data/            — All JSON data exports
```

## GPU usage
- Server has 8× H100 80GB GPUs (cuda:0-7)
- Another user (ubuntu) runs RL training — check `nvidia-smi` before claiming GPUs
- Never interfere with other users' processes
- Phase 0: use GPUs 2-7 (0-1 typically occupied)
