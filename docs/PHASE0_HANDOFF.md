# Phase 0: Multi-Model Steering — Implementation Handoff

**Author**: Claude Code (Opus 4.6)  
**Date**: 2026-04-02  
**Status**: Code complete, all tests pass. No experiments run yet.  
**Plan doc**: `docs/PHASE0_MULTIMODEL_STEERING_PLAN.md`

---

## 1. What Was Implemented

### 1.1 Core multi-model steering pipeline

The existing Gemma-only exp6 steering pipeline was generalized to support all 6 models in MODEL_REGISTRY. The design uses a **SteeringAdapter** that wraps the existing cross-model adapter system (`src/poc/cross_model/adapters/`) with exp6-specific convenience methods.

**Key design decision**: Backward compatibility is preserved. All existing Gemma scripts (0C, 0D, 0F, 0H, 0I) continue to work unchanged — the adapter is only used when `--model-name` is passed.

#### New files

| File | Purpose |
|------|---------|
| `src/poc/exp6/model_adapter.py` | `SteeringAdapter` class — resolves layer/MLP/attn paths, EOS tokens, real-token mask per architecture |
| `scripts/precompute_directions_multimodel.py` | 4-phase direction extraction pipeline (gen→score→acts→merge) parameterized by `--model-name` |
| `scripts/run_phase0_multimodel.sh` | Master orchestration script with 8 steps (precompute, validate, steer, judge, pca, id-steering, commitment, plots) |
| `scripts/phase0_commitment_vs_alpha.py` | **Experiment 1B**: Post-process logit-lens data to compute commitment layer at each α value |
| `scripts/phase0_id_under_steering.py` | **Experiment 1C**: TwoNN intrinsic dimensionality at α=1.0, 0.0, -1.0 |
| `scripts/phase0_pca_direction.py` | **Experiment 1A**: PCA on per-record IT-PT activation differences; reports PC1 variance ratio |

#### Modified files

| File | Change |
|------|--------|
| `src/poc/exp6/config.py` | Added `model_family: str` field. When set, `__post_init__` derives `model_id`, `n_layers`, `d_model`, `proposal_boundary` from MODEL_REGISTRY. |
| `src/poc/exp6/interventions.py` | `register_hooks()` accepts optional `adapter` param. Uses `_get_mlp()/_get_attn()/_get_layer()` closures that dispatch to adapter or legacy Gemma paths. |
| `src/poc/exp6/runtime.py` | `_eos_token_ids()`, `generate_records_A_batch()`, `generate_records_B_batch()`, `generate_record_B()` all accept optional `adapter`. Logit-lens hooks and layer access use adapter when provided. |
| `src/poc/exp6/run.py` | Added `--model-name` and `--no-chat-template` CLI args. Creates `SteeringAdapter`, overrides `real_token_mask`, threads adapter through all condition runners. Added `import torch` to top-level. |

### 1.2 Architecture adapter design

The `SteeringAdapter` in `model_adapter.py` wraps the existing `ModelAdapter` (from `src/poc/cross_model/adapters/`) rather than duplicating its layer-path logic. This means:

- **Gemma**: `model.model.language_model.layers[i]` (the Gemma3ForConditionalGeneration multimodal wrapper)
- **All others**: `model.model.layers[i]` (standard HuggingFace CausalLM)

The adapter provides:
- `get_layers(model_raw)` → list of transformer blocks
- `get_mlp(model_raw, i)` → MLP submodule for hook registration
- `get_attn(model_raw, i)` → self-attention submodule
- `get_final_norm(model_raw)` → final LayerNorm (for logit lens)
- `get_lm_head(model_raw)` → unembedding head (for logit lens)
- `eos_token_ids(tokenizer)` → per-model EOS/EOT tokens
- `real_token_mask(tokenizer, device)` → suppress `<unusedXXXX>` (Gemma) and `<|reserved_special_token_*|>` (Llama)

### 1.3 Piggybacked experiments (1A, 1B, 1C)

These three experiments add near-zero marginal cost to the steering runs:

**1B: Commitment delay under steering** (THE causal link)
- `--collect-logit-lens` is enabled in the steer step (adds ~20% overhead)
- `phase0_commitment_vs_alpha.py` reads `logit_lens_top1_{condition}.npz` files and computes per-token commitment layer using the same no-flip-back algorithm as `collect_L1L2.py`
- Output: `commitment_vs_alpha.json` per model + combined summary

**1C: ID under steering** (completes the triad)
- `phase0_id_under_steering.py` loads model + directions, generates 200 prompts at 3 α values with residual stream hooks on ALL layers
- Subsamples to 5000 tokens per layer, runs `estimate_id_twonn()` from `exp4/analysis/intrinsic_dim.py`
- Output: `id_under_steering.json` per model (ID profile at each α)

**1A: PCA of IT-PT direction** (rank-1 justification)
- `phase0_pca_direction.py` loads selected 600 records, generates with IT and PT models, hooks MLP at corrective layers, collects per-record mean activations
- Computes SVD on the [600, d_model] IT-PT difference matrix per corrective layer
- Reports PC1 variance ratio (> 0.60 → rank-1 justified)
- Output: `pca_scree.json` per model

---

## 2. What Has NOT Been Run

**No Phase 0 experiments have been executed.** All code is written and tested (imports, adapter construction, hook registration, graceful failures) but zero GPU compute has been spent.

The precompute, validation, steering, and analysis steps all need to be run. See Section 5 for the execution plan.

---

## 3. What Was Run: 0G Tuned Lens

### Completed
| Model | Training | Eval PT | Eval IT |
|-------|----------|---------|---------|
| gemma3_4b | 34+34 probes | 200/200 | 200/200 |
| llama31_8b | 32+32 probes | 200/200 | 200/200 |
| qwen3_4b | 36+36 probes | 200/200 | **151/200** (partial, summary generated) |
| mistral_7b | 32+32 probes | 200/200 | 200/200 |
| deepseek_v2_lite | 27+27 probes | **NOT EVAL'D** | **NOT EVAL'D** |
| olmo2_7b | 32+32 probes | **NOT EVAL'D** | **NOT EVAL'D** |

### Plots generated (4 models)
- `results/exp7/plots/0G_commitment_top1.png`
- `results/exp7/plots/0G_commitment_kl.png`
- `results/exp7/plots/0G_commitment_summary.png`
- `results/exp7/plots/0G_commitment_scatter.png`
- `results/exp7/plots/data/0G_commitment_summary.json`

### To resume deepseek + olmo eval
The probe `.pt` files are trained. Just run with `--eval-only`:
```bash
for model in deepseek_v2_lite olmo2_7b; do
  for variant in pt it; do
    uv run python -m src.poc.cross_model.tuned_lens \
      --model "$model" --variant "$variant" \
      --device "cuda:N" --eval-only --n-eval-examples 200 \
      > "logs/exp7/0G/${model}_${variant}_eval.log" 2>&1 &
  done
done
```

---

## 4. Known Issues / Gotchas

### 4.1 Disk space
Only **20 GB free** on `/home/yifan` (439G total, 397G used). The A1 steering runs generate `sample_outputs.jsonl` files that can be ~2GB per model. With 6 models that's ~12GB. **Monitor disk during steering runs.** If needed:
- Delete `results/exp2/` (~1.1GB) and `results/exp4/` (~1GB) — old single-model experiments
- Compress `sample_outputs.jsonl` after scoring
- Move large files to GCS (see `memory/project_disk_gcs.md`)

### 4.2 GPU availability
GPUs 0-1 are often occupied by another user's RL training. The orchestration script uses GPUs 2-7 by default. Check `nvidia-smi` before starting.

### 4.3 Config override ordering
When both `--model-name` and `--proposal-boundary` / `--n-layers` are passed, the explicit CLI values take precedence because `Exp6Config.__post_init__` only overrides architecture params if they're still at Gemma defaults (34/2560/20). If you explicitly set them to non-Gemma values, the registry lookup won't overwrite them.

### 4.4 No chat template everywhere
All Phase 0 runs use `--no-chat-template`. This is critical for cross-model consistency. The existing Gemma A1_it_v4 was run WITH chat template — it cannot be compared directly with the new Phase 0 results.

### 4.5 DeepSeek max_gen_tokens=64
DeepSeek-V2-Lite hits OOM with 200 tokens due to MoE routing memory. The `SteeringAdapter.max_gen_tokens` returns 64 for deepseek. The precompute and steering scripts respect this automatically.

### 4.6 B-experiments remain Gemma-only
The `_make_feature_clamp_hooks()` and `_make_wdec_inject_hooks()` in `runtime.py` still use hardcoded `model_raw.language_model.layers[i]` paths. This is intentional — B-experiments require Gemma Scope 2 transcoders which only exist for Gemma.

---

## 5. Execution Plan (copy-paste ready)

### Estimated wall time: ~28 hours on 6 GPUs

```bash
# ── Step 1: Precompute directions (~6h) ──────────────────────────────────
bash scripts/run_phase0_multimodel.sh --step precompute

# ── Step 2: Validate (~2h) ───────────────────────────────────────────────
bash scripts/run_phase0_multimodel.sh --step validate

# ── Step 3: A1 α-sweep + logit lens (~17h) ──────────────────────────────
bash scripts/run_phase0_multimodel.sh --step steer

# ── Steps 4-7 can run in parallel after steering ────────────────────────
bash scripts/run_phase0_multimodel.sh --step judge &       # ~4h, API-bound
bash scripts/run_phase0_multimodel.sh --step pca &         # ~1.5h, GPU
bash scripts/run_phase0_multimodel.sh --step id-steering & # ~30min, GPU
wait
bash scripts/run_phase0_multimodel.sh --step commitment    # ~30min, CPU only

# ── Step 8: Generate plots ───────────────────────────────────────────────
bash scripts/run_phase0_multimodel.sh --step plots
```

### To run a single model (e.g. for debugging):
```bash
bash scripts/run_phase0_multimodel.sh --step precompute --model llama31_8b
bash scripts/run_phase0_multimodel.sh --step steer --model llama31_8b
```

---

## 6. Validation Checklist

After all experiments complete, verify:

- [ ] Direction quality: Bootstrap cosine > 0.90 for each model
- [ ] Baseline sanity: α=1.0 metrics match unsteered ±2%
- [ ] Format dose-response: STR decreases as α→0 for ≥4/6 models
- [ ] Content preservation: reasoning_em stable within ±5% for ≥4/6 models
- [ ] Random control: no systematic format change
- [ ] Qwen null: attenuated or absent dose-response
- [ ] 1A PCA: PC1 > 60% for ≥4/6 models
- [ ] 1B Commitment: commitment layer shifts with α for ≥4/6 models
- [ ] 1C ID: late-layer ID reduces at α=0 for ≥4/6 models

---

## 7. File Tree

```
# New files (Phase 0)
src/poc/exp6/model_adapter.py                          ← SteeringAdapter
scripts/precompute_directions_multimodel.py            ← Multi-model direction extraction
scripts/run_phase0_multimodel.sh                       ← Orchestration (8 steps)
scripts/phase0_commitment_vs_alpha.py                  ← 1B: commitment vs α
scripts/phase0_id_under_steering.py                    ← 1C: ID under steering
scripts/phase0_pca_direction.py                        ← 1A: PCA rank-1 check
docs/PHASE0_HANDOFF.md                                 ← This file

# Modified files
src/poc/exp6/config.py                                 ← model_family field
src/poc/exp6/interventions.py                          ← adapter param in register_hooks
src/poc/exp6/runtime.py                                ← adapter param in generation fns
src/poc/exp6/run.py                                    ← --model-name, --no-chat-template

# Expected output tree (after experiments complete)
results/cross_model/{model}/directions/
    corrective_directions.npz                          ← [d_model] per layer
    corrective_directions.meta.json
    pca_scree.json                                     ← 1A results
results/cross_model/{model}/exp6/
    merged_A1_{model}_it_v1/
        scores.jsonl
        sample_outputs.jsonl
        logit_lens_top1_{condition}.npz                ← for 1B
    commitment_vs_alpha.json                           ← 1B results
    id_under_steering.json                             ← 1C results
```
