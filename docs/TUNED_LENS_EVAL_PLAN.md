# Tuned-Lens Inference Data Collection Plan (v3 — One-Pass Design)
**Date**: 2026-04-01
**Status**: Tuned-lens training COMPLETE for all 6 models × 2 variants = 12 runs. This document specifies everything to collect in ONE inference pass so we never need to re-generate.

---

## ⚡ ONE PASS DESIGN — Executive Summary

**One function (`eval_commitment()` with `collect_full=True`) collects ALL data for ALL experiments using BOTH raw logit-lens AND tuned-lens in a single generation pass per model-variant.**

The function already runs both lenses every step (lines 1393-1440 of `tuned_lens.py`):
1. Hooks capture residual `h_ℓ` at every layer → `step_buf[layer_idx]`
2. Raw logit-lens: `final_norm(h_ℓ) @ W_U` → `all_raw_logits` [n_layers, vocab]
3. Tuned probes: `probe_ℓ(h_ℓ)` → `all_h_tuned`
4. Tuned logit-lens: `final_norm(all_h_tuned) @ W_U` → `all_tuned_logits` [n_layers, vocab]

Currently it derives ~60+ commitment scalars from these tensors and DISCARDS the per-layer arrays. With `collect_full=True`, we save everything. **This subsumes `collect_L1L2.py` entirely** — no separate L1L2 run needed.

**Result**: 12 inference runs (6 models × 2 variants) produce ALL data for experiments 0G + 1.1 through 1.6 + reshuffling + entropy profiles + emergence tracking, in both raw and tuned-lens variants.

---

## Architecture: Two Separate Pipelines (Don't Confuse Them)

There are **two distinct generation pipelines** in the codebase. They serve different purposes and CANNOT be merged.

### Pipeline 1: Observational eval — `eval_commitment()` ← THIS IS OUR ONE PASS
- **Dataset**: `data/exp3_dataset.jsonl` (2,936 records — **zero overlap** with eval_dataset_v2)
- **Runner**: `modal_eval_0G.py` (Modal cloud) or local `--eval-only`
- **Generation**: Unsteered, greedy, no chat template, PT and IT separately
- **Purpose**: All Phase 1 observational replication + 0G tuned-lens robustness
- **Currently collects**: Commitment scalars from 7 methods × multiple thresholds
- **With `collect_full=True`**: Also saves ALL per-layer arrays (Groups B-G)
- **Runs on**: Modal (H100/A100-80GB) or local GPUs 2-7

### Pipeline 2: Causal steering — `exp6/run.py` (SEPARATE, DO NOT TOUCH)
- **Dataset**: `data/eval_dataset_v2.jsonl` (1,400 records)
- **Runner**: `scripts/run_phase0_multimodel.sh`
- **Purpose**: A1 α-sweep, commitment-vs-α causal link (Phase 0.4a), ID under steering (0.4b)
- **Collects**: `logit_lens_top1_{condition}.npz` — raw top-1 only, NO tuned-lens
- **Post-processing**: `phase0_commitment_vs_alpha.py`, `phase0_id_under_steering.py`

### Why they can't be merged:
- Different datasets (zero prompt overlap — checked programmatically)
- Pipeline 2 applies steering interventions (modifies MLP outputs); Pipeline 1 is unsteered
- Pipeline 2 only needs raw top-1 (for commitment-vs-α); Pipeline 1 needs full distributions
- Pipeline 2 runs 18 conditions × 6 models = 108 runs; Pipeline 1 runs 12

### Why `exp3_dataset.jsonl` is correct for the logit-lens pass:
All logit-lens analyses (mind-change, confidence-stratified, adjacent-KL, reshuffling, entropy, etc.) study how the model's INTERNAL predictions evolve across layers. They do NOT need:
- `expected_answer` / `expected_format` / `expected_behavior` (eval_dataset_v2 fields)
- Correctness scoring, format compliance, or any benchmark evaluation

They only need a diverse set of prompts to generate on. `exp3_dataset.jsonl` (2,936 records) is actually BETTER than `eval_dataset_v2.jsonl` (1,400) for this because:
1. Larger (2× more prompts → better statistics)
2. More diverse question types (10 types: factual, multi_choice, code, yes_no, fictional, format, numerical, unknowable, counterfactual, continuation)
3. More diverse domains (10 domains: ooc, trivia, general, commonsense, safety, programming, math, science, instruction_following, reading_comprehension)
4. Already used by `collect_L1L2.py` and the current `eval_commitment()` — consistency

`eval_dataset_v2.jsonl` is for Pipeline 2 (steering) because its `category` field (GOV-FORMAT, GOV-CONV, CONTENT-FACT, etc.) and expected-answer fields enable governance benchmark scoring — needed to measure the EFFECT of steering, not the internal dynamics.

### `collect_L1L2.py` is now OBSOLETE for multi-model:
The existing `collect_L1L2.py` produces `{model}/{variant}/L1L2_results.jsonl` with: `delta_cosine`, `logit_lens_entropy`, `logit_lens_top1`, `commitment_layer`. ALL of this is a strict subset of what the extended `eval_commitment()` collects. L1L2 has NOT been run for multi-model yet (only plots/data CSVs exist from weight-diff analysis). **Do not run `collect_L1L2.py` — the extended eval pass covers everything.**

Note: `plot_cross_model.py` currently reads `L1L2_results.jsonl`. After the extended eval, update it to read from the new output format instead.

---

## What `eval_commitment()` ACTUALLY Collects Now (Post-Update)

The function has been significantly expanded from the original 3-threshold version. Per prompt, per generation step, it now computes:

### 7 commitment methods × multiple thresholds:

| Method | Thresholds | Keys in JSONL |
|--------|-----------|---------------|
| Raw top-1 (no-flip-back) | — | `commitment_layer_raw` |
| Raw KL | 0.05, 0.1, 0.2, 0.5, 1.0 | `commitment_layer_raw_kl_{t}` |
| Tuned KL | 0.05, 0.1, 0.2, 0.5, 1.0 | `commitment_layer_tuned_{t}` |
| Tuned top-1 | — | `commitment_layer_top1_tuned` |
| Majority (≥90% tuned KL) | 0.05, 0.1, 0.2, 0.5, 1.0 | `commitment_layer_majority_{t}` |
| Cosine (h_ℓ · h_L) | 0.80, 0.90, 0.95, 0.99 | `commitment_layer_cosine_{t}` |
| Entropy (raw logit-lens) | 0.05, 0.1, 0.2, 0.5, 1.0 | `commitment_layer_entropy_{t}` |

### Qualified variants (relaxed commitment):
| Method | Qualifications | Keys |
|--------|---------------|------|
| Raw top-1 qualified | top-K ∈ {3, 5} | `commitment_layer_raw_top1_qual_top{k}` |
| Tuned top-1 qualified | top-K ∈ {3, 5} | `commitment_layer_tuned_top1_qual_top{k}` |
| Raw KL qualified | t × M, M ∈ {3, 5} | `commitment_layer_raw_kl_qual_{t}_{m}x` |
| Tuned KL qualified | t × M, M ∈ {3, 5} | `commitment_layer_tuned_kl_qual_{t}_{m}x` |

**Total**: ~60+ commitment scalar arrays per prompt, all saved to JSONL.

### What is COMPUTED but NOT saved:
Inside `_process_step()`, the function already computes:
- `raw_top1_row` — [n_layers] int — raw logit-lens argmax per layer
- `tuned_top1_row` — [n_layers] int — tuned-lens argmax per layer
- `all_raw_logits` — [n_layers, vocab] — full raw distributions
- `all_tuned_logits` — [n_layers, vocab] — full tuned distributions
- `raw_kl_row` — [n_layers] float — KL(raw_ℓ ‖ final)
- `kl_row` — [n_layers] float — KL(tuned_ℓ ‖ final)
- `raw_ranks` — [n_layers] int — rank of final token in raw distribution (for qualified commitment)
- `tuned_ranks` — [n_layers] int — rank of final token in tuned distribution
- `cosine_row` — [n_layers] float — cos(h_ℓ, h_L) in residual stream
- `entropy_row` — [n_layers] float — entropy of raw logit-lens per layer
- `all_raw_probs` — [n_layers, vocab] — softmax of raw logits (for entropy)

**All of these are discarded after deriving commitment scalars.** The whole plan is about saving them.

---

## Complete Per-Step Data To Collect (Groups B–G)

Everything below is NEW to save. Group A (commitment scalars) stays as-is.

### Group B: Per-layer top-1 tokens + generated token ID
**Already computed as**: `raw_top1_row`, `tuned_top1_row`, `final_top1`

| Field | Shape | Type | Source variable | Analysis |
|-------|-------|------|----------------|----------|
| `raw_top1_per_layer` | [n_layers] | int16 | `raw_top1_row` | 1.1 Mind-change (raw) |
| `tuned_top1_per_layer` | [n_layers] | int16 | `tuned_top1_row` | 1.1 Mind-change (tuned) — robustness |
| `generated_token_id` | scalar | int32 | `final_top1` | Needed for Groups C/D |

**Analyses enabled**:
- **1.1 Mind-change classification** (6 models × 2 lenses): At each layer, compare top1[ℓ] vs top1[ℓ-1]. If different → mind-change. Classify new token as structural/content/discourse/punctuation/function. Currently Gemma-only at 75% structural targeting. Tuned-lens variant is the key robustness check.
- **1.5 Within-model cross-layer correlation**: Per-layer Spearman between mind-change rate and δ-cosine magnitude.

**Plots to create** (NEW, for all 6 models):
1. `mind_change_targeting_raw_6panel.png` — structural-token targeting rate by normalized layer depth, raw logit-lens, PT vs IT
2. `mind_change_targeting_tuned_6panel.png` — same with tuned-lens
3. `mind_change_raw_vs_tuned_scatter.png` — per-model scatter of raw vs tuned structural targeting rate (robustness)
4. `cross_layer_correlation_6panel.png` — Spearman(δ-cosine, mind-change rate) per layer

### Group C: Per-layer KL metrics
**Partially computed**: `raw_kl_row` and `kl_row` (KL-to-final) already exist. **Adjacent-layer KL is NEW.**

| Field | Shape | Type | Source | Analysis |
|-------|-------|------|--------|----------|
| `raw_kl_to_final` | [n_layers] | float16 | `raw_kl_row` (exists) | 1.2 Confidence-stratified |
| `tuned_kl_to_final` | [n_layers] | float16 | `kl_row` (exists) | 1.2 Confidence-stratified (tuned) |
| `raw_kl_adjacent` | [n_layers] | float16 | **NEW** — KL(raw_ℓ ‖ raw_{ℓ-1}) | 1.3 Three-phase |
| `tuned_kl_adjacent` | [n_layers] | float16 | **NEW** — KL(tuned_ℓ ‖ tuned_{ℓ-1}) | 1.3 Three-phase (tuned) |

**New computation needed**: Adjacent-layer KL. After computing `all_raw_logits` (already done), compute:
```python
raw_log_probs = F.log_softmax(all_raw_logits, dim=-1)  # already computed as all_raw_log_q
# Adjacent KL: reuse all_raw_log_q from existing code
for li in range(1, n_layers):
    raw_adj_kl = F.kl_div(raw_log_probs[li], raw_log_probs[li-1], ...).sum(-1)
```
Note: `all_raw_log_q` is already computed on line 1407 of tuned_lens.py for the KL-to-final calculation. Reuse it.

**Plots to create** (NEW, for all 6 models):
5. `adjacent_kl_raw_6panel.png` — mean KL(ℓ‖ℓ-1) across layers, PT vs IT, raw logit-lens
6. `adjacent_kl_tuned_6panel.png` — same with tuned-lens
7. `three_phase_summary.png` — bar chart: mean KL by phase (early/mid/corrective) × model × lens

### Group D: Per-layer confidence for generated token
**Partially computed**: `raw_ranks` and `tuned_ranks` already exist (for qualified commitment). `raw_probs` exists as `all_raw_probs` (for entropy). Just need to index.

| Field | Shape | Type | Source | Analysis |
|-------|-------|------|--------|----------|
| `raw_next_token_prob` | [n_layers] | float16 | `all_raw_probs[:, final_top1]` | 1.2 Confidence strat. |
| `tuned_next_token_prob` | [n_layers] | float16 | `softmax(all_tuned_logits)[:, final_top1]` | 1.2 Confidence (tuned) |
| `raw_next_token_rank` | [n_layers] | int16 | `raw_ranks` (exists!) | Emergence tracking |
| `tuned_next_token_rank` | [n_layers] | int16 | `tuned_ranks` (exists!) | Emergence tracking |
| `raw_final_confidence` | scalar | float32 | `all_raw_probs[-1, final_top1]` | Stratification variable |
| `tuned_final_confidence` | scalar | float32 | `softmax(all_tuned_logits)[-1, final_top1]` | Stratification variable |

**Key codebase fact**: `raw_ranks` and `tuned_ranks` are already computed inside `_process_step()` at lines 1471-1485 for the qualified commitment calculation. They use `argsort(descending=True)` which is more expensive than necessary — for saving purposes, we can reuse them directly. But note the argsort approach gives 1-based rank where 1 = highest logit. This matches exp3's convention.

**Plots to create** (NEW, for all 6 models):
8. `confidence_stratified_raw_6panel.png` — commitment delay by confidence quartile, PT vs IT, raw
9. `confidence_stratified_tuned_6panel.png` — same with tuned-lens
10. `confidence_summary_bar.png` — Δ(commitment delay, IT-PT) by confidence bin × model × lens

### Group E: Top-5 tokens per layer (candidate reshuffling)
**Not currently computed.** Requires `torch.topk`.

| Field | Shape | Type | Analysis |
|-------|-------|------|----------|
| `raw_top5_ids` | [n_layers, 5] | int32 | Candidate reshuffling (raw) |
| `raw_top5_probs` | [n_layers, 5] | float16 | Reshuffling confidence |
| `tuned_top5_ids` | [n_layers, 5] | int32 | Reshuffling (tuned) |
| `tuned_top5_probs` | [n_layers, 5] | float16 | Reshuffling confidence |

**Limit**: Only collect for first 200 prompts per model-variant (storage: ~109 MB). The `all_raw_probs` variable (line 1529) already exists; add `torch.topk(all_raw_probs, k=5, dim=-1)`.

**Plots to create** (NEW, for all 6 models):
11. `candidate_reshuffling_jaccard_6panel.png` — top-5 Jaccard overlap by layer, PT vs IT
12. `candidate_reshuffling_tuned_vs_raw.png` — comparison of reshuffling pattern

### Group F: Per-layer entropy (raw + tuned)
**Raw entropy already computed**: `entropy_row` (line 1532). **Tuned entropy is NEW.**

| Field | Shape | Type | Source | Analysis |
|-------|-------|------|--------|----------|
| `raw_entropy` | [n_layers] | float16 | `entropy_row` (exists) | Entropy profiles |
| `tuned_entropy` | [n_layers] | float16 | **NEW** — from `all_tuned_logits` | Entropy profiles (tuned) |

**Plots to create** (NEW):
13. `entropy_profiles_6panel.png` — raw + tuned entropy by layer, PT vs IT, all 6 models

### Group G: δ-cosine (residual stream)
**Already collected separately by `collect_L1L2.py`** for all 6 models. `step_buf` contains residuals in `eval_commitment()` so we CAN compute it here too. But since `collect_L1L2.py` already has this data, we only need it if we want to co-index with tuned-lens data on the same prompts.

| Field | Shape | Type | Source | Analysis |
|-------|-------|------|--------|----------|
| `delta_cosine` | [n_layers] | float16 | Computed from `step_buf` | 1.4 Heatmaps, 1.5 Correlation |

**Decision**: Include it. The marginal cost is trivial (n_layers dot products on d_model vectors), and having δ-cosine co-indexed with tuned-lens data enables exact per-token correlation (1.5) without prompt-level averaging.

**Plots to create** (update existing):
14. `delta_cosine_heatmap_6x2.png` — already exists in `plot_cross_model.py`, but regenerate from this data for consistency
15. `delta_cosine_vs_mind_change_6panel.png` — per-layer scatter (1.5)

---

## Per-Category Commitment (1.6)
**No new data needed.** The dataset (`exp3_dataset.jsonl`) has a `category` field. The commitment scalars from Group A already give per-step commitment. Just group by category.

**Plot to create** (NEW):
16. `per_category_commitment_heatmap.png` — commitment delay × category × model, raw and tuned

---

## Which Lens for Which Experiment? (Master Matrix)

| # | Experiment | Raw logit-lens | Tuned-lens | δ-cosine (residual) | Data Groups | Status |
|---|-----------|---------------|------------|---------------------|-------------|--------|
| 0G | Commitment delay comparison | ✅ | ✅ | ❌ | A | ✅ Already collected by `eval_commitment()` |
| 1.1 | Mind-change classification | ✅ | ✅ | ❌ | B | 🆕 Needs top1 arrays saved |
| 1.2 | Confidence-stratified commitment | ✅ | ✅ | ❌ | A + D | 🆕 Needs prob/rank saved |
| 1.3 | Adjacent-layer KL profiles | ✅ | ✅ | ❌ | C | 🆕 Needs adj-KL computed + saved |
| 1.4 | Generation-step heatmaps | ❌ | ❌ | ✅ | G | Already in `collect_L1L2.py`; include here for co-indexing |
| 1.5 | Cross-layer correlation | ✅ | ✅ | ✅ | B + G | 🆕 Derived from mind-change + δ-cosine |
| 1.6 | Per-category commitment | ✅ | ✅ | ❌ | A (+ category metadata) | ✅ No new data — just grouping |
| — | Candidate reshuffling | ✅ | ✅ | ❌ | E | 🆕 Needs top5 computed + saved |
| — | Entropy profiles | ✅ | ✅ | ❌ | F | 🆕 Tuned entropy is new |
| — | Token emergence tracking | ✅ | ✅ | ❌ | D | 🆕 Needs rank/prob saved |

---

## Implementation: Changes to `eval_commitment()`

### What's truly new vs what's just "stop discarding"

| Data | Already computed? | Var name in `_process_step()` | Action |
|------|------------------|-------------------------------|--------|
| raw top1 per layer | ✅ line 1403 | `raw_top1_row` | Save to list |
| tuned top1 per layer | ✅ line 1430 | `tuned_top1_row` | Save to list |
| generated token id | ✅ line 1398 | `final_top1` | Save to list |
| raw KL-to-final | ✅ line 1412 | `raw_kl_row` | Save to list |
| tuned KL-to-final | ✅ line 1439 | `kl_row` | Save to list |
| raw ranks | ✅ line 1477 | `raw_ranks` | Save to list (currently local) |
| tuned ranks | ✅ line 1482 | `tuned_ranks` | Save to list (currently local) |
| raw entropy | ✅ line 1531 | `entropy_row` | Save to list |
| cosine h_ℓ · h_L | ✅ line 1517 | `cosine_row` | Already saved as `step_cosine_values` |
| **raw adj-KL** | ❌ NEW | — | Compute from `all_raw_log_q` (line 1407) |
| **tuned adj-KL** | ❌ NEW | — | Compute from `all_log_q` (line 1434) |
| **tuned entropy** | ❌ NEW | — | Compute from `all_tuned_logits` |
| **raw next-token prob** | ❌ NEW | — | Index `all_raw_probs` (line 1529) at `final_top1` |
| **tuned next-token prob** | ❌ NEW | — | softmax(all_tuned_logits)[:, final_top1] |
| **top-5 tokens** | ❌ NEW | — | `torch.topk` on existing probs |
| **δ-cosine** | ❌ NEW (in this func) | — | Compute from `step_buf` |

**Summary**: 11 of 17 fields are already computed — just need to stop discarding. Only 6 require new computation, and 4 of those reuse existing tensors.

### Signature change
```python
def eval_commitment(
    ...,
    collect_full: bool = False,       # Groups B-G (minus top5)
    collect_top5: bool = False,       # Group E (heavier; limit prompts)
    top5_max_prompts: int = 200,      # Only collect top5 for first N
    arrays_dir: Path | None = None,   # Where to save NPY arrays
) -> dict:
```

### NPY output format

Accumulate step-level arrays across all prompts, save once at end:

```
results/cross_model/{model}/tuned_lens/commitment/arrays/
  raw_top1.npy              # [total_steps, n_layers] int16
  tuned_top1.npy            # [total_steps, n_layers] int16
  generated_ids.npy         # [total_steps] int32
  raw_kl_final.npy          # [total_steps, n_layers] float16
  tuned_kl_final.npy        # [total_steps, n_layers] float16
  raw_kl_adj.npy            # [total_steps, n_layers] float16
  tuned_kl_adj.npy          # [total_steps, n_layers] float16
  raw_ntprob.npy            # [total_steps, n_layers] float16
  tuned_ntprob.npy          # [total_steps, n_layers] float16
  raw_ntrank.npy            # [total_steps, n_layers] int16
  tuned_ntrank.npy          # [total_steps, n_layers] int16
  raw_entropy.npy           # [total_steps, n_layers] float16
  tuned_entropy.npy         # [total_steps, n_layers] float16
  delta_cosine.npy          # [total_steps, n_layers] float16
  cosine_h_to_final.npy     # [total_steps, n_layers] float16  (already computed)
  step_index.jsonl          # Maps prompt_id → (start_step, end_step, category)
  # Top-5 (first 200 prompts only):
  raw_top5_ids.npy          # [subset_steps, n_layers, 5] int32
  raw_top5_probs.npy        # [subset_steps, n_layers, 5] float16
  tuned_top5_ids.npy        # [subset_steps, n_layers, 5] int32
  tuned_top5_probs.npy      # [subset_steps, n_layers, 5] float16
  top5_step_index.jsonl     # Maps prompt_id → (start_step, end_step) for top5 subset
```

---

## Storage Estimates (Corrected)

Per model-variant pair. Dataset = 2,936 prompts. Average ~200 steps/prompt for dense models, ~64 for DeepSeek. Estimate ~500K total steps for dense, ~190K for DeepSeek.

| Group | Data | Size (dense, ~500K steps) | Size (DeepSeek, ~190K steps) |
|-------|------|--------------------------|------------------------------|
| A | Commitment scalars (JSONL, ~60 fields) | ~120 MB | ~45 MB |
| B | top1 (raw+tuned, 2 arrays) | 500K × 34 × 2 × 2B = ~68 MB | ~20 MB |
| C | KL (4 arrays: kl_final×2, kl_adj×2) | 500K × 34 × 4 × 2B = ~136 MB | ~41 MB |
| D | prob+rank (4 arrays) | 500K × 34 × 4 × 2B = ~136 MB | ~41 MB |
| E | top5 (200 prompts, ~40K steps) | 40K × 34 × 5 × 2 × 6B = ~82 MB | ~31 MB |
| F | entropy (raw+tuned) | 500K × 34 × 2 × 2B = ~68 MB | ~20 MB |
| G | δ-cosine + cosine-to-final | 500K × 34 × 2 × 2B = ~68 MB | ~20 MB |
| **Total** | | **~678 MB** | **~218 MB** |

For all 12 runs: 5 dense × 2 variants × 678 + 1 × 2 × 218 ≈ **7.2 GB**. Manageable.

---

## Modal vs Local Execution

### Option A: Extend `modal_eval_0G.py` (RECOMMENDED)
The Modal script already runs `eval_commitment()` for all 12 model-variants. Add `collect_full=True` to the call:

```python
# In modal_eval_0G.py, line ~250:
summary = eval_commitment(
    model, tokenizer, adapter, spec, probes, records, device,
    output_path=out_path,
    variant=variant,
    max_new_tokens=max_new_tokens,
    collect_full=True,           # NEW
    collect_top5=True,           # NEW
    top5_max_prompts=200,        # NEW
    arrays_dir=results_dir / "arrays",  # NEW
)
```

**Consideration**: Modal Volume needs to be large enough for ~678 MB per run × 12 = ~8 GB. The `0g-results` volume handles this fine.

**Download**: After completion, `bash scripts/modal_download_results.sh` already downloads the full results dir. Arrays will come along.

### Option B: Local execution
```bash
MODELS=(gemma3_4b llama31_8b qwen3_4b mistral_7b deepseek_v2_lite olmo2_7b)
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    gpu=$((i % 6 + 2))  # GPUs 2-7 (0-1 reserved per CLAUDE.md)
    for variant in pt it; do
        uv run python -m src.poc.cross_model.tuned_lens \
            --model "$model" --variant "$variant" \
            --device "cuda:${gpu}" \
            --eval-only \
            --collect-full --collect-top5 --top5-max-prompts 200 \
            > "logs/exp7/0G/${model}_${variant}_fulleval.log" 2>&1
    done &
done
wait
```

Note: Uses GPUs 2-7 (not 0-1 per CLAUDE.md GPU policy).

---

## Relationship to Other Data Collection

### SUBSUMED by this one pass (DO NOT run separately):
| Previously separate | Was in | Now covered by | Notes |
|--------------------|--------|----------------|-------|
| δ-cosine heatmaps (L1) | `collect_L1L2.py` | Group G | L1L2 NOT YET RUN for multi-model; skip it |
| Raw logit-lens top1 + entropy + commitment (L2) | `collect_L1L2.py` | Groups A + B + F | Strict subset of extended eval |
| Exp3 extended (rank, prob, adj-KL, top5) | `exp3/collect.py` | Groups C + D + E | exp3/collect only ran for Gemma; now all 6 models |

### Collected during Phase 0 steering (SEPARATE pipeline — still needed):
| Data | Source | Dataset | Models |
|------|--------|---------|--------|
| Raw logit-lens top1 UNDER STEERING | `runtime.py` → `logit_lens_top1_{condition}.npz` | eval_dataset_v2.jsonl (1,400) | All 6 |
| Commitment vs α (derived) | `phase0_commitment_vs_alpha.py` | (post-processing) | All 6 |
| ID under steering | `phase0_id_under_steering.py` | eval_dataset_v2.jsonl | All 6 |
| PCA of IT-PT direction | `phase0_pca_direction.py` | (precompute-phase) | All 6 |

Note: Phase 0 steering CANNOT use tuned-lens (different dataset, different purpose, runs under intervention). That's fine — its purpose is establishing the causal direction→commitment→ID link, not comparing lenses.

### What the one pass produces (ALL from extended `eval_commitment(collect_full=True)`):
| Data | Lens | Group | Models |
|------|------|-------|--------|
| Commitment scalars (7 methods × thresholds) | Raw + Tuned | A | All 6 × PT/IT |
| Per-layer top1 tokens | Raw + Tuned | B | All 6 × PT/IT |
| Per-layer KL-to-final | Raw + Tuned | C | All 6 × PT/IT |
| Per-layer adjacent-KL | Raw + Tuned | C | All 6 × PT/IT |
| Per-layer next-token prob/rank | Raw + Tuned | D | All 6 × PT/IT |
| Per-layer top-5 tokens (first 200 prompts) | Raw + Tuned | E | All 6 × PT/IT |
| Per-layer entropy | Raw + Tuned | F | All 6 × PT/IT |
| δ-cosine (residual stream) | N/A | G | All 6 × PT/IT |
| Cosine-to-final h_ℓ · h_L | N/A | (already saved) | All 6 × PT/IT |
| Generated text + token IDs | N/A | B | All 6 × PT/IT |

---

## Full Experiment → Data → Plot Matrix

Every experiment gets BOTH raw logit-lens and tuned-lens results from the same pass. This means every plot can be produced in two variants (raw vs tuned) for robustness comparison.

| # | Experiment | Data Groups | What raw lens shows | What tuned lens adds | Plot file |
|---|-----------|-------------|--------------------|--------------------|-----------|
| 0G | Commitment delay comparison | A | Baseline commitment (7 methods) | Does delay persist under tuned lens? | `plot_tuned_lens_commitment.py` (EXISTS) |
| 1.1 | Mind-change targeting | B | 75% structural targeting at corrective layers? | Same pattern → not logit-lens artifact | `plot_mind_change_multimodel.py` (NEW) |
| 1.2 | Confidence-stratified commitment | A + D | +2.2 high-conf, +6.6 low-conf? | Same scaling → not logit-lens artifact | `plot_confidence_stratified_multimodel.py` (NEW) |
| 1.3 | Adjacent-layer KL (three-phase) | C | Three discrete prediction revision phases? | Same phase structure → weight-intrinsic | `plot_adjacent_kl_multimodel.py` (NEW) |
| 1.4 | Gen-step × layer heatmaps | G | Temporal stability of δ-cosine opposition | N/A (δ-cosine is residual-stream, no lens) | `plot_cross_model.py` (UPDATE to read new format) |
| 1.5 | Cross-layer correlation | B + G | Spearman(δ-cosine, mind-change rate) < -0.5? | Same correlation with tuned mind-change | `plot_cross_layer_corr.py` (NEW) |
| 1.6 | Per-category commitment | A + metadata | Delay by prompt type (factual/code/safety/...) | Same category pattern | `plot_per_category_commitment.py` (NEW) |
| — | Candidate reshuffling | E | Top-5 set reshuffled at corrective layers? | Same reshuffling → weight-intrinsic | `plot_reshuffling_multimodel.py` (NEW) |
| — | Entropy profiles | F | IT lower early, higher at corrective? | Tuned entropy profile comparison | `plot_entropy_profiles.py` (NEW) |
| — | Token emergence | D | Generated token rank evolution across layers | Tuned-lens rank emergence comparison | `plot_emergence_multimodel.py` (NEW) |

**Total: 8 new plot scripts + 2 updates to existing** (`plot_tuned_lens_commitment.py` already exists; `plot_cross_model.py` needs format update)

### Plots per experiment (detailed):

**1.1 Mind-change** (4 plots):
1. `mind_change_targeting_raw_6panel.png` — structural targeting % by normalized layer, PT vs IT, raw lens
2. `mind_change_targeting_tuned_6panel.png` — same, tuned lens
3. `mind_change_raw_vs_tuned_scatter.png` — per-model: is raw targeting % ≈ tuned targeting %?
4. `mind_change_summary_bar.png` — structural targeting % at corrective layers, all 6 models × 2 lenses

**1.2 Confidence-stratified** (3 plots):
5. `confidence_stratified_raw_6panel.png` — commitment delay by confidence quartile, PT vs IT, raw
6. `confidence_stratified_tuned_6panel.png` — same, tuned lens
7. `confidence_scaling_summary.png` — Δ(delay) vs confidence bin slope, per model × lens

**1.3 Adjacent-KL / three-phase** (3 plots):
8. `adjacent_kl_raw_6panel.png` — KL(ℓ‖ℓ-1) by layer, PT vs IT, raw
9. `adjacent_kl_tuned_6panel.png` — same, tuned
10. `three_phase_summary.png` — mean KL by phase × model × lens

**1.4 Heatmaps** (1 plot, already exists in principle):
11. `delta_cosine_heatmap_6x2.png` — update `plot_cross_model.py` to read from new arrays

**1.5 Cross-layer correlation** (1 plot):
12. `cross_layer_corr_6panel.png` — scatter + Spearman ρ per model

**1.6 Per-category** (1 plot):
13. `per_category_commitment_heatmap.png` — category × model, raw and tuned

**Reshuffling** (2 plots):
14. `reshuffling_jaccard_raw_6panel.png` — top-5 Jaccard by layer
15. `reshuffling_jaccard_tuned_6panel.png` — same, tuned

**Entropy** (1 plot):
16. `entropy_profiles_6panel.png` — raw + tuned entropy by layer, PT vs IT

**Emergence** (1 plot):
17. `emergence_rank_6panel.png` — generated token rank by layer, PT vs IT, raw + tuned

---

## Summary of Codebase Corrections from v1

| Issue | v1 (incorrect) | v2 (corrected) |
|-------|---------------|----------------|
| KL thresholds | 3 (0.05, 0.1, 0.2) | **5** (0.05, 0.1, 0.2, 0.5, 1.0) |
| Cosine thresholds | Not mentioned | **4** (0.80, 0.90, 0.95, 0.99) |
| Commitment methods | 5 | **7+** (raw top1, raw KL, tuned KL, tuned top1, majority, cosine, entropy) + qualified variants |
| Dataset | "1,400 prompts" | **2,936 prompts** (`exp3_dataset.jsonl` for tuned-lens eval) |
| Eval runner | Local only | **Modal cloud** (`modal_eval_0G.py`) or local |
| Ranks already computed | "NEW — needs computation" | **Already computed** in `_process_step()` for qualified commitment (lines 1471-1485) |
| Entropy already computed | "NEW" | **Raw entropy already computed** (line 1531) for entropy commitment; only tuned entropy is new |
| Phase 0 logit-lens | Not mentioned | **Separate pipeline** — `runtime.py` collects raw top1 UNDER STEERING; distinct from unsteered eval |
| GPU policy | "GPUs 0-7" | **GPUs 2-7** (0-1 reserved per updated CLAUDE.md) |
| GPUs | "A100" | **H100** 80GB (per updated CLAUDE.md); Modal uses H100 with A100 fallback |
| `cosine_row` | "Not saved" | **Already saved** as `step_cosine_values` (line 1525) |

---

## Overhead Estimate

Only truly new computation (everything else is `list.append()` on existing variables):
- Adjacent-layer KL: reuse `all_raw_log_q` (line 1407) and `all_log_q` (line 1434) — 2 × (n_layers-1) KL reductions — **<3%**
- Tuned entropy: one softmax + entropy reduction on `all_tuned_logits` — **<2%**
- Next-token prob: index into existing probs — **<0.5%**
- Top-5 (200 prompts): `torch.topk` on existing probs — **~3% for those prompts**
- δ-cosine: n_layers dot products on d_model vectors from `step_buf` — **<1%**
- I/O: NPY writes at end — **negligible**

**Total: ~5-8% overhead** on top of existing eval_commitment() runtime. The generation itself (autoregressive decoding with KV cache) remains the bottleneck by far.

---

## ✅ One-Pass Verification Checklist

Before running, verify these are true:
- [ ] `eval_commitment()` has `collect_full` and `collect_top5` params
- [ ] `_process_step()` saves `raw_top1_row`, `tuned_top1_row`, `raw_kl_row`, `kl_row`, `raw_ranks`, `tuned_ranks`, `entropy_row` to per-prompt accumulators
- [ ] `_process_step()` computes adjacent-layer KL from `all_raw_log_q` and `all_log_q`
- [ ] `_process_step()` computes tuned entropy from `all_tuned_logits`
- [ ] `_process_step()` computes next-token prob from `all_raw_probs[:, final_top1]` and tuned equivalent
- [ ] `_process_step()` computes δ-cosine from `step_buf` residuals
- [ ] NPY arrays saved at end of generation loop with `step_index.jsonl` for prompt→step mapping
- [ ] `modal_eval_0G.py` passes `collect_full=True` to `eval_commitment()`
- [ ] `plot_cross_model.py` updated to read from new format instead of `L1L2_results.jsonl`
- [ ] All 17 plots listed above have corresponding plot scripts

---

## Summary: What This One Pass Replaces

| Previous approach | Runs needed | What it collected |
|-------------------|-------------|-------------------|
| `collect_L1L2.py` per model × variant | 12 runs | δ-cosine, raw top1, raw entropy, raw commitment |
| `exp3/collect.py` (Gemma only) | 2 runs | rank, prob, adj-KL, top5, step-KL |
| `eval_commitment()` (current) | 12 runs | 60+ commitment scalars (raw + tuned) |
| **Total separate runs** | **26** | — |

| New approach | Runs needed | What it collects |
|-------------|-------------|------------------|
| `eval_commitment(collect_full=True)` | **12 runs** | ALL of the above + tuned-lens equivalents for ALL 6 models |

**One pass. Both lenses. All 6 models. All experiments.**
