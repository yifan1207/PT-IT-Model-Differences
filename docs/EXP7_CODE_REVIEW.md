# Exp7 Execution Plan Review v2 — Deep Dive on 0G + Full Pipeline

**Date:** 2026-03-28
**Scope:** Full execution plan + deep 0G SOTA analysis
**Key correction:** Several issues from v1 review are already fixed in the codebase — this review reflects the actual code, not the summary.

---

## Critical Corrections from v1 Review

Before diving in, three important corrections based on re-reading the actual code:

1. **`force_decode_acts.py` ALREADY uses KV cache.** Lines 94-105 correctly implement `use_cache=True` on the prefill pass and `past_key_values=past_kv` on subsequent steps. **No fix needed** — this was a stale finding from a prior version.

2. **`exp7/tuned_lens_probes.py` already has many v1 fixes applied:**
   - `n_train` default is now 10,000 (line 719), yielding ~1M+ tokens with prefill
   - Prefill tokens ARE collected (the `shape[1]==1` filter was removed — see `collect_hidden_states()` which does a single forward pass and captures all positions except BOS)
   - Weight decay = 1e-3 is present (line 265)
   - Gradient clipping at 1.0 is present (line 295)
   - C4 validation data loading with gen_merged.jsonl fallback

3. **The PRIMARY 0G implementation is `src/poc/cross_model/tuned_lens.py`** (1,029 lines), NOT `exp7/tuned_lens_probes.py`. The cross-model version is what `run_cross_model_tuned_lens.sh` calls and what will actually run for all 6 families.

---

## Deep Dive: 0G Tuned Lens — SOTA Alignment

### What Belrose et al. (2023) Actually Specifies

| Aspect | Belrose et al. | Our Implementation | Gap |
|--------|---------------|-------------------|-----|
| **Optimizer** | SGD + Nesterov momentum | Adam (lr=1e-3, wd=1e-3) | **Divergent** — see below |
| **Learning rate** | 1.0 (0.25 with final layer) | 1e-3 | **Divergent** — coupled with optimizer choice |
| **LR schedule** | Linear decay → 0 over 250 steps | Cosine annealing + 5% warmup over 5000 steps | **Divergent** |
| **Training steps** | 250 | 5000 | **Divergent** — 20× more steps |
| **Gradient clipping** | norm = 1.0 | norm = 1.0 | ✅ Match |
| **Loss** | KL divergence | KL divergence (batchmean, log_target) | ✅ Match |
| **Initialization** | Identity + zero bias | Identity + zero bias | ✅ Match |
| **Architecture** | Linear(d, d, bias=True) | Linear(d, d, bias=True) | ✅ Match |
| **Hook point** | Residual stream (layer output) | Residual stream (layer output) | ✅ Match |
| **Inference** | norm(probe(h)) → lm_head | norm(probe(h)) → lm_head | ✅ Match |
| **Training data** | Pile validation (~16-33M tokens) | C4 validation (100k tokens default) | **~100-300× gap** |
| **Token positions** | All positions in chunks | All positions (prefill-based) | ✅ Match |
| **Validation** | Not mentioned | 80/20 split + early stopping | ✅ Better |
| **Transfer test** | Shown for checkpoints | PT↔IT transfer | ✅ Novel |

### Analysis of the Optimizer Divergence

This is the most significant departure from Belrose. Their recipe is:

```
SGD + Nesterov momentum, lr=1.0, linear decay over 250 steps
```

Ours is:

```
Adam, lr=1e-3, cosine annealing + warmup over 5000 steps
```

**Is this defensible?** Yes, with proper justification:

- Belrose's SGD+high-LR+fast-schedule is viable because the identity initialization means the probe starts very close to the optimum. SGD with momentum at lr=1.0 takes large steps that are fine when you're making small corrections to an identity map.
- Adam at lr=1e-3 is more conservative but more robust. With 5000 steps, it has time to find equally good (potentially better) minima. The adaptive learning rates handle per-parameter scaling that SGD relies on manual tuning for.
- The `tuned-lens` package on GitHub also supports Adam — it's not an uncommon choice.

**Recommendation:** Keep Adam, but **add Belrose's SGD config as a CLI flag** (`--optimizer {adam,sgd_nesterov}`) so you can report results under both optimizers. If they agree → robustness confirmed. If they disagree → investigate which is better calibrated.

### The Real Issue: Training Data Volume

**This is the only remaining critical problem.**

Belrose's 250 steps × batch_size × seq_len ≈ 16-33M training tokens. Our cross_model/tuned_lens.py collects 100k tokens and stores them in CPU RAM, then samples mini-batches from this pool for 5000 steps.

With 100k tokens, batch_size=64, and 5000 steps: each token is seen ~3.2 times on average. This isn't catastrophic (it's a linear probe with strong identity prior), but it's nowhere near the diversity Belrose achieves.

**The issue isn't overfitting per se — it's distributional coverage.** 100k tokens from C4 covers a narrow slice of the model's representational space. Belrose's 16M+ tokens spans a much broader distribution, ensuring the affine probe learns a generally valid transformation.

**For a NeurIPS submission, the current 100k is risky.** A reviewer can legitimately ask: "How do you know your probes generalize beyond the 100k tokens they were trained on?"

**Recommended fix for `cross_model/tuned_lens.py`:**

Option A (preferred — streaming): Don't store all activations. Instead, stream batches:
```python
for step in range(n_steps):
    # Forward-pass a fresh batch of texts through the model
    batch_texts = next(text_iterator)  # fresh C4 texts each step
    with torch.no_grad():
        h_layer, h_final = forward_batch(model, batch_texts, layer=li)
    # Train probe on this fresh batch
    loss = kl_loss(probe(h_layer), h_final)
    ...
```
This gives 5000 × 64 × ~100 tokens/text = ~32M tokens of diversity — matching Belrose.

**GPU cost:** One extra forward pass per training step per layer. For 5000 steps × 33 layers = 165k forward passes. At ~50ms each on a 4B model = ~2.3 hours per variant. With 12 variants = ~28 GPU-hours. Manageable on 8 GPUs in ~4 hours.

Option B (simpler — larger pool): Increase `--n-tokens` from 100k to 2M. Memory cost: 2M × 2560 × 4 × 33 layers ≈ 650 GB CPU RAM. **Not feasible.**

Option C (compromise — large pool + subsampling): Collect 500k tokens (feasible at ~165 GB CPU RAM for 33 layers), sample from this pool. Each token seen 5000 × 64 / 400k ≈ 0.8 times. This provides 5× more diversity than current setup.

**My recommendation: Option A (streaming) if the infrastructure supports it, Option C otherwise.** Either way, the cross_model shell script should pass `--n-tokens 500000` at minimum.

### Multi-Threshold Commitment Evaluation

The code evaluates at three KL thresholds: [0.05, 0.1, 0.2]. This is good — it maps directly to the 0J sensitivity analysis. But there's a subtlety:

**The commitment metric uses a no-flip-back criterion:** the KL must stay below threshold for ALL subsequent layers. This is strict — a single noisy layer can push commitment arbitrarily late. Belrose reports a softer variant in some analyses.

**Recommendation:** Also compute a "majority" commitment metric: earliest layer where KL < threshold for ≥90% of subsequent layers. Report both. If they agree → robust. If strict is much later → indicates noisy late layers, which is informative.

### Transfer Test Design

The cross_model implementation tests PT probes on IT activations with a transfer ratio threshold of 2.0. This is well-designed, but:

**Missing condition:** IT probes on PT activations. The asymmetry matters — if IT probes fail on PT but PT probes work on IT, it suggests IT adds structure that PT lacks (consistent with the delayed commitment narrative). The exp7 version has this condition; make sure the cross_model version does too.

**Also missing:** Probe performance vs. logit lens baseline per layer. Belrose's key result is that tuned lens substantially outperforms logit lens at early layers but converges at late layers. Reporting this per-layer delta for PT and IT separately would reveal whether IT's late layers are "harder to probe" (suggesting more complex computation).

---

## Execution Plan Review — Per-Experiment

### Step 1: 0G (Cross-model tuned lens) — 8-10 hr estimate

**Shell script:** `run_cross_model_tuned_lens.sh`

**Current plan:**
- Phase 1 (~2 hr): Train 12 probe sets, 100k tokens each
- Phase 2 (~6-8 hr): Evaluate commitment on 1400 records × 512 tokens
- Phase 3 (~20 min): Transfer tests for all 6 models

**Issues with the plan:**

1. **Phase 1 timing is underestimated if data volume increases.** At 500k tokens, collection alone takes ~5× longer (5 forward passes per text × 500k/100 texts = ~5000 forward passes per variant). Estimate: ~1 hour collection + ~1 hour training per variant = ~24 GPU-hours total. On 8 GPUs with batching: ~3-4 hours.

2. **Phase 2 timing depends on generation speed.** 1400 records × 512 tokens is a LOT. At ~50 tokens/sec for a 4B model: 1400 × 512 / 50 = ~14,336 seconds = ~4 hours per variant. With 12 variants: ~48 GPU-hours. On 8 GPUs: ~6 hours. **The 6-8 hr estimate is correct but tight.**

3. **GPU scheduling in the shell script:** Batch 1 runs 4 models in parallel (2 GPUs each for PT+IT). Batch 2 runs OLMo. Batch 3 runs DeepSeek (multi-GPU). This is reasonable but DeepSeek-V2-Lite with MoE may need special handling for the hook placement — verify that `residual_from_output()` in the DeepSeek adapter correctly extracts the post-MoE residual stream, not individual expert outputs.

4. **The `max_new_tokens=512` in Phase 2 is excessive.** Most of your prompts generate 40-80 tokens before EOS. Setting max_new_tokens=512 means the model runs until EOS naturally, which is fine, but the shell script time estimate assumes 512 tokens per record. Actual time will be much less.

**Recommendations for 0G:**

- Increase `--n-tokens` to at least 500,000 (or implement streaming — see above)
- Add `--optimizer sgd_nesterov` option for ablation comparison
- Add "majority" commitment metric alongside strict no-flip-back
- Ensure IT→PT transfer test is included in Phase 3
- Add per-layer tuned-vs-logit-lens delta reporting

### Step 2: 0D (Bootstrap CIs) — CPU ~15 min

**Status:** Code is solid. n_resamples=10,000, BCa method, Cohen's d, Spearman ρ.

**No issues.** This can run on CPU while 0G uses GPUs. Good parallelization.

### Step 3: 0E (Token classifier robustness) — CPU ~5 min

**Status:** Solid. Cohen's kappa, perturbation tests, LLM judge.

**One note:** The LLM judge uses Claude via OpenRouter. Make sure the API key is configured and rate limits won't be a bottleneck.

### Step 4: 0A (Direction calibration) — 8 GPU ~15 min

**Status:** `collect_per_record_acts.py` → `bootstrap_directions.py`

**Check:** The convergence curve tests subset sizes [100, 200, 300, 400, 500, 600, 1000, 1400]. This is good but **1400 is the full dataset — this direction should be treated as the reference, not a bootstrap sample.** Verify that the bootstrap comparisons at size 1400 are comparing to the canonical direction, not self-similarity.

### Step 5: 0B (Force-decode) — 8 GPU ~20 min

**Status:** KV cache is properly implemented. No performance issue.

**The 20 min estimate is reasonable.** 600 records × 80 tokens, with KV cache, ~50ms per step = 600 × 80 × 0.05 = 2400 seconds / 8 GPUs = 5 min per variant × 2 variants = 10 min. The 20 min includes overhead — fine.

### Step 6: 0H (Calibration-evaluation split) — 8 GPU ~45 min

**Status:** `precompute_random_split.py` has the direction computation bug (token-count vs record-count normalization).

**MUST FIX BEFORE RUNNING.** The bug biases directions toward records with longer generations. Fix: normalize sums by number of records, not token counts. Or use `extract_subset_acts()` from `collect_unified_acts.py` to get per-record means first, then average.

### Step 7: 0C (Projection-matched random control) — 8 GPU ~1.5 hr

**Status:** Uses existing intervention infrastructure. Should be straightforward.

**Check:** The projection-matching function `_project_remove_magnitude_matched()` in exp6/interventions.py — verify it scales the random direction perturbation to match the corrective direction's per-token projection magnitude, not just the mean magnitude.

### Step 8: 0I (Intervention formula sensitivity) — 8 GPU ~1.5 hr

**Status:** Tests 3 alternative formulas. Good design.

**Check:** Alternative B (residual stream intervention) hooks at a different point than the current MLP-output intervention. Make sure the hook registration correctly captures residual stream state after MLP addition, not just MLP output.

### Step 9: 0F (Layer range analysis) — depends on A1 runs

**Status:** Solid. 4 layer range variants + single-layer sweep.

**No issues.** This depends on having A1 results for each layer range, so timing is accurate.

---

## Specific Code Edits for 0G

### Edit 1: Increase default `--n-tokens` in cross_model/tuned_lens.py

The default should be at least 500,000. Currently 100,000.

**Location:** CLI argument in `main()` function of `src/poc/cross_model/tuned_lens.py`
**Change:** `--n-tokens` default from `100000` to `500000`

### Edit 2: Add streaming option to avoid memory constraints

If 500k tokens × 33 layers × d_model × 4 bytes exceeds available CPU RAM (~165 GB for Gemma), add a `--streaming` flag that does fresh forward passes each training step instead of pre-collecting all activations.

### Edit 3: Add SGD+Nesterov optimizer option

```python
if args.optimizer == "sgd_nesterov":
    optimizer = torch.optim.SGD(probe.parameters(), lr=1.0, momentum=0.9, nesterov=True)
    n_steps = 250  # Match Belrose
    schedule = "linear_decay"
else:  # adam (default)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-3)
    n_steps = args.n_steps
    schedule = "cosine_warmup"
```

Report results under both optimizers. If they agree → robustness.

### Edit 4: Add majority-vote commitment metric

```python
def _commitment_majority(kl_row, threshold=0.1, frac=0.9):
    """Earliest layer where ≥frac of subsequent layers have KL < threshold."""
    n = len(kl_row)
    for i in range(n):
        subsequent = kl_row[i:]
        if sum(1 for k in subsequent if k < threshold) / len(subsequent) >= frac:
            return i
    return n - 1
```

### Edit 5: Add per-layer tuned-vs-raw delta

In the evaluation phase, compute both raw logit lens KL and tuned lens KL at every layer. Report the delta — this shows where the tuned lens adds the most value (should be early/mid layers per Belrose).

### Edit 6: Fix shell script token count

In `run_cross_model_tuned_lens.sh`, change the `--n-tokens` argument from 100000 to 500000 (or whatever the final default is).

---

## Remaining precompute_random_split.py Bug Fix

The direction computation at line 327:
```python
vec = it_sums[li] / n_it - pt_sums[li] / n_pt
```

Where `n_it` and `n_pt` are **token counts** (accumulated via `int(d[f"it_count_r_{li}"])`).

If records have different generation lengths, this biases toward longer-generation records. The correct approach is:

**Option A:** Track record counts separately from token counts. Divide sums by record count.

**Option B (simpler):** Use `extract_subset_acts()` to get per-record mean activations, then average across records. This gives equal weight to each record regardless of generation length.

Recommend Option B — the utility already exists and is correct.

---

## Overall Execution Plan Assessment

| Exp | Code Ready? | Estimated Time | Blockers |
|-----|------------|---------------|----------|
| 0G | **Needs edits** — data volume, optimizer option | 8-12 hr (revised up) | Edit 1-6 above |
| 0D | ✅ Ready | 15 min CPU | None |
| 0E | ✅ Ready | 5 min CPU + API calls | OpenRouter API key |
| 0A | ✅ Ready | 15 min 8-GPU | None |
| 0B | ✅ Ready | 20 min 8-GPU | None |
| 0H | **Needs fix** — direction bias bug | 45 min 8-GPU | Fix token-vs-record normalization |
| 0C | ✅ Ready | 1.5 hr 8-GPU | Needs completed A1 baselines |
| 0I | ✅ Ready | 1.5 hr 8-GPU | Needs completed A1 baselines |
| 0F | ✅ Ready | Variable | Needs A1 runs at each range |

**Critical path:** 0G is the longest job and the #1 priority. Start it first with the edits above. Everything else can run in parallel on remaining GPUs or CPU.

**Total wall-clock estimate:** ~2 days with 8 GPUs, assuming 0G takes 10-12 hours and everything else fits in the gaps.
