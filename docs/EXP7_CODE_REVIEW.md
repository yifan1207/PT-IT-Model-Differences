# Exp7 Codebase Review — Consolidated Findings & Actionable Fixes

**Date:** 2026-03-28
**Scope:** All 11 Python files in `src/poc/exp7/`
**Focus:** Literature alignment (especially tuned lens), rigor, correctness, performance

---

## Executive Summary

The exp7 codebase is well-structured overall, with clean parallelization patterns, proper hook placement for direction extraction (MLP outputs), and solid statistical methods (BCa bootstrap, Cohen's d). However, three issues require immediate attention before running final experiments:

1. **CRITICAL — Tuned lens training data ~100× too small** (160k tokens vs literature's 16.4M)
2. **BUG — `precompute_random_split.py` direction computation uses proportional scaling instead of actual subset filtering**
3. **PERFORMANCE — `force_decode_acts.py` lacks KV cache, causing quadratic compute**

Below is a per-file assessment followed by prioritized action items.

---

## Per-File Assessments

### 1. `tuned_lens_probes.py` (776 lines) — Exp 0G

**Alignment with NEXT_STEPS_v12:** Partially aligned. Supports all 6 model families, has PT→IT transfer test, uses KL loss and identity init. But training data volume is far below literature standards.

**What's correct (per Belrose et al., 2023):**
- Probe architecture: `nn.Linear(d_model, d_model, bias=True)` with identity weight init, zero bias — matches Belrose exactly
- Loss function: `F.kl_div(log_pred, target, reduction="batchmean", log_target=True)` — correct
- Inference pipeline: `final_norm(probe(h))` → `lm_head(normed)` — correct sequencing
- Hooks on full layer output (residual stream), not MLP — fixed from v1, now correct
- Adam optimizer, lr=1e-3 — matches literature
- Cosine annealing with warmup — good addition (Belrose used constant LR, but cosine is standard practice)
- Train/val split (80/20) with best-model checkpointing — good addition
- Cross-model probe transfer test (PT probes on IT, IT probes on PT) — exactly what 0G needs

**CRITICAL issues:**

**(a) Training data volume: ~100× too small**

Current: `n_train=2000` prompts × ~80 generated tokens = ~160k tokens.
Literature: Belrose et al. train on **16.4M tokens** from the Pile validation set, chunked at seq_len=2048. The `tuned-lens` package default is 50k chunks × 2048 = ~100M tokens.

This matters because tuned lens probes need to learn a *per-layer affine transform* that maps intermediate representations to the final unembedding space. With only 160k tokens, the probe may overfit to the specific distribution of your 2000 prompts rather than learning the true layer-to-output mapping. This directly undermines the validity of commitment onset measurements.

**Fix:** Increase to at minimum 10,000 prompts (yielding ~800k tokens with prefill included — see issue b). Ideally use 50,000+ prompts from a diverse corpus (e.g., C4, RedPajama, or the Pile validation set) to reach ~1M+ tokens. NEXT_STEPS_v12 already specifies "50,000+ tokens per variant" — the code should match this.

**(b) Prefill tokens discarded — only generated tokens collected**

Lines ~420-430 (the hook) filter with `if hidden.shape[1] == 1`, which only captures autoregressive generation steps (one token at a time). This discards ALL prefill positions where the model processes the prompt in parallel (shape[1] > 1).

Belrose et al. train on **all token positions** — every position in every chunk contributes a training example. Discarding prefill means:
- You lose ~90% of your already-small training data (prompts are ~50-100 tokens, generation is ~80)
- The probe never sees how the model represents tokens it processes in parallel, only sequential generation
- This creates a distributional mismatch between training and evaluation

**Fix:** Remove the `shape[1] == 1` filter. Collect hidden states at ALL positions during both prefill and generation. Each position is an independent training example for the affine probe.

**(c) No weight decay or gradient clipping**

The Belrose implementation uses weight decay (1e-2 in some configurations). Without it, the identity-initialized probe weights can drift far from identity, potentially learning degenerate mappings. Gradient clipping (max_norm=1.0) is also standard practice for transformer training.

**Fix:** Add `weight_decay=1e-2` to Adam optimizer and `torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)` before optimizer step.

---

### 2. `force_decode_acts.py` (294 lines) — Exp 0B

**Alignment with NEXT_STEPS_v12:** Well aligned. Supports both forward (PT on IT tokens) and reverse (IT on PT tokens) directions, governance and random record sets.

**PERFORMANCE issue — No KV cache, quadratic compute:**

Lines 117-121:
```python
for t in range(len(gen_ids)):
    forced = torch.tensor([gen_ids[:t + 1]], dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_ids, forced], dim=1)
    with torch.no_grad():
        model_raw(input_ids)
```

For each step t, the entire sequence (prompt + gen_ids[:t+1]) is re-processed from scratch. With prompt length P and generation length K, this is O(K × (P+K)²) attention operations instead of O(K × (P+K)) with KV cache. For P=50, K=80, that's ~50× slower than necessary.

**Fix:** Use HuggingFace's `past_key_values` / `use_cache=True`:
```python
past_kv = None
# First: process prompt
with torch.no_grad():
    out = model_raw(prompt_ids, use_cache=True)
    past_kv = out.past_key_values

# Then: step through forced tokens one at a time
for t in range(len(gen_ids)):
    tok = torch.tensor([[gen_ids[t]]], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model_raw(tok, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
```

**Caveat:** Verify that MLP hooks still fire correctly with cached attention. They should — the MLP at each layer still executes for the new token — but test on a small batch first.

**Minor:** The hook captures `out[0, -1, :]` which is correct for the last position. With KV cache, the output tensor has shape (1, 1, d_model), so `out[0, -1, :]` and `out[0, 0, :]` are equivalent. No change needed.

---

### 3. `precompute_random_split.py` (385 lines) — Exp 0H

**Alignment with NEXT_STEPS_v12:** Conceptually aligned (random-600 / held-out-800 / bottom-600 splits), but has a correctness bug.

**BUG — Direction computation uses proportional scaling instead of subset filtering (lines ~326-337):**

The code computes directions for random-600 by scaling the *total* sum of activations by the fraction of records in the random subset, rather than summing only the activations belonging to records in that subset. Concretely:

```python
r_frac = in_r.sum() / max(n_rec, 1)
it_sum = it_total_sum * r_frac  # WRONG: scales total, not filters
pt_sum = pt_total_sum * r_frac
```

This assumes random-600 records have the same mean activation as the full 1400, which defeats the purpose of testing direction stability on a different subset.

**Fix:** Use `collect_unified_acts.extract_subset_acts()` to get the actual activations for the random-600 record IDs, then compute `mean_IT - mean_PT` from those filtered activations directly. The utility function already exists and handles the record-ID matching correctly.

---

### 4. `bootstrap_directions.py` (415 lines) — Exp 0A + 0B analysis

**Alignment:** Excellent. Implements exactly what 0A specifies.

**Assessment:** Solid implementation.
- Per-layer bootstrap with pairwise cosine similarity is the right approach for direction stability
- Convergence curve with subset sizes [100, 200, 300, 400, 500, 600, 1000, 1400] × 20 draws
- OOD direction test properly compares governance-selected vs random-600 directions
- Layer group summaries (early/mid/corrective) match the paper's framing
- Matched-token comparison with corrective-vs-early drop flagging for 0B

**No issues found.**

---

### 5. `bootstrap_ci.py` (586 lines) — Exp 0D

**Alignment:** Excellent. BCa bootstrap CIs, effect sizes, monotonicity — all per spec.

**Assessment:** Strong statistical rigor.
- BCa (bias-corrected accelerated) method via `scipy.stats.bootstrap` — gold standard for small-sample CIs
- Cohen's d with magnitude interpretation (small/medium/large)
- Spearman ρ for α-sweep monotonicity
- Programmatic scorers for all benchmarks (STR, format_compliance_v2, reasoning_em, alignment_behavior, MMLU, GSM8K)
- Cross-model CIs bootstrapped over prompts (correct unit of analysis)

**Minor suggestion:** Consider adding bootstrap sample size as a configurable parameter (currently hardcoded). For the paper, 10,000 bootstrap samples is conventional; verify this is the default being used.

---

### 6. `collect_unified_acts.py` (398 lines) — Exp 0A + 0H activation collection

**Alignment:** Good. Combined collection for all 1400 records with 8-GPU parallelization.

**Assessment:** Clean implementation.
- Hooks on `.mlp` — correct for direction extraction (MLP outputs, not residual stream)
- Worker slicing and merge step well-implemented
- `extract_subset_acts()` utility is properly designed for downstream subsetting
- Verification against canonical precompute_v2 directions

**No issues found.**

---

### 7. `collect_per_record_acts.py` (262 lines) — Per-record activation collection

**Alignment:** Good. Straightforward data collection pipeline.

**Assessment:** Clean, minimal code. Hooks on `.mlp` for MLP output activations. 8-GPU parallelization with merge step matches the pattern in other files.

**No issues found.**

---

### 8. `token_classifier_robustness.py` (638 lines) — Exp 0E

**Alignment:** Excellent. Comprehensive classifier validation.

**Assessment:** Thorough robustness testing.
- Cohen's kappa for inter-rater agreement (LLM judge vs rule-based classifier)
- Two perturbation tests: STRUCTURAL↔DISCOURSE and FUNCTION↔CONTENT boundaries
- Enrichment ratios with proper baseline comparison
- Human template CSV generation for manual validation
- Claude-based LLM judge via OpenRouter

**Minor:** The LLM judge prompt should be documented/versioned somewhere accessible for reproducibility. Consider saving the exact prompt text alongside results.

---

### 9. `layer_range_analysis.py` (373 lines) — Exp 0F

**Alignment:** Good. 4 layer range variants + single-layer importance sweep.

**Assessment:** Clean analysis pipeline.
- 3-panel sensitivity plot (governance/content/safety) is well-designed
- Per-layer importance ranking by STR delta
- Proper handling of missing runs (graceful skip with warning)

**No issues found.**

---

### 10. `onset_threshold_sensitivity.py` (460 lines) — Exp 0J

**Alignment:** Good. Multi-threshold sensitivity analysis.

**Assessment:** Solid implementation.
- 5 σ-thresholds (0.5, 0.75, 1.0, 1.5, 2.0) + 4 absolute thresholds (0.02, 0.05, 0.10, 0.15)
- Consecutive-layer requirement (n_consecutive=2) is a good noise-reduction strategy
- Baseline from stable mid-early region (layers 6-14) is well-motivated
- Gemma alt-ranges extraction for A1 reruns

**No issues found.**

---

### 11. `__init__.py` — Package init

Trivially fine.

---

## Prioritized Action Items

### P0 — Must fix before running experiments

| # | File | Issue | Effort |
|---|------|-------|--------|
| 1 | `tuned_lens_probes.py` | Increase training data to 10,000+ prompts (~1M+ tokens) | ~1 hour code + rerun time |
| 2 | `tuned_lens_probes.py` | Remove `shape[1]==1` filter — collect ALL token positions (prefill + generation) | ~30 min |
| 3 | `precompute_random_split.py` | Replace proportional scaling with actual subset filtering via `extract_subset_acts()` | ~30 min |

### P1 — Should fix (affects rigor or performance)

| # | File | Issue | Effort |
|---|------|-------|--------|
| 4 | `tuned_lens_probes.py` | Add weight_decay=1e-2 and gradient clipping (max_norm=1.0) | ~10 min |
| 5 | `force_decode_acts.py` | Add KV cache to avoid quadratic compute | ~1 hour |

### P2 — Nice to have

| # | File | Issue | Effort |
|---|------|-------|--------|
| 6 | `bootstrap_ci.py` | Make bootstrap n_resamples configurable, document default | ~10 min |
| 7 | `token_classifier_robustness.py` | Version the LLM judge prompt alongside results | ~15 min |

---

## Tuned Lens Literature Alignment Summary

| Aspect | Belrose et al. (2023) | Current code | Status |
|--------|----------------------|--------------|--------|
| Architecture | Linear(d, d, bias=True) | Same | ✅ |
| Init | Identity weight, zero bias | Same | ✅ |
| Loss | KL divergence (batchmean) | Same | ✅ |
| Optimizer | Adam, lr=1e-3 | Same | ✅ |
| LR schedule | Constant | Cosine + warmup | ✅ (better) |
| Training data | 16.4M tokens (Pile val) | ~160k tokens | ❌ ~100× too small |
| Token positions | All positions in chunks | Generated only (shape[1]==1) | ❌ Discards ~90% |
| Weight decay | 1e-2 (some configs) | None | ⚠️ |
| Gradient clipping | Standard practice | None | ⚠️ |
| Validation split | Not in original paper | 80/20 with checkpointing | ✅ (better) |
| Hook placement | Residual stream (layer output) | Same | ✅ |
| Inference | norm(probe(h)) → lm_head | Same | ✅ |
| Transfer test | Not in original | PT↔IT probe transfer | ✅ (novel) |

---

## Overall Rigor Assessment

**Strengths:**
- Statistical methods are sound (BCa bootstrap, Cohen's d, Spearman ρ)
- Parallelization pattern (8-GPU workers + merge) is consistent and well-tested
- Hook placement is correct for each experiment's purpose (MLP for direction extraction, residual stream for tuned lens)
- The intervention formula and layer groups align with the paper draft
- Robustness checks (0E classifier, 0F layer range, 0J threshold sensitivity) cover the major reviewer concerns

**Weaknesses:**
- The tuned lens training data issue is the single biggest threat to experiment validity. A reviewer could dismiss the 0G results entirely if the probes are undertrained.
- The precompute_random_split bug would silently produce incorrect OOD directions, undermining 0H's claim that directions generalize beyond governance-selected records.
- Force-decode without KV cache is a practical blocker — it may be too slow to run on 600 records × 80 steps on reasonable timescales.

**Recommendation:** Fix P0 items 1-3 before any experiment runs. Items 4-5 can be done in parallel with early experiments. The rest of the codebase is solid and ready to execute.
