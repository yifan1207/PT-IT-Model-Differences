# Next Steps v12 — Post-Citation-Fix, Strategic Priorities Revised
**Date**: 2026-03-28
**Status**: Paper draft v5 complete. Citations fixed (Huang→Ouyang). IB framing refined (Cheng/Song/Saxe nuance). Biology parallels restored (Levelt/Friston/Gold&Shadlen). Cross-model observational suite complete (6 families). Gemma causal steering complete. Now: strengthen methodology with tuned lens across all models, deepen causal evidence, extend to second family.

---

## What's Done — Summary

| Experiment | Status | Key Result |
|---|---|---|
| Cross-model L1 (δ-cosine) 6 families | ✅ DONE | Gemma strong (−0.4), DeepSeek terminal spike (−0.45), universal terminal-layer presence |
| Cross-model L2 (commitment delay) | ✅ DONE | **6/6 families** IT > PT; DeepSeek 7-layer delay (onset layer 16/27 = 59% depth) |
| Cross-model L3 (weight changes) | ✅ DONE | Gemma: late-layer concentration. Others: uniform. Convergent computation. |
| Cross-model L8 (intrinsic dim) | ✅ DONE | **6/6 families** IT > PT late-layer. Δ ID: +1.3 to +4.7. Most robust finding. |
| Cross-model L9 (attention entropy) | ✅ DONE | Noisy. No universal pattern. Downgraded. |
| Gemma A1 corrective α-sweep (v4) | ✅ DONE | Clean dose-response on 4 format metrics, MMLU flat |
| Gemma A1 layer specificity | ✅ DONE | Corrective layers (20–33) only |
| Gemma A1 random direction control | ✅ DONE | Zero format effect (projection magnitude caveat) |
| Gemma A5a progressive skip | ✅ DONE | Corrective range dominates |
| Template ablation (3 experiments) | ✅ DONE | Weight-encoded, template-independent |
| Gemma A2 PT injection | ✅ DONE | Noisy — framed as circuit evidence |
| Paper draft v5 | ✅ DONE | Citations fixed, IB refined, biology restored, output governance removed, ~31 refs |

---

## Replication Hierarchy

| Finding | Cross-model | Paper role |
|---|---|---|
| ID expansion (late-layer IT > PT) | **6/6** | Primary — most robust, universal |
| Commitment delay (IT commits later) | **6/6** | Primary — central narrative |
| MLP opposition (terminal δ-cosine) | **5/5** (variable magnitude) | Supporting |
| Weight change concentration | **1/6** (Gemma only) | Dissociation finding |
| Causal format control (steering) | **1/1** (Gemma only) | Core causal — **must extend** |

---

## TIER 0: METHODOLOGY — FIX OR THE PAPER IS DEAD

These are not experiments that add novelty — they are fixes to methodological weaknesses that a reviewer would rightly reject on. Do these first, before anything else.

### 0A. Direction calibration sensitivity [0.5 days] ⭐ DO FIRST

**Why this is urgent**: We extract our corrective direction from 600 prompts out of 1,400. If the direction is unstable under different prompt subsets, *every downstream result is unreliable*. This is cheap to test and answers the most basic question about our method.

**Codebase facts** (verified from `scripts/precompute_directions_v2.py`):
- `TOP_N = 600` (line 69), dataset = `data/eval_dataset_v2.jsonl` (1,400 records)
- Selection: Phase 2 ranks by contrast score = `(STR_IT - STR_PT) + norm_PT_NLL + 0.25×(G1_IT - G1_PT)`
- Direction formula: `d̂_ℓ = normalize(mean_IT_mlp_ℓ − mean_PT_mlp_ℓ)` with L2 norm + eps 1e-12
- Activations = **MLP outputs** (not residual stream) at generated-token positions only (`shape[1]==1` detection skips prefill)
- MAX_GEN = 80 tokens per record, symmetric (min of IT/PT gen length)
- Directions saved to `results/exp5/precompute_v2/precompute/corrective_directions.npz` with keys `layer_1` through `layer_33`
- d_model = 2560 (verified in `src/poc/shared/constants.py`)
- All 33 layers hooked simultaneously in one forward pass

**Method**:
1. Phase 3 of precompute already collects per-record per-layer MLP activations (IT and PT). Load the raw per-record activation tensors from Phase 3 output (or re-collect if only means were saved — check `results/exp5/precompute_v2/precompute/` for per-record files)
2. **If per-record activations are saved**: draw 50 bootstrap resamples of 600 records from 1,400 (using the same contrast scoring, just resampling which 600). Recompute d̂_ℓ per resample.
3. **If only means are saved**: re-run Phase 3 with a flag to save per-record activation means, then bootstrap. Alternatively: modify Phase 4 (`_compute_directions()` at line ~500) to loop over bootstrap samples.
4. For each resample, compute d̂_ℓ = normalize(mean_resample_IT − mean_resample_PT) at ALL layers 1–33 (not just corrective — let's see if stability varies by layer)
5. Pairwise cosine similarity matrix across 50 directions, per layer
6. Subset size convergence: 100, 300, 600, 1000, 1400. For each size, take 20 random draws, compute direction, cosine to full-1400 direction.
7. **Out-of-distribution test**: draw 600 *completely new random prompts* (not from the 1,400 calibration set — sample from a different split or construct new prompts from TriviaQA/GSM8K/etc not already in eval_dataset_v2.jsonl). Run Phases 1+3+4 on these. Cosine to canonical direction.
8. Report: mean pairwise cosine ± std per layer, convergence plot, OOD cosine

**Success criteria**:
- Pairwise cosine > 0.95 within bootstraps → direction is stable
- Out-of-distribution cosine > 0.90 → direction generalizes beyond calibration set
- If either fails → recompute with more prompts / different prompts and understand why

**Deliverables**: Figure (convergence curve), table (per-layer cosine stats), paragraph for §4.1.

### 0B. Matched-token direction validation (precompute confound control) [1 day]

**Why**: Free-running generation confound is the most serious methodological weakness. IT and PT generate different token sequences from the same prompt → our MLP-output direction may conflate genuine weight-level activation differences with token-history effects.

**Method**:
1. From Phase 1 output, extract IT's generated token IDs for the 600 selected records. Format as force-decode JSON.
2. Run `collect.py --force-decode-file <it_tokens.json>` on the **PT model** with MLP hooks active at layers 1–33.
3. Compute matched direction: d̂_matched_ℓ = normalize(mean(IT_mlp_ℓ) − mean(PT_on_IT_tokens_mlp_ℓ)) per layer
4. Also reverse: force-decode PT's tokens through IT
5. Cosine: cos(d̂_matched, d̂_free-running) and cos(d̂_reverse, d̂_free-running) per layer
6. **Critical**: if cosine is high, re-run A1 α-sweep with d̂_matched. Overlay dose-response curves.
7. Plot per-layer cosine similarity (all 33 layers)

**Success criteria**: cosine > 0.90 at corrective layers → confound is minor. < 0.70 → must recompute everything.

### 0C. Projection-matched random direction control [0.5 days]

**Why**: In d=2560, a random unit vector has expected projection magnitude |h·d̂_rand| ≈ ‖h‖/√2560 ≈ ‖h‖/50.6, substantially smaller than the corrective direction's projection.

**Method**:
1. Use `_project_remove_magnitude_matched()` (already implemented in exp6/interventions.py lines 70-104) that scales random direction perturbation to match corrective direction's per-token projection magnitude.
2. Run full α-sweep with projection-matched random directions on 1,400 eval prompts.

**Success criteria**: Format metrics still flat → direction specificity is real, not magnitude artifact.

### 0D. Bootstrap CIs on all main figures [1 day]

**Why**: No formal statistics in current draft. Basic requirement for any empirical paper.

**Method**:
1. Load per-record metric values from A1 results JSONL for each α condition
2. Bootstrap 10,000 resamples (upgrade to BCa method for bias correction)
3. Report mean ± 95% CI for every format and content metric
4. Spearman ρ (+ p-value) for monotonicity claims in dose-response
5. Cohen's d for cross-model PT vs IT comparisons (ID, commitment delay)
6. **Expand MMLU**: increase from 300 → 1,000+ items (log-prob scoring, very cheap)
7. Also re-run GSM8K at 500 items minimum
8. For cross-model metrics: bootstrap CI on per-family Δ ID and Δ commitment values

**Deliverables**: Updated figures with error bars/shading, statistics table for appendix.

### 0E. Token classifier specification + robustness check [0.5 days]

**Why**: §3.4 defines 5 categories (STRUCTURAL, DISCOURSE, PUNCTUATION, FUNCTION, CONTENT) with regex patterns. Reviewer will ask: "how sensitive are results to these boundaries?"

**Method**:
1. Document the full token classification: vocabulary size per category, regex patterns, examples, edge cases
2. Compute reclassification sensitivity: perturb the boundaries (move 500 tokens between adjacent categories)
3. Report Hyland (2005) taxonomy justification, inter-annotator agreement on a 200-token sample (LLM + human)
4. If results are stable under reclassification → robustness confirmed

**Deliverables**: Appendix C expansion, robustness table, paragraph for §3.4.

### 0F. Corrective layer range justification [0.5 days]

**Why**: We define "corrective layers" as 20–33 (Gemma). Why 20 and not 18? Researcher degree of freedom.

**Method**:
1. Show δ-cosine curve and mark the point where IT first deviates from PT by >1σ → defines onset
2. Run A1 α-sweep with layers 18–33, 20–33, 22–33, 20–31
3. For cross-model: show "final ~40% of layers" ≡ δ-cosine onset at ~60% normalized depth

**Deliverables**: Sensitivity table (layer range vs format effect), paragraph for §4.1 or §7 Limitations.

### 0G. Tuned-lens replication of commitment delay [3–5 days] — ⭐⭐ UPGRADED TO CRITICAL

**Why**: This is now our **#1 methodological priority**. Reviewers specifically flagged reliance on raw logit lens. The tuned lens (Belrose et al., 2023) is best practice. If our commitment delay finding vanishes under tuned lens, the entire narrative collapses.

**STRATEGIC UPGRADE**: Previously scoped to "Gemma only, maybe OLMo." Now: **all 6 models**. The cross-model commitment delay is our central claim — it must be validated with the best available measurement tool.

**Method**:
1. **Training data**: Use 50,000+ tokens per model variant (12 variants total: 6 PT + 6 IT). Sample from C4-validation or The Pile for domain diversity. This addresses the current implementation's critical flaw of only 200 prompts × ~40 tokens = 8,000 tokens (woefully insufficient for a d_model-to-d_model affine probe with ~6.5M parameters).
2. **Architecture**: Per-layer affine probe T_ℓ: R^d_model → R^d_model, identity initialization, bias=True. Standard from Belrose et al.
3. **Training**: KL divergence loss (KL(softmax(W_U · T_ℓ(h_ℓ)) ‖ softmax(W_U · h_L))). Learning rate 1e-3 with cosine schedule. 5,000 steps minimum per probe. **Train-test split**: 80/20 with early stopping on validation KL.
4. **Critical fix**: Hook on **residual stream** (not MLP output). Current implementation hooks `model.language_model.layers[li].mlp` which captures MLP output, not the residual stream state. The tuned lens must decode the full residual stream representation.
5. **Commitment metric**: Use KL-to-final < 0.1 nats threshold (same as raw logit lens) for direct comparison. Also report top-1 agreement metric.
6. **PT-probe-on-IT transfer test**: Train probes on PT, evaluate on IT (and vice versa). Belrose et al. showed probes transfer across checkpoints. High transfer = IT uses same latent prediction pathways. Low transfer = IT reorganizes computation. This is an important diagnostic.
7. **Report for all 6 families**:
   - Raw logit lens commitment layer vs tuned lens commitment layer (scatter plot)
   - IT-PT commitment delay under both methods (table)
   - Threshold sensitivity (0.05, 0.1, 0.2 nats) under tuned lens
   - Cross-checkpoint probe transfer quality (KL on held-out data)

**Compute estimate**: ~12 probe training runs × ~1 hour each = 12 GPU-hours for training. Inference for commitment measurement: ~6 hours across all models. Total: ~2 days on 1 GPU, parallelizable to ~1 day on 2 GPUs. Plus 1–2 days for analysis and debugging.

**Success criteria**: Commitment delay replicates for ≥5/6 families under tuned lens. Magnitude within ~2 layers of raw logit lens estimates.
**If it fails**: Need to understand whether raw logit lens creates systematic bias against IT. If tuned lens shows *larger* delay → our paper's findings are conservative. If tuned lens shows *no delay* → fundamental problem with the narrative.

**Deliverables**: 6-model tuned lens commitment table, raw-vs-tuned comparison figure, transfer test results, updated §2.3 and §3.1.

### 0H. Calibration-evaluation split validation [0.5 days] — REVIEWER-REQUESTED

**Why**: The 600 calibration prompts use format-related scores from the same 1,400 records used for evaluation.

**Method**:
1. Strict split: randomly select 600 prompts for direction extraction (NO format-based selection), use remaining 800 for evaluation only
2. Recompute corrective direction from the random 600
3. Run A1 α-sweep on the held-out 800
4. Also test: direction from bottom-600 (lowest contrast) prompts

**Success criteria**: Random-selected direction produces comparable format dose-response.

### 0I. Intervention formula sensitivity [0.5 days] — REVIEWER-REQUESTED

**Why**: Our formula `h' = h − (1−α)·(h·d̂)·d̂` is one choice. Others exist.

**Method**:
1. Test alternative intervention formulas:
   - Current: projection removal (MLP outputs)
   - Alternative A: additive `h' = h + α·d̂·‖h‖`
   - Alternative B: residual stream intervention (hook after MLP, not on MLP output)
   - Alternative C: attention output intervention
2. Report whether the format-content dissociation is robust across ≥2 formulas

**Deliverables**: Comparison table, paragraph for §4.1 or §7 Limitations.

### 0J. Corrective onset threshold sensitivity analysis [0.5 days] — REVIEWER-REQUESTED

**Why**: "Suspiciously tight" ~59% onset across families — need to show this isn't an artifact.

**Method**:
1. For each of 6 families, compute onset layer under thresholds: 0.5σ, 0.75σ, 1.0σ, 1.5σ, 2.0σ
2. Also test absolute thresholds: IT−PT δ-cosine difference > {0.02, 0.05, 0.10, 0.15}
3. Report onset layer (and normalized depth) per family × threshold → 6×9 table
4. For Gemma: re-run A1 α-sweep with narrower (2σ) and broader (0.5σ) layer ranges

**Success criteria**: Onset layer varies by ≤3 layers per family. Normalized depth stays in ≤15pp window.

---

## TIER 1: NEW EXPERIMENTS — DEEPEN THE STORY

### 1A. PCA spectrum of IT-PT differences [0.5–1 day] ⭐ HIGH PRIORITY

**Why**: Is "the corrective direction" one thing or several? Determines whether format control is a unified mechanism or several co-localized sub-processes.

**Codebase facts**:
- Phase 3 saves per-record MLP output activations. 600 records × ~60 tokens/record ≈ 36,000 difference vectors per layer.
- d_model = 2560

**Data sufficiency note**: For stable PCA in d=2560, we need at minimum 10x the number of meaningful components. With 36,000 token-level vectors we have good coverage. If only per-record means are saved (600 vectors), PCA is still meaningful but can only resolve up to 600 components — sufficient for determining whether PC1 dominates but not for fine-grained spectral analysis.

**Method**:
1. At each corrective layer (20–33), construct difference matrix M_ℓ (IT_mlp − PT_mlp)
2. Run PCA; report cumulative explained variance for PC1–PC20
3. **Interpretation**:
   - PC1 > 60%: rank-1 direction well-justified
   - PC1 ~30%, PC2 ~20%, PC3 ~15%: multiple sub-processes
4. Extract PC2, PC3 as unit vectors; run mini A1 α-sweeps at α ∈ {−2, −1, 0, 1, 2}
5. Compute cosine(PC1, d̂_canonical) — how much does rank-1 mean differ from top PC?
6. **Extension**: run on OLMo and Llama if direction extraction complete

**Deliverables**: Scree plot, cumulative variance plot, PC1–3 steering results, paragraph for §4 or new subsection.

### 1B. Delayed commitment = corrective direction (causal link) [1–2 days] ⭐ CRITICAL

**Why**: We claim delayed commitment and the corrective stage are two sides of the same coin. Currently only correlative. Direct causal evidence: manipulating the corrective direction should manipulate commitment timing.

**Method**:
1. **IT − V̂ (remove corrective direction → measure commitment acceleration)**
   - Load existing A1 logit_lens_top1 data for each α condition
   - Compute per-token commitment layer from logit lens trajectories
   - Plot: mean commitment layer vs α. **Prediction**: monotone relationship
   - **May already be extractable from existing data**

2. **PT + V̂ (inject corrective direction → measure commitment delay)**
   - A2 injection on PT at corrective layers (20–33): β ∈ {0, 0.25, 0.5, 1.0, 1.5, 2.0}
   - Collect logit lens during generation; compute commitment layer at each β
   - **Prediction**: commitment shifts later as β increases

3. **Feature-level analysis** (top-200 transcoder features at corrective layers)

**Deliverables**: Commitment-vs-α plot, commitment-vs-β plot (overlaid with ±SE bands), feature heatmap.

### 1C. ID measurement during α-sweep [0.5 days]

**Why**: If corrective direction drives ID expansion, removing it should reduce late-layer ID toward PT levels. Completes the triad (corrective direction ↔ commitment delay ↔ ID expansion).

**Method**:
1. During A1 α-sweep, compute TwoNN ID at corrective layers at each α
2. Plot ID vs α alongside format metrics
3. **Prediction**: ID decreases as α→0, increases as α→2
4. Also measure at early/mid layers as control — ID should be flat

### 1D. Second-family causal steering: Llama 3.1 8B [3–5 days] ⭐⭐ HIGHEST SINGLE IMPACT

**Why**: All causal evidence is Gemma-only. Replicating in one more family transforms the paper.

**STRATEGIC CHANGE (v12)**: Switched primary replication target from OLMo to **Llama 3.1 8B**. Rationale:
- **Strongest strategic value**: Llama dominates the open-source ecosystem. Replication on Llama validates generality far more than OLMo for the broader community.
- **Architectural similarity**: Standard transformer, GQA, all global attention. Closest to Gemma structurally. If the mechanism transfers to a standard-architecture model, it strengthens universality claims.
- **Minimal confounds**: No MoE (DeepSeek), no MLA (DeepSeek), no sliding window (Mistral), no open-data-only concern (OLMo). Clean replication.
- **Compute**: 8B is manageable on single H100 for A1 experiments.
- **OLMo as secondary target**: OLMo is better for 2A (training trajectory) due to open checkpoints. Keep as secondary replication if time permits.

**Method**:
1. Direction extraction: `precompute_directions_v2.py` adapted for Llama 3.1 8B PT/IT
   - d_model = 4096, n_layers = 32
   - Corrective layers: final ~40% → layers ~19–31
2. A1 α-sweep: 4 format + 2 content metrics
3. Layer specificity: early/mid/corrective
4. Random direction control (norm-matched + projection-matched)
5. Commitment delay under ablation (from 1B)
6. Tuned lens commitment (from 0G, already planned)

**Success criteria**: Clean format dose-response with flat content at corrective layers. Commitment delay modulated by α.

### 1E. SVCCA/CKA representational similarity analysis [1 day] — NEW in v12

**Why**: Before investing 5–7 days in crosscoder training, we can get substantial insight from lightweight representational comparison methods. SVCCA (Raghu et al., 2017) and CKA (Kornblith et al., 2019) quantify how similar PT and IT representations are at each layer, without training any dictionary.

**What this reveals that direction analysis cannot**:
- **Layer-wise divergence profile**: Where do PT and IT representations diverge most? Does this correlate with the corrective onset?
- **Shared vs. divergent subspace dimensionality**: CKA gives a scalar similarity; SVCCA gives top canonical correlations whose falloff reveals how many independent dimensions of divergence exist.
- **Cross-model comparison**: Do all 6 families show the same PT-IT divergence profile shape? Is the corrective onset visible as a CKA drop?

**Method**:
1. For each model family, collect residual stream activations at all layers for 1,000 prompts (PT and IT)
2. Compute CKA(PT_ℓ, IT_ℓ) at each layer → CKA profile per family
3. Compute SVCCA between PT_ℓ and IT_ℓ → canonical correlation spectrum per layer
4. **Prediction**: CKA drops at corrective onset (~60% depth), with the drop magnitude correlating with the δ-cosine IT-PT difference
5. **Cross-model**: Overlay CKA profiles for all 6 families (normalized depth x-axis)
6. **Bonus**: SVCCA canonical direction analysis — extract the top canonical direction of PT-IT divergence at corrective layers. Compare with our mean-difference direction (cosine). If they agree → our direction extraction is principled. If they disagree → SVCCA reveals a better direction.

**Compute**: ~2 GPU-hours per family (forward passes only, no training). Total: ~12 GPU-hours.

**Deliverables**: CKA profile figure (6 families overlaid), SVCCA spectrum at corrective layers, cosine(SVCCA_top_direction, mean_difference_direction), paragraph for §3 or §5.

### 1F. ID abstraction phase analysis [0.5 days] — NEW in v12

**Why**: Cheng et al. (2024) identified a high-dimensional "abstraction phase" at intermediate layers in transformers. Song et al. (2025) independently found expansion-contraction dynamics. Our paper claims IT extends this high-dimensional phase — we should test this directly and rigorously.

**Method**:
1. For each of 6 families (PT and IT), compute full-layer TwoNN ID profile (all layers, not just late)
2. Identify the "abstraction phase peak" (layer of maximum ID) for PT and IT separately
3. **Key metrics**:
   - Peak ID layer (PT vs IT) — does IT peak later?
   - Peak-to-final ID drop (PT vs IT) — does IT show less contraction?
   - Width of the high-ID plateau (number of layers within 90% of peak) — is IT's plateau wider?
4. Compare our findings explicitly to Cheng et al.'s reported profiles
5. **Prediction**: IT extends the high-ID phase by 3–6 layers (consistent with the commitment delay), with the peak shifting later and the contraction being delayed

**Deliverables**: Full-layer ID profiles for all 12 model variants (6 PT + 6 IT), abstraction phase comparison table, paragraph updating §3.2.

---

## TIER 2: EXTENSIONS & EXPLORATIONS

### 2A. OLMo training trajectory analysis [2–3 days]

**Why**: OLMo 2 7B has checkpoints at every 1,000 steps through Tülu 3 (SFT → DPO → RLVR). Track exactly when the corrective stage / delayed commitment emerges.

**Method**:
1. 5–8 checkpoints: base PT, early SFT, mid SFT, late SFT, early DPO, late DPO, early RLVR, final IT
2. At each: L1 (δ-cosine), L2 (commitment delay), L8 (ID) on 500-prompt subset
3. Plot emergence curves: when does commitment delay first exceed PT by >2 layers?

**Key question**: Does the corrective stage emerge during SFT (behavioral cloning) or DPO/RLVR (preference optimization)?

### 2B. MoE-specific analysis: DeepSeek-V2-Lite [1–2 days]

**Why**: MoE has expert routing — the corrective stage could manifest as *expert selection changes*.

**Method**:
1. Analyze expert routing patterns: which experts are activated in PT vs IT at corrective layers?
2. Is there a "format expert" that IT activates more than PT?
3. Compute per-expert activation difference (IT − PT) at terminal layers

### 2C. Refusal direction cosine similarity [0.5 days]

**Why**: We claim the corrective direction subsumes refusal. Direct geometric test.
**Method**: Extract Arditi et al.'s refusal direction on Gemma 3 4B IT. Cosine with corrective direction per layer.
**Expected**: Moderate cosine (0.3–0.6) — partial overlap but not identity.

### 2D. Additional content benchmarks [0.5 days]

Add TriviaQA (500), HellaSwag (200), ARC-Challenge (200) to the α-sweep. Report ± 95% bootstrap CI. Strengthens "content preserved" claim.

### 2E. Perplexity on factual corpora during α-sweep [0.5 days] — REVIEWER-REQUESTED

**Why**: Continuous, sensitive metric that complements discrete accuracy.

**Method**:
1. 3 factual corpora: Wikipedia (500 passages), C4-validation (500), PubMed (200)
2. At each α, compute per-token perplexity with corrective direction ablation
3. Report mean perplexity ± 95% CI per corpus per α
4. Compare to PT baseline perplexity (the "floor" α=0 should approach)

**Success criteria**: Perplexity change < 5% across α range.

### 2F. Open-ended content generation quality evaluation [1 day] — REVIEWER-REQUESTED

**Why**: All current content metrics are closed-form. Blind spot for open-ended helpfulness.

**Method**:
1. 200 open-ended prompts (factual explanation, reasoning, creative-factual, instruction-following)
2. Generate at α ∈ {−2, 0, 1, 2}
3. LLM-judge evaluation (factual accuracy, coherence, completeness, instruction adherence)
4. Human spot-check 50 pairs

### 2G. Crosscoder training [5–7 days]

Train BatchTopK crosscoder at corrective layers on Gemma PT+IT. **v12 note**: Consider SVCCA/CKA results from 1E first — if these show the divergence is low-rank and well-captured by the mean direction, crosscoder may be lower priority. If SVCCA reveals rich multi-dimensional structure, crosscoder becomes more valuable.

**Lighter alternative**: If compute is tight, train crosscoder only at the 3 highest-signal corrective layers (identified by CKA drop from 1E), not all 14 layers.

### 2H. DoLA verification [1 day]

Run DoLA with pre-corrective (~layer 19) vs post-corrective (~layer 33) contrast. If benefit concentrates when layers bracket corrective stage → mechanistic explanation for DoLA.

### 2I. OLMo causal steering (secondary replication) [3–5 days] — RENAMED from 1D

**Why**: After Llama replication (1D), OLMo provides a third replication point and connects to training trajectory analysis (2A). But Llama is the priority.

---

## TIER 3: FUTURE WORK — Delayed Commitment Implications

### 3A. Is delayed commitment beneficial? [research direction]
Forced delayed commitment via auxiliary loss. Train on small model (1B).

### 3B. Post-commitment logit dynamics [0.5 days, doable now]
After the model commits, what changes in top-k logits? Sharpening vs ranking changes.

### 3C. Does delayed commitment generalize beyond IT? [research direction]
Test: RLHF-only, reasoning-trained (o1-style), RLVR. OLMo checkpoints enable this.

### 3D. Commitment timing vs confidence — deeper analysis [0.5 days, doable now]
Calibration (ECE, Brier), correct vs incorrect commitment trajectories, entropy dynamics.

---

## Priority Execution Order

```
═══════════════════════════════════════════════════════════
PHASE 1 — METHODOLOGY FIXES (Week 1, ~5 days)
═══════════════════════════════════════════════════════════

Day 1 (parallel — all cheap/fast):
  [0A] Direction calibration sensitivity          0.5 day  ⭐ DO FIRST
  [0E] Token classifier robustness check          0.5 day
  [0H] Calibration-evaluation split validation    0.5 day  ⭐ REVIEWER

Day 2 (parallel — compute-heavy):
  [0B] Matched-token validation (force-decode)    1 day    ⭐ CRITICAL
  [0C] Projection-matched random control          0.5 day
  [0D] Bootstrap CIs + expanded MMLU/GSM8K        1 day

Day 3:
  [0B] continued
  [0F] Corrective layer range justification       0.5 day
  [0I] Intervention formula sensitivity           0.5 day  ⭐ REVIEWER
  [0J] Onset threshold sensitivity analysis       0.5 day  ⭐ REVIEWER

Day 4–5:
  [0G] Tuned-lens replication — START             3–5 days ⭐⭐ #1 PRIORITY
       Begin with Gemma PT+IT (most important)
       Then Llama, OLMo, Qwen, Mistral, DeepSeek

═══════════════════════════════════════════════════════════
PHASE 2 — NEW EXPERIMENTS + TUNED LENS CONT. (Week 2)
═══════════════════════════════════════════════════════════

Day 1 (parallel):
  [0G] Tuned-lens remaining models (continued)
  [1A] PCA spectrum of IT-PT differences          0.5 day  ⭐ HIGH
  [1F] ID abstraction phase analysis              0.5 day  (NEW)

Day 2:
  [0G] Tuned-lens analysis + comparison figures
  [1E] SVCCA/CKA representational similarity      1 day    (NEW)
  [1C] ID during α-sweep                          0.5 day

Day 3–5:
  [1B] Commitment ↔ direction causal link          2 days   ⭐ CRITICAL
       (IT−V̂ → commitment acceleration)
       (PT+V̂ → commitment delay)

═══════════════════════════════════════════════════════════
PHASE 3 — SECOND-FAMILY REPLICATION (Week 3)
═══════════════════════════════════════════════════════════

Day 1–5:
  [1D] Llama 3.1 8B causal steering               3–5 days ⭐⭐ HIGHEST IMPACT
       Direction extraction + A1 α-sweep
       Layer specificity + random control
       Commitment under ablation

═══════════════════════════════════════════════════════════
PHASE 4 — EXTENSIONS + FINALIZATION (Week 4)
═══════════════════════════════════════════════════════════

Day 1–2:
  [2D] Additional content benchmarks               0.5 day
  [2E] Perplexity on factual corpora               0.5 day  ⭐ REVIEWER
  [2F] Open-ended generation quality eval           1 day    ⭐ REVIEWER
  [2C] Refusal direction cosine                    0.5 day

Day 3–5:
  Paper rewrite with all new results + figures
  Hero figure design
  Submission preparation
  Final proofread
```

---

## Updated Concern-to-Action Mapping

| Concern | Status | Action | Priority |
|---|---|---|---|
| Direction stability (prompt sensitivity) | 🔴 UNTESTED | **0A**: Bootstrap + OOD test | ⭐ FIRST |
| Free-running generation confound | 🔴 UNTESTED | **0B**: Matched-token validation | ⭐ CRITICAL |
| Random control weak (projection magnitude) | 🔴 UNTESTED | **0C**: Projection-matched control | ⭐ CRITICAL |
| No formal statistics | 🔴 MISSING | **0D**: Bootstrap CIs (BCa) + expanded benchmarks | ⭐ CRITICAL |
| Token classifier arbitrariness | 🟡 PARTIAL | **0E**: Robustness check + Hyland justification | HIGH |
| Layer range researcher DOF | 🟡 PARTIAL | **0F**: Sensitivity analysis | HIGH |
| Logit lens brittleness (vs tuned lens) | 🔴 REVIEWER FLAG | **0G**: Tuned-lens on ALL 6 FAMILIES | ⭐⭐ #1 PRIORITY |
| Calibration-evaluation overlap | 🔴 REVIEWER FLAG | **0H**: Strict split validation | ⭐ REVIEWER |
| Off-manifold intervention sensitivity | 🟡 PARTIAL | **0I**: Intervention formula comparison | REVIEWER |
| Onset threshold sensitivity | 🔴 REVIEWER FLAG | **0J**: Multi-threshold onset analysis | ⭐ REVIEWER |
| Single model for causal evidence | 🔴 CRITICAL | **1D**: Llama 3.1 8B causal steering | ⭐⭐ HIGHEST IMPACT |
| Commitment ↔ direction only correlative | 🟡 CORRELATIVE | **1B**: PT+V̂ / IT−V̂ commitment measurement | ⭐ CRITICAL |
| Corrective direction = rank-1 assumption | 🟡 ASSUMED | **1A**: PCA spectrum | HIGH |
| ID expansion causal link | 🟡 CORRELATIVE | **1C**: ID during α-sweep | HIGH |
| No representational similarity analysis | 🔴 NEW | **1E**: SVCCA/CKA across 6 families | HIGH |
| ID abstraction phase not tested | 🔴 NEW | **1F**: Full-layer ID profiles vs Cheng et al. | MEDIUM |
| Content preservation only discrete metrics | 🔴 REVIEWER FLAG | **2E**: Perplexity on factual corpora | MEDIUM |
| No open-ended quality evaluation | 🔴 REVIEWER FLAG | **2F**: LLM-judge + human eval | MEDIUM |
| Content preservation thin (2 benchmarks) | 🟡 PARTIAL | **2D**: Additional benchmarks | MEDIUM |
| Crosscoder priority uncertain | 📋 PLANNED | **2G**: Depends on 1E results | LOW (for now) |
| Citation errors | ✅ FIXED | Huang→Ouyang in v5 | — |
| IB framing oversold | ✅ FIXED | Cheng/Song/Saxe nuance in v5 §3.2 | — |
| Biology parallels removed | ✅ FIXED | Levelt/Friston restored in v5 §6 | — |
| Cross-model recipe inaccuracies | ✅ FIXED | Corrected in v5 (Qwen 4-stage, DeepSeek GRPO, Gemma multi-RL) | — |
| Cross-model convergence oversold | ✅ ADDRESSED | Reframed in v5: pretraining diversity + explicit caveats | — |
| OLMo checkpoint compatibility | ✅ ADDRESSED | Verified non-preview checkpoints; caveat in v5 §2.1 | — |
| Output governance overclaimed | ✅ ADDRESSED | Removed as framing in v5; "format and register control" | — |
| Jain et al. scope | ✅ VERIFIED | Correctly scoped to safety fine-tuning in v5 | — |

---

## Acceptance Probability Estimates (revised v12)

| State | Acceptance | Oral |
|---|---|---|
| Current (v5 draft, 6/6 cross-model, citations fixed) | ~58-63% | <5% |
| + Tier 0A–0F methodology fixes | ~72-77% | ~8% |
| + 0G tuned lens ALL 6 models (the big upgrade) | ~80-85% | ~15% |
| + 0H–0J reviewer-requested | ~83-87% | ~18% |
| + PCA + commitment causal link + SVCCA/CKA (1A, 1B, 1E) | ~87-90% | ~22-25% |
| + Llama 3.1 8B causal steering (1D) | ~91-94% | ~30-35% |
| + ID abstraction phase + extensions (1F, 2D–2F) | ~93-95% | ~35-40% |
| + OLMo trajectory + crosscoder (2A, 2G) | ~95-97% | ~42-48% |

**Key insight v12**: The tuned lens upgrade from "Gemma only" to "all 6 models" is the single biggest acceptance probability jump (+8-10pp). It transforms "we acknowledge this limitation" into "we validated with both methods across all models." This is why 0G is now ⭐⭐ — it's more impactful than any single Tier 1 experiment.

**Second insight**: Switching causal replication from OLMo to Llama 3.1 8B is strategically superior. Llama is the community standard — a positive replication on Llama carries more weight with reviewers than OLMo. OLMo remains valuable for training trajectory (2A), which no other model can provide.

**Third insight**: The new SVCCA/CKA experiment (1E) is high-value-per-compute. At ~12 GPU-hours, it provides cross-model representational divergence profiles that either (a) perfectly corroborate our δ-cosine findings with a principled method, or (b) reveal additional structure we're missing. Either outcome strengthens the paper.

**Fourth insight**: The ID abstraction phase analysis (1F) directly connects our findings to the Cheng et al. (2024) and Song et al. (2025) literature we now cite in §3.2. Showing that IT specifically extends the abstraction phase is a clean, publishable result that makes the IB discussion concrete rather than speculative.
