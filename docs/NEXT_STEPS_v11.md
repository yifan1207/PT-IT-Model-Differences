# Next Steps v11 — Focused on Delayed Commitment + Rigour
**Date**: 2026-03-28
**Status**: Paper draft v4 complete. Narrative recentered on delayed commitment (primary) and corrective stage (secondary). Output governance reframed as implication. Cross-model observational suite complete (6 families). Gemma causal steering complete. Now: strengthen methodology, deepen the delayed commitment story, extend causal evidence.

---

## What's Done — Summary

| Experiment | Status | Key Result |
|---|---|---|
| Cross-model L1 (δ-cosine) 6 families | ✅ DONE | Gemma strong (−0.4), DeepSeek terminal spike (−0.45), universal terminal-layer presence |
| Cross-model L2 (commitment delay) | ✅ DONE | **6/6 families** IT > PT; DeepSeek 7-layer delay (onset layer 16/27 = 59% depth) |
| Cross-model L3 (weight changes) | ✅ DONE | Gemma: late-layer concentration. Others: uniform. Convergent computation. |
| Cross-model L8 (intrinsic dim) | ✅ DONE | **6/6 families** IT > PT late-layer. Δ ID: +1.3 to +4.7. Most robust finding. |
| Cross-model L9 (attention entropy) | ✅ DONE | Noisy. No universal pattern. Downgraded. |
| Gemma A1 corrective α-sweep (v4) | ✅ DONE | Clean dose-response on 4 governance metrics, MMLU flat |
| Gemma A1 layer specificity | ✅ DONE | Corrective layers (20–33) only |
| Gemma A1 random direction control | ✅ DONE | Zero governance effect (projection magnitude caveat) |
| Gemma A5a progressive skip | ✅ DONE | Corrective range dominates |
| Template ablation (3 experiments) | ✅ DONE | Weight-encoded, template-independent |
| Gemma A2 PT injection | ✅ DONE | Noisy — framed as circuit evidence |
| Paper draft v4 | ✅ DONE | 9 sections, delayed commitment + corrective stage centered, 9 figures, ~38 refs |

---

## Replication Hierarchy

| Finding | Cross-model | Paper role |
|---|---|---|
| ID expansion (late-layer IT > PT) | **6/6** | Primary — most robust, universal |
| Commitment delay (IT commits later) | **6/6** | Primary — central narrative |
| MLP opposition (terminal δ-cosine) | **5/5** (variable magnitude) | Supporting |
| Weight change concentration | **1/6** (Gemma only) | Dissociation finding |
| Causal governance (steering) | **1/1** (Gemma only) | Core causal — **must extend** |

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

**Deliverables**: Figure (convergence curve), table (per-layer cosine stats), paragraph for §5.1.

### 0B. Matched-token direction validation (precompute confound control) [1 day]

**Why**: Free-running generation confound is the most serious methodological weakness. IT and PT generate different token sequences from the same prompt → our MLP-output direction may conflate genuine weight-level activation differences with token-history effects (different KV caches → different layer inputs → different MLP outputs even if weights were identical).

**Codebase facts** (verified from `src/poc/collect.py`):
- Force-decode already implemented: lines 256–263 load a JSON mapping `{record_id: [tok1, tok2, ...]}` and lines 815–837 inject `forced_tokens[step]` instead of argmax during generation
- Phase 1 of `precompute_directions_v2.py` saves `gen_ids` in its output — need to verify exact format and location
- MLP hooks in Phase 3 capture `out[0, 0, :].float().cpu()` (single generated token per step, skipping prefill via `shape[1]==1` check)
- The 600 selected records are identified by contrast score in Phase 2

**Method**:
1. From Phase 1 output, extract IT's generated token IDs for the 600 selected records. Format as force-decode JSON: `{record_id: [tok_id_1, tok_id_2, ..., tok_id_T]}` where T = min(IT_len, PT_len, 80)
2. Run `collect.py --force-decode-file <it_tokens.json>` on the **PT model** with MLP hooks active at layers 1–33. This makes PT process IT's exact token sequence, producing PT MLP activations conditioned on IT's tokens.
3. Compute matched direction: d̂_matched_ℓ = normalize(mean(IT_mlp_ℓ) − mean(PT_on_IT_tokens_mlp_ℓ)) per layer
4. Also reverse: force-decode PT's tokens through IT → d̂_reverse_ℓ = normalize(mean(IT_on_PT_tokens_mlp_ℓ) − mean(PT_mlp_ℓ))
5. Cosine: cos(d̂_matched, d̂_free-running) and cos(d̂_reverse, d̂_free-running) per layer
6. **Critical**: if cosine is high, re-run A1 α-sweep (`src/poc/exp6/interventions.py`) with d̂_matched loaded in place of canonical directions. Overlay dose-response curves.
7. Plot per-layer cosine similarity (all 33 layers) to see if stability varies — corrective layers (20–33) may show higher/lower agreement than early layers

**Success criteria**: cosine > 0.90 at corrective layers → confound is minor. < 0.70 → must recompute everything.
**If it fails**: Recompute all corrective directions with matched tokens via modified Phase 3+4, re-run A1 suite. Adds ~3 days.

### 0C. Projection-matched random direction control [0.5 days]

**Why**: In d=2560, a random unit vector has expected projection magnitude |h·d̂_rand| ≈ ‖h‖/√2560 ≈ ‖h‖/50.6, substantially smaller than the corrective direction's projection. Our "zero governance effect" for random directions may simply reflect that the perturbation magnitude was ~50× smaller. This is an obvious reviewer attack.

**Codebase facts** (verified from `src/poc/exp6/interventions.py`):
- Random direction generation: `_random_unit_vector(d_model, seed, layer_idx)` at line 62 — samples from N(0,1), normalizes to unit norm. Seeded per-layer for reproducibility.
- Current intervention: `_project_remove(mlp_out, direction, alpha)` at line 44 — applies h' = h − (1−α)·(h·d̂)·d̂ to **MLP output** tensors
- The `SteeringHookManager` registers forward hooks on `model.language_model.layers[L].mlp`
- Random control already exists as a condition in the experiment configs but uses norm-matched (not projection-matched) vectors

**Method**:
1. Modify `_project_remove` (or add a new function `_project_remove_magnitude_matched`) that at each forward pass:
   - Computes p_corr = |(h · d̂_corr)| and p_rand = |(h · d̂_rand)| for the current MLP output
   - Scale factor s = p_corr / p_rand (per-token, per-layer, per-step)
   - Applies: h' = h − (1−α) · s · (h · d̂_rand) · d̂_rand
2. This ensures the actual perturbation vector ‖Δh‖ is identical in magnitude for random and corrective
3. Run full α-sweep via `run_exp6_A_v4.sh` with new condition flag on 1,400 eval prompts
4. Both hooks are on MLP outputs (same hook point, same layer set 20–33)

**Success criteria**: Governance metrics still flat → direction specificity is real, not magnitude artifact.
**If it fails**: Specificity weakens — fall back on layer specificity (§5.3) + content-layer control.

### 0D. Bootstrap CIs on all main figures [1 day]

**Why**: No formal statistics in current draft. Basic requirement for any empirical paper.

**Codebase facts** (verified from `src/poc/exp6/`):
- Eval metrics computed in-loop: `structural_token_ratio`, `format_compliance_v2`, `mmlu_forced_choice`, `reasoning_em`, `alignment_behavior`
- MMLU uses **log-probability forced-choice scoring** (no generation/regex) — cheap to scale
- Results stored in JSONL: one line per (record, condition) with all metric values
- LLM judge metrics (G1 governance quality, G2 register, S1/S2 safety) computed post-hoc
- Full A1 sweep costs ~$17 for LLM judge scoring
- Current MMLU: 300 items. GSM8K: 200 items. Eval dataset: 1,400 records across 7 categories

**Method**:
1. Load per-record metric values from A1 results JSONL for each α condition
2. Bootstrap 10,000 resamples of eval records at each α value
3. Report mean ± 95% CI for every governance and content metric
4. Spearman ρ (+ p-value) for monotonicity claims in dose-response
5. Cohen's d for cross-model PT vs IT comparisons (ID, commitment delay)
6. **Expand MMLU**: increase from 300 → 1,000+ items. Since `mmlu_forced_choice` uses log-prob scoring with no generation, this is very cheap (just forward pass per item). Can potentially run full 14,042.
7. Also re-run GSM8K at 500 items minimum (this requires generation — ~2 hrs on 1 GPU)
8. For cross-model metrics: bootstrap CI on per-family Δ ID and Δ commitment values

**Deliverables**: Updated figures with error bars/shading, statistics table for appendix.

### 0E. Token classifier specification + robustness check [0.5 days]

**Why**: §3.4 defines 5 categories (STRUCTURAL, DISCOURSE, PUNCTUATION, FUNCTION, CONTENT) with regex patterns. A reviewer will ask: "how sensitive are results to these boundaries?"

**Method**:
1. Document the full token classification: vocabulary size per category, regex patterns, examples, edge cases
2. Compute reclassification sensitivity: perturb the boundaries (move 500 tokens between adjacent categories) and re-measure the mind-change analysis
3. Report Hyland (2005) taxonomy justification, inter-annotator agreement on a 200-token sample (LLM + human)
4. If results are stable under reclassification → robustness confirmed. If sensitive → identify which boundary matters and justify it.

**Deliverables**: Appendix C expansion, robustness table, paragraph for §3.4.

### 0F. Corrective layer range justification [0.5 days]

**Why**: We define "corrective layers" as 20–33 (Gemma). Why 20 and not 18? Why not 22–33? This looks like a researcher degree of freedom.

**Method**:
1. Show the δ-cosine curve and mark the point where IT first deviates from PT by >1σ → this defines onset
2. Show A1 results are robust to ±2 layers on each boundary: run A1 α-sweep with layers 18–33, 20–33, 22–33, 20–31
3. For cross-model: show that "final ~40% of layers" is equivalent to the δ-cosine onset at ~60% normalized depth (already in §3.1, but needs formal justification)
4. Cite Lad et al. (2024) and Wei et al. (2026) who independently identify similar layer ranges

**Deliverables**: Sensitivity table (layer range vs governance effect), paragraph for §5.1 or §8 Limitations.

### 0G. Tuned-lens replication of commitment delay [1–2 days] — REVIEWER-REQUESTED

**Why**: Reviewers specifically flagged reliance on raw logit lens as a methodology concern. The tuned lens (Belrose et al., 2023) is now considered best practice — raw logit lens can be brittle and systematically less faithful. If our commitment delay finding vanishes under tuned lens, the entire narrative weakens.

**Method**:
1. Train tuned-lens affine probes at each layer for Gemma PT and IT (12 probes: PT layers 1–33, IT layers 1–33 — or use the `tuned-lens` package if public probes exist for Gemma 3 4B)
2. Recompute commitment layer using tuned-lens KL-to-final instead of raw logit-lens KL-to-final
3. Recompute mind-change statistics (layer-by-layer top-1 prediction changes)
4. Report: (a) commitment delay under tuned lens vs raw logit lens, (b) threshold sensitivity (0.05, 0.1, 0.2 nats), (c) are the mind-change category proportions (75% structural) robust?
5. Ideally do for at least 2 models (Gemma + OLMo)

**Success criteria**: Commitment delay replicates (IT still commits later) and magnitude is comparable (within ~2 layers).
**If it fails**: Need to understand whether raw logit lens was creating a systematic bias. Possible that tuned lens shows *larger* delay (more faithful), not smaller.

### 0H. Calibration-evaluation split validation [0.5 days] — REVIEWER-REQUESTED

**Why**: The 600 calibration prompts are selected using governance-related scores (STR, G1) from the same 1,400 records used for evaluation. A reviewer flagged this as potential "target-aware selection" — the direction is optimized for high-governance prompts and then evaluated on governance.

**Method**:
1. Strict split: randomly select 600 prompts for direction extraction (NO governance-based selection), use remaining 800 for evaluation only
2. Recompute corrective direction from the random 600
3. Run A1 α-sweep on the held-out 800
4. Compare dose-response curves to the original (governance-selected) direction
5. Also test: direction from bottom-600 (lowest contrast) prompts — does it still work?

**Success criteria**: Random-selected direction produces comparable governance dose-response → selection bias is not driving results.
**If it fails**: The governance-based selection is doing real work → need to be much more careful about framing the direction as capturing governance-specific signal.

**Deliverables**: Overlay dose-response plot (governance-selected vs random vs low-contrast), paragraph for §5.1.

### 0I. Intervention formula sensitivity [0.5 days] — REVIEWER-REQUESTED

**Why**: Reviewer noted that activation-patching/intervention results are sensitive to methodological choices. Our formula `h' = h − (1−α)·(h·d̂)·d̂` is one choice. Others exist.

**Method**:
1. Test alternative intervention formulas on the same A1 α-sweep:
   - Current: projection removal `h' = h − (1−α)·proj(h, d̂)` (applied to MLP outputs)
   - Alternative A: additive `h' = h + α·d̂·‖h‖` (magnitude-scaled addition)
   - Alternative B: residual stream intervention (hook on residual stream after MLP, not MLP output)
   - Alternative C: attention output intervention (hook on attention output instead of MLP)
2. Report whether the governance-content dissociation is robust across intervention types
3. Key question: does the hook point (MLP output vs residual stream) matter?

**Success criteria**: Core dissociation (governance degrades, content flat) holds across ≥2 intervention formulas.

**Deliverables**: Comparison table, paragraph for §5.1 or §8 Limitations.

### 0J. Corrective onset threshold sensitivity analysis [0.5 days] — REVIEWER-REQUESTED

**Why**: We define corrective onset as the layer where IT first deviates from PT by >1σ on the δ-cosine profile (§3.1, §0F). But how sensitive is this onset to the threshold? If switching from 1σ to 0.5σ or 2σ shifts onset by 5+ layers, the "55–65% depth" claim is fragile. A reviewer flagged that the consistent ~59% onset across families looked "suspiciously tight" — we need to show this isn't an artifact of one arbitrary threshold.

**Codebase facts**:
- δ-cosine profiles stored in `results/cross_model/plots/data/L1_mean_delta_cosine.csv` — per-layer IT and PT values for all 6 families
- Onset layer defined via 1σ threshold on (IT − PT) δ-cosine difference relative to early-layer baseline
- Current onset values: Gemma L20/34 (59%), Llama L19/32 (59%), Qwen L22/36 (61%), Mistral L20/32 (63%), DeepSeek L16/27 (59%), OLMo L19/32 (59%)

**Method**:
1. For each of the 6 families, compute onset layer under thresholds: 0.5σ, 0.75σ, 1.0σ (current), 1.5σ, 2.0σ
2. Also test absolute thresholds: IT−PT δ-cosine difference > {0.02, 0.05, 0.10, 0.15}
3. Report onset layer (and normalized depth) for each family × threshold combination → 6×9 table
4. Compute range of onset layers per family across all thresholds → how many layers does the window shift?
5. **Key question**: Does the "55–65% depth" range hold under all reasonable thresholds, or does it expand to e.g. "40–75%"?
6. For Gemma specifically: re-run A1 α-sweep with the narrower (2σ) and broader (0.5σ) layer ranges to test whether the dose-response is robust to onset definition

**Success criteria**: Onset layer varies by ≤3 layers across threshold choices per family, and normalized depth remains in a ≤15pp window (e.g., 50–65% or 55–70%). Gemma A1 dose-response is qualitatively unchanged.
**If it fails**: Report the full sensitivity range honestly. If onset is highly threshold-dependent, reframe as "gradual transition" rather than "discrete onset."

**Deliverables**: 6×9 onset table, normalized-depth sensitivity plot, Gemma A1 dose-response under alternative layer ranges, paragraph for §3.1 and §8 Limitations.

---

## TIER 1: NEW EXPERIMENTS — DEEPEN THE STORY

### 1A. PCA spectrum of IT-PT differences [0.5 days] ⭐ HIGH PRIORITY

**Why**: Is "the corrective direction" one thing or several? This determines whether output governance is a unified mechanism or several co-localized sub-processes. Our hypothesis: **multiple smaller components**, not one monolithic direction.

**Codebase facts**:
- Phase 3 of `precompute_directions_v2.py` collects per-record MLP output activations across layers 1–33
- Per record: up to 80 generated tokens, each producing a [2560]-dim MLP output vector
- 600 records × ~60 tokens/record ≈ **36,000 difference vectors** per layer (not 24,000 — corrected from MAX_GEN=80, actual is min(IT_len, PT_len, 80))
- Need to verify whether Phase 3 saves **per-record per-token** activations or only **per-record means**. If only means → 600 × 2560 matrix per layer. If per-token → ~36,000 × 2560 per layer.
- Check: `results/exp5/precompute_v2/precompute/` for raw activation files

**Method**:
1. Check what Phase 3 saves. If per-record means only: the PCA matrix is 600 × 2560 per layer (still meaningful but lower rank). If per-token: ~36,000 × 2560.
2. At each corrective layer (20–33), construct the difference matrix M_ℓ where each row is (mlp_IT − mlp_PT) for one record (or token)
3. Run PCA on M_ℓ; report cumulative explained variance for PC1–PC20
4. **Interpretation**:
   - PC1 > 60%: one dominant governance mode → rank-1 direction is well-justified
   - PC1 ~30%, PC2 ~20%, PC3 ~15%: multiple sub-processes → output governance decomposes into sub-directions (format, register, discourse, safety?)
5. Extract PC2 and PC3 as separate unit vectors; run mini A1 α-sweeps at α ∈ {−2, −1, 0, 1, 2} by loading PC2/PC3 as the direction in `SteeringHookManager`
6. If PC2/PC3 differentially affect format vs register vs safety → sub-process decomposition confirmed
7. **Extension**: run same PCA on OLMo and Llama (requires direction extraction for those models)
8. Also interesting: cosine(PC1, d̂_canonical) — how much does the rank-1 mean differ from the top PC?

**Deliverables**: Scree plot, cumulative variance plot, PC1–3 steering mini-results, cosine(PC1, mean-direction), paragraph for §4 or new §4.2.

### 1B. Delayed commitment = corrective direction (causal link) [1–2 days] ⭐ CRITICAL

**Why**: We claim delayed commitment and the corrective stage are two sides of the same coin. Currently this is only argued correlatively. We need direct causal evidence: manipulating the corrective direction should manipulate commitment timing.

**Codebase facts**:
- A2 injection formula (verified from `src/poc/exp6/interventions.py` lines 34–59): `h' = h + β × d̂ × ‖h‖` where h = MLP output, d̂ = unit direction, ‖h‖ = per-token L2 norm
- A1 ablation formula: `h' = h − (1−α) × (h·d̂) × d̂` — projection removal from MLP output
- **Logit lens data already collected**: the exp6 batch generation code (`src/poc/exp6/`) saves `logit_lens_top1` per layer per generated step as `.npz` files. This means commitment layer can be derived from existing A1 results without re-running generation.
- Commitment layer definition: earliest layer ℓ where KL(logit_lens_ℓ || logit_lens_final) < 0.1 nats
- Transcoder features: variant-matched (PT-trained for PT, IT-trained for IT), `gemma-scope-2-4b-{pt,it}`, ~60–150 active features per layer per token

**Method**:
1. **IT − V̂ (remove corrective direction → measure commitment acceleration)**
   - Load existing A1 α-sweep logit_lens_top1 data from `results/exp6/merged_A1_it_v4/`
   - At each α ∈ {−5, −2, −1, 0, 0.5, 1, 2, 5}, compute per-token commitment layer from logit lens trajectories
   - Plot: mean commitment layer vs α. **Prediction**: monotone increase (earlier commitment as α→0, later as α→2)
   - **This may already be extractable from existing data** — check if logit_lens NPZs exist for all A1 conditions

2. **PT + V̂ (inject corrective direction → measure commitment delay)**
   - Use A2 injection on PT model at corrective layers (20–33): h' = h + β·‖h‖·d̂
   - β ∈ {0, 0.25, 0.5, 1.0, 1.5, 2.0} — run via `run_exp6_A_v4.sh` A2 conditions
   - Collect logit lens at all layers during generation
   - Compute commitment layer at each β
   - **Prediction**: commitment shifts *later* (toward IT levels) as β increases
   - **Note**: A2 results are known to be noisier than A1 (existing A2 data shows inconsistent governance improvement)

3. **Feature-level analysis of corrective stage (top-200 features)**
   - At corrective layers (20–33), extract top 200 most-activated transcoder features for PT and IT
   - Use variant-matched transcoders: `gemma-scope-2-4b-pt` for PT, `gemma-scope-2-4b-it` for IT
   - Compare: Jaccard overlap (expect ~0.15 based on existing data), feature descriptions via W_dec → token-space projection → LLM interpretation
   - Visualize as feature-difference heatmap: which features are IT-unique at corrective layers?
   - LLM classify each IT-unique feature: format/register/safety/discourse/content/other
   - Human spot-check 50 features
   - **This directly answers "what does the corrective stage compute?" at the feature level**

**Deliverables**: Commitment-vs-α plot, commitment-vs-β plot (overlaid with ±SE bands), feature heatmap, feature classification table.

### 1C. ID measurement during α-sweep [0.5 days]

**Why**: If the corrective direction drives ID expansion, removing it should reduce late-layer ID toward PT levels. This causally links corrective computation to dimensional expansion — completing the triad (corrective direction ↔ commitment delay ↔ ID expansion).

**Method**:
1. During A1 α-sweep on Gemma, compute TwoNN ID at corrective layers at each α ∈ {−5, −2, −1, 0, 0.5, 1, 2, 5}
2. Plot ID vs α alongside governance metrics
3. **Prediction**: ID decreases as α→0, increases as α→2
4. Also measure at early/mid layers as control — ID should be flat

**Deliverables**: ID-vs-α plot (corrective vs control layers), paragraph for §3.2 or §8 Discussion.

### 1D. Second-family causal steering: OLMo 2 7B [3–5 days] ⭐ HIGHEST SINGLE IMPACT

**Why**: All causal evidence is Gemma-only. Replicating in one more family transforms the paper. OLMo is ideal: strong ID expansion (+4.7), fully open-source, intermediate checkpoints.

**Method**:
1. Direction extraction: `precompute_directions_v2.py` on OLMo PT/IT
2. Define corrective layers (final ~40%: layers ~19–31 for 32L model)
3. A1 α-sweep: 4 governance + 2 content metrics
4. Layer specificity: early/mid/corrective
5. Random direction control (norm-matched + projection-matched)
6. **Also measure commitment delay under ablation** (from 1B above) — two-family causal triad

**Success criteria**: Clean governance dose-response with flat content at corrective layers.

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

**Why**: MoE has expert routing — the corrective stage could manifest as *expert selection changes* rather than (or in addition to) direction-level changes. DeepSeek commitment delay data still pending.

**Method**:
1. Finish DeepSeek commitment delay measurement (L2)
2. Analyze expert routing patterns: which experts are activated in PT vs IT at corrective layers?
3. Is there a "governance expert" that IT activates more than PT?
4. Compute per-expert activation difference (IT − PT) at terminal layers

### 2C. Refusal direction cosine similarity [0.5 days]

**Why**: We claim the corrective direction subsumes refusal. Direct geometric test.

**Method**: Extract Arditi et al.'s refusal direction on Gemma 3 4B IT. Compute cosine with corrective direction at each corrective layer.
**Expected**: Moderate cosine (0.3–0.6) — partial overlap but not identity.

### 2D. Additional content benchmarks [0.5 days]

Add TriviaQA (500), HellaSwag (200), ARC-Challenge (200) to the α-sweep. Report ± 95% bootstrap CI. Strengthens the "content preserved" claim.

### 2E. Perplexity on factual corpora during α-sweep [0.5 days] — REVIEWER-REQUESTED

**Why**: MMLU forced-choice and GSM8K exact-match are coarse content metrics. A reviewer noted that a direction could degrade factual knowledge in ways that forced-choice accuracy misses — e.g., reducing confidence on correct answers, or degrading next-token prediction quality on factual text. Perplexity on held-out factual corpora is a continuous, sensitive metric that complements discrete accuracy.

**Method**:
1. Select 3 factual corpora (diverse, non-overlapping with calibration/eval):
   - **Wikipedia** (random 500 passages, ~512 tokens each): general factual knowledge
   - **C4-validation** (500 passages): web text distribution
   - **PubMed abstracts** (200 passages): domain-specific factual text
2. At each α ∈ {−5, −2, −1, 0, 0.5, 1, 2, 5}, apply corrective direction ablation at layers 20–33 and compute per-token perplexity on each corpus
3. Report mean perplexity ± 95% bootstrap CI per corpus per α
4. **Key comparison**: Baseline IT perplexity vs α=0 (direction fully removed) — if perplexity increases significantly, the corrective direction contains some content-relevant information
5. Also compute perplexity of PT baseline on same corpora — this establishes the "floor" that α=0 should approach
6. Plot perplexity vs α alongside governance metrics for a unified view

**Success criteria**: Perplexity change < 5% across α range → content preservation claim strengthened with a continuous metric. Perplexity at α=0 approaches PT levels → direction removal returns model toward PT behavior without degrading below PT.
**If it fails**: Perplexity increases significantly → the corrective direction encodes some content-relevant signal, not purely governance. Requires nuanced reframing.

**Deliverables**: Perplexity-vs-α plot (3 corpora), comparison table, paragraph for §5.4 and §8 Limitations.

### 2F. Open-ended content generation quality evaluation [1 day] — REVIEWER-REQUESTED

**Why**: All current content metrics are closed-form (forced-choice MMLU, exact-match GSM8K). A reviewer correctly noted that "a direction could affect open-ended helpfulness, calibration, or explanation quality while leaving multiple-choice accuracy unchanged." This is a real blind spot — the corrective direction could degrade explanation coherence, factual elaboration, or reasoning chains without touching MCQ accuracy.

**Method**:
1. Construct 200 open-ended prompts spanning:
   - Factual explanation (50): "Explain how photosynthesis works", "What causes tides?"
   - Reasoning (50): "Why is the sky blue? Walk me through the physics.", multi-step math word problems
   - Creative-factual (50): "Write a paragraph about the history of the printing press"
   - Instruction-following (50): "List 5 causes of WWI and explain each in one sentence"
2. Generate responses at α ∈ {−2, 0, 1, 2} (4 conditions × 200 prompts = 800 generations)
3. **LLM-judge evaluation** (GPT-4 or Claude as judge):
   - Factual accuracy (1–5 scale)
   - Coherence / fluency (1–5 scale)
   - Completeness (1–5 scale)
   - Instruction adherence (1–5 scale)
4. Also compute: response length distribution, self-BLEU (diversity), and perplexity of generated text
5. **Human spot-check**: 50 randomly selected (prompt, α=0 response, α=1 response) pairs, blind evaluation

**Success criteria**: LLM-judge scores flat across α for factual accuracy and coherence → content preservation extends to open-ended generation. Format/register scores should degrade at α=0 (confirming governance control).
**If it fails**: Open-ended quality degrades at α=0 → the corrective direction encodes some aspects of helpful response structure, not just format. This actually strengthens the "output governance" interpretation but complicates the clean dissociation narrative.

**Deliverables**: LLM-judge scores table (4 metrics × 4 α values), response length distributions, human evaluation summary, paragraph for §5.4.

### 2G. Crosscoder training [5–7 days]

Train BatchTopK crosscoder at corrective layers on Gemma PT+IT. Would enable unified shared-vs-IT-specific feature identification. Highest impact for oral; save for camera-ready.

### 2H. DoLA verification [1 day]

Run DoLA with pre-corrective (~layer 19) vs post-corrective (~layer 33) contrast. If benefit concentrates when layers bracket corrective stage → mechanistic explanation for DoLA.

---

## TIER 3: FUTURE WORK — Delayed Commitment Implications

These are deeper research directions that go beyond the current paper but should be mentioned in §8 Discussion / §9 Conclusion as future work, and some preliminary data would strengthen the narrative.

### 3A. Is delayed commitment beneficial? [research direction]

**Core question**: Is IT's delayed commitment an intentional optimization or a side effect? Can we force *more* of it?

**Preliminary experiments** (if time):
1. **Forced delayed commitment via auxiliary loss**: Add a penalty term that discourages early-layer commitment (e.g., KL(logit_lens_layer_ℓ || uniform) for ℓ < 0.6·L). Train on a small model (1B) to see if forced delay improves instruction-following.
2. **Optimal commitment timing**: Is there a sweet spot? Plot task performance vs commitment layer. If over-delaying hurts performance → delayed commitment is a tradeoff, not pure benefit.

**Why it matters**: If delayed commitment can be engineered directly via loss functions, this is a new training technique for instruction tuning — potentially more efficient than current SFT/DPO approaches.

### 3B. Post-commitment logit dynamics [0.5 days, doable now]

**Core question**: After the model commits (KL-to-final < 0.1 nats), what changes in the top-k logits? Does the model just sharpen confidence on the same token, or does the ranking shuffle?

**Method**:
1. At the commitment layer and at each subsequent layer, record top-10 logits and their tokens
2. Classify post-commitment changes: sharpening-only (top-1 unchanged, probabilities redistributed) vs ranking-change (top-1 shifts)
3. Compare PT vs IT: does IT have more post-commitment stability? (Consistent with the corrective stage "locking in" a decision)
4. **Rigor check**: If top-1 frequently changes after the "commitment layer", our commitment definition (KL < 0.1) is too loose. May need to tighten threshold or use a different criterion.

**Deliverables**: Post-commitment stability analysis, potential revision to commitment definition.

### 3C. Does delayed commitment generalize beyond IT? [research direction]

**Core question**: Is delayed commitment specific to instruction tuning, or does it appear in all post-training? What about:
- **RLHF/DPO-only** (no SFT): Does preference optimization alone create delayed commitment?
- **Reasoning-trained models** (o1-style, DeepSeek-R1): Do they delay commitment even more? (Hypothesis: yes, because chain-of-thought requires maintaining options longer)
- **RLVR** (reinforcement learning from verifiable rewards): OLMo's Tülu 3 includes RLVR — does this stage amplify commitment delay beyond SFT+DPO?

**Testable with OLMo checkpoints** (connects to 2A): Compare commitment delay at SFT-only checkpoint vs DPO checkpoint vs RLVR checkpoint. If each stage increases delay → post-training *generally* delays commitment, with IT as a special case.

### 3D. Commitment timing vs confidence — deeper analysis [0.5 days, doable now]

**Core question**: Our confidence-stratified analysis (Figure 2) shows the correlation. Deeper questions:
1. **Calibration**: Are tokens that undergo longer corrective processing better *calibrated* (lower ECE, lower Brier score)?
2. **Correct vs incorrect**: Among tokens where IT disagrees with PT, do correctly-committed tokens show a different commitment trajectory than incorrectly-committed ones?
3. **Entropy dynamics**: How does the entropy of the logit-lens distribution change across layers for high-delay vs low-delay tokens?

---

## Priority Execution Order

```
═══════════════════════════════════════════════════════════
PHASE 1 — METHODOLOGY FIXES (Week 1, ~5 days)
═══════════════════════════════════════════════════════════

Day 1 (parallel — all cheap/fast):
  [0A] Direction calibration sensitivity          0.5 day  ⭐ DO FIRST
  [0E] Token classifier robustness check          0.5 day
  [0H] Calibration-evaluation split validation    0.5 day  ⭐ REVIEWER-REQUESTED

Day 2 (parallel — compute-heavy):
  [0B] Matched-token validation (force-decode)    1 day    ⭐ CRITICAL
  [0C] Projection-matched random control          0.5 day
  [0D] Bootstrap CIs + expanded MMLU/GSM8K        1 day

Day 3:
  [0B] continued (analysis + A1 rerun if needed)
  [0F] Corrective layer range justification       0.5 day
  [0I] Intervention formula sensitivity           0.5 day  ⭐ REVIEWER-REQUESTED
  [0J] Onset threshold sensitivity analysis       0.5 day  ⭐ REVIEWER-REQUESTED

Day 4–5:
  [0G] Tuned-lens replication (Gemma PT+IT)       1–2 days ⭐ REVIEWER-REQUESTED
  [0D] continued (cross-model bootstrap CIs)

═══════════════════════════════════════════════════════════
PHASE 2 — NEW EXPERIMENTS (Week 2, ~5 days)
═══════════════════════════════════════════════════════════

Day 1 (parallel):
  [1A] PCA spectrum of IT-PT differences          0.5 day  ⭐ HIGH
  [1C] ID during α-sweep                          0.5 day
  [3B] Post-commitment logit dynamics              0.5 day

Day 2–3:
  [1B] Delayed commitment = corrective direction   2 days   ⭐ CRITICAL
       (IT−V̂ ablation → commitment acceleration)
       (PT+V̂ injection → commitment delay)
       (Top-200 feature analysis)

Day 4–5:
  [1D] OLMo causal steering (begin)               start of 3–5 day block

═══════════════════════════════════════════════════════════
PHASE 3 — EXTENSIONS (Week 3)
═══════════════════════════════════════════════════════════

Day 1–3:
  [1D] OLMo causal steering (complete)
  [2D] Additional content benchmarks               0.5 day

Day 3–5:
  [2A] OLMo training trajectory                    2–3 days (if checkpoints accessible)
  [2C] Refusal direction cosine                    0.5 day
  [2E] Perplexity on factual corpora               0.5 day  ⭐ REVIEWER-REQUESTED
  [2F] Open-ended generation quality eval           1 day    ⭐ REVIEWER-REQUESTED

═══════════════════════════════════════════════════════════
PHASE 4 — PAPER FINALIZATION (Week 4)
═══════════════════════════════════════════════════════════

Day 1–2:
  [2B] DeepSeek MoE analysis (if time)
  Paper rewrite with all new results + figures

Day 3–5:
  Hero figure design
  Submission preparation
  Final proofread
```

---

## Updated Concern-to-Action Mapping

| Concern | Status | Action | Priority |
|---|---|---|---|
| Direction stability (prompt sensitivity) | 🔴 UNTESTED | **0A**: Bootstrap + out-of-distribution test | ⭐ FIRST |
| Free-running generation confound | 🔴 UNTESTED | **0B**: Matched-token validation | ⭐ CRITICAL |
| Random control weak (projection magnitude) | 🔴 UNTESTED | **0C**: Projection-matched control | ⭐ CRITICAL |
| No formal statistics | 🔴 MISSING | **0D**: Bootstrap CIs + expanded benchmarks | ⭐ CRITICAL |
| Token classifier arbitrariness | 🟡 PARTIAL | **0E**: Robustness check + Hyland justification | HIGH |
| Layer range researcher DOF | 🟡 PARTIAL | **0F**: Sensitivity analysis | HIGH |
| Logit lens brittleness (vs tuned lens) | 🔴 REVIEWER FLAG | **0G**: Tuned-lens replication | ⭐ REVIEWER |
| Calibration-evaluation overlap | 🔴 REVIEWER FLAG | **0H**: Strict split validation | ⭐ REVIEWER |
| Off-manifold intervention sensitivity | 🟡 PARTIAL | **0I**: Intervention formula comparison | REVIEWER |
| Single model for causal evidence | 🔴 CRITICAL | **1D**: OLMo causal steering | ⭐ HIGHEST IMPACT |
| Commitment ↔ direction only correlative | 🟡 CORRELATIVE | **1B**: PT+V̂ / IT−V̂ commitment measurement | ⭐ CRITICAL |
| Corrective direction = rank-1 assumption | 🟡 ASSUMED | **1A**: PCA spectrum | HIGH |
| ID expansion causal link | 🟡 CORRELATIVE | **1C**: ID during α-sweep | HIGH |
| Output governance overclaimed | 🟢 ADDRESSED | Reframed as diagnostic label in v4 draft | — |
| δ-cosine "necessary" overclaimed | 🟢 ADDRESSED | Softened to "empirical marker" in v4 draft | — |
| Content preservation overclaimed | 🟢 ADDRESSED | Caveats added in v4 draft + 0D expansion | — |
| "Causal triad" is correlational | 🟢 ADDRESSED | Reframed as "co-modulation" in v4 draft | — |
| Citation errors (5 papers) | 🟢 FIXED | Corrected in v4 draft | — |
| d_model error (4096 vs 2560) | 🟢 FIXED | Corrected in v4 draft | — |
| Distributional shift alternative explanation | 🟢 ADDRESSED | Added to Limitations in v4 draft | — |
| Training dynamics unknown | 📋 PLANNED | **2A**: OLMo checkpoint trajectory | MEDIUM |
| MoE-specific mechanisms unknown | 📋 PLANNED | **2B**: DeepSeek expert analysis | MEDIUM |
| Onset threshold sensitivity | 🔴 REVIEWER FLAG | **0J**: Multi-threshold onset analysis | ⭐ REVIEWER |
| Content preservation only discrete metrics | 🔴 REVIEWER FLAG | **2E**: Perplexity on factual corpora | MEDIUM |
| No open-ended quality evaluation | 🔴 REVIEWER FLAG | **2F**: LLM-judge + human eval on open-ended | MEDIUM |
| Content preservation thin (2 benchmarks) | 🟡 PARTIAL | **2D**: Additional benchmarks | MEDIUM |
| Novelty vs Assistant Axis | ✅ RESOLVED | Positioned as complementary in §7 | — |
| Template confound | ✅ RESOLVED | Three-experiment ablation | — |
| Single model (observational) | ✅ RESOLVED | 6-family cross-model complete (6/6) | — |
| Phase boundary | ✅ RESOLVED | Removed from paper | — |

---

## Acceptance Probability Estimates

| State | Acceptance | Oral |
|---|---|---|
| Current (6/6 cross-model, citation fixes, framing softened, methodology open) | ~55-60% | <5% |
| + Tier 0A–0F methodology fixes (direction stability, matched-token, CIs) | ~70-75% | ~8% |
| + Tier 0G–0I reviewer-requested (tuned lens, calibration split, formula sensitivity) | ~78-83% | ~12% |
| + PCA + commitment causal link (1A, 1B) | ~83-87% | ~18-22% |
| + OLMo causal steering (1D) | ~88-92% | ~25-30% |
| + OLMo training trajectory (2A) | ~90-93% | ~30-35% |
| + Crosscoder + DoLA + all extensions | ~93-95% | ~40-45% |

**Key insight**: The reviewer feedback reveals that methodology concerns are more serious than we estimated in v10. The ~15% jump from Tier 0 is not about adding novelty — it's about reaching the "credible empirical paper" threshold. Without 0A–0F, the paper is at high risk of rejection on methodological grounds alone.

**Second key insight**: Three new reviewer-requested experiments (0G tuned-lens, 0H calibration split, 0I intervention formula) are cheap (~2.5 days total) but address very specific reviewer concerns. Adding these moves us from "we noted this limitation" to "we tested it and it's fine" — a substantial credibility difference.

**Third insight**: The commitment ↔ direction causal link (1B) is the strongest new experiment for the *narrative*. It transforms delayed commitment from "interesting observation" to "causally linked to the corrective direction" — making the entire paper's story tight.
