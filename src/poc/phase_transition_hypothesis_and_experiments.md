# Post-Training Sharpens a Universal Representational Phase Transition: Hypothesis & Experimentation Plan

**Project**: Mechanistic Interpretability of PT vs IT Computational Pipelines  
**Model**: Gemma 3 4B-PT / 4B-IT with Gemma Scope 2 MLP Transcoders  
**Date**: March 2026  
**Status**: Final revised hypothesis incorporating empirical weight-norm results

---

## 1. Literature Review

### 1.1 The Intrinsic Dimension Story: A Universal Geometric Phase Transition

Multiple independent lines of work have converged on the same finding: transformer representations undergo a characteristic geometric transformation across layers, following an expansion-contraction pattern.

**Valeriani et al. (2023)** first demonstrated this in protein language models and vision transformers: the intrinsic dimension (ID) of the data manifold expands in early layers, peaks in the first third, then contracts significantly in intermediate layers. They identified three computational phases — expansion, contraction, and a final plateau or shallow second peak — and showed this pattern holds across modalities.

**Cheng et al. (2024; ICLR 2025, "Emergence of a High-Dimensional Abstraction Phase in Language Transformers")** extended this to five pretrained language models (OPT-6.7B, Llama-3-8B, Pythia-6.9B, OLMo-7B, Gemma-7B). They found a distinct high-ID phase in the first ~30% of layers where: (1) representations achieve their first full linguistic abstraction; (2) surface-form features (sentence length, word content) become unrecoverable by probes; (3) representations first become viable for transfer to downstream tasks; (4) different LMs converge to similar representations. Crucially, better-performing models show an earlier and sharper onset of this phase. The ID peak marks the transition from surface to semantic processing.

**Song et al. (2025, "Bridging the Dimensional Chasm")** formalized this as an expansion-contraction pattern: tokens diffuse from low-dimensional input space to a high-dimensional "working space," then progressively project onto lower-dimensional "semantic submanifolds." They found a negative correlation between working space dimensionality and generalization — effective models compress more aggressively, reaching ~10-dimensional submanifolds resembling human semantic spaces.

**Razzhigaev et al. (EACL 2024, "The Shape of Learning")** showed that in decoder transformers, the anisotropy profile follows a bell-shaped curve peaking in the middle layers, and that during training, intrinsic dimension first expands then contracts as representations become more compact.

**Biagetti et al. (2025, "The Geometry of Tokens")** demonstrated that the intrinsic dimension of token representations correlates with next-token prediction loss — prompts with higher loss values have tokens in higher-dimensional spaces. They confirmed the expansion-contraction pattern at the token level.

**Key takeaway for our work**: The L0 dip at layer ~11 in Gemma 3 4B corresponds to the *contraction side* of the ID peak — the moment where high-dimensional working representations compress into lower-dimensional semantic representations. In PT, this is a smooth transition. Our finding is that IT makes this transition discontinuous.

### 1.2 Stages of Inference: Behavioral Evidence for Discrete Computation

**Lad, Gurnee & Tegmark (2024; "The Remarkable Robustness of LLMs: Stages of Inference?")** proposed four stages of transformer inference based on layer deletion and swapping experiments across multiple model families:

1. **Detokenization** (early layers): Integrating local context to lift raw token embeddings into higher-level representations. Highly sensitive to intervention — deleting early layers severely degrades performance.
2. **Feature engineering** (early-middle layers): Iteratively refining task- and entity-specific features. Moderately robust to intervention.
3. **Prediction ensembling** (middle-late layers): Aggregating hidden states into plausible next-token predictions. Remarkably robust — middle layers can be deleted with minimal accuracy loss.
4. **Residual sharpening** (final layers): Suppressing irrelevant features to finalize the output distribution. Highly sensitive.

Their detokenization→feature engineering boundary maps directly onto our L0 dip. The fact that middle layers are robust while boundary layers are sensitive is consistent with our finding of a sharp computational transition at layer ~11.

### 1.3 What Instruction Tuning Actually Changes

**URIAL / Lin et al. (ICLR 2024, "The Unlocking Spell on Base LLMs")**: Analysis of token distribution shifts between base and aligned LLMs revealed that aligned models perform nearly identically to base models on the majority of token positions. Most distribution shifts occur with stylistic tokens (discourse markers, safety disclaimers). This supports the "Superficial Alignment Hypothesis" — IT primarily teaches the model a new style/format for expressing knowledge it already possesses.

**Wu et al. (NAACL 2024, "From Language Modeling to Instruction Following")**: The mechanistic study of IT's effects on LLM internals revealed three changes: (1) IT empowers LLMs to recognize instruction parts of prompts, conditioning response generation on instructions; (2) self-attention heads in lower-to-middle layers develop new word-word relationships specifically tied to instruction verbs (13–35% increase in instruction-verbal relations); (3) feed-forward networks *rotate* their pre-trained concept directions toward user-oriented tasks (writing, coding) rather than creating new ones.

**Fierro et al. (2024, "Does Instruction Tuning Make LLMs More Consistent?")**: IT increases representational consistency — paraphrased inputs cluster more tightly, and outputs show greater invariance to non-semantic perturbations. The gains stem from improved subject enrichment in hidden states.

**Minder et al. (2025) / Crosscoder model diffing**: Fine-tuned and base models share the vast majority of their feature representations. IT creates genuinely novel latents for chat-specific behaviors (refusal, false information detection, personal questions) while repurposing existing latents for most tasks.

### 1.4 Safety Layers: Post-Training Creates Localized Computational Boundaries

**Li et al. (ICLR 2025, "Safety Layers in Aligned Large Language Models")**: Identified a small set of contiguous layers in the middle of aligned models that are crucial for distinguishing malicious from normal queries. By analyzing layer-wise cosine similarity between hidden vectors for different query types, they showed that: early layers process all queries identically; starting at a specific middle layer, normal-malicious query pairs diverge dramatically; this divergence converges in subsequent layers. These "safety layers" exist at a consistent fractional depth (~30-40% of total layers) across model families (Llama, Phi, Gemma).

This is directly relevant: their safety layers appear at approximately the same fractional depth as our L0 dip, suggesting that post-training creates (or sharpens) a computational boundary at this depth where the model makes categorical decisions about input type.

### 1.5 Our Weight Norm Finding: The Crucial Constraint

**Our empirical result (this work)**: Comparing MLP weight norms (Δ_rel and d_cos) between Gemma 3 4B-PT and 4B-IT across all 34 layers reveals that weight modifications from post-training are concentrated almost entirely in layers 25–33. Layers 0–24 show minimal weight change (Δ_rel oscillates around 0.06–0.10; d_cos below 0.007). Layers 25–33 show dramatic change (Δ_rel jumps to 0.20+; d_cos reaches 0.018).

**Critical implication**: There is NO weight modification signal at layers 10–11 (the dip). The sharpening of the phase transition at the dip is NOT caused by direct MLP weight modification at that layer. The same MLP weights produce a sharper L0 dip in IT than in PT. This means the sharpening must be driven by changes in what information reaches the dip-layer MLP — i.e., by attention routing changes in the surrounding layers.

---

## 2. Integrated Hypothesis

### H-MAIN: Two-Mechanism Restructuring of the Computational Pipeline

**Post-training restructures the transformer's computational pipeline through two distinct mechanisms operating at different network depths:**

**Mechanism 1 — Direct Rewriting (layers 25–33):** IT massively modifies late-layer MLP weights to create a corrective computational stage that opposes the residual stream (negative cosine similarity), implements format/safety/output constraints, and operates universally across input types (IC ≈ OOC ≈ R). This is the *supply-side* change — IT directly installs new computational routines.

**Mechanism 2 — Attention-Mediated Sharpening (layer ~11):** IT sharpens the pre-existing representational phase transition (the smooth ID peak/contraction present in PT) into a discrete computational boundary, without modifying the MLP weights at the transition layer. The sharpening is driven by attention routing changes in layers ~8–11: IT learns to concentrate attention on instruction-relevant tokens, delivering qualitatively different information to the same MLP, which then produces a sharper feature population turnover. This is the *demand-side* change — the corrective stage's requirement for clean, categorically-organized input propagates backward through the network, reshaping how earlier layers route information.

**The resulting architecture has three discrete stages:**
- **Stage 1 (layers 0–10): Surface Processing** — Detokenization and initial feature building. Functionally identical in PT and IT (minimal weight changes, similar attention patterns).
- **Stage 2 (layers 12–24): Semantic Computation** — Abstract feature refinement, entity resolution, reasoning. Similar weights in PT and IT, but IT activates more features per layer (higher L0 plateau vs. PT's monotonic decline) because the corrective stage downstream demands richer abstract representations.
- **Stage 3 (layers 25–33): Corrective Stage** — Output shaping, safety enforcement, format compliance. Exists only in IT (massively modified weights, negative cosine with residual stream, sustained high L0). In PT, these layers merely perform residual sharpening.

**The boundary between Stage 1 and Stage 2 (the L0 dip at layer ~11) is the signature finding**: PT has a smooth transition (gradual bell-curve peak). IT compresses this into a single-layer discontinuity — a hard "gate" where surface features die and semantic features take over in a single step. This gate is not created by changing the gate's own weights, but by changing what passes through it (attention routing) and what demands are placed on its output (corrective stage requirements).

### Specific Predictions

**P1 — Feature Population Shift**: Features active at layer 10 and layer 12 are qualitatively different populations. The Jaccard similarity between active feature sets across the dip (layer 10→12) is lower than at adjacent non-dip layers. This shift is sharper in IT than PT.

**P2 — Attention Divergence at the Dip**: Attention patterns at layers 8–11 differ significantly between PT and IT, despite near-identical MLP weights. IT attention at these layers is more concentrated (lower entropy) and more focused on instruction-relevant tokens. This is the proximal cause of the dip sharpening.

**P3 — Fractional Depth Consistency**: The dip occurs at a consistent fractional depth (~30% of total layers) across model sizes and families, corresponding to the universal ID peak location identified by Cheng et al. and Valeriani et al.

**P4 — Top-Down Causality**: The corrective stage (layers 25+) causally drives the dip sharpening. Transplanting IT late layers into PT should begin to sharpen the dip; transplanting IT early layers into PT should not.

**P5 — Feature Category Transition**: Features exclusive to the pre-dip regime are predominantly lexical/syntactic (token patterns, POS-like). Features exclusive to the post-dip regime are predominantly semantic/entity-level. In IT specifically, post-dip features include instruction-following and format features not present in PT's post-dip population.

**P6 — ID Geometry Confirmation**: The intrinsic dimension profile mirrors the L0 profile — PT shows smooth ID contraction; IT shows sharper contraction at the dip layer, independently confirming the phase transition sharpening without reliance on transcoder decomposition.

---

## 3. Experimentation Plan

### Phase 0: Immediate Validation (3.5 days total)

#### E0a: Attention Pattern Comparison PT vs IT at Dip Layers [1.5 days] — Tests P2

**Rationale**: The weight norm plot shows no MLP changes at the dip. If the dip sharpens without weight changes, it must be attention-driven. This experiment identifies the proximal mechanism.

**Method**: For each prompt in the dataset, extract full attention matrices at layers 6–14 for both PT and IT. Compute per-layer:
- Attention entropy (H): mean entropy of attention distributions across heads
- Instruction attention mass: fraction of total attention weight on instruction-relevant vs. filler tokens (requires token role annotation)
- PT-IT attention divergence: mean KL(attn_PT || attn_IT) per head per layer

**Expected result**: Layers 8–11 show the largest attention divergence between PT and IT despite near-identical MLP weights. IT attention entropy is lower (more focused) at these layers. Instruction attention mass is higher in IT at layers 8–11.

**Failure mode**: If attention patterns are also identical at the dip layers, the sharpening may be driven by residual stream composition changes from earlier layers (which would point to a more subtle emergent effect).

#### E0b: Adjacent-Layer Jaccard Across the Dip [0.5 days] — Tests P1

**Rationale**: The most basic test of whether the dip represents a feature population transition or just a temporary suppression.

**Method**: For each prompt, record active transcoder features at layers 8–14 (both models). Compute Jaccard(L, L+1) for all consecutive pairs, plus Jaccard(10, 12) to skip the dip.

**Metrics**:
- Jaccard(L, L+1) curve for L ∈ {8,9,10,11,12,13} — shape comparison PT vs IT
- Jaccard(10, 12) — how much does skipping the dip restore similarity?
- |features_at_10 \ features_at_12| — how many features die at the dip?
- |features_at_12 \ features_at_10| — how many new features appear after the dip?

**Expected result**: IT shows a deep Jaccard valley at layers 10→11 that PT doesn't. Jaccard(10, 12) is moderate in both models (some features return, some are new), but the valley is much sharper in IT. This confirms the dip is a population transition, not suppression.

#### E0c: Intrinsic Dimension Profile PT vs IT [1 day] — Tests P6

**Rationale**: Connects our transcoder-based finding to the established geometric literature. If the ID profile mirrors the L0 profile, we validate our finding with an independent methodology that doesn't require transcoders.

**Method**: Forward-pass both models on the full prompt set. At each layer, collect last-token residual stream vectors across all prompts. Estimate ID using TwoNN estimator (fast, well-established). Plot ID(layer) for PT and IT.

**Expected result**: PT shows smooth expansion→peak→contraction (matching Cheng et al.). IT shows the same peak but with sharper contraction at the dip layer. The ID dip in IT is more discontinuous than in PT, independently confirming the L0 finding.

#### E0d: Weight Norm Plot — Already Complete ✓

Result: Weight changes concentrated in layers 25–33. No signal at dip. This rules out direct MLP modification as the dip-sharpening mechanism.

### Phase 1: Core Mechanistic Evidence (4.5 days total)

#### E1a: Feature Label Analysis Before/After Dip [2 days] — Tests P5

**Rationale**: The qualitative evidence that makes or breaks the paper. If pre-dip features are demonstrably lexical and post-dip features are demonstrably semantic, the "phase transition from surface to semantic processing" story is concrete, not just statistical.

**Method**: For both PT and IT, identify three feature populations on the dataset:
- Pre-dip exclusive: active at layer 10 but NOT layer 12
- Post-dip exclusive: active at layer 12 but NOT layer 10
- Surviving: active at both layers 10 and 12

For each population, pull Gemma Scope 2 Neuronpedia labels (where available) and manually categorize ~100 features per group into:
- Lexical/syntactic: token patterns, bigrams, POS-like, morphological
- Semantic/entity: concepts, relations, topics, entities
- Format/output: instruction-following, discourse markers, output shaping
- Ambiguous/unclear

**Expected result**: Pre-dip exclusive features are predominantly lexical (>60%). Post-dip exclusive features are predominantly semantic (>60%). In IT specifically, post-dip features include a category of format/instruction features absent from PT's post-dip population. Surviving features are the most abstract/general ones in both models.

#### E1b: Logit Lens / Answer Emergence PT vs IT [1.5 days]

**Rationale**: If the dip is a compression boundary, we should see different answer commitment patterns before and after it. In IT, the model may delay commitment until after the dip (because the corrective stage needs to verify/modify).

**Method**: For factual prompts where the correct answer token is known, apply the unembedding matrix at each layer to get a probability distribution. Track:
- Rank of correct token at each layer
- Probability of correct token at each layer
- KL(layer_L distribution || final distribution) at each layer

Compare trajectories PT vs IT.

**Expected result**: PT shows gradual answer emergence through the dip (smooth rank improvement). IT shows a different pattern: less commitment before the dip, then rapid commitment after — consistent with the model compressing input first, then making predictions. The KL-to-final curve shows a sharper slope change at the dip in IT.

#### E1c: Layer Ablation at the Dip [1 day] — Causal Validation

**Rationale**: If the dip is a critical computational bottleneck (a "gate"), ablating it should be devastating. If it's just a statistical pattern, ablating it shouldn't matter much.

**Method**: For IT only:
- Zero out MLP output at layer 11 → measure output quality (perplexity on held-out text, accuracy on factual QA)
- Zero out MLP output at layer 15 (control — middle layers should be robust per Lad et al.)
- Zero out MLP output at layer 11 in PT → compare degradation magnitude

**Expected result**: Ablating layer 11 in IT causes severe degradation (it's the gate — downstream layers expect compressed categorical input). Ablating layer 15 causes minimal degradation (redundant middle layer). Ablating layer 11 in PT causes less degradation than in IT — because PT's transition is smooth, no single layer is critical.

### Phase 2: Deepening the Mechanism (3–4 days total)

#### E2a: Weight Transplant / Hybrid Models [2 days] — Tests P4

**Rationale**: Tests top-down causality. If the corrective stage drives the dip sharpening, transplanting IT's late layers into PT should sharpen the dip without any changes to early layers.

**Method**: Construct hybrid models by stitching layers from PT and IT:
- **Hybrid A**: PT layers 0–24 + IT layers 25–33 → Does the dip sharpen?
- **Hybrid B**: IT layers 0–24 + PT layers 25–33 → Does the dip soften?
- **Hybrid C**: PT layers 0–11 + IT layers 12–33 → Isolate the boundary

Run L0-per-layer analysis on each hybrid. Compare dip sharpness (measured as |L0(layer10) - L0(layer11)| / L0(layer10)).

**Expected result**: Hybrid A shows partial dip sharpening — the corrective stage's demands propagate backward even through unchanged early layers, though the effect is weaker than full IT (because attention patterns haven't been trained to match). Hybrid B shows a softer dip despite having IT's early layers — removing the corrective stage removes the pressure that organized the gate.

**This experiment is potentially novel** — layer-granularity PT/IT hybrid models with computational profile measurement.

#### E2b: Feature Analysis at Late-Layer Rise (layers 28–30 in IT) [0.5 days]

**Rationale**: The IT L0 plot shows a slight rise at layers 28–30. This may be the onset of the corrective stage, where new format/safety features activate.

**Method**: Same as E1a but for the late-layer rise. Compare features active at layer 27 vs. 30 in IT. Cross-reference with the cosine sign-flip layer from Plot 10 (corrective stage onset).

**Expected result**: New features appearing at layers 28–30 in IT that aren't active at layer 25 are predominantly format/safety/output-shaping features. The layer where these features appear correlates with the cosine sign flip.

#### E2c: Alignment Tax Quantification [0.5 days]

**Method**: For IT, compute total |delta_norm| summed across layers 0–24 vs. 25–33 across all prompts and generation steps.

**Metric**: What fraction of total MLP computational effort (by delta norm) is devoted to the corrective stage? Is it constant across IC/OOC/R?

**Expected result**: 30–50% of total MLP edit magnitude occurs in the corrective stage (layers 25–33), and this fraction is approximately constant across task types. This is the "alignment tax" — the computational cost of IT's corrective stage on every forward pass.

### Phase 3: Generalization (3–4 days total)

#### E3a: Cross-Model Replication [2–3 days] — Tests P3

**Rationale**: Is the dip sharpening universal or Gemma-specific?

**Method**: Run L0-per-layer (or raw MLP output norm, if transcoders unavailable) on:
- Gemma 2 2B PT/IT (Gemma Scope 1 transcoders available)
- If accessible: Llama 3.2 1B/3B base vs. instruct (no transcoders — use MLP output norm proxy)

For each model pair, measure:
- Dip fractional depth = dip_layer / total_layers
- Dip sharpness in PT vs. IT
- Weight norm concentration (does IT always modify late layers most?)

**Expected result**: All models show a dip at fractional depth ~0.28–0.35. IT always sharpens the dip relative to PT. Weight modification always concentrates in the last quarter. This would establish the phenomenon as universal.

#### E3b: Expanded Dataset [1 day]

**Method**: Scale from current ~200 prompts to 500, drawing from TriviaQA, GSM8K, ARC-Challenge, MMLU-Pro, and custom OOC prompts. Include the format-control subset (same 70 prompts in raw completion, Q&A, chat template, and instruction formats).

**Purpose**: Increase statistical power and verify that dip sharpness is not an artifact of prompt selection. The format-control subset tests whether the dip's sharpness depends on prompt format (it shouldn't, per the weight-norm finding).

---

## 4. Priority and Timeline

| Priority | Experiment | Days | Tests | Depends On |
|----------|-----------|------|-------|-----------|
| 1 | E0b: Jaccard across dip | 0.5 | P1 | — |
| 2 | E0a: Attention comparison at dip | 1.5 | P2 | — |
| 3 | E0c: Intrinsic dimension profile | 1.0 | P6 | — |
| 4 | E1a: Feature labels before/after dip | 2.0 | P5 | E0b |
| 5 | E1c: Layer ablation at dip | 1.0 | Causal | — |
| 6 | E1b: Logit lens answer emergence | 1.5 | — | — |
| 7 | E2a: Weight transplant hybrids | 2.0 | P4 | E1c |
| 8 | E2b: Late-layer feature analysis | 0.5 | — | E1a |
| 9 | E2c: Alignment tax | 0.5 | — | — |
| 10 | E3a: Cross-model replication | 2.5 | P3 | E0b |
| 11 | E3b: Expanded dataset | 1.0 | — | — |

**Critical path**: E0b → E0a → E1a → E2a (5 days for the core mechanistic story)

**Total estimated time**: ~14 days for all experiments. The first 7 days (Phase 0 + Phase 1) produce a complete paper-ready story. Phase 2 and 3 strengthen the contribution.

---

## 5. Key Literature to Cite

### Must Cite (Core Related Work)

| Paper | Relevance |
|-------|-----------|
| Cheng et al. (ICLR 2025) — High-Dimensional Abstraction Phase | ID peak = our dip; they characterize geometry, we characterize feature-level computation and show IT sharpens it |
| Lad, Gurnee & Tegmark (2024) — Stages of Inference | Four-stage framework for PT; our work shows IT creates discrete boundaries between their stages |
| Li et al. (ICLR 2025) — Safety Layers | Safety-specific layers at same depth as our dip; our dip is the general phenomenon, their safety layers are one instance |
| URIAL / Lin et al. (ICLR 2024) — Superficial Alignment | Most IT token changes are stylistic; our corrective stage (layers 25+) is the mechanism implementing the style change |
| Wu et al. (NAACL 2024) — Behavior Shift after IT | IT changes attention in middle layers + rotates FFN concepts; our work shows exactly where these changes manifest (attention at dip, weights at late layers) |
| Valeriani et al. (2023) — Geometry of Hidden Representations | Original ID expansion-contraction finding; our L0 profile is the feature-level signature of their geometric phenomenon |
| Song et al. (2025) — Bridging the Dimensional Chasm | Working space → semantic space compression; our dip is the compression boundary |
| Minder et al. (2025) — Crosscoder Model Diffing | Shared vs. exclusive features PT/IT; our feature label analysis (E1a) should find IT-exclusive features concentrated post-dip |

### Should Cite (Supporting)

| Paper | Relevance |
|-------|-----------|
| Arditi et al. (2024) — Refusal Direction | Single direction mediates refusal; likely one component of our corrective stage subspace |
| Gromov et al. (2024) — Unreasonable Ineffectiveness of Deeper Layers | Deep layers removable in PT; our corrective stage explains why IT's deep layers are NOT removable |
| DoLA / Chuang et al. (ICLR 2024) | Contrasting layers improves factuality; works because it leverages the propose-correct structure our work identifies |
| Belrose et al. (2023) — Tuned Lens | Methodology for answer emergence experiment (E1b) |
| Safety Tax / Huang et al. (2025) | Behavioral alignment tax; our E2c provides the mechanistic quantification |
| Razzhigaev et al. (EACL 2024) — Shape of Learning | Anisotropy bell curve in decoders; our L0 bell curve in PT is the feature-level analog |
| Fierro et al. (2024) — Instruction Tuning Consistency | IT increases representational consistency; our dip sharpening is one mechanism — by forcing a clean boundary, IT ensures consistent abstract representations post-dip |
| Layer Freezing literature (multiple, 2024–2025) | Bottom 25–50% freezable during IT with no loss; consistent with our finding that pre-dip layers are unchanged |

---

## 6. Paper Narrative (Proposed Structure)

**Title**: "Post-Training Creates Discrete Computational Boundaries: How Instruction Tuning Restructures Layer-Wise Processing Through Two Distinct Mechanisms"

**Core claim**: Instruction tuning transforms the pretrained model's smooth computational pipeline into a three-stage architecture with discrete boundaries, through two mechanisms: (1) direct weight rewriting in late layers creates a corrective stage, and (2) attention routing changes in middle layers sharpen the pre-existing representational phase transition into a hard computational gate — without modifying the gate's own weights.

**Novelty**:
1. First feature-level characterization of the representational phase transition (complements the geometric ID literature)
2. Discovery that IT sharpens this transition without changing the transition layer's weights (attention-mediated, demand-driven)
3. Two-mechanism story (late-layer weight rewriting + middle-layer attention reorganization) that unifies URIAL's "superficial alignment" observation with the safety layers finding
4. Weight transplant experiments (E2a) demonstrating top-down causality — the first direct test of whether late-layer demands shape early-layer computation

**Connections to existing narratives**:
- Explains WHY Cheng et al.'s ID peak is sharper in better models (more organized compression)
- Explains WHY Lad et al.'s middle layers are deletable (redundant semantic refinement between two hard boundaries)
- Explains WHAT the safety layers are mechanistically (one application of the general categorical routing at the dip)
- Provides the MECHANISM for URIAL's superficial alignment (IT changes style tokens because the corrective stage modifies output distributions, while most computation remains unchanged)
