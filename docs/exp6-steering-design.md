# Steering Experiment Design — Exp6
## Causal Evidence for the Corrective Computational Stage

**Last updated**: 2026-03-23
**Goal**: Prove the corrective stage is causally responsible for output governance (format, structure, conversational style) through bidirectional steering experiments.
**Template papers**: Arditi et al. 2024 (refusal direction, NeurIPS), Turner et al. 2023 (ActAdd), Anthropic Scaling Monosemanticity 2024 (feature clamping)

---

## 0. WHY STEERING, NOT ABLATION

Full-phase ablation (Exp5) caused catastrophic collapse for all phases — removing 8-14 layers exceeds the model's tolerance (~25-30% per ShortGPT). This is not informative for phase-specific claims.

Steering avoids this problem because:
- It operates on a **specific direction or feature**, not entire layers
- The model's general computation is preserved — only the targeted component changes
- It enables **dose-response curves** (α sweep) that show graded effects
- It supports **bidirectional** testing: add direction → behavior increases; remove → decreases
- It has strong literature precedent: Arditi (refusal), ActAdd (sentiment), RepE (honesty), Anthropic (feature clamping)

We already proved the foundation works: corrective_directional ablation (Exp5) preserves coherence at 0.99 while removing the corrective direction. Now we build on this.

---

## 1. WHAT WE MEAN BY "OUTPUT GOVERNANCE"

Before designing experiments, we need to formalize the claim. The corrective stage doesn't control "alignment" in the broad RLHF sense — that's too vague and our Exp5 results show alignment behavior persists even after directional ablation. Instead:

**Output governance** = the set of computational interventions that transform a model's raw next-token prediction into a well-formed conversational response. This includes:

1. **Conversational structure**: producing "Answer:" after "Question:", generating numbered lists when asked, following turn-taking conventions
2. **Format adherence**: markdown formatting, LaTeX delimiters, code blocks, proper punctuation patterns
3. **Register/style**: producing helpful explanatory text rather than raw web-text continuation, hedging language, discourse connectives ("First,", "However,", "In summary,")
4. **Structural token selection**: the corrective stage preferentially modifies STRUCTURAL tokens (E3.10: 75% of mind changes are structural tokens like "Question", "Answer", numbers, punctuation)

We specifically distinguish output governance from:
- **Factual knowledge**: encoded in content layers, tested by MMLU/TriviaQA
- **Safety/refusal**: may overlap but is not the primary claim (Arditi et al. already covered refusal)
- **Reasoning ability**: GSM8K, logic — likely distributed across layers

---

## 2. EXPERIMENT DESIGN: TWO COMPLEMENTARY APPROACHES

### Approach A: Direction-Level Steering (residual stream)

**Rationale**: The corrective direction (IT-PT difference in MLP outputs at corrective layers) captures the aggregate computational change. Steering this direction tests whether the OVERALL corrective computation is causal.

**Extraction** (already done in Exp5 precompute):
```python
# For each corrective layer l ∈ {20, ..., 33}:
# 1. Run N prompts through both IT and PT
# 2. Cache MLP output at layer l for both
# 3. corrective_direction[l] = mean(mlp_IT[l] - mlp_PT[l]) across prompts
# 4. Normalize to unit vector
```

**Experiment A1: Remove corrective direction from IT (does IT become PT-like?)**

```
For α ∈ {5.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0,-2,  -3, -5}:
    For each prompt:
        Run IT model with intervention at corrective layers:
            h' = h - (1-α) × proj(h, d_corr)
        α=1.0: baseline (no change)
        α=0.0: direction fully removed (Arditi-style abliteration)
        α=-1.0: direction reversed (anti-correction)
        and all the alpha levels ilisted above
```

**Prediction**: As α decreases from 5,1→0,-5 format adherence and structural token production degrade BEFORE factual quality degrades. At α=0, output should resemble PT-style raw completion (less structured, more web-text-like).

**Experiment A2: Add corrective direction to PT (does PT become IT-like? and if we remove these more does pt ebcome less "output governed")**

```
For β ∈ {-5.0, -3.0, -2.0, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0}:
    For each prompt:
        Run PT model with intervention at corrective layers:
            h' = h + β × d_corr × ‖h‖  (scale by hidden state norm for magnitude matching)
        β=0.0: baseline PT
        β=1.0: corrective direction at natural IT magnitude
        β>1.0: amplified correction
```

**Prediction**: As β increases, PT output becomes more structured: "Question:"/"Answer:" patterns emerge, formatting improves, response becomes more conversational. Factual content should be relatively preserved.

**Controls**:
- **Random direction**: same norm, random orientation. Should NOT produce format changes.
- **Rotated direction**: 90° rotation of d_corr in a random plane containing d_corr. Tests specificity.
- **Content-layer direction**: extract IT-PT difference at content layers (0-11), inject at corrective layers. Should NOT work because it's the wrong computational signature.

**Why this works and is well-justified**: This is exactly the Arditi et al. methodology applied to a different direction. They extracted the refusal direction as a mean difference, removed/added it, and measured behavioral changes. We do the same for the corrective direction. The key advantage: we're not removing layers (which causes collapse) — we're removing a specific computational component.

### Approach B: Feature-Level Steering (transcoder features)

**Rationale**: Direction-level steering tests the aggregate. Feature-level steering tests specific components: can we identify individual features responsible for output governance, and steer them individually?

**This is the Anthropic Scaling Monosemanticity approach**: they clamped individual SAE features and showed causal effects on model behavior. We do the same with our transcoder features.

**Step B0: Feature Selection (which features to steer)**

We need to identify features related to output governance. Three methods:

**Method 1: Label-based selection (LLM-assisted)**
```python
# For each layer l ∈ {20, ..., 33}:
#   For each feature f in IT transcoder at layer l:
#     label = top_tokens[f] from Gemma Scope (already cached)
#     Use LLM to classify label as:
#       GOVERNANCE: formatting, structural, discourse, turn-taking
#       SAFETY: refusal, harm-avoidance
#       CONTENT: factual, semantic, domain-specific
#       OTHER: multilingual, code, unknown
#     Keep features classified as GOVERNANCE
```

We already have these labels cached for L20-L33 (16,384 features × 14 layers = 229,376 features to classify). An LLM can classify the top-token labels in bulk. lets use a top model dont care about cost but also just dont take too long think abuot howt o divide these into how may llm calls make sense think

**Method 2: Activation-based selection (quantitative, no LLM)**
```python
# Features that fire preferentially on structural tokens:
# 1. From our 3k collection, we have token-type labels (CONTENT/STRUCTURAL/FUNCTION/etc)
# 2. For each feature, compute: P(structural_token | feature_active) / P(structural_token)
# 3. Features with enrichment ratio > 2.0 are "structural governance features"
# 4. Similarly for DISCOURSE and PUNCTUATION tokens
```

This uses our existing token-type classifier and the feature activation data. No LLM needed. The enrichment ratio directly measures whether a feature preferentially fires during governance-relevant tokens.

**Method 3: Importance-based selection**
```python
# Features that are most DIFFERENT between IT and PT:
# 1. From feature_importance_summary.npz for both IT and PT
# 2. Compute: diff_score[f] = (sum_IT[f] / total_IT) - (sum_PT[f] / total_PT)
# 3. Features with high positive diff_score are "IT-amplified features"
# 4. Features with high negative diff_score are "PT-pruned features"
```

This directly identifies features that post-training amplified or suppressed.

**We should use Methods 1+2 combined**: LLM labels for interpretability (which features are about formatting?), activation-based enrichment for quantitative backing (do these features actually fire on structural tokens?). Features that score high on BOTH are our targets.

And we should also use methods 3 or use this as a method as well for our later experiments. Rember this we should use this as a methods for all of expeirment B for the data and results create two copies one for our methods 1,2 combin one fore method 3.

**Experiment B1: Steer governance features in IT (suppress AND amplify)**

We do **bidirectional** feature-level steering — not just zero-clamping but active amplification at multiple magnitudes. This gives us dose-response curves at the feature level, matching what we do at the direction level in A1.

```python
# For each prompt:
#   Run IT forward pass with transcoder interception at corrective layers
#   At each corrective layer l:
#     x = pre-feedforward-layernorm output
#     features = transcoder.encode(x)  # sparse activations
#     ā = precomputed mean activation for each governance feature
#     For each governance feature f in our target set:
#       features[f] = γ × ā[f]  # clamp to scaled mean activation
#     x_modified = transcoder.decode(features)
#     Replace MLP output with x_modified
```

**Magnitude sweep γ**: {0.0 (zero/suppress), 0.25, 0.5, 1.0 (natural baseline), 2.0, 5.0, 10.0}

Following Anthropic's Scaling Monosemanticity approach, which clamped features at up to 10× max activity and showed behavioral changes. γ=0 is pure ablation; γ>1 is amplification. Recent work (Behavioral Steering in MoE Models, 2026) shows these dose-response curves can be smooth (sigmoid-like), phase-transition (sudden flip at a threshold), or inverted-U (peak then collapse). The *shape* of our curve itself is informative about how governance features interact.

**Predictions by γ**:
- γ=0.0: Format/structure degrades, content preserved. Output becomes PT-like (raw completion style).
- γ=0.5: Mild format degradation. Some structural tokens lost.
- γ=1.0: Baseline. No change (sanity check that our transcoder interception is clean).
- γ=2.0: Amplified governance. Outputs become MORE structured — more headers, more bullet points, more verbose explanations. May become overly formal or templated.
- γ=5.0–10.0: Extreme amplification. Expect either extreme formatting (every response becomes a numbered list) or eventual coherence collapse. The threshold where collapse happens tells us about the feature's dynamic range.

**Feature set sweep**: Apply to top-10, top-50, top-100, top-500, and all governance features separately. This gives a 2D dose-response surface (number of features × magnitude).

**Experiment B2: Inject governance features into PT via W_dec projection**

The user correctly identified that we can bypass the cross-dictionary problem. Instead of trying to match feature indices across IT and PT transcoders (which fails because Jaccard=0.15), we **project IT features back to residual stream space** using the IT transcoder's decoder matrix, then inject that direction into PT's residual stream directly.

The math:
```python
# Step 1: Compute the "governance direction" from IT's feature space
# For governance features f₁, f₂, ..., fₖ at layer l:
#   v_gov[l] = Σᵢ (ā[fᵢ] × W_dec_IT[fᵢ])
# where ā[fᵢ] = mean activation of feature fᵢ across IT prompts
# and W_dec_IT[fᵢ] = decoder vector for feature fᵢ (shape: d_model=4096)
#
# This gives us a direction in residual stream space that represents
# "what these governance features collectively push toward"

# Step 2: Inject into PT
# For each prompt, at each corrective layer l:
#   h_PT' = h_PT + β × v_gov[l]
# where β controls injection magnitude
```

**Why this works**: Each transcoder feature's decoder vector W_dec[f] tells us what direction that feature pushes the residual stream. The weighted sum over governance features gives us a "governance direction" that lives in the shared 4096-dim residual stream space — not in feature index space. This direction can be added to PT without any cross-dictionary assumption. It's conceptually similar to A2 (direction-level steering) but constructed bottom-up from specific features rather than top-down from the aggregate IT–PT difference.

**β sweep**: {0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0}

**Prediction**: PT output becomes more structured as β increases. Compared to A2 (which uses the full IT–PT mean difference), B2 should produce *more targeted* governance changes because v_gov is constructed only from governance-classified features, not the entire IT–PT diff (which includes content changes, safety changes, etc.).

**Key comparison**: If B2 produces governance changes similar to A2, it validates that the corrective direction is primarily composed of governance features. If A2 produces broader changes than B2, the corrective direction contains non-governance components (e.g., safety, content refinement).

**Variant B2b — Subtract governance direction from IT**:
```python
# h_IT' = h_IT - β × v_gov[l]
```
This is the feature-level analogue of A1 (corrective direction removal), but targeted to governance features only. Prediction: format degrades while safety and content are preserved (because we're only removing the governance component, not the full corrective direction).

**Experiment B3: Control features (specificity test)**

Three controls, all using the same γ-sweep and β-sweep as B1/B2:

1. **Random features**: Select same number of features at random (not governance-classified). Prediction: no systematic governance change. If governance DOES change, our feature classification is wrong.

2. **Content features**: Select features classified as CONTENT by our Method 1+2 pipeline. Apply same amplification sweep. Prediction: factual output quality changes, but governance metrics stay stable.

3. **IT-suppressed features**: Features with high negative diff_score from Method 3 (features PT uses more than IT — the web-text, HTML, unused-token features). Amplify these in IT. Prediction: output becomes MORE web-text-like and LESS governed — a complement to B1 (suppressing governance features should have similar effect to amplifying anti-governance features).

**Experiment B4: Layer specificity test**

Steer governance features at different corrective sub-ranges:
```
Only L20-25: early corrective (where cosine first goes negative)
Only L26-29: mid corrective (peak weight divergence from PT)
Only L30-33: late corrective (closest to output, L32-33 most important by E3.11)
```
Both suppress (γ=0) and amplify (γ=5.0) at each sub-range.

**Prediction based on E3.11**: L30-33 steering has the largest effect on governance because these layers contain the most important governance features. L20-25 steering has minimal effect — these layers set up the corrective computation but don't directly govern output tokens.

---

## 3. EVALUATION PROTOCOL

### 3.1 Metrics (multi-axis)

We evaluate on 4 axes to test specificity. The core claim is: **steering the corrective stage changes output governance (Axis 1) while preserving content (Axis 3) and coherence (Axis 2)**. If all axes move together, we don't have specificity. If Axis 1 moves alone, we do.

**Axis 1: Output Governance (SHOULD change with steering)**

*Programmatic metrics (reproducible, no LLM):*
- **Structural token ratio**: fraction of generated tokens classified as STRUCTURAL/DISCOURSE/PUNCTUATION by our existing token-type classifier. This is a direct readout of what the corrective stage modifies (75% of mind changes target structural tokens per §5.3).
- **Turn structure detection**: regex-based detection of conversational patterns — "Question:"/"Answer:" pairs, numbered/bulleted lists, markdown headers, code block delimiters. Count per response.
- **Format instruction compliance**: from IFEval subset — did the model follow explicit format instructions (JSON, markdown, numbered list, specific casing)? Binary per prompt.
- **Response structure entropy**: measure regularity of response structure — do responses have consistent paragraph/section patterns? Shannon entropy over sentence-length distribution.
- **Response length distribution**: mean and variance. IT produces longer, more structured responses.

*LLM-judge metrics (richer signal, less reproducible):*
- **Format adherence score** (1–5): "Does this response use appropriate formatting for the question asked?"
- **Conversational register score** (1–5): "Does this response sound like a helpful assistant vs. a web text continuation?"
- **Structural completeness**: "Does the response have a proper greeting/answer/closing structure?"

**Axis 2: Coherence (should be PRESERVED for moderate steering, may degrade at extremes)**
- **Perplexity** under a held-out reference model
- **LLM-judge coherence** (binary): "Is this response coherent and grammatical?"
- **Gibberish detection**: fraction of outputs where >50% of tokens are non-English or repeated
- **Sentence completion rate**: fraction of responses that end with complete sentences (not truncated)

**Axis 3: Content/Factual (should be PRESERVED)**
- **MMLU subset** (200 questions): factual knowledge — multiple choice accuracy
- **TriviaQA subset** (200): factual recall — exact match
- **GSM8K subset** (100): mathematical reasoning — answer correctness
- **ARC-Challenge subset** (100): science reasoning

**Axis 4: Safety (informational — not our core claim, but important to track)**
- **AdvBench refusal rate** (100): does model still refuse harmful requests?
- **XSTest** (100): false-positive refusal rate on safe prompts
- **LLM-judge alignment behavior**: Comply/Refuse/Gibberish classification

### 3.2 Prompt set — designed to isolate output governance

The dataset must demonstrate that steering affects FORMAT AND STRUCTURE but not CONTENT. This means we need prompts where format and content are independently measurable. We use 7 categories, each chosen to stress-test a different aspect of the output governance claim:

**Category GOV-FORMAT (200 prompts) — Format-heavy instructions**
Source: IFEval + custom. Prompts that explicitly request specific formats.
Examples: "List the planets in a numbered list", "Write a JSON object with fields for name and age", "Explain photosynthesis using markdown headers", "Write code to sort a list, with comments".
*Why*: If steering works, this category shows the largest governance metric changes. These prompts make format compliance directly measurable (did it produce JSON? Did it use numbered list? Did it use headers?). Content can be assessed independently.

**Category GOV-CONV (200 prompts) — Conversational structure**
Source: Custom conversational Q&A + multi-turn templates.
Examples: "What is the capital of France?", "Help me understand quantum computing", "What should I cook for dinner tonight?", "Compare Python and JavaScript".
*Why*: Tests whether the model produces conversational-assistant-style responses (greeting → explanation → closing) vs. PT-style raw completion (fragment, web text continuation, no structure). Governance metric: does the response have answer structure? Content metric: is the answer factually correct?

**Category GOV-REGISTER (100 prompts) — Register/style sensitivity**
Source: Prompts where the same content could be expressed in assistant register or web-text register.
Examples: "Explain machine learning" (assistant: "Machine learning is a field of..." vs. web-text: "Machine learning. From Wikipedia..."), "What are the benefits of exercise" (assistant: structured, hedged vs. web-text: listicle/SEO style).
*Why*: Tests whether steering shifts register specifically. LLM-judge can rate "assistant-like" vs. "web-text-like".

**Category CONTENT-FACT (200 prompts) — Pure factual knowledge**
Source: MMLU + TriviaQA. Multiple choice and short-answer factual questions.
Examples: "What year did WWII end?", "Which element has atomic number 79?", MMLU multiple-choice.
*Why*: Content accuracy should be PRESERVED under governance steering. If it degrades at same rate as governance, we lack specificity.

**Category CONTENT-REASON (100 prompts) — Reasoning**
Source: GSM8K + ARC-Challenge. Math word problems and science reasoning.
*Why*: Reasoning should also be preserved — it's not a governance function. Tests that the corrective stage doesn't encode reasoning capability.

**Category SAFETY (100 prompts) — Safety/refusal**
Source: AdvBench (50 harmful) + XSTest (50 safe-but-triggering).
*Why*: Informational — we're not claiming the corrective stage IS safety, but we should track whether safety changes under governance steering. If safety degrades alongside governance, the refusal direction may be a subcomponent.

**Category BASELINE-EASY (100 prompts) — Trivially easy**
Source: Simple one-word-answer questions, basic arithmetic, greetings.
Examples: "What is 2+2?", "Hello!", "What color is the sky?"
*Why*: Matched-token analysis (§4.1) showed that easy tokens need minimal correction (1-layer overhead). These prompts should be LEAST affected by steering — they serve as a ceiling/floor calibration.

**Total: 1,000 prompts**

### 3.3 Compute budget

```
Direction steering (A1):  1,000 prompts × 7 α values × 1 method           =  7,000 runs
Direction steering (A2):  1,000 prompts × 8 β values × 1 method           =  8,000 runs
Controls (A1+A2):         1,000 prompts × 3 controls × 2 (remove+add)     =  6,000 runs
Feature steering (B1):    1,000 prompts × 7 γ values × 5 feature sets     = 35,000 runs
Feature→direction (B2):   1,000 prompts × 8 β values × 1 method           =  8,000 runs
Controls (B3):            1,000 prompts × 3 control types × 2 γ values    =  6,000 runs
Layer specificity (B4):   1,000 prompts × 3 sub-ranges × 2 γ values       =  6,000 runs
─────────────────────────────────────────────────────────────────────────────
Total:                                                                      ~76,000 runs
```

At ~0.5 sec/prompt on H100: ~10.5 hours on 8 GPUs. Manageable in 2 days with some parallelism.

We can reduce by running GOV-FORMAT and CONTENT-FACT categories first (400 prompts, ~40% compute) to get early signal before committing to the full sweep.

---

## 4. WHAT CONSTITUTES SUCCESS

### Strong success (publishable causal evidence):
- A1 shows clean dose-response: governance metrics degrade monotonically as α→0 while content metrics are preserved. The governance curve leads content by ≥0.3α.
- A2 shows PT becomes more structured when corrective direction is injected. Governance metrics increase monotonically with β.
- B1 amplification (γ>1) makes IT outputs MORE formatted/structured (more headers, more lists, more verbose). B1 suppression (γ<1) makes IT outputs more PT-like. Two-sided dose-response.
- B2 (feature→direction into PT) produces governance improvements comparable to or more targeted than A2.
- Controls (random direction, content features) show no governance effect.
- B2 governance direction produces narrower effects than A2 full corrective direction → confirms corrective direction is decomposable.

### Acceptable success:
- A1 shows dose-response on governance with some content degradation at extreme values. As long as governance degrades FIRST and MORE than content, specificity holds.
- A2 shows modest structural improvement in PT — doesn't fully become IT, but structure metrics improve.
- B1 amplification produces overly-formatted but still coherent output (inverted-U curve — peak then decline). This is still informative about the mechanism.
- B2 works but not as cleanly as A2 (transcoder reconstruction adds noise). Directional pattern matches.
- B1 works for some feature sets but not others — reveals which specific features matter.

### Failure modes:
- A1: governance and content degrade together at same rate → corrective direction is not specific to governance
- A2: no change in PT → corrective direction is not transferable (not sufficient for governance)
- B1: amplification (γ>1) produces no change or same change as suppression → features are not doing what labels suggest
- All controls also produce governance changes → our direction/features aren't special
- Everything degrades simultaneously → catastrophic collapse again (shouldn't happen based on Exp5 corrective_directional result)
- B2 produces identical effects to A2 → the governance features don't isolate governance specifically, they just reconstruct the full corrective direction

---

## 5. IMPLEMENTATION PLAN

### What exists (from Exp5):
- ✅ `PrecomputedAblationStats` with corrective_directions already computed
- ✅ `directional_ablate_tensor` with α-sweep support
- ✅ `InterventionSpec` framework with nnsight hook injection
- ✅ LLM-judge evaluation pipeline
- ✅ Custom benchmark suite (IFEval, structural tokens, etc.)
- ✅ Feature labels cached for L20-L33

### What needs building:

**For Approach A (direction steering):**
1. Add `method="directional_add"` to InterventionSpec — inject direction into PT (currently only supports removal from IT)
2. Add control conditions: random direction, rotated direction, content-layer direction
3. Run precompute on PT model to get baseline activations

**For Approach B (feature steering):**
1. **Feature classifier**: LLM-batch classify all 16,384 × 14 feature labels into GOVERNANCE/SAFETY/CONTENT/OTHER
2. **Feature enrichment**: compute structural-token enrichment ratio from existing 3k data
3. **Transcoder interception hook**: at each corrective layer, intercept MLP input, run through transcoder, modify features, decode back, substitute
4. **Feature clamping logic**: scale-clamp at arbitrary γ (not just zero). Need to precompute mean activation ā[f] for each governance feature across the 3k IT collection to define γ=1.0 baseline.
5. **W_dec projection for B2**: Load IT transcoder decoder weights, compute governance direction as weighted sum of decoder vectors, add to InterventionSpec as a new direction type.

**Estimated implementation time**:
- A1/A2 (direction steering): 1 day (builds directly on Exp5 infrastructure)
- B0 (feature selection): 0.5 day (LLM classification + enrichment computation)
- B1 (feature steering with γ sweep): 2 days (transcoder interception hook + magnitude scaling is the new piece)
- B2 (W_dec projection + injection): 0.5 day (conceptually simple once B0 is done — just matrix multiply + direction injection via A2 infrastructure)
- B3-B4 + evaluation + plots: 1.5 days
- **Total: ~5-6 days**

### Run order:
1. **B0**: classify features FIRST (can run overnight while coding A1/A2)
2. **A1** (fastest, most comparable to Exp5): remove corrective direction from IT with α sweep
3. **A2**: add corrective direction to PT with β sweep
4. **B1**: steer governance features in IT with γ sweep (suppress AND amplify)
5. **B2**: compute governance direction via W_dec, inject into PT
6. **B3**: control features (run alongside B1/B2 — same infrastructure)
7. **B4**: layer specificity (last — informational, not critical for the paper)

---

## 6. CONNECTION TO PAPER NARRATIVE

If steering works, the paper argument becomes:

1. **Observational**: We document a corrective computational stage (negative cosine, weight norm concentration, feature reorganization) that post-training amplifies.
2. **Structural**: A feature population transition at layer 11 (Jaccard dip, ID compression) marks a computational phase boundary. The corrective stage lies beyond this boundary.
3. **Causal**: Removing the corrective direction degrades output governance (format, structure) while preserving factual content. Adding it to the base model improves structure. Individual governance features can be identified and their suppression specifically degrades formatting.
4. **Connection to prior work**: The corrective stage encompasses the refusal direction (Arditi et al.) as one component. Our feature-level analysis suggests output governance is a richer phenomenon — format, structure, register — of which safety/refusal is one specialized sub-direction.

This is a clean narrative: observe → characterize → prove causally → connect to literature.

---

## 7. LITERATURE GROUNDING

| Method | Precedent | What they proved | How we extend |
|---|---|---|---|
| Direction extraction (mean diff) | Arditi et al. 2024 | Refusal is a single direction | We extract the corrective direction (broader than refusal) |
| Direction removal (α=0) | Arditi et al. 2024 | Removing direction eliminates refusal | We remove corrective direction, measure governance degradation |
| Direction addition (β>0) | Turner et al. 2023 (ActAdd) | Adding direction steers behavior | We add corrective direction to PT, measure governance improvement |
| Feature clamping | Anthropic Scaling Mono 2024 | Clamping features causes behavioral change | We clamp governance features, measure specific format effects |
| Dose-response (α sweep) | RepE (Zou 2023) | Graded control of concept expression | We sweep α to show graded governance control |
| Cross-model transfer | Representation surgery literature | Representations transfer between models | We inject IT direction into PT |
| Controls (random direction) | Standard in steering literature | Validates specificity | We use random + rotated + cross-phase controls |

| Feature→direction (W_dec) | SAE-targeted steering (Kharlapenko 2024), Feature Guided Activation Additions (2025) | Construct steering vectors from SAE feature decoder vectors | We compose governance features via W_dec into a direction for cross-model injection (B2) |
| Bidirectional feature steering | Anthropic Scaling Mono 2024 (Golden Gate Bridge), Behavioral Steering in MoE (2026) | Amplifying features causes behavioral overshoot | We amplify governance features (γ>1) to show over-formatting, confirming they control governance |

Every methodological choice has direct precedent in published work. No novel methods — only novel application to the corrective stage phenomenon.

---

## 8. ADDITIONAL EXPERIMENTS (CONTINGENT)

These are experiments worth considering if time permits or if the core A+B results raise specific follow-up questions. Ordered by expected value.

### Exp C: Logit Attribution Through Governance Features

**Idea**: For each governance feature f, compute its direct contribution to the logit of each vocabulary token: `logit_contribution[f, v] = a_f × (W_dec[f] · W_U[v])`, where W_U is the unembedding matrix. This tells us exactly which tokens each governance feature promotes/suppresses.

**Why it's valuable**: This is the *interpretability* counterpart to the steering experiments. Steering tells us governance features are causal; logit attribution tells us *what they do* at the token level. If governance feature #1234 promotes "Answer:" and suppresses raw content tokens, that's a specific mechanistic story. OpenAI's SAE latent attribution work (2024) validates this approach.

**Implementation**: ~0.5 day. We have W_dec from the transcoder and W_U from the model. Matrix multiply + sorting. No GPU inference needed.

**Contingent on**: B0 (feature classification). Only meaningful for features we've already identified as governance-related.

### Exp D: Governance Probe at Corrective Layers

**Idea**: Train a simple linear probe on corrective-layer activations to predict "is the next token structural/format?" vs. "is the next token content?". Then measure: (a) does probe accuracy decrease when we steer governance away? (b) is probe accuracy higher in IT than PT at corrective layers?

**Why it's valuable**: This complements the behavioral steering results with a representational test. If the corrective stage contains a linearly-readable governance signal, that's evidence the representation itself encodes governance information, not just that the features happen to affect governance through a complex nonlinear pathway.

**Implementation**: ~1 day. Need to collect corrective-layer activations (already have from 3k collection), train logistic regression, then evaluate under steering conditions.

**Contingent on**: A1 results. Only worth doing if A1 shows clean governance-specific degradation.

### Exp E: Corrective Direction Decomposition

**Idea**: The full corrective direction d_corr is presumably a mixture of governance, safety, and other sub-directions. Can we decompose it? Project d_corr onto the governance direction v_gov (from B2) and the safety direction v_safety (from Arditi-style extraction on harmful/harmless pairs). Measure: what fraction of d_corr is governance? What fraction is safety? Is there a residual?

```python
# d_corr = full corrective direction (IT-PT mean diff)
# v_gov = governance direction (from B2, via W_dec projection)
# v_safety = safety/refusal direction (Arditi-style extraction)
#
# proj_gov = (d_corr · v_gov) / ‖v_gov‖²  × v_gov
# proj_safety = (d_corr · v_safety) / ‖v_safety‖²  × v_safety
# residual = d_corr - proj_gov - proj_safety
#
# Measure: ‖proj_gov‖/‖d_corr‖, ‖proj_safety‖/‖d_corr‖, ‖residual‖/‖d_corr‖
```

**Why it's valuable**: This directly tests the "corrective stage encompasses safety as a subcomponent" claim from §10. If governance accounts for 60% of the corrective direction and safety 15%, we have quantitative evidence for the hierarchy. If they're nearly orthogonal, governance and safety are independent sub-mechanisms within the corrective stage.

**Implementation**: ~0.5 day. Just linear algebra on precomputed directions. But requires safety direction extraction (Arditi methodology on Gemma 3).

**Contingent on**: B2 results + having a clean safety direction.

### What we decided NOT to add:

- **Crosscoder training** (Anthropic 2024–2025): Would give us cross-model feature alignment, but requires training a new model — scope creep for this paper. Better as follow-up.
- **Training dynamics analysis** (intermediate checkpoints): Gemma 3 checkpoints aren't public. Can't do this.
- **Feature splitting/absorption at the dip**: Interesting but tangential to the corrective stage claim. Could confuse reviewers if we dilute the narrative.
- **Full circuit tracing** (Anthropic 2025): Would be ideal but requires infrastructure we don't have and would take weeks. Steering gives us causal evidence at lower cost.
