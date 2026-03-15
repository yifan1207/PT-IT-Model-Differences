# Hierarchical Distributional Narrowing in Next-Token Prediction
## Research Document v2 — Deepened Theory + Implementation Plan

**Author**: Yifan (working notes)  
**Date**: March 2026  
**Status**: POC → Scaled experiments

---

## Part I: The Central Idea (Revised)

### 1. What We're Actually Studying

Next-token prediction, even for reasoning questions, is a **composite process** that combines multiple types of computation at every token position:

1. **Structural/format narrowing** — "The answer to a math question is a number" (50k tokens → ~100)
2. **N-gram / pattern-based narrowing** — "After 'is' in this context, a single digit is likely" (~100 → ~10)
3. **Semantic/associative narrowing** — "This involves multiplication of small numbers, answer is likely single digit" (refines further)
4. **Computational/reasoning narrowing** — "3 × 2 = 6 specifically" (~10 → 1)

These aren't separate modules — they're interleaved, partially overlapping computations implemented by features that may serve multiple levels simultaneously. But they ARE distinguishable in attribution graphs because they have different **logit-effect signatures** (how broadly vs narrowly they push the output distribution) and different **upstream activation patterns** (what triggers them).

The central question: **At each token position in a reasoning task, how is the model's total computation distributed across these levels of narrowing? How does this distribution change with task difficulty, model scale, and whether reasoning is in-context vs in-weights?**

### 2. Why This Matters Beyond Description

This isn't just "some features are structural, some are semantic" — that's obvious. The non-obvious claims are:

**Claim 1: The distribution across levels is systematically skewed.** Format/structural narrowing features dominate attribution by construction (larger logit shifts), which means SAE-based circuit analysis systematically underrepresents the computational work done by reasoning features. Anyone using attribution graphs to study reasoning is seeing a distorted picture.

**Claim 2: The levels become more separable at scale.** In small models, the same features serve multiple levels (entanglement). In large models, features specialize to individual levels. This is a mechanistic account of "emergent reasoning" — not that large models learn new capabilities, but that their existing capabilities become decomposable and composable.

**Claim 3: Where lookup ends and composition begins is measurable.** For memorized facts, level 4 is just another lookup. For genuine computation, level 4 requires composing outputs from multiple level-3 features. This composition is where reasoning actually happens, and it's the part most invisible to standard attribution analysis.

**Claim 4: CoT and ICL change the distribution across levels.** Chain-of-thought externalizes level 4 computation into the context, converting it into level 1-2 operations (just read the number from context). ICL examples strengthen level 2-3 features (pattern recognition) which scaffolds level 4 (computation). These are testable predictions about how the narrowing hierarchy shifts.

### 3. The Theoretical Foundation

#### 3.1 Information-Theoretic View

At each token position, the model reduces the entropy of the output distribution from H₀ (uniform over vocab, ~15.6 bits for 50k tokens) to H_final (concentrated on the correct token, ~0 bits). Each feature contributes some ΔH to this reduction.

Format-routing features provide large ΔH (e.g., 50k→10 is ~12 bits).
Value-computing features provide smaller ΔH (e.g., 10→1 is ~3.3 bits).
But the computational difficulty is INVERSE to the ΔH: narrowing to "a number" is trivial pattern matching, while narrowing to "6 specifically" requires actual computation.

**The paradox**: The features doing the hardest computational work get the smallest attribution. This is because attribution tracks logit-shift magnitude, which correlates with ΔH, not computational difficulty.

#### 3.2 Connection to "Embers of Autoregression" (McCoy et al. 2023)

McCoy et al. argued that many apparent LLM capabilities are better explained by the statistics of next-token prediction than by genuine understanding. Our framework makes this precise: levels 1-3 of narrowing ARE statistical pattern matching (n-gram statistics, distributional regularities). Only level 4 potentially involves computation that goes beyond pattern matching. The question "does this model reason?" translates to "does this model have level-4 features that compose rather than just lookup?"

#### 3.3 Connection to "I Predict Therefore I Am" (2025)

This paper proves that under mild conditions, LLM representations learned through next-token prediction can be modeled as log-posteriors over latent discrete concepts. In our framework: each narrowing level corresponds to a different granularity of latent concept. Level 1 concepts = broad categories (number, word, punctuation). Level 4 concepts = specific values (the answer is 6). The hierarchical narrowing we observe might reflect the model's learned hierarchy of latent concepts.

#### 3.4 The "Equi-Learning Law" (He & Su 2024)

This paper shows each transformer layer contributes equally to reducing prediction error. This seems to contradict our observation that format and value features occupy different layers. Resolution: each layer contributes equally to *total* entropy reduction, but different layers reduce entropy at *different levels of the hierarchy*. Early layers do broad narrowing (large ΔH per feature, easy computation), later layers do fine narrowing (small ΔH per feature, hard computation). The total ΔH per layer is constant, but the *nature* of the reduction changes.

#### 3.5 NTP's Sparse + Low-Rank Decomposition (Zhao et al. 2024)

The logit matrix learned through NTP decomposes into a sparse component (empirical next-token probabilities — our level 1-2 features) and a low-rank component (co-occurrence support patterns — our level 3-4 features). Context embeddings with identical next-token supports collapse to collinear directions ("subspace collapse"). This predicts that format-routing features should form a lower-dimensional subspace than value-computing features — testable via PCA on feature decoder vectors.

### 4. Revised Hypothesis Set

**H1 (Hierarchical Narrowing)**: The model's per-token computation can be decomposed into features operating at different levels of distributional specificity, from broad category (format) to specific value (computation), with intermediate levels for pattern-based and associative narrowing.

**H2 (Attribution Inversion)**: Features doing the most computationally difficult work (level 4, genuine reasoning) have the smallest attribution magnitudes, because they narrow the distribution over a smaller range.

**H3 (Scale-Dependent Separability)**: At small model scales, features serve multiple narrowing levels simultaneously (entangled). At large scales, features specialize to individual levels. The degree of specialization is measurable via ablation dissociation tests.

**H4 (Composition as the Signature of Reasoning)**: The boundary between lookup and genuine computation is detectable as the point where features must compose (combine outputs from multiple upstream features) rather than simply activate based on a learned pattern. This composition shows up as features with multiple strong incoming edges from different content-bearing features in the attribution graph.

**H5 (ICL Restructures the Hierarchy)**: In-context examples change which narrowing level does the most work. With good examples, ICL shifts work from level 4 (internal computation) to levels 2-3 (pattern matching from examples), making reasoning more robust but potentially less general.

---

## Part II: Papers to Read (Priority Ordered)

### Must-Read for Core Theory

1. **Lindsey et al. 2025** — "On the Biology of a Large Language Model" (addition circuits, lookup-table features, multi-context generalization) — READ THE ADDITION SECTION CAREFULLY
2. **Temporal SAEs** (arxiv 2511.05541) — SAE bias toward structural features, temporal smoothness as fix
3. **Mahowald et al. 2024** — "Dissociating Language and Thought in LLMs" (formal vs functional competence)
4. **Bloom & Templeton** — "Understanding SAE Features with the Logit Lens" (partition/suppression/prediction feature taxonomy)
5. **"I Predict Therefore I Am"** (arxiv 2503.08980) — NTP learns log-posteriors over latent discrete concepts, identifiability result

### Must-Read for Methodology

6. **Marks et al. ICLR 2025** — "Sparse Feature Circuits" (SHIFT ablation, feature-level circuit analysis)
7. **Ameisen et al. 2025** — "Circuit Tracing" (attribution graph methods, transcoder details)
8. **"SAEs Are Good for Steering — If You Select the Right Features"** (EMNLP 2025) — input vs output features, output score metric
9. **Dunefsky et al. 2024** — "Transcoders Find Interpretable LLM Feature Circuits" (transcoder methodology, greater-than circuit)

### Should-Read for Context

10. **He & Su 2024** — "A Law of Next-Token Prediction" (equi-learning across layers)
11. **Zhao et al. 2024** — NTP sparse + low-rank decomposition, subspace collapse
12. **McCoy et al. 2023** — "Embers of Autoregression" (what NTP statistics explain vs don't)
13. **"Revisiting ICL Inference"** (ICLR 2025) — 3-step ICL decomposition
14. **Quirke & Barez 2024/2025** — "Arithmetic in Transformers Explained" (full mechanistic account of addition/subtraction circuits)
15. **Levelt 1999** — "Models of Word Production" (lemma/lexeme distinction in speech production)
16. **Lin et al. 2025** — "Reasoning Bias of Next-Token Prediction Training" (how NTP training creates reasoning shortcuts)
17. **OOCR primer** (outofcontextreasoning.com) + Greenblatt 2025 lesswrong posts on 2-hop latent reasoning

### Worth Skimming

18. **Tigges et al. NeurIPS 2024** — Circuit consistency across training and scale
19. **"Circuit Compositions"** (2025) — Modular circuit reuse across tasks
20. **Flesch et al. bioRxiv 2025** — Transformer-brain alignment in relational reasoning (positional encoding ↔ visual cortex, attention ↔ frontoparietal)
21. **"Base Models Know How to Reason"** (ICLR 2026) — Reasoning taxonomy from restricted SAEs, steering vectors

---

## Part III: Practical Implementation — Automated Circuit Analysis

### 5. Prompt Design Principles

**Constraints for good circuit-tracing prompts:**
- Short (< 15 tokens total) — long prompts create huge attribution graphs that are uninterpretable
- Single correct answer that is 1-3 tokens — enables clean ablation analysis
- Answer should be a specific token, not open-ended — we need ground truth
- Vary reasoning type systematically — memorized vs computed vs ICL vs OOCR

**Prompt set (30 prompts, 6 tiers of 5):**

```
TIER 1: In-weights, trivially memorized (1-hop lookup)
  "3 x 2 is"           → 6
  "7 + 5 is"           → 12
  "9 - 3 is"           → 6
  "The capital of France is" → Paris
  "H2O is called"      → water

TIER 2: In-weights, probably memorized (still lookup)
  "8 x 7 is"           → 56
  "15 + 28 is"         → 43
  "The capital of Peru is"  → Lima
  "CO2 stands for carbon"   → dioxide
  "Opposite of hot is"      → cold

TIER 3: In-weights, requires some computation
  "13 x 4 is"          → 52
  "47 + 36 is"         → 83
  "sqrt of 144 is"     → 12
  "Days in Feb 2024 is"    → 29
  "Next prime after 7 is"  → 11

TIER 4: ICL with examples (in-context reasoning)
  "2+3=5, 4+1=5, 7+8="     → 15
  "cat:animal, rose:plant, dog:" → animal
  "AB=BA, CD=DC, EF="       → FE
  "1:one, 2:two, 5:"        → five
  "hot→cold, big→small, fast→" → slow

TIER 5: ICL with novel operations (must reason from examples)
  "a#b=a+2b: 3#1=5, 2#4=10, 1#3="  → 7
  "f(x)=2x+1: f(3)=7, f(5)=11, f(4)=" → 9
  "XY→YX: AB→BA, CD→DC, MN→"       → NM
  "♦2=4,♦3=9,♦5="                    → 25
  "abc→cba: hello→olleh, cat→"       → tac

TIER 6: Out-of-context reasoning (no examples, needs latent knowledge)
  "Taylor Swift birth year Nobel literature:" → Cela (very hard)
  "Protons in carbon plus oxygen is"         → 14
  "Planet count minus inner planets is"      → 4
  "Vowels in 'education' is"                → 5
  "Fibonacci after 5,8 is"                  → 13
```

### 6. Implementation with Gemma Scope + Circuit Tracer

#### 6.1 Setup

```bash
# Install circuit-tracer (Anthropic's open-source library)
pip install circuit-tracer  # or clone from github

# Gemma Scope SAEs are available via:
# - Neuronpedia (web interface for exploration)
# - SAELens (Python library for programmatic access)
pip install sae-lens transformer-lens
```

#### 6.2 Core Analysis Script (Pseudocode)

```python
import json
from circuit_tracer import CircuitTracer
from sae_lens import SAE
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gemma-2-2b")
sae = SAE.from_pretrained("gemma-scope-2b-pt-res")  # residual stream SAE

tracer = CircuitTracer(model, sae)

def analyze_prompt(prompt: str, correct_token: str) -> dict:
    """Full analysis pipeline for one prompt."""
    
    # 1. Get attribution graph
    graph = tracer.get_attribution_graph(
        prompt, 
        target_token=correct_token,
        threshold=0.01  # minimum attribution to include
    )
    
    # 2. Extract features and compute metrics
    features = []
    for node in graph.top_features(k=30):
        # Get logit effect: how does this feature push the output distribution?
        logit_effect = node.decoder_weight @ model.W_U  # project to vocab
        probs = torch.softmax(logit_effect, dim=-1)
        
        feature_data = {
            "id": node.feature_id,
            "label": node.auto_label,
            "layer": node.layer,
            "position": node.token_position,
            "attribution": node.attribution_to_target,
            "activation": node.activation_magnitude,
            # Narrowing level metrics
            "logit_entropy": -torch.sum(probs * torch.log(probs + 1e-10)).item(),
            "digit_concentration": probs[DIGIT_TOKENS].sum().item(),
            "correct_token_rank": get_rank(logit_effect, correct_token),
            "top5_promoted": get_top_k_tokens(logit_effect, k=5),
            # Upstream connections
            "incoming_edges": [
                {"from_feature": e.source.feature_id, 
                 "from_label": e.source.auto_label,
                 "weight": e.weight}
                for e in node.incoming_edges[:10]
            ],
        }
        features.append(feature_data)
    
    # 3. Auto-classify narrowing level
    for f in features:
        if f["logit_entropy"] < 4.0 and f["digit_concentration"] > 0.3:
            f["narrowing_level"] = 1  # broad format
        elif f["logit_entropy"] < 6.0 and f["digit_concentration"] > 0.1:
            f["narrowing_level"] = 2  # fine format
        elif f["correct_token_rank"] <= 5:
            f["narrowing_level"] = 4  # value-computing
        else:
            f["narrowing_level"] = 3  # intermediate/associative
    
    # 4. Compute narrowing level attribution fractions
    total_attr = sum(abs(f["attribution"]) for f in features)
    level_fractions = {}
    for level in [1, 2, 3, 4]:
        level_attr = sum(abs(f["attribution"]) for f in features 
                        if f["narrowing_level"] == level)
        level_fractions[f"level_{level}_fraction"] = level_attr / total_attr
    
    return {
        "prompt": prompt,
        "correct_token": correct_token,
        "features": features,
        "level_fractions": level_fractions,
        "total_features": len(features),
    }

# Run on all prompts
results = []
for prompt, answer in PROMPT_SET:
    result = analyze_prompt(prompt, answer)
    results.append(result)
    
# Save for analysis
with open("narrowing_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
```

#### 6.3 Ablation Script (Pseudocode)

```python
def ablation_experiment(prompt, correct_token, features_by_level):
    """Test dissociability of narrowing levels."""
    
    results = {}
    
    # Baseline
    baseline_output = model.generate(prompt, max_tokens=5)
    baseline_logprobs = model.get_logprobs(prompt)
    results["baseline"] = {
        "output": baseline_output,
        "correct_prob": baseline_logprobs[correct_token]
    }
    
    # Ablate each level separately
    for level in [1, 2, 3, 4]:
        level_features = [f for f in features_by_level if f["narrowing_level"] == level]
        
        # Zero-ablation: set feature activations to zero
        with model.hooks(zero_ablate_features=level_features):
            ablated_output = model.generate(prompt, max_tokens=5)
            ablated_logprobs = model.get_logprobs(prompt)
        
        results[f"ablate_level_{level}"] = {
            "output": ablated_output,
            "correct_prob": ablated_logprobs.get(correct_token, 0),
            "still_outputs_number": any(t in ablated_output for t in DIGIT_STRINGS),
            "still_correct": correct_token in ablated_output,
            "coherent": not is_nonsense(ablated_output),  # heuristic check
        }
    
    # Ablate all format (levels 1+2) keeping value (levels 3+4)
    format_features = [f for f in features_by_level 
                       if f["narrowing_level"] in [1, 2]]
    with model.hooks(zero_ablate_features=format_features):
        results["ablate_all_format"] = {
            "output": model.generate(prompt, max_tokens=5),
            # Does the right value leak through in any form?
        }
    
    # Ablate all value (levels 3+4) keeping format (levels 1+2)  
    value_features = [f for f in features_by_level 
                      if f["narrowing_level"] in [3, 4]]
    with model.hooks(zero_ablate_features=value_features):
        results["ablate_all_value"] = {
            "output": model.generate(prompt, max_tokens=5),
            # Does the model still output a number (right format, wrong value)?
        }
    
    return results
```

#### 6.4 Using Claude Code as Analysis Agent

Rather than having Claude Code run the experiments (fragile), use it to:

1. **Generate the analysis scripts** from the pseudocode above, adapted to your exact environment
2. **Post-process the JSON results** — "here are 30 prompts' worth of attribution data, identify patterns in how the narrowing level fractions differ across tiers"
3. **Generate visualizations** — "plot the fraction of attribution at each narrowing level as a function of prompt tier"
4. **Write up findings** — "summarize what the ablation experiments show about dissociability"

The human-in-the-loop part: interpreting whether a feature is "really" format vs value, designing follow-up experiments based on surprises, and deciding what story the results tell.

#### 6.5 Practical Notes for Gemma Scope

- **SAEs available**: Gemma Scope provides SAEs for Gemma-2-2B at layers 0-25 with 16k and 65k feature dictionaries. Use the 16k version for interpretability (sparser, cleaner features).
- **Circuit tracer**: Supports Gemma-2-2B with per-layer transcoders. The circuit-tracer library outputs attribution graphs compatible with Neuronpedia's visualization frontend.
- **Batch size**: Circuit tracing for a single prompt takes ~30-60 seconds on an A100. For 30 prompts, budget ~30 minutes compute.
- **Ablation**: Steering via SAE features is straightforward with SAELens hooks. Zero-ablation = set feature activation to 0 before reconstructing. Negative steering = multiply activation by -1 (different from zero-ablation, creates artifacts).

### 7. What To Look For In Results

Beyond the narrowing level fractions, look for these specific patterns:

**Pattern A: Layer progression of narrowing levels.**
Plot: for each layer (0-25), what fraction of active features at that layer are level 1 vs 2 vs 3 vs 4? Prediction: levels 1-2 dominate early layers, levels 3-4 dominate later layers.

**Pattern B: Gathering point identification.**
For each prompt, is there a specific token position where both format and value features have high activation? Does this position consistently correspond to a delimiter (space, =, :)?

**Pattern C: Composition signatures.**
For level-4 (value-computing) features, count the number of strong incoming edges from other content-bearing features. High in-degree from diverse sources = composition. Low in-degree = lookup. Compare this across tiers: tier 1-2 should show more lookup, tier 3-5 should show more composition.

**Pattern D: ICL restructuring.**
Compare tier 4 (ICL with examples) to tier 2 (in-weights, similar difficulty). Do ICL prompts have different narrowing level fractions? Specifically: do ICL examples create new level-2/3 features (pattern matching from examples) that reduce the load on level-4 features?

**Pattern E: Failure mode diagnosis.**
For prompts where the model gets the answer wrong, which narrowing level fails? Does format succeed while value fails (outputs a number, but wrong one)? Or does everything fail together?

**Pattern F: Feature reuse across domains.**
Do the same level-1 format features appear across different domains (math, factual QA, code)? Are level-4 value features domain-specific?

### 8. Timeline

**Week 1**: Set up circuit-tracer + SAELens environment, run analysis on tier 1 (5 prompts), verify the pipeline works and produces interpretable results.

**Week 2**: Run full 30-prompt analysis. Post-process, look for patterns A-F. Write up initial findings.

**Week 3**: Run ablation experiments on the 10 most interesting prompts (based on week 2 findings). Focus on dissociability tests.

**Week 4**: Based on results, decide: scale to Gemma-2-9B? Try Temporal SAEs? Focus on ICL tier specifically? Write POC report.

---

## Part IV: Open Questions and Adjacent Directions

### On the nature of "reasoning" in this framework

The deepest question this project touches: **is there a qualitative boundary between level-3 (associative/pattern) and level-4 (computational) narrowing, or is it a continuum?**

If it's a continuum: "reasoning" is just pattern matching at higher levels of abstraction, and there's no sharp distinction between "the model memorized 3×2=6" and "the model computed 23×17=391." The difference is just how many lookup-table features need to compose.

If there's a boundary: there exists a point where the model transitions from activating pre-learned features to actually performing novel computation through feature composition. This boundary would be detectable as a qualitative change in the attribution graph structure — from "single feature with high activation" (lookup) to "multiple features with moderate activations combining through edges" (composition).

The tier 5 prompts (novel operations) are specifically designed to test this. If the model can solve "a#b=a+2b: 1#3=?" by composing addition and multiplication features in a new way, that's evidence for genuine composition. If it can't, that's evidence that "reasoning" in this model is limited to lookup.

### On what we CAN'T learn from this

Even if all experiments succeed perfectly, we still can't claim:
- That LLMs have "dual systems" like humans (we can show functional decomposition, not architectural modularity)
- That the narrowing hierarchy is the "right" decomposition (there may be other equally valid decompositions)
- That larger models genuinely "reason" (we can only show their features are more specialized and composable)
- That this generalizes beyond Gemma-2 (architecture-specific features might drive the decomposition)

### What would make this a great paper

The paper becomes great if we can show a **clean, surprising, and practically useful** result. Candidates:

- **Surprising**: Level-4 value-computing features contribute <10% of total attribution even on reasoning tasks where the model gets the answer right. Reasoning is invisible to standard attribution analysis.
- **Clean**: The dissociability index increases monotonically with model scale across 3+ model sizes.
- **Useful**: The narrowing level analysis can predict which prompts a model will fail on, or diagnose WHY it failed (format error vs computation error), enabling targeted interventions.

The worst outcome: "some features are broad, some are specific, they're partially entangled" — this is true but uninteresting. The project needs to find something that changes how people think about attribution or reasoning.
