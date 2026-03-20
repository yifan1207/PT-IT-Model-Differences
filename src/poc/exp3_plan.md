# Experiment 3: The Corrective Computational Stage

## Core Claim

Instruction tuning does not merely add behavioral constraints on top of a fixed
computational pipeline. It **structurally reorganises** the pipeline:

- Layers 0–~10: tokenisation / low-level composition (same in PT and IT)
- Layer ~10–11: **phase boundary** — sharpened from a smooth peak (PT) into a
  sharp dip-and-recovery (IT), suggesting a discrete computational stage transition
- Layers 11–~20: semantic processing / core prediction (shared pipeline)
- Layers ~20–33: **corrective stage** (IT only) — late layers actively oppose
  the residual stream direction, suppressing the raw PT prediction and steering
  toward format, helpfulness, and safety constraints

This corrective stage is **universal** — its magnitude and direction are
consistent regardless of task type (IC / OOC / R) or generation step — which
explains why IT models appear computationally identical to PT at the macro level,
while diverging at the output level.

---

## Hypotheses

### NH1 — IT's sustained late-layer feature activity reflects output governance

IT's higher L0 and stronger delta norms in layers 20–33 reflect three concurrent
processes that do not exist in PT:

1. **Format enforcement**: every token must fit a conversational format (register,
   paragraph structure, no truncation). Requires active suppression of tokens that
   would break format.
2. **Output planning**: the Anthropic Biology paper showed Claude activates
   "rhyming-word" features before writing intermediate words. IT may maintain high
   feature activity by continuously planning the full response structure.
3. **Consistency maintenance**: IT must stay on-topic, avoid factual contradiction,
   and maintain persona — a continuous "check" against accumulated context that
   PT never needs to perform.

**Test (Exp 2d / 1b):** check whether IT's late-generation features are
"forward-looking" — their logit effects promote tokens that appear *later* in
the generation, not the immediate next token.

---

### NH2 — The layer 10–11 transition is a detokenisation/composition boundary sharpened by IT

In PT, layers 10–11 are the activity peak followed by a smooth decline.
In IT, layer 11 is a sharp dip (~90 feature drop, 160→70) then immediate
recovery — a **phase transition**, not a smooth rolloff.

Interpretation: everything before layer 11 is one computational phase
(detokenisation + early feature engineering); everything after is a second phase
(semantic processing + output shaping). Instruction tuning sharpened this boundary
from gradual to discrete.

---

### NH3 — The late-layer residual difference between PT and IT is alignment machinery

Consistent with "Safety Layers in Aligned LLMs" (Li et al., ICLR 2025). Middle
layers build a representation of "what comes next based on the pattern." Late
layers in IT then actively push **against** this raw prediction to enforce:
helpful, harmless, well-formatted, non-repetitive output. PT never acquired this
corrective stage because it was not trained on what a "good response" looks like.

The negative `cos(delta_i, h_{i-1})` in IT late layers is this corrective stage
in action: the layer is explicitly moving the residual stream **away** from where
the middle layers pointed it.

---

## Paper Direction

### Primary paper
> **"Post-Training Creates a Corrective Computational Stage: How Instruction
> Tuning Restructures Layer-Wise Computation in LLMs"**

### Ambitious extensions

**A — The Alignment Tax**
If IT's late layers always oppose the residual regardless of content, the model
pays a fixed corrective cost on every forward pass — including trivially safe
inputs like "What is 2+2?". Quantify: what fraction of total MLP contribution
(by |delta_norm|) comes from layers 20–33? If 30–40%, this is a significant
efficiency finding. Connects to early-exit literature (AdaInfer).

**C — Surgical Removal of the Corrective Stage**
Ablate MLP outputs at layers 25–33 in IT. If output degrades specifically in
format/safety while semantic content survives, this demonstrates the corrective
stage is **modular and removable** — the strongest mechanistic result.

---

## Experiment Phases

### Phase 0 — Sanity Checks and Confound Controls

| # | Experiment | Est. | Why critical |
|---|---|---|---|
| 0a | Transcoder reconstruction MSE per layer, PT vs IT | 0.5 d | If IT late-layer error is 2× PT, L0 finding is partially artifactual. Must check before publishing. Finding 2 (cosine) is safe regardless — uses raw residuals. |
| 0b | Run IT on raw completion prompts (no chat template) | 1 d | Tests whether corrective stage is weight-driven or format-triggered. Critical confound: if correction disappears without the chat template, it is prompt-induced not architecture-induced. |
| 0c | Per-layer weight Frobenius norm PT vs IT | 0.1 d | **Already implemented** via `weight_shift.py`. Did post-training increase late-layer weight magnitudes? If yes, model learned to make bigger edits there. |

---

### Phase 1 — Core Mechanistic Evidence

| # | Experiment | Est. | Why |
|---|---|---|---|
| 1a | Answer emergence and stabilisation — logit lens PT vs IT | 1–2 d | Track rank and probability of the *final generated token* across layers. Key metrics: earliest layer where token enters top-k, becomes top-1, and where the layer distribution stabilises (KL-to-final drops below threshold). If PT commits early and IT keeps modifying, that is the cleanest corrective stage evidence. |
| 1b | Attraction vs repulsion decomposition per layer | 1 d | Decompose each layer's logit-lens delta into: push toward correct token (attraction) and push away from top-wrong tokens (repulsion). Tests whether IT late layers carry more *repulsive* mass than PT — i.e., the corrective stage is suppressive, not just directive. |
| 1c | IC / OOC / R comparison restricted to layers 0–20 | 1 d | Tests whether task-type universality is a genuine base property or a correction-stage artifact. Redo L0 and Jaccard on pre-correction layers only. Uses existing exp2 results — no new inference. |

---

### Phase 2 — Deepening the Corrective Stage Finding

| # | Experiment | Est. | Why |
|---|---|---|---|
| 2a | Token-type stratification of corrective stage magnitude | 1–2 d | Split generated tokens into: (a) first content token, (b) punctuation/whitespace, (c) discourse markers ("Well,", "Let me", "I think"), (d) EOS/turn tokens. Check if late-layer delta norm is larger for style/format/control tokens than core content. Strongly supports the governance interpretation. Categories (b)-(d) classified by token-list lookup — no LLM judge needed. |
| 2b | Quantify the alignment tax | 0.5 d | `tax = Σ|delta_norm[20:33]| / Σ|delta_norm[0:33]|`. Compare PT vs IT. Uses existing exp2 `layer_delta_norm` results — no new inference. |
| 2c | Layer-to-final KL divergence trajectory PT vs IT | 1 d | For each layer i, `KL(softmax(logit_lens_i) ∥ softmax(logit_lens_33))`. When does each model "commit"? The layer where KL drops sharply is the commitment point. PT and IT may have different commitment depths. Computable from residuals already captured in exp2 (see `collect.py` notes). |
| 2d | Generation-step-resolved corrective stage | 1 d | Per-token cosine heatmap (layers × generation position) split by token type. **Largely done** — Plot 11 is the overall version. Extend by stratifying by token type from 2a. |

---

### Phase 3 — Causal Validation

| # | Experiment | Est. | Why |
|---|---|---|---|
| 3a | Ablate corrective layers in IT | 1 d | Zero out MLP outputs at layers 25–33 during IT generation. Does output become PT-like? Does semantic content survive while format/safety breaks? This makes the corrective stage causal, not merely correlational. |
| 3b | Activation patching: PT residual → IT late layers | 1–2 d | Run PT on a prompt, save `h_20` (residual after layer 20). Run IT on the same prompt; at layer 20, replace IT's residual with PT's saved value. Continue IT generation from that point. Does the IT corrective stage "fix" PT's raw mid-layer prediction into IT format? |

---

### Phase 4 — Generalisation and Scale

| # | Experiment | Est. | Why |
|---|---|---|---|
| 4a | Replicate cosine finding on second model family | 2–3 d | Raw residual cosine metric (no transcoders needed) on Llama 3.1 8B base vs instruct, or Qwen 2.5 base vs instruct. If the negative late-layer cosine replicates, the corrective stage is a universal property of RLHF/SFT post-training. |
| 4b | Increase dataset size and diversity | 1–2 d | More prompts; matched pairs across IC/OOC/R; varied prompt formats. Strengthens all statistical tests and reduces selection bias. |
| 4c | PT vs IT benchmark accuracy on prompt set | 0.5 d | Does IT actually perform better on reasoning tasks, or only format better? Contextualises all findings — if accuracy is the same, the corrective stage is pure overhead. |

---

## What Can Be Computed From Existing Exp2 Data (No New Inference)

The following analyses work directly on `exp2_results.json`:

| Analysis | Input field | Notes |
|---|---|---|
| 1c pre-correction L0/Jaccard | `l0`, `active_features` | Filter to layers 0–20 |
| 2a token stratification | `generated_tokens`, `layer_delta_cosine` | Classify tokens by lookup table |
| 2b alignment tax | `layer_delta_norm` | Sum layers 0–20 vs 20–33 |
| 2d cosine heatmap per token type | `layer_delta_cosine`, `generated_tokens` | Extension of Plot 11 |

The following require new inference (extended `collect.py`):

| Analysis | New quantity | How collected |
|---|---|---|
| 0a transcoder MSE | `transcoder_mse[step][layer]` | New nnsight hook: `mlp.output` |
| 1a emergence / rank | `next_token_rank[step][layer]`, `next_token_prob[step][layer]` | Post-trace from existing residuals |
| 1b attraction/repulsion | `logit_delta_contrib[step][layer]` | `logit_lens[i][token] - logit_lens[i-1][token]`, post-trace |
| 2c KL-to-final | `kl_to_final[step][layer]` | `KL(lens_i ∥ lens_33)`, post-trace |
| 3a ablation | New generation run with nnsight intervention | Separate script |
| 3b patching | Two-model coordinated trace | Separate script |

---

## Literature

### Tier 1 — Must Read and Cite

| Paper | Year | Venue | Connection |
|---|---|---|---|
| URIAL (Lin et al.) | 2024 | ICLR | 77.7% of IT tokens unchanged from PT. The corrective stage explains the other 22.3%. |
| Safety Tax (Huang et al.) | 2025 | arXiv | Coined "Safety Tax" for reasoning degradation from alignment. This work provides the mechanism. |
| Mitigating Alignment Tax of RLHF (Luo et al.) | 2024 | EMNLP | Weight averaging reduces the tax. Explanation: it dilutes corrective stage weights. |
| Refusal Mediated by Single Direction (Arditi et al.) | 2024 | NeurIPS | Refusal direction is one component of the broader corrective subspace. |
| Safety Layers in Aligned LLMs (Li et al.) | 2025 | ICLR | Safety in specific late layers. Generalisation: ALL output correction lives there. |
| LARF: Layer-Aware Representation Filtering | 2025 | EMNLP | Identifies safety-sensitive layers via scaling; cosine metric is complementary. |
| DoLA: Decoding by Contrasting Layers (Chuang et al.) | 2024 | ICLR | Works by contrasting early/late layers. Corrective stage explains why it works. |
| FAST: SAEs for Instruct Models (Li et al.) | 2025 | arXiv | Base SAEs fail on IT — critical methodological check for transcoder validity. |
| Crosscoder Model Diffing (Lindsey et al.) | 2024–2025 | Anthropic | Shared/exclusive features PT↔IT. Layer-wise dynamics complement their feature analysis. |
| Stages of Inference (Lad, Gurnee & Tegmark) | 2024 | arXiv | Four stages for PT. This work adds/modifies: five stages for IT, or qualitative sharpening of the stage boundary. |

### Tier 2 — Important Context

| Paper | Year | Venue | Connection |
|---|---|---|---|
| SFT Memorizes, RL Generalizes (Chu et al.) | 2025 | ICML | Disentangle which post-training method creates the corrective stage |
| Embers of Autoregression (McCoy et al.) | 2023–2024 | PNAS | Task-type universality might be a correction-stage artifact, not a base property |
| Kim, Yoo & Oh | 2025 | ICML | Layer-wise dynamics for certain/uncertain — check if tested on IT only |
| Transformer Dynamics (Guitchounts et al.) | 2025 | — | Residual stream as dynamical system; corrective stage creates a repellor |
| Demystifying Layer Roles | 2025 | — | Different tasks need different layers under pruning; connects to where correction matters |
| Base Models Know How to Reason | 2026 | ICLR | Reasoning exists in PT; corrective stage might suppress as much as it formats |
| Anthropic Biology (Lindsey et al.) | 2025 | Anthropic | Circuit tracing shows task-specific circuits; corrective stage might mask these |

### Tier 3 — Alignment Tax Angle

| Paper | Year | Connection |
|---|---|---|
| AdaInfer: Not All Layers Necessary | 2025 | Early exit for efficient inference; corrective stage defines what to skip |
| Light Alignment via Single Neuron | 2026 | Lightweight alternatives; shows why they might suffice |
| Echo Chamber: RL Amplifies Pretraining | 2025 | Correction might amplify a latent PT signal; check if PT has weak negative cosine |
| Representation Engineering (Zou et al.) | 2023 | Control vectors might work by modulating the corrective stage |
| Behavior Shift after Instruction Tuning | 2024 | NAACL | Systematic PT vs IT representation comparison |

### Tier 4 — Methods

| Paper | Year | Connection |
|---|---|---|
| Logit Lens + Tuned Lens | 2020/2023 | Project residual before/after correction into vocab space |
| Sparse Feature Circuits (Marks et al.) | 2025 | ICLR | Causal feature identification methodology |
| RouteSAE | 2025 | EMNLP | Multi-layer feature extraction; could capture cross-layer correction features |
