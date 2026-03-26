# Cross-Model Replication Plan v2: Proving the Corrective Stage is Universal
## NeurIPS 2026 — Targeting Oral-Level Rigor

**Date**: 2026-03-25 (updated)
**Goal**: Demonstrate the corrective computational stage in 5–6 model families (including MoE), proving it is an emergent architectural principle of instruction tuning, not a Gemma-specific artifact.

**What changed from v1**: Expanded from 4 to 6 model families. Added DeepSeek (MoE) and OLMo (fully open). Added MoE-specific methodology (§3). Added theoretical framing of what cross-model replication means (§10). Added related work analysis (§11).

---

## 1. WHY CROSS-MODEL REPLICATION MATTERS

### The reviewer argument we must defeat
"Your findings are on Gemma 3 4B only. This could be a Gemma-specific training artifact, not a general property of instruction tuning."

### What NeurIPS oral-level papers do
- Yu et al. (2025) tested across **6 models** with four convergent methods.
- Arditi et al. (2024, refusal direction) showed the refusal direction in **5 models** (Llama 2 7B/13B/70B + Gemma 2B/7B).
- Lad et al. (2024, stages of inference) tested on **4 models** (GPT-2, Pythia 1.4B/6.9B, Llama 2 7B).

**Standard for strong NeurIPS mech-interp papers: 3–5 model variants across 2–3 families. We target 6 models across 6 families.**

### What we need to show
For each model, the corrective stage appears as:
1. Negative δ-cosine band in late layers, significantly stronger in IT than PT
2. IT commits later than PT (delayed commitment)
3. Weight changes concentrated at corrective layers
4. (Ideally) Direction steering dose-response on governance with content flat

If (1)–(3) hold across 5+ families, that's the "universal architectural principle" claim. If (4) also holds on at least one non-Gemma model, that's the "causal" generalization. If it holds on an MoE model, that's a substantial additional contribution.

---

## 2. MODEL SELECTION

### Selection criteria
- Both PT (base) and IT (instruct) variants publicly available
- Different model families (different company, training data, training methodology)
- Comparable size range (3B–8B active parameters)
- Widely recognized by the community
- Mix of dense and MoE architectures

### Selected models (6 families)

| # | Model | Family | Arch | Layers | d_model | Active Params | PT HF ID | IT HF ID |
|---|---|---|---|---|---|---|---|---|
| 1 | **Gemma 3 4B** | Google | Dense | 34 | 2560 | 4B | google/gemma-3-4b-pt | google/gemma-3-4b-it |
| 2 | **Llama 3.1 8B** | Meta | Dense | 32 | 4096 | 8B | meta-llama/Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct |
| 3 | **Qwen 3 4B** | Alibaba | Dense | 36 | 2560 | 4B | Qwen/Qwen3-4B-Base | Qwen/Qwen3-4B |
| 4 | **Mistral 7B v0.3** | Mistral AI | Dense | 32 | 4096 | 7B | mistralai/Mistral-7B-v0.3 | mistralai/Mistral-7B-Instruct-v0.3 |
| 5 | **DeepSeek-V2-Lite** | DeepSeek | **MoE** | 27 | 2048 | 2.4B | deepseek-ai/DeepSeek-V2-Lite | deepseek-ai/DeepSeek-V2-Lite-Chat |
| 6 | **OLMo 2 7B** | AI2 | Dense | 32 | 4096 | 7B | allenai/OLMo-2-0325-7B | allenai/OLMo-2-0325-7B-Instruct |

### Why these six

**Diversity**:
- **6 companies**: Google, Meta, Alibaba, Mistral AI, DeepSeek, AI2 — maximum diversity of training data and methodology.
- **Architecture diversity**: 5 dense + 1 MoE. The MoE model (DeepSeek-V2-Lite) tests whether the corrective stage survives expert routing.
- **Openness spectrum**: Fully open (OLMo: weights + data + code + recipe) to partially open (DeepSeek: weights only). OLMo's fully open nature means reviewers can verify training details.
- **Layer count variation**: 27, 32, 32, 34, 36 layers. Tests the fractional depth hypothesis across a range.

**Why DeepSeek-V2-Lite specifically**:
- It's the smallest DeepSeek MoE model (16B total, 2.4B active per token) — fits on a single GPU.
- Has both base and Chat variants on HuggingFace.
- Uses the same MoE architecture as the full V2/V3 (shared experts + routed experts, Multi-Head Latent Attention).
- Tests MoE without the compute burden of 671B parameter models.
- DeepSeek is extremely popular in the community — including it raises the paper's relevance.

**Why OLMo**:
- Fully open: training data composition, training recipe, all intermediate checkpoints published.
- If the corrective stage appears in OLMo, we can make stronger claims about the training process because we know exactly what went in.
- Different training recipe from all others (Tülu 3 for IT, very transparent SFT/DPO pipeline).

### Predicted corrective stage locations

| Model | Total layers | Predicted phase boundary (~1/3) | Predicted corrective onset (~60%) | Predicted corrective range |
|---|---|---|---|---|
| Gemma 3 4B | 34 | Layer 11 (confirmed) | Layer 20 (confirmed) | 20–33 |
| Llama 3.1 8B | 32 | Layer ~10–11 | Layer ~19–20 | ~19–31 |
| Qwen 3 4B | 36 | Layer ~12 | Layer ~21–22 | ~21–35 |
| Mistral 7B v0.3 | 32 | Layer ~10–11 | Layer ~19–20 | ~19–31 |
| DeepSeek-V2-Lite | 27 | Layer ~9 | Layer ~16 | ~16–26 |
| OLMo 2 7B | 32 | Layer ~10–11 | Layer ~19–20 | ~19–31 |

These are predictions to be tested, not assumptions. Deviations from these predictions are interesting data, not failures.

---

## 3. MOE-SPECIFIC METHODOLOGY

### Why MoE matters for our claims

The corrective stage operates in the **residual stream** — the shared communication channel that all components read from and write to. In dense models, each layer's MLP uniformly transforms the residual stream. In MoE models, different tokens are routed to different experts within the same layer. This raises critical questions:

1. **Does the corrective stage appear per-expert, per-layer, or both?**
2. **Does routing change across the corrective boundary?** (Do governance-relevant tokens get routed to different experts in IT vs PT?)
3. **Is the opposition signal (negative δ-cosine) coming from specific experts or from the expert ensemble?**

### Key insight: residual stream analysis is MoE-compatible

The residual stream after an MoE layer is:

```
h_ℓ = h_{ℓ-1} + Σ_i [g_i(x) · E_i(h_{ℓ-1})]  +  attn_ℓ(h_{ℓ-1})
```

where `g_i(x)` is the gating weight for expert `E_i`. Crucially:
- **There is ONE residual stream per token position**, regardless of routing.
- δ_ℓ = h_ℓ − h_{ℓ-1} still represents "what this layer wrote" — it's just that "this layer" involves a mixture of experts.
- δ-cosine, commitment delay, and all residual stream metrics apply identically.

Recent work (MoE-Lens, 2025) confirms cosine similarity between single top-expert + residual and full ensemble + residual is ~0.95 across all layers, meaning our residual stream analysis captures the dominant computation.

### MoE-specific experiments

In addition to the standard L1–L3, L8, L9 experiments (which apply directly to MoE residual streams), we add:

**L-MoE-1: Expert routing divergence**
- At each MoE layer, record which experts are activated for each token position.
- Compare IT vs PT routing patterns: do governance-relevant tokens (structural markers, format tokens) get routed to different experts in IT?
- Metric: Jensen-Shannon divergence of expert routing distributions (IT vs PT) per layer.
- Prediction: routing divergence peaks at corrective layers, meaning IT learned to route governance tokens through specific experts.

**L-MoE-2: Expert-decomposed δ-cosine**
- Decompose δ_ℓ into expert contributions: δ_ℓ = Σ_i [g_i · E_i(h)] + attn contribution.
- Compute δ-cosine per expert: cos(g_i · E_i(h), h_{ℓ-1}).
- Do all experts oppose the residual stream at corrective layers, or only specific "governance experts"?
- Prediction: opposition is concentrated in 2–3 dominant experts per corrective layer in IT; PT experts show no such concentration.

**L-MoE-3: Shared expert analysis (DeepSeek-specific)**
- DeepSeek-V2 has shared experts (always active for every token) plus routed experts.
- Is the corrective signal in the shared experts (universal opposition) or the routed experts (token-specific correction)?
- Prediction: shared experts carry the baseline corrective signal; routed experts specialize it per token type.

### Implementation notes for MoE
- DeepSeek-V2-Lite uses 2 shared + 64 routed experts per MoE layer, with top-6 routing.
- Not all layers are MoE in DeepSeek-V2-Lite — some are dense MLP layers. Check the architecture config to identify which layers have MoE vs dense MLP.
- Expert gating weights are accessible via the router module. Hook into the router to collect routing decisions.
- Memory: only 2.4B params active per token, but the full model is ~16B (all experts). With bf16, this is ~32GB — might need quantization or offloading for the full model. However, for inference only the active experts need to be in GPU memory per forward pass.

---

## 4. EXPERIMENT SPECIFICATIONS (Updated)

### Tier 1: CRITICAL (all 6 models)

#### L1: δ-Cosine Heatmap

**Goal**: For each model (PT and IT), produce a heatmap of `cos(δ_ℓ, h_{ℓ-1})` across layers × generation steps. This is the foundational observable — the corrective stage appears as a band of strongly negative δ-cosine in IT's late layers that is absent or much weaker in PT.

**Detailed implementation spec (for Claude Code)**:

##### Step 1: Model loading
- Load PT and IT models separately with `torch_dtype=torch.bfloat16, device_map="auto"`.
- No need for `attn_implementation="eager"` here (we don't need attention weights, just residual streams).
- **No chat template** for either PT or IT. Tokenize raw prompt text identically for both.

##### Step 2: Autoregressive generation with residual hooks
- For each prompt in the 2.5k set, generate up to **512 tokens** autoregressively (greedy or temperature=0).
- At **each generation step** (each new token), hook the residual stream **after** each layer. The hook point is the output of the full transformer block (post-attention + post-MLP + residual connection).
  - For HuggingFace models, this is typically the output of each decoder layer: `model.model.layers[ℓ]` output.
  - Use `register_forward_hook` on each layer to capture `h_ℓ` (the residual stream after layer ℓ) for the **last token position only** (the newly generated token).
- Store: for each generation step t, for each layer ℓ: `h_ℓ^(t)` as a float32 vector of shape `[d_model]`.
- **Memory management**: Do NOT store all residuals for all steps × all layers × all prompts in memory. Process one prompt at a time: generate, compute δ-cosine per step per layer, store only the scalar cosine values, then discard residual vectors before moving to the next prompt.

##### Step 3: δ-cosine computation
At each generation step t, for each layer ℓ ≥ 1:
```python
delta = h[ℓ] - h[ℓ - 1]  # what this layer wrote to the residual stream
denom = delta.norm() * h[ℓ - 1].norm()
cos_val = (torch.dot(delta, h[ℓ - 1]) / denom).item() if denom > 0 else float("nan")
```
- Layer 0: set to NaN (no previous layer to diff against).
- This gives a `[n_steps, n_layers]` matrix of cosine values per prompt.

##### Step 4: Aggregation
- Average the `[n_steps, n_layers]` matrices across all prompts → `mean_cosine[step][layer]`.
- Also compute std/SEM for error bars if needed.
- Separate PT and IT averages.

##### Step 5: Output format
Save per-prompt results as JSONL:
```json
{
  "prompt_id": "abc123",
  "model": "llama-3.1-8b",
  "variant": "pt",
  "delta_cosine": [[cos_l0_t0, cos_l1_t0, ...], [cos_l0_t1, cos_l1_t1, ...], ...],
  "n_layers": 32,
  "n_steps": 187,
  "generated_text": "..."
}
```
Also save the averaged heatmap as `.npy`: shape `[max_steps, n_layers]` for PT and IT separately.

##### Step 6: Plots
- **Per-model heatmap pair**: Two side-by-side heatmaps (PT left, IT right). X-axis = generation step (0 to max_steps). Y-axis = layer index (0 to n_layers-1). Colormap = diverging (blue-white-red), centered at 0. Shared colorbar. Title: "{Model} PT" / "{Model} IT".
- **Per-model difference heatmap**: Single heatmap of `IT_mean - PT_mean`. Highlights where IT is more negative than PT.
- **Cross-model summary**: 6 rows (one per model), X-axis = normalized depth (0 to 1). For each model, average δ-cosine across generation steps 10–100 (skip early transient) → a single curve of δ-cosine vs depth. Plot all 6 models on same axes, PT as dashed, IT as solid.

**Template handling**: No templates for PT or IT. Raw text, identical tokenization.

**Success criterion**: IT's δ-cosine in predicted corrective range is significantly more negative than PT's (≥1.5× ratio, p < 0.01 via paired t-test across prompts at each layer).


#### L2: Commitment Delay

**Goal**: Show that IT models commit to their final token prediction later (at a deeper layer) than PT models. This is computed via logit lens during the same generation runs as L1 — **collect both L1 and L2 data in a single pass** to avoid redundant computation.

**Detailed implementation spec (for Claude Code)**:

##### Step 1: Collect alongside L1
- During the L1 generation loop, at each generation step, you already have `h_ℓ` (residual stream after layer ℓ) for all layers. Reuse these.
- You additionally need the model's final LayerNorm and unembedding matrix `W_U`. Extract once at model load time:
  - `final_norm = model.model.norm` (or equivalent for each architecture — the RMSNorm before the lm_head)
  - `W_U = model.lm_head.weight.T` — shape `[d_model, vocab_size]`. Note: some models tie embeddings; check if `lm_head` exists or if it's the embedding matrix transposed.

##### Step 2: Logit lens at each layer
For each generation step t, at each layer ℓ:
```python
h_normed = final_norm(h[ℓ].to(device))  # apply final LayerNorm
logits_ℓ = h_normed.float() @ W_U       # [vocab_size]
probs_ℓ = torch.softmax(logits_ℓ, dim=-1)
top1_ℓ = logits_ℓ.argmax().item()
entropy_ℓ = -(probs_ℓ * torch.log(probs_ℓ + 1e-12)).sum().item()
```
- Also compute the **final layer's** top1 token: `top1_final = logits[n_layers-1].argmax()`.

##### Step 3: Commitment layer definition
For each generation step t, the **commitment layer** is the earliest layer ℓ where:
1. `top1_ℓ == top1_final` (same top-1 prediction as the final layer), AND
2. For all subsequent layers ℓ' > ℓ: `top1_{ℓ'} == top1_final` (it stays committed — no flip-back).

If these conditions are never met (the prediction keeps changing), set commitment_layer = n_layers (i.e., only commits at the very end).

Alternative metric (also collect): **KL to final**:
```python
kl_to_final = torch.nn.functional.kl_div(
    torch.log(probs_ℓ + 1e-12), probs_final, reduction='sum'
).item()
```
Commitment = earliest ℓ where `kl_to_final < 0.1` and stays below for all subsequent layers.

##### Step 4: Aggregation
- Per prompt: `commitment_layer_PT[step]` and `commitment_layer_IT[step]` for each generation step.
- Across prompts: mean commitment layer for PT vs IT, per generation step.
- Also: histogram of commitment layers across all prompts × steps.
- **Commitment delay** = `mean(commitment_IT) - mean(commitment_PT)`.

##### Step 5: Output format
Extend the L1 JSONL with logit lens data:
```json
{
  "prompt_id": "abc123",
  "model": "llama-3.1-8b",
  "variant": "pt",
  "delta_cosine": [...],
  "logit_lens_entropy": [[ent_l0_t0, ent_l1_t0, ...], ...],
  "logit_lens_top1": [[tok_l0_t0, tok_l1_t0, ...], ...],
  "commitment_layer": [cl_t0, cl_t1, ...],
  "n_layers": 32,
  "n_steps": 187
}
```

##### Step 6: Plots
- **Per-model commitment curve**: X-axis = layer (normalized depth). Y-axis = fraction of (prompt, step) pairs where model has committed. Two curves: PT (blue) and IT (red). IT curve should shift right (later commitment).
- **Per-model logit lens entropy**: Heatmap of logit-lens entropy across layers × generation steps (same layout as L1 heatmap). IT should show higher entropy at corrective layers.
- **Cross-model bar chart**: For each model, bar showing mean commitment delay (IT − PT) in layers. Error bars = SEM.

*Run on our same 2.5k prompts we previously used for exp2-4.*

**Success criterion**: IT commits ≥3 layers later than PT. I'll evaluate visually.

#### L3: Weight Change Localization

**Goal**: Compare the weights of PT and IT models layer-by-layer. Measure how much the weights changed at each layer during instruction tuning. We expect weight changes to be concentrated at corrective layers.

**Detailed implementation spec (for Claude Code)**:

##### Step 1: Load both models' state dicts
- Load PT and IT state dicts. Do NOT load full models into GPU — just need the parameter tensors. Use `torch.load` or `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float32, device_map="cpu")` and access `.state_dict()`.
- For large models (8B), you can iterate layer-by-layer to avoid loading both full models simultaneously.

##### Step 2: Per-layer weight diff (dense models)
For each layer ℓ, identify ALL parameter tensors belonging to that layer. For a standard transformer layer these are:
- Self-attention: `q_proj`, `k_proj`, `v_proj`, `o_proj` (or fused `qkv_proj` for some models)
- MLP: `gate_proj`, `up_proj`, `down_proj` (for SwiGLU)
- LayerNorm(s): `input_layernorm`, `post_attention_layernorm`

Compute per-layer normalized Frobenius diff:
```python
total_sq_diff = 0.0
total_params = 0
for name, param_pt in pt_state_dict.items():
    if f"layers.{ℓ}." in name:
        param_it = it_state_dict[name]
        total_sq_diff += ((param_it.float() - param_pt.float()) ** 2).sum().item()
        total_params += param_pt.numel()
delta_w[ℓ] = math.sqrt(total_sq_diff / total_params)  # RMS weight diff
```

Also compute **per-component** diffs (attention vs MLP vs norm) to see whether IT primarily modified MLP or attention:
```python
delta_attn[ℓ] = rms_diff(attention params at layer ℓ)
delta_mlp[ℓ] = rms_diff(MLP params at layer ℓ)
delta_norm[ℓ] = rms_diff(LayerNorm params at layer ℓ)
```

##### Step 3: Per-layer weight diff (MoE — DeepSeek-V2-Lite)
Same approach but decompose the MLP component:
```python
delta_router[ℓ] = rms_diff(router/gate weights at layer ℓ)
delta_shared_expert[ℓ] = rms_diff(shared expert weights at layer ℓ)
delta_routed_experts[ℓ] = mean over all experts of rms_diff(expert_i weights)
delta_attn[ℓ] = rms_diff(attention weights)  # same as dense
```
Note: Not all DeepSeek-V2-Lite layers are MoE. Some have standard dense MLP. Check `config.json` `moe_layer_freq` to identify which layers are MoE vs dense, and handle accordingly.

##### Step 4: Non-layer parameters
Also compute diffs for:
- Embedding layer: `model.embed_tokens.weight`
- Final LayerNorm: `model.norm.weight`
- LM head: `lm_head.weight` (if not tied to embeddings)
Report these separately as they're not layer-indexed.

##### Step 5: Output format
Save as JSON:
```json
{
  "model": "llama-3.1-8b",
  "n_layers": 32,
  "delta_w": [0.0012, 0.0013, ...],
  "delta_attn": [0.0008, 0.0009, ...],
  "delta_mlp": [0.0015, 0.0016, ...],
  "delta_norm": [0.0001, 0.0001, ...],
  "delta_embed": 0.0005,
  "delta_final_norm": 0.0003,
  "delta_lm_head": 0.0007
}
```

##### Step 6: Plots
- **Per-model bar chart**: X-axis = layer index. Y-axis = RMS weight diff. Stacked or grouped bars for attention vs MLP vs norm components. Highlight predicted corrective range with shaded background.
- **Cross-model overlay**: All 6 models on same axes, X-axis = normalized depth, Y-axis = RMS weight diff (each model normalized to its own max, so we compare the *shape* of the profile, not the absolute magnitude).
- **MoE decomposition** (DeepSeek only): Separate plot with router / shared / routed expert diffs.

**No prompts needed** — this is purely a weight comparison, no inference required. Fast to compute (~minutes per model pair).

**Success criterion**: Weight changes peak in predicted corrective range.

#### L8: Intrinsic Dimensionality Profile

**Goal**: At each layer, estimate the intrinsic dimensionality (ID) of the residual stream representations. We expect an ID minimum at ~1/3 depth (the phase boundary / "dip"), indicating a geometric bottleneck that separates the content-building phase from the corrective phase. Compare PT vs IT profiles.

**Detailed implementation spec (for Claude Code)**:

##### Step 1: Collect residual vectors
- For each prompt in the 2.5k set, run a **single forward pass** (no generation, just encode the prompt).
- At each layer ℓ, extract the residual stream at the **last token position**: `h_ℓ = model.model.layers[ℓ].output[0][0, -1, :]` → shape `[d_model]`.
- Store as a matrix: `R_ℓ` = `[n_prompts, d_model]` for each layer. This is ~2500 × 2560 (for 4B models) ≈ 25 MB per layer in float32. With 34 layers that's ~850 MB — fits in memory.
- **No chat template** for either PT or IT.

##### Step 2: TwoNN intrinsic dimensionality estimator
Use the TwoNN method (Facco et al., 2017). For each layer ℓ:

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

def twonn_id(X: np.ndarray) -> float:
    """TwoNN intrinsic dimensionality estimator.
    X: [n_samples, d_features]
    Returns: estimated intrinsic dimensionality (float).
    """
    nn = NearestNeighbors(n_neighbors=3, metric='euclidean').fit(X)
    distances, _ = nn.kneighbors(X)
    # distances[:, 0] is self (0), distances[:, 1] is nearest, distances[:, 2] is 2nd nearest
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    # Filter out zero distances
    valid = (r1 > 1e-10) & (r2 > 1e-10)
    mu = r2[valid] / r1[valid]
    # MLE estimator
    n = len(mu)
    id_estimate = n / np.sum(np.log(mu))
    return id_estimate
```

- Compute `ID_PT[ℓ]` and `ID_IT[ℓ]` for each layer.
- For confidence intervals: bootstrap (resample prompts 100 times, recompute ID each time, take 5th/95th percentile).

##### Step 3: Alternative — scikit-dimension library
If available, use `skdim.id.TwoNN()` which handles edge cases better:
```python
import skdim
estimator = skdim.id.TwoNN()
id_estimate = estimator.fit_transform(X)
```
Install: `pip install scikit-dimension --break-system-packages`

##### Step 4: Output format
```json
{
  "model": "llama-3.1-8b",
  "variant": "pt",
  "n_layers": 32,
  "intrinsic_dim": [45.2, 48.1, 52.3, ...],
  "intrinsic_dim_ci_low": [42.1, 45.0, ...],
  "intrinsic_dim_ci_high": [48.3, 51.2, ...],
  "n_prompts": 2500
}
```

##### Step 5: Plots
- **Per-model ID profile**: X-axis = layer index. Y-axis = intrinsic dimensionality. Two lines: PT (blue) and IT (red) with bootstrap CI shading. Mark predicted phase boundary with vertical dashed line.
- **Cross-model summary**: All 6 models on same axes, X-axis = normalized depth (0 to 1), Y-axis = ID (each model normalized to its own max for shape comparison). Look for consistent dip at ~0.33.

**Success criterion**: ID minimum at ~1/3 depth (±2 layers normalized). I'll evaluate visually.

#### L9: Attention Entropy Divergence

**Goal**: For each model (PT and IT), compute per-layer, per-head attention entropy over the same 2.5k prompt set. Then compute IT−PT entropy divergence per layer. We expect maximum divergence at or near the phase boundary (~1/3 depth).

**Detailed implementation spec (for Claude Code)**:

##### Step 1: Model loading
- Load model with `attn_implementation="eager"` — this is **required** to get full attention weight matrices. Flash attention / SDPA do NOT return attention weights.
- Example: `model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", torch_dtype=torch.bfloat16, device_map="auto")`
- For DeepSeek-V2-Lite (MoE with Multi-Head Latent Attention / MLA): MLA compresses KV into a low-rank latent space internally, but the final attention distribution (post-softmax) is still a standard `[n_heads, T_q, T_k]` tensor. When using `output_attentions=True`, HuggingFace returns the **decompressed** attention weights, so no special handling needed — just use the returned tensor as-is.
- **No chat template** for either PT or IT. Raw text input, same as PT.

##### Step 2: Forward pass and attention extraction
- For each prompt in the 2.5k set, tokenize and run a **single forward pass** (no generation needed — we only need the prompt's attention patterns, not generated text).
- Pass `output_attentions=True` to the forward call.
- The model returns `attentions`: a tuple of length `n_layers`, where each element is `[batch=1, n_heads, T, T]`.
- Extract the **last-token query row** at each layer: `attn_row = attentions[layer_idx][0, :, -1, :]` → shape `[n_heads, T]`. This gives us how the last token attends to all previous tokens, per head.

##### Step 3: Entropy computation (per-head, per-layer)
```python
def attn_entropy(attn_row: torch.Tensor) -> float:
    """Entropy of a single attention distribution over key positions.
    attn_row: [T_k] — already post-softmax probabilities.
    Returns entropy in nats.
    """
    probs = attn_row.clamp(min=0.0)  # clamp fp noise
    return float(-(probs * torch.log(probs + 1e-12)).sum().item())
```
- For each layer ℓ and head h: `entropy[ℓ][h] = attn_entropy(attn_row[h])`
- Output per prompt: a `[n_layers, n_heads]` matrix of entropy values.

##### Step 4: Aggregation across prompts
- For each layer ℓ: compute `mean_entropy_PT[ℓ]` = mean over all prompts and all heads. Also compute SEM.
- Same for IT: `mean_entropy_IT[ℓ]`.
- Divergence: `divergence[ℓ] = mean_entropy_IT[ℓ] − mean_entropy_PT[ℓ]`.
- Also compute relative divergence: `rel_div[ℓ] = divergence[ℓ] / mean_entropy_PT[ℓ]`.

##### Step 5: Per-head heatmap
- Compute `head_diff[ℓ][h]` = mean over prompts of `(entropy_IT[ℓ][h] − entropy_PT[ℓ][h])`.
- This gives a `[n_layers, n_heads]` heatmap showing which heads at which layers diverge most.

##### Step 6: Model-specific attention notes

| Model | n_heads (Q) | n_kv_heads (GQA) | Attention type | Special notes |
|---|---|---|---|---|
| Gemma 3 4B | 8 | 4 | GQA, mix of global + local (sliding window=1024) | Global attention at layers where `layer_idx % 6 == 5` (i.e., 5,11,17,23,29). Local layers have limited context window — entropy is bounded by window size. **Separate global vs local layers in analysis.** |
| Llama 3.1 8B | 32 | 8 | GQA, all global | Straightforward. All layers see full context. |
| Qwen 3 4B | 32 | 8 | GQA, all global | Straightforward. |
| Mistral 7B v0.3 | 32 | 8 | GQA, sliding window (4096) | All layers use sliding window attention. Entropy bounded by window size. Compare within-model only. |
| DeepSeek-V2-Lite | 16 | 2 (MLA) | Multi-Head Latent Attention | `output_attentions=True` returns decompressed attention weights as standard `[n_heads, T, T]`. Use as-is. Not all layers are MoE — check `config.json` for which layers have MoE vs dense MLP. But attention is the same regardless. |
| OLMo 2 7B | 32 | 8 | GQA, all global | Straightforward. |

**IMPORTANT for sliding window models (Gemma local layers, Mistral)**: Attention entropy is bounded by `log(window_size)` rather than `log(T)`. This means absolute entropy values are NOT comparable across global vs local layers. When plotting:
- For Gemma: plot global and local layers separately, OR normalize entropy by `log(effective_context_length)`.
- For cross-model comparison: normalize entropy by `log(effective_context_length_at_that_layer)` so values are comparable.

##### Step 7: Output format
Save results as JSONL, one line per prompt:
```json
{
  "prompt_id": "abc123",
  "model": "llama-3.1-8b",
  "variant": "pt",
  "attn_entropy": [[h0_l0, h1_l0, ...], [h0_l1, h1_l1, ...], ...],
  "n_layers": 32,
  "n_heads": 32,
  "seq_len": 157
}
```

##### Step 8: Plots (3 panels per model, same layout as our Gemma exp4 plots)
- **Panel A**: Per-layer mean entropy — PT (blue) vs IT (red) with ±1 SEM shading. X-axis = layer index. Y-axis = mean entropy (nats).
- **Panel B**: Per-layer absolute divergence (IT − PT). Bar chart or line. Highlight the layer with maximum divergence. Mark predicted phase boundary with vertical dashed line.
- **Panel C**: Per-head entropy difference heatmap — `[n_layers × n_heads]`, colormap = diverging (blue-white-red), where red = IT higher entropy, blue = PT higher entropy.

##### Step 9: Cross-model summary plot
- One row per model (6 rows). X-axis = **normalized depth** (layer_idx / n_layers, 0 to 1). Y-axis = divergence (IT − PT mean entropy).
- All 6 models overlaid on same axes. This directly tests whether entropy divergence peaks at ~0.33 normalized depth across all models.

**Success criterion**: Maximum entropy divergence at or near predicted phase boundary (~1/3 depth, ±2 layers normalized). The peak doesn't need to be exactly at 1/3 — within ±10% normalized depth is fine. What matters is that the peak is consistent across models when measured in fractional depth.

---

## 5. IMPLEMENTATION STRATEGY

### Code architecture (updated)

```
src/poc/cross_model/
├── config.py              # ModelPair config with MoE flag
├── collect.py             # Model-agnostic generation + metric collection
├── weight_diff.py         # Weight change analysis (dense + MoE decomposed)
├── steer.py               # Direction extraction + α-sweep
├── moe_analysis.py        # MoE-specific: routing divergence, expert decomposition
├── plots/
│   ├── cosine_heatmap.py
│   ├── commitment_delay.py
│   ├── weight_change.py
│   ├── dose_response.py
│   ├── phase_boundary.py
│   ├── universal_summary.py    # The 6-row cross-model figure
│   └── moe_routing.py          # MoE-specific plots
└── models/
    ├── base.py            # Abstract base class for model adapters
    ├── gemma3.py          # Gemma 3 4B adapter
    ├── llama31.py         # Llama 3.1 8B adapter
    ├── qwen3.py           # Qwen 3 4B adapter
    ├── mistral.py         # Mistral 7B adapter
    ├── deepseek_v2.py     # DeepSeek-V2-Lite adapter (MoE-aware)
    └── olmo2.py           # OLMo 2 7B adapter
```

Model adapter interface:
```python
class ModelAdapter(ABC):
    @abstractmethod
    def get_residual_hook_point(self, layer_idx: int) -> str: ...
    @abstractmethod
    def get_mlp_output_hook_point(self, layer_idx: int) -> str: ...
    @abstractmethod
    def get_attention_hook_point(self, layer_idx: int) -> str: ...
    @abstractmethod
    def apply_template(self, tokenizer, prompt: str) -> str: ...
    @abstractmethod
    def get_stop_tokens(self, tokenizer) -> list[int]: ...
    @property
    @abstractmethod
    def is_moe(self) -> bool: ...
    # MoE-specific (optional, only for MoE adapters)
    def get_router_hook_point(self, layer_idx: int) -> str | None: ...
    def get_expert_output_hook_points(self, layer_idx: int) -> list[str] | None: ...
```

### GPU requirements (updated)

| Model | Total Params | Active Params | bf16 Memory | Quantized (4-bit) |
|---|---|---|---|---|
| Gemma 3 4B | 4B | 4B | ~8 GB | ~3 GB |
| Llama 3.1 8B | 8B | 8B | ~16 GB | ~6 GB |
| Qwen 3 4B | 4B | 4B | ~8 GB | ~3 GB |
| Mistral 7B v0.3 | 7B | 7B | ~14 GB | ~5 GB |
| DeepSeek-V2-Lite | 16B | 2.4B | ~32 GB* | ~10 GB |
| OLMo 2 7B | 7B | 7B | ~14 GB | ~5 GB |

*DeepSeek-V2-Lite loads all experts into memory even though only 2.4B are active per token. For single-GPU, use 4-bit quantization or CPU offloading for inactive experts.

With 8× A100 40GB GPUs:
- Run 4 dense models in parallel (2 GPUs each for PT+IT pair, or 1 GPU each for 4B models)
- DeepSeek-V2-Lite on a dedicated GPU with quantization
- Plenty of headroom for the α-sweep generation phase

### Prompt dataset
Same eval_dataset_v2.jsonl (1,400 prompts, 7 categories) for all models. Same top-600 high-contrast selection for direction extraction.


## 7. WHAT THIS ADDS TO THE PAPER

### Current state (single model)
- §3: Corrective stage in Gemma 3 4B (observational)
- §6: Causal steering in Gemma 3 4B

### With 6-model replication
- §3: Corrective stage in **6 model families** — NEW §3.5 cross-model universality subsection
- §3.6 (NEW): MoE analysis — corrective stage survives expert routing
- §6: Causal steering in **Gemma + Llama** (+ optional third)
- §7 (Discussion): Theoretical framing — corrective stage as emergent property of IT (see §10)
- §9: "Single model" limitation eliminated; replaced by "limited to 3B–8B active params"

### Impact on acceptance probability
- Current (Gemma only): 50–60%
- 4-model observational: 65–75%
- **6-model observational + MoE: 75–80%**
- **6-model observational + MoE + Llama causal: 80–90% (oral territory)**

---

## 8. SIZE VARIATION (Future Extension)

For initial validation, we use one size per family. Once the corrective stage is confirmed, we can add:

| Family | Smaller | Our size | Larger |
|---|---|---|---|
| Gemma | Gemma 3 1B | Gemma 3 4B | Gemma 3 12B |
| Llama | Llama 3.2 3B | Llama 3.1 8B | Llama 3.1 70B |
| Qwen | Qwen 3 1.7B | Qwen 3 4B | Qwen 3 8B |
| DeepSeek | — | V2-Lite (2.4B active) | V3 (37B active) |

This would test: Does the corrective onset fraction (e.g., 60% depth) change with scale? Does the effect strength (δ-cosine ratio) increase with scale?

**Priority**: Low. Initial 6-model validation at one size each is sufficient for NeurIPS submission. Size variation is camera-ready or follow-up paper material.

---

## 9. RIGOR CHECKLIST (NeurIPS Oral Standard)

### Methodology
- [ ] Same prompt dataset across all 6 models (controlled comparison)
- [ ] Template-free primary analysis (confound elimination)
- [ ] Template-with secondary analysis (ecological validity)
- [ ] Statistical tests with effect sizes for every claim
- [ ] Normalized depth fraction for cross-model comparison
- [ ] Predictions stated BEFORE running (registered predictions in paper)
- [ ] MoE analysis decomposes residual stream contributions

### Controls
- [ ] PT as natural control for each model
- [ ] Random direction control at corrective layers (Gemma + Llama minimum)
- [ ] Content metric (MMLU) as negative control in steering
- [ ] Multiple governance metrics per model
- [ ] MoE: compare to random-expert baseline

### Reproducibility
- [ ] All code open-sourced with model-agnostic framework
- [ ] Exact model IDs + HuggingFace revisions specified
- [ ] Prompt dataset released
- [ ] Random seeds documented
- [ ] Compute requirements documented per model

### Presentation
- [ ] Cross-model summary figure with normalized depth axis (6 rows)
- [ ] Table of quantitative results across all 6 models
- [ ] Effect sizes and confidence intervals
- [ ] Honest reporting of any model where pattern doesn't hold
- [ ] Supplementary: full heatmaps for every model × variant × metric

---

## 10. THEORETICAL FRAMING: WHAT CROSS-MODEL REPLICATION MEANS

### The central question
If the corrective computational stage appears in 6 independently trained model families, what does this tell us about the nature of instruction tuning?

### Three levels of explanation

**Level 1: Convergent optimization (weakest claim, most defensible)**

Instruction tuning via SFT + RLHF optimizes for a similar loss landscape across all models: produce helpful, harmless, conversational responses. Given the same optimization objective applied to pretrained transformers, gradient descent converges on a similar computational solution — dedicating late layers to "output governance" (format, register, structure, safety) while preserving earlier layers for content processing.

This is analogous to how different CNN architectures all learn Gabor filters in early layers — not because of shared initialization, but because the data distribution + loss function constrain the solution space.

**Supporting evidence**:
- Huh et al. (2024, "Platonic Representation Hypothesis") show that different models converge on similar representation spaces as they scale. They argue this convergence is driven by modeling the same underlying reality.
- The alignment tax literature (EMNLP 2024) shows that safety can be achieved with low-rank weight modifications, suggesting the "governance" computation is a low-dimensional subspace added on top of the content computation. If it's low-dimensional, independent training runs should find similar subspaces.

**Level 2: Structural necessity (medium claim)**

The corrective stage may be structurally *necessary* for the autoregressive generation of formatted conversational output. The argument:

1. Pretrained LMs learn to model internet text distributions, which include both raw content and formatted/structured text.
2. The early/middle layers learn content representations sufficient for next-token prediction across all text types.
3. Instruction tuning's SFT phase requires the model to consistently produce *one specific format* (conversational assistant output) regardless of input.
4. The most parameter-efficient way to achieve this is: keep content layers intact (they're already good at understanding) and add a corrective transformation in late layers that redirects the output distribution from "general internet text" toward "structured assistant response."
5. Late layers are the natural location because (a) they're closest to the output distribution, (b) modifying early layers would disrupt content understanding, and (c) the residual stream architecture means late layers can "overwrite" early layer contributions without destroying them.

This is a stronger claim: the corrective stage isn't just convergent, it's the *optimal* solution given the constraints of fine-tuning a pretrained transformer. Any sufficiently capable IT procedure will discover it.

**Supporting evidence**:
- Lad et al. (2024) show a 4-stage structure (detokenization → feature engineering → prediction → residual sharpening) in multiple models. Our corrective stage maps onto a modification of their "residual sharpening" stage.
- The model stitching literature (NeurIPS 2025) shows that affine mappings between residual streams transfer features across models, suggesting shared structure in the representation space.
- The activation steering literature (ICLR 2025) shows that instruction-following can be modulated by adding/removing direction vectors — consistent with governance being encoded as directions, not distributed computation.

**Level 3: Data-driven emergence (strongest claim, most speculative)**

The corrective stage is an *inevitable* consequence of:
1. Pretraining on internet-scale data (which contains a distribution over registers, formats, and conversational structures)
2. Followed by optimization toward a narrow "helpful assistant" distribution

The internet contains all the building blocks of assistant-like behavior — Q&A forums, documentation, customer service transcripts, structured writing. The pretrained model already represents these patterns in its weights. Instruction tuning doesn't teach the model *new* capabilities so much as it installs a *computational selector* that consistently activates the "helpful assistant" register.

This selector manifests as the corrective stage because:
- The pretrained model's late layers already compute different "finalization strategies" for different text types
- IT sharpens this into a single dominant strategy (the assistant register)
- The negative δ-cosine represents the model actively suppressing the other finalization strategies

If this is true, then ANY instruction tuning method (SFT, RLHF, DPO, constitutional AI) applied to ANY sufficiently capable pretrained transformer should produce a corrective stage, because they all optimize for the same target distribution.

**Supporting evidence**:
- DeepSeek-R1-Zero (pure RL, no SFT) still learns to produce structured, formatted reasoning output — suggesting the model discovers formatting governance through optimization alone.
- The low-rank nature of the IT-PT weight difference (our own finding + alignment tax literature) is consistent with a "selector" interpretation: a low-rank projection selects one behavioral mode from many that the pretrained model already represents.

### What MoE tells us about mechanism

If the corrective stage appears in DeepSeek-V2-Lite (MoE), it constrains the mechanism further:

**Scenario A: Corrective stage operates through the shared experts**
- Implication: governance is a universal transformation applied to ALL tokens, regardless of routing.
- This supports Level 2/3: governance is a global register selector, not content-dependent.

**Scenario B: Corrective stage operates through specific routed experts**
- Implication: governance involves token-type-specific corrections (e.g., structural tokens get routed to "formatting experts," content tokens to "register experts").
- This supports a more nuanced view: governance is computed token-by-token, with expert specialization.
- Would be consistent with MoE interpretability literature showing experts develop functional specialization.

**Scenario C: No corrective stage in MoE**
- Implication: the corrective stage requires dense computation; MoE architectures achieve governance through a different mechanism (possibly through the router learning to select "governance experts" at every layer, rather than concentrating governance in late layers).
- Still publishable: "The corrective stage is specific to dense architectures; MoE achieves governance through distributed expert selection." This would be an important negative result.

### How to discuss this in the paper

In §7 (Discussion), present Level 1 as the primary claim and Level 2 as a motivated hypothesis:

> "The replication of the corrective computational stage across six independently trained model families — spanning different companies, training corpora, optimization procedures, and architectures (including MoE) — suggests this is not an artifact of any specific training run but an emergent structural property of instruction tuning applied to pretrained transformers. We hypothesize this arises because SFT + RLHF optimizes for a common objective (structured assistant output) under shared constraints (preserve content capabilities while redirecting output format), and the corrective late-layer stage is the parameter-efficient solution to this optimization problem."

Mention Level 3 as speculation in the conclusion:

> "A more speculative interpretation is that the corrective stage is *inevitable*: pretrained language models already encode the building blocks of conversational behavior from their training data, and instruction tuning merely installs a computational selector in late layers to consistently activate the assistant register."

---

## 11. RELATED WORK CONNECTIONS

### Papers supporting the universality claim

| Paper | Finding | Connection to our work |
|---|---|---|
| Huh et al. 2024 (Platonic Representation Hypothesis) | Models converge on shared representation spaces | Supports Level 1: if representations converge, computational stages should too |
| Lad et al. 2024 (Stages of Inference) | 4-stage computational pipeline in 4 models | Our corrective stage refines their "residual sharpening" stage; IT modifies stage 4 |
| Belrose et al. 2023 (Tuned Lens) | Latent predictions converge smoothly across layers | Our commitment delay extends this: IT delays convergence specifically in corrective layers |
| Arditi et al. 2024 (Refusal Direction) | Single direction controls refusal in 5 models | Parallel to our governance direction; both are low-rank modifications from fine-tuning |
| Yu et al. 2025 (Cross-model mech interp) | Circuits generalize within model families, vary across | Our L1–L3 tests cross-family; finding universality would strengthen both papers |
| NeurIPS 2025 (Model Stitching) | Affine maps transfer features across models | Implies shared residual stream geometry — consistent with corrective stage at similar fractional depth |
| MoE-Lens 2025 | Individual expert contributions recoverable via logit lens | Our L-MoE-2 extends this: which experts contribute the corrective signal? |
| Mixture of Monosemantic Experts (MONET) | MoE experts develop functional specialization | Predicts our L-MoE-1 should show governance-specific expert routing |
| Improving Instruction Following via Activation Steering (ICLR 2025) | Instruction compliance modulated by activation vectors | Direct parallel: their instruction vectors ≈ our governance direction |

### Papers we should cite but that partially conflict

| Paper | Finding | Tension with our work |
|---|---|---|
| Yu et al. 2025 | Circuits differ across model families | If L4/L5 don't replicate, this explains why |
| MoE interpretability (various) | MoE models achieve monosemantic experts, different from dense | Our corrective stage might not exist in MoE (Scenario C) |

---

## 12. COMPARISON TO RELATED WORK: MODEL COUNT

| Paper | Models | Families | Dense/MoE | Observational | Causal |
|---|---|---|---|---|---|
| Arditi et al. 2024 | 5 | 2 (Meta, Google) | Dense only | Direction extraction | Ablation |
| Lad et al. 2024 | 4 | 3 (OpenAI, EleutherAI, Meta) | Dense only | Layer analysis | Layer deletion |
| Chaudhury et al. 2025 | 3 | 2 (Meta, Google) | Dense only | Probing + patching | Causal patching |
| **Ours (proposed)** | **6** | **6 (Google, Meta, Alibaba, Mistral, DeepSeek, AI2)** | **Dense + MoE** | **5 experiments** | **Direction steering** |

Our model diversity would be the most comprehensive in the mech-interp literature for this type of architectural claim. The MoE inclusion is novel — no prior mech-interp paper has tested residual stream stage structure in MoE models.

---

## 13. RISK MITIGATION (Updated)

### If DeepSeek-V2-Lite doesn't fit on a single GPU
- Use 4-bit quantization (bnb nf4). With 16B params, this brings it to ~10 GB.
- Alternatively, use CPU offloading for inactive experts (only 2.4B active per forward pass).
- Worst case: drop DeepSeek and note "MoE analysis deferred to follow-up" — still have 5 dense models.

### If Qwen 3 4B base model isn't available yet
- Qwen 3 was released recently; verify base model is on HuggingFace.
- Fallback: Qwen 2.5 7B (well-established, both Base and Instruct available).

### If OLMo 2 7B shows a weak or absent corrective stage
- OLMo's IT (Tülu 3) uses a different IT recipe (DPO focus). If the corrective stage is weaker, this is interesting data about IT methodology effects.
- Still report it: "OLMo shows a weaker corrective stage (1.3× vs 2.7× for Gemma), possibly reflecting its DPO-focused IT pipeline."

### If one model family shows no corrective pattern at all
- 5/6 is still strong evidence for near-universality.
- Investigate the outlier: architecture? Training data? IT methodology?
- A 5/6 result with one explained outlier is potentially MORE convincing than 6/6 (shows we're not p-hacking).

### If the corrective onset varies across models
- Plot corrective onset as fraction of total depth for all 6 models.
- Test correlation with: model size, number of IT training steps, training data volume.
- Variable onset is MORE interesting than fixed onset — it constrains the mechanism.
