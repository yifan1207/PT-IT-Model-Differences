# POC Documentation: Hierarchical Distributional Narrowing

## Table of Contents

1. [Research Hypothesis](#1-research-hypothesis)
2. [System Architecture](#2-system-architecture)
3. [Model & Transcoders](#3-model--transcoders)
4. [Data Schema: FeatureRecord](#4-data-schema-featurerecord)
5. [Pipeline Overview](#5-pipeline-overview)
6. [File Reference](#6-file-reference)
   - [config.py](#configpy)
   - [model.py](#modelpy)
   - [attribution.py](#attributionpy)
   - [analyze.py](#analyzepy)
   - [run_poc.py](#run_pocpy)
   - [run_plots.py](#run_plotspy)
   - [inference_test.py](#inference_testpy)
7. [Plots Reference](#7-plots-reference)
8. [Output Data Format](#8-output-data-format)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [Metric Glossary](#10-metric-glossary)
11. [Key Technical Decisions](#11-key-technical-decisions)

---

## 1. Research Hypothesis

### The Core Idea: Hierarchical Distributional Narrowing

Next-token prediction is not a single-step computation. The model decomposes it into a hierarchy of increasingly specific operations:

| Level | What it does | Example (predicting " 6" for "3 × 2 is") |
|-------|-------------|------------------------------------------|
| **L1 — Format** | Establishes the broad output type | "this will be a number/digit" |
| **L2 — N-gram** | Adds statistical context | "math expressions often end in single digits" |
| **L3 — Semantic** | Associative retrieval | "3 and 2 are related to 6 via multiplication" |
| **L4 — Computation** | The exact specific value | "3 × 2 = 6 specifically, not 5, not 7" |

Each level narrows the probability distribution further. A format feature might spread probability across all 10 digit tokens. A computation feature concentrates almost all probability on one token.

### The Attribution Inversion Claim (Claim 1)

**High-entropy (broad/low-specificity) features receive systematically higher attribution scores than low-entropy (specific/high-specificity) features — even though the specific features do the hardest computation.**

Concretely, the `attribution` metric is:
```
attribution(f) = |activation(f) × logit_target(f)|
```

where `logit_target = W_dec[f] @ W_U[:, target]`. This quantity measures the direct effect of feature `f` on the target token's logit. But:

- A **broad format feature** fires strongly (large `activation`) and has moderate `logit_target` → high attribution.
- A **specific computation feature** fires weakly (small `activation`) and has high `logit_target` → low attribution.

The attribution metric is biased toward features that fire hard, regardless of how specifically they point at the correct answer.

### What We Measure

1. **N₉₀** — *Broadness*: fewest tokens whose `|c(f)|` values sum to ≥90% of `Σ|c(f)|`. Low N₉₀ = specific, high N₉₀ = broad.
2. **Specificity** — `logit_target / logit_norm`: fraction of the feature's logit force that points directly at the target token. Range: −1 (suppresses target) to +1 (promotes target).
3. **Attribution** — `|activation × logit_target|`: the direct contribution of this feature to the target token's final logit.

If Claim 1 holds: **scatter plot of specificity vs attribution has a negative slope** (high-specificity features have lower attribution).

---

## 2. System Architecture

```
Prompts (config.py)
       │
       ▼
circuit-tracer attribute()          ← build attribution graph
       │
       ▼
Graph: active_features [n_active, 3]  → (layer, pos, feat_idx)
       adjacency_matrix [n_nodes, n_nodes]
       activation_values [n_active]
       selected_features [n_selected] ← indices into active_features
       │
       ▼
_extract_records()                  ← attribution.py
  Phase 1: group by layer, batch compute W_dec @ W_U
  Phase 2: assemble FeatureRecord objects
       │
       ▼
FeatureRecord list                  ← one per selected feature, per prompt
       │
       ├──► save to poc_results.json  ← analyze.py
       │
       └──► regression analysis       ← analyze.py
              pearson/spearman r
              OLS slope
              scatter plot
       │
       ▼
run_plots.py                        ← 10 analysis plots from saved JSON
```

---

## 3. Model & Transcoders

### Base Model: `google/gemma-3-4b-pt`

- **Architecture**: Gemma3 (text-only), base pretrained (NOT instruction-tuned)
- **Variant**: `-pt` = pretrained base; we want raw sentence completion, not chat
- **Layers**: 34 transformer layers (indices 0–33)
- **d_model**: 4096
- **vocab_size**: 262,144
- **Gated repo**: requires HuggingFace login (`huggingface-cli login`)

We use the base model (not `-it`) because we're studying raw next-token prediction. Chat models add special formatting that changes which features activate.

### Transcoders: `google/gemma-scope-2-4b-pt`

Transcoders are MLP replacements — they replace each transformer layer's MLP with a learned encoder-decoder architecture where the hidden units are interpretable "features." Unlike SAEs (which work on residual stream), transcoders capture the **functional computation** of each MLP.

**Available variants per layer:**
```
width  : 16k | 65k | 262k     (feature dictionary size)
l0     : small (~10-20 active) | medium (~30-60 active) | big (~60-150 active)
affine : plain | _affine (adds W_skip learned affine skip connection)
```

**We use**: `width_65k_l0_medium_affine`
- 65,536 features per layer (~4× more than the 16k default)
- ~30–60 features active per prompt (medium sparsity)
- Affine variant: captures the skip connection that carries residual signal around the MLP
- Fits on a single H100 80GB (~36 GB for all 34 layers vs ~146 GB for 262k)

**Hook names for Gemma 3** (required by `TranscoderSet`):
- `feature_input_hook = "mlp.hook_in"` (pre-MLP layernorm output)
- `feature_output_hook = "hook_mlp_out"` (post-MLP layernorm output)

### Why No `config.yaml`?

`google/gemma-scope-2-4b-pt` does not have a top-level `config.yaml` that circuit-tracer can auto-parse (unlike Gemma Scope 1 for Gemma 2). We build `TranscoderSet` manually:
1. `snapshot_download` with glob pattern to pull only the chosen variant
2. Load each layer's `params.safetensors` with `load_gemma_scope_2_transcoder`
3. Assemble into `TranscoderSet` with explicit hook names

### Backend: nnsight

Gemma 3 requires the `nnsight` backend (not `transformerlens`). The `NNSightReplacementModel` extends nnsight's `LanguageModel`. Key access patterns differ from TransformerLens:
- **W_U**: `model.unembed_weight` = `lm_head.weight` shape `[vocab_size, d_model]` → transpose to `[d_model, vocab_size]`
- **Tokenizer**: `model.tokenizer`
- **Transcoders**: `model.transcoders[layer]` (via `TranscoderSet.__getitem__`)
- **W_dec**: `transcoder.W_dec[feat_idx]` shape `[d_model]`

---

## 4. Data Schema: FeatureRecord

Every feature that circuit-tracer selects in the attribution graph for a prompt becomes one `FeatureRecord`. This is the atomic unit of data throughout the pipeline.

```python
@dataclass
class FeatureRecord:
    # Identity
    feature_idx: int        # index within the transcoder layer's feature dictionary
    layer: int              # transformer layer (0–33)
    position: int           # token position in the prompt (0 = first token)
    prompt_id: str          # e.g. "M1", "R5" — group letter + 1-based index

    # Activation
    activation: float       # how strongly this feature fired on this position
                            # (signed; can be negative for suppressor features)

    # Logit-lens: what does this feature's decoder direction "point at" in vocab space?
    # logit_vec = W_dec[feature_idx] @ W_U   [vocab_size]
    logit_target: float     # logit_vec[correct_token_id]  — signed
    logit_norm: float       # ||logit_vec||₂  — L2 norm of the full logit vector
    logit_entropy: float    # H(softmax(logit_vec))  — broad (high) vs narrow (low)

    # Interpretability
    top1_token_id: int      # argmax(logit_vec) — most promoted token
    top1_token_str: str     # decoded string of top1_token_id
    correct_token_rank: int # rank of target token in logit_vec (1 = top, lower is better)

    # Core analysis axes
    specificity: float      # logit_target / logit_norm  ∈ [−1, +1]
                            # +1 = feature entirely focused on target token
                            #  0 = feature agnostic to target token
                            # −1 = feature actively suppresses target token
    attribution: float      # |activation × logit_target|  ≥ 0
                            # direct effect on the target token's logit

    # Broadness metric (N₉₀)
    # c(f) = activation × logit_vec   [vocab_size]  — contribution vector
    c_total_mass: float     # Σ|c(f)| — total contribution magnitude across vocab
    n50: int                # min tokens for 50% of Σ|c(f)| mass  (-1 if c_total_mass==0)
    n90: int                # min tokens for 90% of Σ|c(f)| mass  (-1 if c_total_mass==0)
                            # LOW n90 = specific (narrow contribution)
                            # HIGH n90 = broad (spreads across many tokens)

    # Mechanism
    promote_ratio: float    # Σmax(c(f),0) / Σ|c(f)|
                            # 1.0 = pure promotion ("vote FOR these tokens")
                            # 0.0 = pure suppression ("vote AGAINST these tokens")
                            # 0.5 = mixed

    # Graph structure
    incoming_edge_count: int  # number of other active features with edges into this one
                              # HIGH = compositional (depends on upstream features)
                              # LOW  = lookup/direct (doesn't depend on upstream)

    # Heatmap data
    top50_contributions: list  # top-50 tokens by |c_value|
                               # each: {"token_id": int, "token_str": str, "c_value": float}
```

### N₉₀ Formula

```python
sorted_abs = abs(c_vec).sort(descending=True).values
cumsum = sorted_abs.cumsum(0)
n90 = (cumsum < 0.9 * total_mass).sum().item() + 1
```

This counts how many tokens you need to include (taking the largest contributors first) before you've covered 90% of the total contribution mass. The `+1` converts from a zero-based index to a count.

Special case: if `total_mass == 0` (feature has an all-zero decoder row — shouldn't happen in a trained transcoder), both `n50` and `n90` are set to `-1`. All plot code filters out `n90 <= 0`.

---

## 5. Pipeline Overview

### Data Flow

```
run_poc.py  ──►  attribution.py  ──►  FeatureRecord list
              (per prompt)
                    │
                    ▼
            analyze.py: build_result()
                    │
                    ▼
            poc_results.json
                    │
                    ▼
            run_plots.py  ──►  10 PNG plots
```

### Prompt Structure

Prompts are organized in two groups:

**Memorization (M)**: Direct retrieval. The answer is a stored fact.
- Capitals: "The capital of France is" → " Paris"
- Antonyms: "The opposite of hot is" → " cold"
- Colors, facts, arithmetic (memorized single-digit results)
- Languages: "People in France speak" → " French"

**Reasoning (R)**: Multi-step inference or computation.
- OOCR (out-of-context reasoning): compose two facts
  - "The language spoken in the capital of Japan is" → " Japanese"
- ICL novel operations: pattern completion
  - "f(1)=3, f(2)=5, f(0)=" → " 1"
- Multi-step: "Half of a dozen is" → " 6"

All prompts use raw sentence-completion format for the base model — no chat template, no system prompt. Target tokens have a leading space (SentencePiece convention: " Paris" not "Paris").

### Multi-GPU Support

`run_poc.py` automatically distributes prompts across all available CUDA GPUs using `ProcessPoolExecutor`. Each GPU subprocess gets its own model copy. Results are merged and sorted by prompt order after all GPUs finish.

---

## 6. File Reference

### `config.py`

Central configuration dataclass. All hyperparameters and prompts in one place.

```python
@dataclass
class PocConfig:
    model_name: str = "google/gemma-3-4b-pt"
    transcoder_set: str = "google/gemma-scope-2-4b-pt"
    backend: str = "nnsight"
    device: str = "cpu"
    dtype_str: str = "float32"          # "bfloat16" on GPU
    transcoder_variant: str = "width_262k_l0_medium_affine"

    # circuit-tracer settings
    max_n_logits: int = 10              # max top-k logit nodes to include in graph
    desired_logit_prob: float = 0.95    # stop adding nodes when 95% logit prob is covered
    batch_size: int = 512               # source nodes per backward pass (512=H100, 64=MPS, 32=CPU)
    max_feature_nodes: int = 200        # max transcoder features in graph

    output_path: str = "results/poc_results.json"
    plot_path: str = "results/specificity_vs_attribution.png"

    prompts: dict[str, list[tuple[str, str]]] = ...  # "memorization" and "reasoning" groups
```

**Important `batch_size` note**: This is NOT the model inference batch size. It controls how many source nodes are processed per backward pass in circuit-tracer's attribution computation. Larger = faster but requires more VRAM/RAM.

**`desired_logit_prob = 0.95`**: circuit-tracer builds the graph by iteratively adding the most important features until the attribution covers 95% of the target token's final logit probability. With `max_feature_nodes = 200`, you get at most 200 features but often fewer.

---

### `model.py`

Loads Gemma 3 4B PT with Gemma Scope 2 transcoders.

**Key function: `_load_transcoder_set(cfg)`**

```python
pattern = f"transcoder_all/layer_*_{variant}/params.safetensors"
local_dir = snapshot_download(repo_id, allow_patterns=[pattern])

transcoders = {}
for layer in range(34):
    path = Path(local_dir) / "transcoder_all" / f"layer_{layer}_{variant}" / "params.safetensors"
    transcoders[layer] = load_gemma_scope_2_transcoder(
        str(path), layer=layer, device=device, dtype=dtype,
        lazy_encoder=False, lazy_decoder=False,
    )

return TranscoderSet(transcoders,
    feature_input_hook="mlp.hook_in",
    feature_output_hook="hook_mlp_out",
    scan=f"{repo_id}//transcoder_all/{variant}")
```

`lazy_decoder=False` loads all `W_dec` matrices upfront. This is required because we access `W_dec[feat_idx]` for every selected feature.

**Key function: `load_model(cfg) → LoadedModel`**

```python
model = ReplacementModel.from_pretrained_and_transcoders(
    model_name=cfg.model_name,
    transcoders=transcoder_set,
    backend="nnsight",
    device=device, dtype=dtype,
)

W_U = model.unembed_weight.detach().float().T.contiguous().to(device)
tokenizer = model.tokenizer
return LoadedModel(model=model, W_U=W_U, tokenizer=tokenizer)
```

`W_U` is transposed: `lm_head.weight` is `[vocab_size, d_model]`; we need `[d_model, vocab_size]` so that `W_dec[f] @ W_U` gives a `[vocab_size]` logit vector.

**`LoadedModel` fields:**
- `model`: the `ReplacementModel` (nnsight backend)
- `W_U`: `[d_model, vocab_size]` float32, on model device
- `tokenizer`: HuggingFace tokenizer

---

### `attribution.py`

Runs circuit-tracer and computes all per-feature metrics.

**`run_attribution(prompt, correct_token_id, prompt_id, loaded, cfg)`**

Calls `circuit_tracer.attribute()` to build the attribution graph, then calls `_extract_records()`.

**`_extract_records(graph, ...)` — Two-phase design**

*Phase 1: Batch compute logit vectors, grouped by layer*

Grouping by layer allows a single matrix multiply per layer instead of one per feature:

```python
layer_to_sel_indices: dict[int, list[int]] = defaultdict(list)
for sel_idx in range(n_features):
    layer = int(graph.active_features[graph.selected_features[sel_idx]][0])
    layer_to_sel_indices[layer].append(sel_idx)

for layer, sel_indices in layer_to_sel_indices.items():
    feat_idxs = [int(graph.active_features[graph.selected_features[i]][2]) for i in sel_indices]
    transcoder = loaded.model.transcoders[layer]
    batch_logit_vecs = transcoder.W_dec[feat_idxs].float() @ loaded.W_U  # [n, vocab_size]
    for j, sel_idx in enumerate(sel_indices):
        data[sel_idx] = _compute_feature_metrics(activation, batch_logit_vecs[j], ...)
```

*Phase 2: Assemble FeatureRecord list*

```python
adj = graph.adjacency_matrix   # [n_nodes, n_nodes]
n_active = graph.active_features.shape[0]

for sel_idx in range(n_features):
    active_idx = int(graph.selected_features[sel_idx])  # index into active_features
    layer, pos, feat_idx = graph.active_features[active_idx].tolist()
    incoming = int((adj[active_idx, :n_active].abs() > 1e-4).sum().item())
    ...
```

**Critical indexing note**: The adjacency matrix is indexed by `active_idx` (position in the full `active_features` tensor), NOT by `sel_idx` (position in the `selected_features` subset). `selected_features` is a subset of `active_features` — you must use `graph.selected_features[sel_idx]` to get the correct row.

**`_compute_feature_metrics(activation, logit_vec, correct_token_id, tokenizer)`**

All scalar metrics are computed here:

```python
c_vec = activation * logit_vec       # [vocab_size] contribution vector
abs_c = c_vec.abs()
total_mass = abs_c.sum().item()

logit_target = logit_vec[correct_token_id].item()
logit_norm = logit_vec.norm().item()
p = torch.softmax(logit_vec, dim=-1)
logit_entropy = float(-(p * torch.log(p.clamp(min=1e-10))).sum())

correct_token_rank = int((logit_vec > logit_vec[correct_token_id]).sum().item()) + 1

# N50/N90
if total_mass > 0:
    sorted_abs = abs_c.sort(descending=True).values
    cumsum = sorted_abs.cumsum(0)
    n50 = int((cumsum < 0.5 * total_mass).sum().item()) + 1
    n90 = int((cumsum < 0.9 * total_mass).sum().item()) + 1
else:
    n50 = n90 = -1   # sentinel; filtered by all plots

promote_ratio = c_vec.clamp(min=0).sum().item() / (total_mass + 1e-10)
specificity = logit_target / (logit_norm + 1e-10)
attribution = abs(activation * logit_target)
```

**Top-50 contributions** are decoded with `_batch_decode()` which calls `tokenizer.batch_decode([[tid] for tid in ids])` — ~50× faster than per-token decoding.

---

### `analyze.py`

Post-processing after collecting all `FeatureRecord` lists.

**`build_result(prompt, prompt_id, correct_token, records, elapsed) → dict`**

Wraps a prompt's results into the standard dict format stored in JSON:
```python
{
    "prompt": str,
    "prompt_id": str,           # e.g. "M1", "R3"
    "correct_token": str,       # e.g. " Paris"
    "elapsed_s": float,
    "n_features": int,
    "features": [asdict(r) for r in records]
}
```

**`run_regression(all_results) → (stats, xs, ys, group_labels)`**

Pools all `(specificity, attribution)` pairs across all prompts and runs:
- Pearson r
- Spearman r
- OLS linear regression

Requires ≥2 features (raises `ValueError` with a helpful message otherwise).

Prints a formatted summary including whether the attribution inversion hypothesis is confirmed:
- `pearson_r < −0.1 AND pearson_p < 0.05` → **CONFIRMED**
- `pearson_r < 0` but not significant → weak trend
- `pearson_r ≥ 0` → hypothesis not supported

**`save_scatter_plot(xs, ys, group_labels, stats, path)`**

Scatter plot of specificity vs attribution, with:
- Red dots = memorization prompts (M)
- Blue dots = reasoning prompts (R)
- Black OLS regression line with equation in legend

**`save_results(all_results, stats, output_path)`**

Saves to JSON:
```json
{
    "regression_stats": {...},
    "prompts": [...]
}
```

---

### `run_poc.py`

Main entry point. Orchestrates the full pipeline.

```bash
uv run python -m src.poc.run_poc                    # auto-detect device
uv run python -m src.poc.run_poc --device mps       # Apple Silicon
uv run python -m src.poc.run_poc --device cuda      # NVIDIA, multi-GPU
uv run python -m src.poc.run_poc --gpus 0 1         # specific GPUs
```

**`_default_device()`**: Auto-selects `cuda` → `mps` → `cpu` in that priority order.

**`_tagged_prompts(cfg)`**: Flattens the `cfg.prompts` dict into `[(prompt, tok_str, prompt_id), ...]` where `prompt_id` is `{group_letter}{1-based-index}` (e.g. `M1`, `R3`).

**`_run_sequential(cfg)`**: Single-device path (CPU or MPS). Runs prompts one by one in the main process. Each prompt is wrapped in `try/except` so a single attribution failure doesn't abort the whole run.

**Multi-GPU path**: Uses `ProcessPoolExecutor` with `spawn` context. Prompts are distributed round-robin across GPU indices. Each subprocess runs `_gpu_worker` which loads its own model copy and processes its assigned prompts. Results are merged and sorted by original prompt order.

---

### `run_plots.py`

Regenerates all 10 plots from a saved `poc_results.json`.

```bash
uv run python -m src.poc.run_plots
uv run python -m src.poc.run_plots --results results/poc_results.json --output results/plots
```

Loads `data["prompts"]` from the JSON (supporting both the wrapped format from `run_poc.py` and a plain list). Calls each plot's `make_plot(results, output_path)` function.

---

### `inference_test.py`

Smoke test: verifies prompt/token quality **before** the full run. Does NOT load transcoders — only the base model.

```bash
uv run python -m src.poc.inference_test --device mps
uv run python -m src.poc.inference_test --topk 10
```

For each prompt:
1. Checks target token is single-token (multi-token = will be skipped in attribution)
2. Runs a forward pass
3. Reports the rank of the expected token in the next-token distribution

A good prompt has its expected token at rank 1 or 2. Low top-1 rate across prompts suggests tokenization issues (missing leading spaces, etc.).

---

## 7. Plots Reference

All plots read from the `FeatureRecord` fields stored in `poc_results.json`. All filter out `n90 <= 0` (sentinel for zero-mass features). Group colors: `M` (memorization) = red `#e74c3c`, `R` (reasoning) = blue `#3498db`.

### Plot 1: log(N₉₀) vs log(attribution) — Core Bias Test

**File**: `plot1_n90_vs_attribution.py`

**What it shows**: The central claim. If broad features (high N₉₀) get disproportionately high attribution, the OLS slope is positive.

**Axes**:
- x: `log(N₉₀)` — log broadness
- y: `log(attribution)` — log attribution magnitude

**What to look for**: Positive slope with significant p-value → attribution bias is real.

---

### Plot 2: total_impact vs target_efficiency — Decomposition

**File**: `plot2_decomposition.py`

**What it shows**: Decomposes attribution into two factors:
- `total_impact = |activation| × logit_norm` — how hard does the feature push the full vocab distribution?
- `target_efficiency = |logit_target| / logit_norm` — what fraction of that push lands on the target?

```
attribution = activation × logit_target
            = total_impact × target_efficiency
```

**Axes**:
- x: `total_impact` (log scale)
- y: `target_efficiency`
- color: attribution magnitude (LogNorm colormap)

**Quadrant interpretation**:
- top-left = precise but weak → hidden reasoning (low attribution despite high efficiency)
- top-right = precise AND strong → high attribution
- bottom-right = strong but imprecise → high attribution (format features)
- bottom-left = weak and imprecise → low attribution

---

### Plot 3: |activation| vs log(N₉₀) — Activation vs Broadness

**File**: `plot3_activation_vs_n90.py`

**What it shows**: Separates two explanations for attribution bias:
- Explanation A: broad features fire more strongly → attribution bias is the model's inherent preference
- Explanation B: broad features get high attribution at equal activation → attribution bias is a measurement artifact

**Axes**:
- x: `log(N₉₀)` — broadness
- y: `|activation|` — firing strength

**What to look for**: If slope is near zero → the bias is a measurement artifact (Explanation B), not a model preference.

---

### Plot 4: Histogram of log(N₉₀) — Distribution Shape

**File**: `plot4_n90_histogram.py`

**What it shows**: Whether feature broadness is bimodal (two discrete classes) or continuous (a spectrum). Two subplots: pooled all features, and overlaid by group (M vs R).

**What to look for**: Bimodal → discrete levels are justified. Unimodal/uniform → broadness is a continuous spectrum.

---

### Plot 5: Contribution Heatmap — One Per Prompt

**File**: `plot5_contribution_heatmap.py`

**What it shows**: For each prompt, a 2D heatmap of the top-20 features × top-50 tokens. Each cell is `c(f, t) = activation(f) × logit_vec[t]`. Red = positive contribution (promote), blue = negative (suppress). Target token column is highlighted in gold.

Row labels: `L{layer} F{feat_idx} a={activation:+.2f}`

**Output**: One PNG per prompt, saved to `results/plots/plot5_heatmaps/`.

---

### Plot 6: promote_ratio vs log(N₉₀) — Promotion vs Suppression

**File**: `plot6_promote_suppress.py`

**What it shows**: How features narrow the distribution — by actively promoting tokens (promote_ratio ≈ 1.0) or suppressing competitors (promote_ratio ≈ 0.0). Colored by attribution magnitude.

**Axes**:
- x: `promote_ratio` (0 = pure suppress, 1 = pure promote, 0.5 = mixed)
- y: `log(N₉₀)` — broadness

**What to look for**: Are suppressors generally broader (high N₉₀) or narrower? Natural clustering into suppressor vs promoter vs mixed?

---

### Plot 7: Token Position × Layer — Spatial Computation Map

**File**: `plot7_position_layer.py`

**What it shows**: Where in the computation (position × layer) each feature activates. Color = log(N₉₀), dot size = attribution.

**Axes**:
- x: token position in prompt (0 = first)
- y: transformer layer (0–33)
- color: cool = narrow/specific (low N₉₀), warm = broad (high N₉₀)

**What to look for**: Pattern where early/middle layers have warm (broad) features and late layers have cool (specific) features → confirms hierarchical narrowing by depth.

---

### Plot 8: log(N₉₀) vs target_efficiency — Efficiency vs Broadness

**File**: `plot8_n90_vs_target_efficiency.py`

**What it shows**: Complementary to Plot 1. Broad features (high N₉₀) should have low `target_efficiency` — they spread force across many tokens instead of focusing on the target.

**Axes**:
- x: `log(N₉₀)` — broadness
- y: `target_efficiency = |logit_target| / logit_norm`

**What to look for**: Negative slope → broad features are inefficient at promoting the correct token. Combined with Plot 1 showing broad features dominate attribution: they win attribution despite being bad at the task.

---

### Plot 9: Activation CV vs mean N₉₀ — Consistent vs Context-Specific

**File**: `plot9_activation_cv_vs_n90.py`

**What it shows**: For features appearing in ≥2 prompts, plots:
- CV (coefficient of variation) = std(|activation|) / mean(|activation|) — how consistently the feature fires across prompts
- mean N₉₀ — average broadness across appearances

**Quadrant interpretation**:
- low CV + high N₉₀ = consistent AND broad → **format/context features** (fire on every prompt of this type)
- high CV + low N₉₀ = variable AND specific → **computation features** (fire only when the specific answer is needed)
- low CV + low N₉₀ = consistent AND specific → **syntax features** (e.g., "answer ends with period")
- high CV + high N₉₀ = variable AND broad → rare format variations

Dot color = number of prompts the feature appears in (darker = more universal). Dot size = mean |activation|.

---

### Plot 10: Layer-wise N₉₀ Progression — Hierarchical Narrowing

**File**: `plot10_layer_n90_progression.py`

**What it shows**: The key test of the hierarchical narrowing hypothesis. For features at the **answer position** (last token position before generation), plots mean N₉₀ as a function of layer.

**What to look for**: Decreasing mean N₉₀ from early to late layers → the model progressively narrows from broad to specific across depth. Separate M/R group lines show whether memorization vs reasoning prompts narrow differently.

---

## 8. Output Data Format

### `results/poc_results.json`

```json
{
    "regression_stats": {
        "n_features_total": 4200,
        "n_prompts": 46,
        "pearson_r": -0.3142,
        "pearson_p": 1.23e-15,
        "spearman_r": -0.2891,
        "spearman_p": 8.45e-13,
        "ols_slope": -0.012345,
        "ols_intercept": 0.034567,
        "ols_r_squared": 0.0987,
        "ols_p_value": 1.23e-15,
        "specificity_mean": 0.1234,
        "specificity_std": 0.2345,
        "attribution_mean": 0.012345,
        "attribution_std": 0.023456
    },
    "prompts": [
        {
            "prompt": "The capital of France is",
            "prompt_id": "M1",
            "correct_token": " Paris",
            "elapsed_s": 45.2,
            "n_features": 87,
            "features": [
                {
                    "feature_idx": 12345,
                    "layer": 24,
                    "position": 6,
                    "prompt_id": "M1",
                    "activation": 3.45,
                    "logit_target": 2.12,
                    "logit_norm": 15.3,
                    "logit_entropy": 7.23,
                    "top1_token_id": 9876,
                    "top1_token_str": " Paris",
                    "correct_token_rank": 1,
                    "specificity": 0.1386,
                    "attribution": 7.314,
                    "c_total_mass": 52.6,
                    "n50": 3,
                    "n90": 28,
                    "promote_ratio": 0.73,
                    "incoming_edge_count": 5,
                    "top50_contributions": [
                        {"token_id": 9876, "token_str": " Paris", "c_value": 7.31},
                        ...
                    ]
                },
                ...
            ]
        },
        ...
    ]
}
```

---

## 9. Running the Pipeline

### Prerequisites

```bash
# 1. Authenticate with HuggingFace (gemma-3-4b-pt is a gated repo)
huggingface-cli login

# 2. Dependencies (circuit-tracer from GitHub, not PyPI)
uv add git+https://github.com/safety-research/circuit-tracer.git@26a976e
uv add transformer-lens torch numpy scipy matplotlib tqdm huggingface-hub safetensors einops
```

### Step 1: Verify Tokenization

Always run this first. It downloads only the tokenizer (~1MB) and the base model (~8GB) without transcoders, and checks that all target tokens are single-token and near top-1:

```bash
uv run python -m src.poc.inference_test --device mps  # or --device cpu
```

Look for `Rank = 1 ✓` for most prompts. Multi-token targets are flagged and will be skipped in attribution. After verifying, delete the cached model if you want to save ~8GB (it will be redownloaded with transcoders in the full run):

```bash
rm -rf ~/.cache/huggingface/hub/models--google--gemma-3-4b-pt
```

### Step 2: Full Attribution Run

The first run downloads ~8GB (base model) + ~22GB (262k transcoders for 34 layers). Subsequent runs use cache.

```bash
# Apple Silicon (MPS)
uv run python -m src.poc.run_poc --device mps

# Single NVIDIA GPU
uv run python -m src.poc.run_poc --device cuda

# Specific GPUs (multi-GPU)
uv run python -m src.poc.run_poc --device cuda --gpus 0 1 2 3

# CPU (slow, testing only)
uv run python -m src.poc.run_poc --device cpu
```

Results are saved to `results/poc_results.json`. The regression summary and scatter plot are generated automatically.

### Step 3: Generate All Plots

```bash
uv run python -m src.poc.run_plots
# or specify paths:
uv run python -m src.poc.run_plots --results results/poc_results.json --output results/plots
```

Outputs 10 PNGs to `results/plots/`. Plot 5 generates one PNG per prompt in `results/plots/plot5_heatmaps/`.

---

## 10. Metric Glossary

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| `activation` | raw transcoder activation | unbounded (signed) | how strongly the feature fired |
| `logit_target` | `W_dec[f] @ W_U[:, target]` | unbounded (signed) | feature's direct effect on target token logit |
| `logit_norm` | `‖W_dec[f] @ W_U‖₂` | ≥ 0 | total magnitude of feature's logit effect |
| `logit_entropy` | `H(softmax(W_dec[f] @ W_U))` | ≥ 0 | breadth of feature's logit distribution |
| `specificity` | `logit_target / logit_norm` | [−1, +1] | fraction of logit force pointing at target |
| `attribution` | `|activation × logit_target|` | ≥ 0 | direct contribution to target token's final logit |
| `c_total_mass` | `Σ|activation × logit_vec|` | ≥ 0 | total contribution magnitude across all vocab tokens |
| `n50` | min tokens for 50% of `c_total_mass` | ≥ 1 (or −1) | sharp core of contribution |
| `n90` | min tokens for 90% of `c_total_mass` | ≥ 1 (or −1) | broad reach of contribution |
| `promote_ratio` | `Σmax(c,0) / Σ|c|` | [0, 1] | fraction of contribution that is positive (promotion) |
| `incoming_edge_count` | `(adj[active_idx, :n_active].abs() > 1e-4).sum()` | ≥ 0 | upstream feature dependencies in attribution graph |
| `correct_token_rank` | rank of target in `logit_vec` | ≥ 1 | 1 = feature's top prediction is the correct answer |
| `top1_token_str` | `argmax(logit_vec)` decoded | string | most promoted token for interpretability |

---

## 11. Key Technical Decisions

### Why transcoders instead of SAEs?

SAEs (Sparse Autoencoders) decompose the **residual stream** — they capture what information is present at each position, but not what **computation** was done to put it there. Transcoders replace the MLP entirely — each transcoder feature represents a discrete **computational operation**: "take input X, produce output Y." This is crucial for studying the *mechanism* of narrowing, not just its *result*.

### Why the base model (`-pt`) instead of instruction-tuned (`-it`)?

The base model does raw next-token prediction. Instruction-tuned models have system prompts and chat formatting that would activate different feature sets and confound the analysis. We want to study the model's raw knowledge and computation on simple sentence completions.

### Why `width_65k_l0_medium_affine`?

- **65k width**: 4× richer than the 16k default. Fits on a single H100 (~36 GB for all 34 layers vs ~146 GB for 262k which exceeds 80 GB VRAM).
- **medium l0 (~30–60 active)**: Moderate sparsity. Small l0 (~10–20) may miss important features; big l0 (~60–150) includes too many noisy near-zero activations.
- **affine**: The `_affine` variant learns a skip connection `W_skip` that captures the residual/linear part of the MLP computation separately. This means the transcoder features represent only the *nonlinear* computation, making them more interpretable.

### Why `desired_logit_prob = 0.95`?

Circuit-tracer builds the attribution graph iteratively, adding features until they collectively explain 95% of the target logit. This is a coverage threshold, not a pruning threshold. Setting it to 1.0 would include almost all features; 0.95 gives a compact, representative graph.

### The adjacency matrix indexing bug

The graph has `n_active` total active features and `n_selected ≤ n_active` selected features. The adjacency matrix rows/columns are indexed by `active_idx` (0..n_active-1), not by position in the `selected_features` array. Using `sel_idx` directly as the adjacency row would read the wrong row — possibly from a completely different feature's edges. Always use `active_idx = int(graph.selected_features[sel_idx])` before accessing `adj[active_idx, ...]`.

### Entropy formula: `clamp` vs `+ epsilon`

```python
# WRONG: logit_entropy = float(-(p * torch.log(p + 1e-10)).sum())
# RIGHT:
logit_entropy = float(-(p * torch.log(p.clamp(min=1e-10))).sum())
```

`p + 1e-10` shifts all probability values, including the non-zero ones, distorting the entropy calculation. `p.clamp(min=1e-10)` only prevents log(0) on the effectively-zero probabilities, leaving the real probabilities unchanged.
