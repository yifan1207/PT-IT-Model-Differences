# Phase 0: Multi-Model Steering — Detailed Implementation Plan

**Date**: 2026-04-01
**Goal**: Extend the full causal steering story (precompute corrective directions + A1 α-sweep + evaluation) from Gemma-only to all 6 model families. This transforms the paper from single-model causal to multi-model causal evidence — the single highest-impact addition.

**CRITICAL DESIGN DECISION: NO CHAT TEMPLATES**
All experiments — direction extraction AND steering evaluation — use raw format B (no chat template) for all models. This ensures:
1. Identical input format across IT and PT (apples-to-apples comparison)
2. Consistent cross-model comparison (no per-model template idiosyncrasies)
3. Cleaner mechanistic claim: effects are weight-intrinsic, not template-gated
4. Matches direction extraction (precompute_directions_v2.py already uses `chat_template: False`)

**Note**: The existing Gemma A1_it_v4 results were run WITH chat template (`apply_chat_template=True` is the Exp6Config default). We must **rerun Gemma A1 without chat template** alongside the other 5 models so all 6 have consistent experimental conditions. The existing A1_notmpl Gemma ablation already showed similar dose-response, so we expect consistent results.

---

## 1. What We Need to Prove

The paper currently makes three core claims:

1. **Opposition**: IT activations oppose PT predictions at late layers (δ-cosine shift) — *already multi-model* (6/6 families, §3.1)
2. **Commitment delay**: IT commits to output tokens later than PT — *already multi-model* (6/6 families, §3.2)
3. **Causal format control**: Removing the corrective direction degrades formatting while preserving content — **Gemma-only** (§3.3)

Phase 0 replicates claim (3) across all 6 families. If format dose-response replicates on ≥4/6 models, the paper's central argument becomes bulletproof.

---

## 2. Architecture Audit — What Needs to Change

### 2.1 Current Gemma-hardcoded paths

The steering pipeline has several Gemma-specific assumptions that must be generalized:

| Component | File | Gemma-specific code | Required change |
|-----------|------|-------------------|-----------------|
| Model loading | `src/poc/shared/model.py` | `_N_LAYERS = 34`, hardcoded Gemma3ForConditionalGeneration, TranscoderSet | New `load_model_for_steering()` that loads plain HF model via `AutoModelForCausalLM` (no transcoders needed for A-series) |
| Config defaults | `src/poc/exp6/config.py` | `n_layers=34, d_model=2560, proposal_boundary=20`, `model_id = f"google/gemma-3-4b-{variant}"` | Accept these from CLI args or derive from `MODEL_REGISTRY` |
| Hook registration | `src/poc/exp6/interventions.py` L300,324,349,373 | `model_raw.language_model.layers[i].mlp` | Architecture-dependent layer path (see §2.2) |
| Runtime generation | `src/poc/exp6/runtime.py` L87 | `loaded.model._model` (nnsight wrapper) | Direct HF model access when skip_transcoders=True |
| EOS tokens | `src/poc/exp6/runtime.py` L62 | `<end_of_turn>` (Gemma-specific) | Per-model EOS/EOT tokens |
| Logit lens hooks | `src/poc/exp6/runtime.py` L107-108 | `model_raw.language_model.model.norm`, `.lm_head` | Per-model norm/lm_head paths |
| Direction extraction | `scripts/precompute_directions_v2.py` L66-69 | `ALL_LAYERS = range(1,34)`, `D_MODEL = 2560` | Derive from model spec |
| Direction extraction | `scripts/precompute_directions_v2.py` L406 | `model_raw.language_model.layers[li].mlp` | Architecture-dependent path |

### 2.2 Layer access paths per architecture

| Model | Root path | MLP hook | Layer module | Final norm | LM head |
|-------|-----------|----------|-------------|------------|---------|
| **Gemma 3 4B** | `model_raw.language_model.layers[i]` | `.mlp` | `.language_model.layers[i]` | `.language_model.model.norm` | `.language_model.lm_head` |
| **Llama 3.1 8B** | `model_raw.model.layers[i]` | `.mlp` | `.model.layers[i]` | `.model.norm` | `.lm_head` |
| **Qwen 3 4B** | `model_raw.model.layers[i]` | `.mlp` | `.model.layers[i]` | `.model.norm` | `.lm_head` |
| **Mistral 7B** | `model_raw.model.layers[i]` | `.mlp` | `.model.layers[i]` | `.model.norm` | `.lm_head` |
| **DeepSeek-V2-Lite** | `model_raw.model.layers[i]` | `.mlp` | `.model.layers[i]` | `.model.norm` | `.lm_head` |
| **OLMo 2 7B** | `model_raw.model.layers[i]` | `.mlp` | `.model.layers[i]` | `.model.norm` | `.lm_head` |

**Key insight**: Gemma 3 is the only outlier with `.language_model.layers`. All other models use `.model.layers`. The adapter needs only two paths.

### 2.3 Layer ranges per model

From `MODEL_REGISTRY` in `src/poc/cross_model/config.py`:

| Model | n_layers | d_model | corrective_onset (~60%) | proposal_boundary | corrective_layers |
|-------|----------|---------|------------------------|-------------------|-------------------|
| gemma3_4b | 34 | 2560 | 20 | 20 | 20–33 |
| llama31_8b | 32 | 4096 | 19 | 19 | 19–31 |
| qwen3_4b | 36 | 2560 | 22 | 22 | 22–35 |
| mistral_7b | 32 | 4096 | 19 | 19 | 19–31 |
| deepseek_v2_lite | 27 | 2048 | 16 | 16 | 16–26 |
| olmo2_7b | 32 | 4096 | 19 | 19 | 19–31 |

**Note**: `proposal_boundary` = `corrective_onset` = `round(n_layers * 0.60)`. This is the default from `ModelSpec.corrective_onset`. We should use this for consistency with the observational analysis, but also test sensitivity to this boundary (already validated by Exp7 0F on Gemma).

### 2.4 No chat templates — raw format B everywhere

**Design decision**: All steering experiments use `apply_chat_template=False`. Both IT and PT models receive identical raw text prompts (format B from `eval_dataset_v2.jsonl`). This matches exactly how directions were extracted in `precompute_directions_v2.py` (which already records `"chat_template": False` in its metadata).

**Rationale**:
- Direction extraction uses raw prompts → steering evaluation should too for consistency
- Removes per-model template variation as a confound in cross-model comparison
- The existing Gemma `A1_notmpl` ablation already validated that the dose-response is weight-intrinsic (not template-gated), so we lose nothing by dropping templates
- Simpler code: no need to handle per-model template formats

**Implementation**: Set `apply_chat_template=False` in the run script for all models. The `Exp6Config.__post_init__` already sets this to False for PT; we just need to also force it for IT via CLI or config override.

**EOS token handling** still needs per-model logic (even without chat template, models may emit different stop tokens):
- Gemma 3: `<end_of_turn>` (token 107) — may still fire even without chat template
- Llama 3.1: `<|eot_id|>` (token 128009)
- Qwen 3: `<|im_end|>` (varies)
- Mistral 7B: `</s>` (standard)
- DeepSeek-V2-Lite: `<｜end▁of▁sentence｜>` or `<|im_end|>` (varies)
- OLMo 2: `<|endoftext|>` (standard)

**Fix**: Replace the hardcoded `<end_of_turn>` check in `runtime.py` L62 with a generic pattern: use `tokenizer.eos_token_id` plus any model-specific EOT token found via `tokenizer.convert_tokens_to_ids()` for a list of known EOT strings.

---

## 3. Implementation Plan — Code Changes

### 3.1 Create `src/poc/exp6/model_adapter.py` (NEW FILE)

A lightweight adapter that resolves architecture-specific layer paths:

```python
@dataclass
class ModelAdapter:
    """Architecture-specific layer path resolver for Exp6 steering."""
    model_name: str          # short name from MODEL_REGISTRY
    layers_root: str         # e.g. "model.layers" or "language_model.layers"
    final_norm_path: str     # e.g. "model.norm"
    lm_head_path: str        # e.g. "lm_head"
    n_layers: int
    d_model: int
    corrective_onset: int    # proposal_boundary
    eot_tokens: list[str]    # extra EOS/EOT token strings beyond tokenizer.eos_token_id

    def get_layer(self, model_raw, layer_idx):
        """Traverse model_raw to get layer module."""
        parts = self.layers_root.split(".")
        mod = model_raw
        for p in parts:
            mod = getattr(mod, p)
        return mod[layer_idx]

    def get_mlp(self, model_raw, layer_idx):
        return self.get_layer(model_raw, layer_idx).mlp

    def get_attn(self, model_raw, layer_idx):
        return self.get_layer(model_raw, layer_idx).self_attn

    def get_final_norm(self, model_raw):
        parts = self.final_norm_path.split(".")
        mod = model_raw
        for p in parts:
            mod = getattr(mod, p)
        return mod

    def get_lm_head(self, model_raw):
        parts = self.lm_head_path.split(".")
        mod = model_raw
        for p in parts:
            mod = getattr(mod, p)
        return mod

def adapter_for_model(model_name: str) -> ModelAdapter:
    """Build adapter from MODEL_REGISTRY spec."""
    from src.poc.cross_model.config import get_spec
    spec = get_spec(model_name)

    # Gemma 3 is the only model that uses language_model.layers
    if "gemma" in model_name:
        layers_root = "language_model.layers"
        final_norm = "language_model.model.norm"
        lm_head = "language_model.lm_head"
    else:
        layers_root = "model.layers"
        final_norm = "model.norm"
        lm_head = "lm_head"

    return ModelAdapter(
        model_name=model_name,
        layers_root=layers_root,
        final_norm_path=final_norm,
        lm_head_path=lm_head,
        n_layers=spec.n_layers,
        d_model=spec.d_model,
        corrective_onset=spec.corrective_onset,
        eot_tokens=_eot_tokens_for(model_name),
    )
```

### 3.2 Modify `src/poc/exp6/config.py`

Add CLI-driven overrides so the config isn't Gemma-defaulted:

```python
# Add to Exp6Config:
model_name: str = "gemma3_4b"  # NEW: key into MODEL_REGISTRY

def __post_init__(self):
    if self.model_name != "gemma3_4b" and not self.model_id:
        from src.poc.cross_model.config import get_spec, model_id_for_variant
        spec = get_spec(self.model_name)
        self.model_id = model_id_for_variant(spec, self.model_variant)
        self.n_layers = spec.n_layers
        self.d_model = spec.d_model
        self.proposal_boundary = spec.corrective_onset
    elif not self.model_id:
        self.model_id = f"google/gemma-3-4b-{self.model_variant}"
    # ... rest of existing __post_init__
```

### 3.3 Modify `src/poc/exp6/interventions.py`

Replace all `model_raw.language_model.layers[i].mlp` with adapter calls:

```python
def register_hooks(self, model_raw, cfg, adapter=None):
    # If no adapter, fall back to Gemma paths (backward compat)
    get_mlp = (adapter.get_mlp if adapter else
               lambda m, i: m.language_model.layers[i].mlp)
    get_layer = (adapter.get_layer if adapter else
                 lambda m, i: m.language_model.layers[i])
    get_attn = (adapter.get_attn if adapter else
                lambda m, i: m.language_model.layers[i].self_attn)

    # Then use get_mlp(model_raw, layer_idx) everywhere
```

### 3.4 Modify `src/poc/exp6/runtime.py`

1. Replace `loaded.model._model` with direct HF model access when `skip_transcoders=True`
2. Use adapter for logit-lens norm/lm_head paths
3. Generalize EOS token logic

### 3.5 Modify `src/poc/exp6/run.py`

Add `--model-name` CLI argument:
```python
p.add_argument("--model-name", default="gemma3_4b",
               choices=list(MODEL_REGISTRY.keys()),
               help="Model family from cross_model registry")
```

Pass through to Exp6Config and construct adapter.

### 3.6 Create `scripts/precompute_directions_multimodel.py` (NEW FILE)

Generalized version of `precompute_directions_v2.py` that:
- Accepts `--model-name` from MODEL_REGISTRY
- Derives ALL_LAYERS, D_MODEL, layer paths from the model spec
- Loads model via plain `AutoModelForCausalLM` (no transcoders)
- Outputs to `results/cross_model/{model_name}/directions/corrective_directions.npz`

**Key changes from v2**:
- `ALL_LAYERS = list(range(1, spec.n_layers))` instead of hardcoded `range(1, 34)`
- `D_MODEL = spec.d_model` instead of hardcoded 2560
- Hook path uses adapter pattern
- Output dir per model

---

## 4. Experiments to Run

### 4.1 Step 1: Precompute corrective directions (5 models)

For each non-Gemma model, run the 4-phase direction extraction pipeline:

| Model | Dataset | Top-N calibration | Layers | d_model | Est. time (1 GPU) |
|-------|---------|-------------------|--------|---------|-------------------|
| llama31_8b | eval_dataset_v2.jsonl (1,400) | 600 | 1–31 | 4096 | ~3h |
| qwen3_4b | eval_dataset_v2.jsonl (1,400) | 600 | 1–35 | 2560 | ~2h |
| mistral_7b | eval_dataset_v2.jsonl (1,400) | 600 | 1–31 | 4096 | ~3h |
| deepseek_v2_lite | eval_dataset_v2.jsonl (1,400) | 600 | 1–26 | 2048 | ~2h |
| olmo2_7b | eval_dataset_v2.jsonl (1,400) | 600 | 1–31 | 4096 | ~3h |

**Output per model**: `results/cross_model/{model_name}/directions/corrective_directions.npz`

**Parallelism**: Each model needs 2 GPU passes (IT gen + PT gen + PT NLL → IT acts + PT acts). With 8 GPUs and 2 workers per model, run 4 models simultaneously.

**Total estimated time**: ~4–6 hours with 8 GPUs.

```bash
# Example for Llama 3.1 8B:
# Phase 1: gen (2 workers)
uv run python scripts/precompute_directions_multimodel.py \
    --model-name llama31_8b --phase gen \
    --worker-index 0 --n-workers 2 --device cuda:0
uv run python scripts/precompute_directions_multimodel.py \
    --model-name llama31_8b --phase gen \
    --worker-index 1 --n-workers 2 --device cuda:1

# Phase 2: score (CPU)
uv run python scripts/precompute_directions_multimodel.py \
    --model-name llama31_8b --phase score

# Phase 3: acts (2 workers)
uv run python scripts/precompute_directions_multimodel.py \
    --model-name llama31_8b --phase acts \
    --worker-index 0 --n-workers 2 --device cuda:0
uv run python scripts/precompute_directions_multimodel.py \
    --model-name llama31_8b --phase acts \
    --worker-index 1 --n-workers 2 --device cuda:1

# Phase 4: merge (CPU)
uv run python scripts/precompute_directions_multimodel.py \
    --model-name llama31_8b --phase merge
```

### 4.2 Step 2: Direction validation gate (ALL 6 models, including Gemma)

**Before committing to the full α-sweep, validate each model's direction with a lightweight sanity check.**

**4.2a Direction stability (0A analog)**:
For each model, bootstrap resample the 600 selected calibration records (with replacement), recompute the mean direction, repeat 100 times. Compute pairwise cosine similarities between bootstrapped directions.
- **Pass criterion**: Mean pairwise cosine > 0.90
- **GPU cost**: Zero — reuses existing acts data, just resamples and averages on CPU
- **Time**: ~5 min per model

**4.2b Quick sanity steering (3-point check)**:
Run A1 at just 3 α values (1.0 baseline, 0.0 full removal, -1.0 amplification) on a small subset (~100 prompts) with `apply_chat_template=False`.
- **Pass criterion**: STR drops at α=0 vs α=1.0; STR rises at α=-1 vs α=1.0
- **GPU cost**: ~20 min per model on 1 GPU
- **Purpose**: Catches bad directions, wrong layer paths, hook bugs, or architecture-specific issues before the expensive full sweep

**If a model fails either check**: Investigate before proceeding. Possible causes: MoE routing instability (DeepSeek), wrong layer path in adapter, insufficient calibration data, or genuinely absent corrective stage (Qwen — expected).

### 4.3 Step 3: A1 α-sweep on all 6 models (NO chat template)

**All 6 models including Gemma** — the existing Gemma A1_it_v4 was run with `apply_chat_template=True`. We rerun Gemma with `apply_chat_template=False` so all 6 models have identical experimental conditions.

**Alpha values (same 14 for all models)**: `[5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0, -2.0, -3.0, -5.0]`

**Conditions per model**: 14 α values + 1 baseline + 3 random controls = 18 conditions

**Evaluation dataset**: `data/eval_dataset_v2.jsonl` (1,400 prompts, 7 categories)

**No chat template**: `--apply-chat-template false` (or equivalent config override) for all models including Gemma. Raw format B input only.

**Benchmarks per condition**:
- `structural_token_ratio` (STR) — programmatic, fast
- `format_compliance_v2` — programmatic
- `mmlu_forced_choice` — forced-choice log-prob
- `reasoning_em` — exact match
- `alignment_behavior` — exact match

**Post-hoc LLM judge** (after merge):
- G1 (response quality), G2 (instruction following), S1 (safety refusal), S2 (safety compliance)

**Estimated time per model**: ~5–8 hours with 2 workers (1,400 prompts × 18 conditions × 200 max_gen_tokens)

**Total for 6 models**: ~30–48 hours with 2 GPUs per model. With 8 GPUs running 4 models in parallel → ~18–24 hours wall clock.

```bash
# Example for Llama 3.1 8B (2 workers, NO chat template):
for wi in 0 1; do
    uv run python src/poc/exp6/run.py \
        --experiment A1 \
        --model-name llama31_8b \
        --variant it \
        --no-chat-template \
        --dataset data/eval_dataset_v2.jsonl \
        --n-eval-examples 1400 \
        --device "cuda:${wi}" \
        --worker-index "$wi" \
        --n-workers 2 \
        --run-name "A1_llama31_8b_it_notmpl_v1" \
        --corrective-direction-path "results/cross_model/llama31_8b/directions/corrective_directions.npz" \
        --output-base "results/cross_model/llama31_8b/exp6" \
        --proposal-boundary 19 \
        --n-layers 32 \
        > "logs/phase0/llama31_8b_A1_w${wi}.log" 2>&1 &
done
wait

# Merge workers
uv run python scripts/merge_exp6_workers.py \
    --experiment A1 --variant it --n-workers 2 \
    --merged-name "merged_A1_llama31_8b_it_notmpl_v1" \
    --source-dirs results/cross_model/llama31_8b/exp6/A1_llama31_8b_it_notmpl_v1_w{0,1}

# LLM judge (post-hoc)
uv run python scripts/llm_judge_exp6.py \
    --merged-dir "results/cross_model/llama31_8b/exp6/merged_A1_llama31_8b_it_notmpl_v1" \
    --model google/gemini-2.5-flash --workers 16 --tasks g1 g2 s1 s2
```

### 4.4 Step 4: Direction stability check (0A analog) — full bootstrap per model

After the sanity check passes and the full sweep completes, run the full 1000-resample bootstrap for paper-quality stability numbers. This can run in parallel with the LLM judge (both are CPU/API-bound).

```bash
# Full bootstrap: 1000 resamples, compute pairwise cosine distribution
# Report: mean, std, 95% CI of pairwise cosine similarity
# This is the number that goes in the paper
```

### 4.5 Step 5: Piggybacked measurements during steering (near-zero marginal cost)

These measurements use data already collected during the A1 sweep. Enable `--collect-logit-lens` for all steering runs.

**4.5a Commitment delay under steering (causal link experiment 1B)**
At each of the 14 α values, compute the median KL-commitment layer across all generated tokens. This directly tests: "does removing the corrective direction accelerate commitment?" If commitment layer shifts earlier as α→0 across multiple models, this establishes the causal link: corrective direction → commitment delay.
- **Data source**: logit_lens_top1.npz already captured during generation
- **Compute**: CPU post-processing only. ~10 min per model.
- **Deliverable**: Commitment layer vs α curve for all 6 models — the causal link figure.

**4.5b ID under steering (experiment 1C)**
At 3 key α values (α=1.0 baseline, α=0 removal, α=-1 reversal), cache layer-wise hidden states for ~200 prompts and compute TwoNN ID profiles.
- **Additional GPU cost**: ~30 min per model (3 conditions × 200 prompts with hidden state caching)
- **Note**: This requires a modified generation pass that saves hidden states, not just logit-lens top-1. Run as a separate lightweight pass after the main sweep.
- **Deliverable**: ID profile plots at 3 α values for all 6 models.

**4.5c PCA of IT-PT direction (experiment 1A)**
After the precompute phase, run PCA on the per-record IT-PT activation differences at corrective layers.
- **Data source**: The per-worker acts files from precompute phase 3 (before averaging)
- **Compute**: CPU only. ~5 min per model.
- **Key question**: Is PC1 > 60% of variance? If yes, rank-1 direction justified. If distributed, the corrective stage decomposes into sub-directions.
- **Deliverable**: Scree plots for all 6 models.

### 4.6 Special considerations per model

| Model | Risk | Mitigation |
|-------|------|------------|
| **DeepSeek-V2-Lite** | MoE routing may mean MLP output varies with input routing decisions. `max_new_tokens=64` constraint. | Use `max_gen_tokens=64` in config. Monitor for NaN/inf in generated text. If MoE routing is input-dependent, the direction may be less stable — bootstrap 0A check will reveal this. |
| **Qwen 3 4B** | Expected null result (zero commitment delay → no concentrated corrective stage). | This is a **feature**, not a bug. If Qwen shows no format dose-response, it's a powerful negative control confirming the corrective-stage interpretation. If it DOES show dose-response, we need to revise the narrative. |
| **Mistral 7B** | Sliding window attention (4096). | Shouldn't affect MLP-level steering. Verify that generated outputs don't degrade with long prompts. |
| **OLMo 2 7B** | MHA (32 query = 32 KV heads) — no GQA. Largest ID expansion (+4.7). | Standard architecture, should work cleanly. High ID expansion predicts strong dose-response. |
| **Llama 3.1 8B** | GQA (32 query, 8 KV heads). Most widely used model. | Standard architecture. Borderline commitment delay (+0.5) — may show weaker but still present dose-response. |

---

## 5. Plots to Create

### 5.1 Multi-model dose-response (THE CENTERPIECE) — Figure for §3.3

**What**: 6-panel figure (one per model), each showing format metrics (STR, format_compliance_v2) and content metrics (reasoning_em, MMLU) as a function of α.

**Layout**: 2×3 grid or 3×2 grid. Each panel has:
- X-axis: α (log-ish scale, from -5 to 5)
- Y-axis: metric value (normalized or raw)
- Lines: STR (blue), format_compliance (green), reasoning_em (red), MMLU (orange)
- Vertical dashed line at α=1.0 (baseline)
- 95% CI bands from bootstrap (if available; otherwise ±1 SE across records)

**Prediction**: 5/6 models show format degradation as α→0 with content preservation. Qwen shows flat.

**Script**: `scripts/plot_phase0_multimodel_dose_response.py`

### 5.2 Format-content dissociation summary — Figure for §3.3

**What**: Single figure comparing the "format-content gap" across all 6 models.

**Layout**: Bar chart or scatter plot. For each model at α=0 (full removal):
- Bar 1: Format change (e.g., STR Δ from baseline)
- Bar 2: Content change (e.g., reasoning_em Δ from baseline)

**Purpose**: Shows that the dissociation is universal — format degrades, content doesn't.

### 5.3 Multi-model random control comparison — Figure for appendix

**What**: For each model, compare the A1 α-sweep to the random-direction control at matched magnitudes.

**Purpose**: Confirms direction specificity generalizes across architectures. Can be appendix.

### 5.4 Direction stability across models — Figure for appendix

**What**: 6-panel figure showing bootstrap pairwise cosine distributions for each model's corrective direction.

**Purpose**: Validates that the extracted direction is stable (not random noise) for all models.

### 5.5 Multi-model effect sizes — Table for §3.3

**What**: Table of Cohen's d effect sizes for key metrics at α=0 (full removal) across all 6 models.

| Model | STR d | format_compliance d | reasoning_em d | MMLU d | Safety d |
|-------|-------|-------------------|---------------|--------|----------|
| Gemma 3 4B | -X.XX | -X.XX | -X.XX | -X.XX | -X.XX |
| Llama 3.1 8B | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... |

### 5.6 Cross-model monotonicity Spearman — Table/Figure for §3.3

**What**: Spearman correlation between α and each metric, per model. Tests whether the dose-response is monotonic.

**Purpose**: Quantitative evidence that the α→metric relationship is dose-dependent, not just on/off.

### 5.7 LLM judge dose-response — Figure for appendix

**What**: G1 (quality) and G2 (instruction following) across α for all 6 models.

**Purpose**: Validates the programmatic metrics with independent LLM evaluation.

### 5.8 Commitment delay vs α (THE CAUSAL LINK) — Figure for §3.3.7 (NEW)

**What**: 6-panel figure showing median KL-commitment layer as a function of α for all models.

**Layout**: Same 2×3 grid. Each panel has:
- X-axis: α
- Y-axis: median commitment layer
- Horizontal dashed lines at PT baseline and IT baseline commitment layers
- Expected: monotonic shift from IT's commitment layer (at α=1) toward PT's (at α=0)

**Purpose**: The strongest causal evidence in the paper — a single scalar (α) simultaneously modulates format quality AND commitment timing. Currently Gemma-only (§3.3.7). Multi-model replication makes this bulletproof.

### 5.9 ID profiles under steering — Figure for §4.1 (NEW)

**What**: 6-panel figure showing TwoNN ID profiles at α=1.0, α=0.0, α=-1.0.

**Purpose**: Tests whether removing the corrective direction reduces late-layer ID. Completes the triad: direction ↔ commitment ↔ ID.

### 5.10 PCA spectrum of corrective direction — Figure for §5 or appendix (NEW)

**What**: Scree plots showing variance explained by top PCs of the per-record IT-PT activation differences at corrective layers, for all 6 models.

**Purpose**: Answers "is the corrective direction one thing or several?" If PC1 > 60% → rank-1 justified. If distributed → corrective stage decomposes into sub-directions.

---

## 6. Integration with Paper

### 6.1 Changes to §3.3 (Steering)

Currently §3.3 presents Gemma-only steering results. After Phase 0:

**Before**: "We remove the corrective direction from Gemma 3 4B IT..."
**After**: "We remove each model's corrective direction across all 6 families..."

Specific additions:
- Replace the single-model dose-response figure with the 6-panel multi-model figure (§5.1)
- Add the format-content dissociation summary (§5.2)
- Add the multi-model effect size table (§5.5)
- Move single-model details to appendix

### 6.2 Changes to Introduction

Update the third finding:
**Before**: "Removing this direction from generation degrades structural formatting while preserving content quality"
**After**: "Removing this direction degrades formatting while preserving content across 5 of 6 model families, with the exception (Qwen) predicted by our observational analysis"

### 6.3 Changes to Discussion

- Strengthen the "What replicates" section with specific multi-model steering evidence
- Qwen null result discussion becomes more powerful with causal data

### 6.4 New appendix sections

- Per-model dose-response curves (detailed versions of §5.1)
- Per-model random control comparison (§5.3)
- Direction stability per model (§5.4)
- Per-model LLM judge scores (§5.7)

---

## 7. Execution Timeline & GPU Schedule

### Week 1: Code changes + direction precompute + validation

| Day | Task | GPUs |
|-----|------|------|
| Day 1 | Create `model_adapter.py`, modify `config.py`, `interventions.py`, `runtime.py`, `run.py`. Add `--model-name` and `--no-chat-template` CLI args. | 0 |
| Day 2 | Create `precompute_directions_multimodel.py`, test on Llama (1 worker, 50 records) | 1 |
| Day 3 | Run direction precompute for Llama + OLMo + Mistral (2 workers each) | 6 |
| Day 4 | Run direction precompute for DeepSeek + Qwen (2 workers each). Bootstrap stability (0A) for Day 3 models. | 4+0 |
| Day 5 | Bootstrap stability for Day 4 models. **Validation gate**: 3-point sanity steering check on all 5 new models (~100 prompts each, ~20 min/model). | 4 |

### Week 2: Full steering runs + evaluation (all 6 models, no chat template)

| Day | Task | GPUs |
|-----|------|------|
| Day 6 | A1 α-sweep (no-tmpl): Gemma (2w) + Llama (2w) + OLMo (2w) | 6 |
| Day 7 | A1 α-sweep continued (if needed). Start Mistral (2w). | 4–6 |
| Day 8 | A1 α-sweep: DeepSeek (2w) + Qwen (2w). Merge + LLM judge for completed models. | 4+0 |
| Day 9 | Merge + LLM judge for remaining models. Full bootstrap (0A) for all 6. | 0 |
| Day 10 | Generate all plots. Cross-model comparison analysis. | 0 |

**Total GPU-hours** (estimated):
- Direction precompute: 5 new models × ~3h × 2 GPUs = ~30 GPU-hours
- Validation gate: 5 models × ~0.3h × 1 GPU = ~1.5 GPU-hours
- A1 steering (6 models): 6 models × ~7h × 2 GPUs = ~84 GPU-hours
- **Total: ~116 GPU-hours** (~15h wall on 8 GPUs)

---

## 8. Run Script Template

Create `scripts/run_phase0_multimodel.sh`:

```bash
#!/usr/bin/env bash
# Phase 0: Multi-model steering — precompute + A1 α-sweep for all 6 models.
#
# Usage:
#   bash scripts/run_phase0_multimodel.sh --step precompute   # Step 1: directions
#   bash scripts/run_phase0_multimodel.sh --step steering     # Step 2: A1 α-sweep
#   bash scripts/run_phase0_multimodel.sh --step judge        # Step 3: LLM judge
#   bash scripts/run_phase0_multimodel.sh --step plots        # Step 4: generate plots
#   bash scripts/run_phase0_multimodel.sh --model llama31_8b  # single model only

set -euo pipefail

MODELS=(gemma3_4b llama31_8b olmo2_7b mistral_7b deepseek_v2_lite qwen3_4b)
DATASET="data/eval_dataset_v2.jsonl"
N_EVAL=1400
N_WORKERS=2
LOG_DIR="logs/phase0"
JUDGE_MODEL="google/gemini-2.5-flash"

# Model-specific configs (n_layers, proposal_boundary, max_gen_tokens)
declare -A N_LAYERS=( [llama31_8b]=32 [olmo2_7b]=32 [mistral_7b]=32
                      [deepseek_v2_lite]=27 [qwen3_4b]=36 [gemma3_4b]=34 )
declare -A PROP_BOUND=( [llama31_8b]=19 [olmo2_7b]=19 [mistral_7b]=19
                        [deepseek_v2_lite]=16 [qwen3_4b]=22 [gemma3_4b]=20 )
declare -A MAX_GEN=( [llama31_8b]=200 [olmo2_7b]=200 [mistral_7b]=200
                     [deepseek_v2_lite]=64 [qwen3_4b]=200 [gemma3_4b]=200 )

mkdir -p "$LOG_DIR"

precompute_model() {
    local model=$1 gpu_start=$2
    local dir_out="results/cross_model/${model}/directions"
    mkdir -p "$dir_out"

    echo "=== Precompute directions: $model (GPU ${gpu_start}-$((gpu_start+N_WORKERS-1))) ==="

    # Phase 1: gen
    for ((wi=0; wi<N_WORKERS; wi++)); do
        local gpu=$((gpu_start + wi))
        uv run python scripts/precompute_directions_multimodel.py \
            --model-name "$model" --phase gen \
            --worker-index "$wi" --n-workers "$N_WORKERS" \
            --device "cuda:${gpu}" \
            > "$LOG_DIR/${model}_precompute_gen_w${wi}.log" 2>&1 &
    done
    wait

    # Phase 2: score (CPU)
    uv run python scripts/precompute_directions_multimodel.py \
        --model-name "$model" --phase score \
        > "$LOG_DIR/${model}_precompute_score.log" 2>&1

    # Phase 3: acts
    for ((wi=0; wi<N_WORKERS; wi++)); do
        local gpu=$((gpu_start + wi))
        uv run python scripts/precompute_directions_multimodel.py \
            --model-name "$model" --phase acts \
            --worker-index "$wi" --n-workers "$N_WORKERS" \
            --device "cuda:${gpu}" \
            > "$LOG_DIR/${model}_precompute_acts_w${wi}.log" 2>&1 &
    done
    wait

    # Phase 4: merge (CPU)
    uv run python scripts/precompute_directions_multimodel.py \
        --model-name "$model" --phase merge \
        > "$LOG_DIR/${model}_precompute_merge.log" 2>&1
}

steer_model() {
    local model=$1 gpu_start=$2
    local corr_dir="results/cross_model/${model}/directions/corrective_directions.npz"
    local out_base="results/cross_model/${model}/exp6"
    local run_name="A1_${model}_it_v1"
    local nl=${N_LAYERS[$model]}
    local pb=${PROP_BOUND[$model]}
    local mg=${MAX_GEN[$model]}

    echo "=== A1 α-sweep: $model (GPU ${gpu_start}-$((gpu_start+N_WORKERS-1))) ==="

    for ((wi=0; wi<N_WORKERS; wi++)); do
        local gpu=$((gpu_start + wi))
        uv run python src/poc/exp6/run.py \
            --experiment A1 \
            --model-name "$model" \
            --variant it \
            --dataset "$DATASET" \
            --n-eval-examples "$N_EVAL" \
            --device "cuda:${gpu}" \
            --worker-index "$wi" \
            --n-workers "$N_WORKERS" \
            --run-name "$run_name" \
            --corrective-direction-path "$corr_dir" \
            --output-base "$out_base" \
            --proposal-boundary "$pb" \
            --n-layers "$nl" \
            --max-gen-tokens "$mg" \
            > "$LOG_DIR/${model}_A1_w${wi}.log" 2>&1 &
    done
    wait

    # Merge
    local src_dirs=()
    for ((wi=0; wi<N_WORKERS; wi++)); do
        src_dirs+=("${out_base}/${run_name}_w${wi}")
    done
    uv run python scripts/merge_exp6_workers.py \
        --experiment A1 --variant it --n-workers "$N_WORKERS" \
        --merged-name "merged_${run_name}" \
        --source-dirs "${src_dirs[@]}" \
        > "$LOG_DIR/${model}_merge.log" 2>&1
}

judge_model() {
    local model=$1
    local run_name="A1_${model}_it_v1"
    local merged="results/cross_model/${model}/exp6/merged_${run_name}"

    echo "=== LLM Judge: $model ==="
    uv run python scripts/llm_judge_exp6.py \
        --merged-dir "$merged" \
        --model "$JUDGE_MODEL" --workers 16 --tasks g1 g2 s1 s2 \
        > "$LOG_DIR/${model}_judge.log" 2>&1
}

# Dispatch based on --step
# ... (parse args, loop over models, assign GPUs)
```

---

## 9. Validation Checklist

Before declaring Phase 0 complete, verify:

- [ ] **Direction quality**: Bootstrap cosine > 0.90 for each model's corrective direction
- [ ] **Baseline sanity**: α=1.0 baseline metrics match unsteered model within ±2%
- [ ] **Format dose-response**: STR shows monotonic decrease as α→0 for ≥4/6 models
- [ ] **Content preservation**: reasoning_em stable within ±5% for α∈[-1, 5] for ≥4/6 models
- [ ] **Random control**: Random direction controls show no systematic format change
- [ ] **Qwen null**: Qwen shows attenuated or absent format dose-response (consistent with near-zero commitment delay)
- [ ] **Effect sizes**: Cohen's d > 0.2 for format metrics at α=0 for ≥4/6 models
- [ ] **Monotonicity**: Spearman |ρ| > 0.7 for STR vs α for ≥4/6 models
- [ ] **LLM judge agreement**: G2 (instruction following) tracks STR across α for ≥4/6 models
- [ ] **All plots generated**: §5.1–5.7 all rendered and reviewed

---

## 10. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Direction instability for MoE (DeepSeek) | Medium | Low (one model) | Bootstrap check first; if unstable, report as limitation |
| Weak dose-response for Llama (small commitment delay) | Medium | Medium | May need to test stronger α values or different proposal_boundary |
| OOM on 4096-dim models (Llama, Mistral, OLMo) | Low | Medium | Use batch_size=4 instead of 8; verify GPU memory before full run |
| Qwen shows unexpected dose-response | Low | High (narrative change) | Would require revising the "corrective stage absent" interpretation |
| LLM judge API rate limits | Medium | Low (delays only) | Run judge with backoff; it's post-hoc and can be retried |
| Total GPU time exceeds estimate | Medium | Low | Pipeline supports resume — just rerun with same args |

---

## 11. File Tree After Phase 0

```
results/cross_model/
├── gemma3_4b/           # Already exists (observational data)
│   ├── directions/      # Copy from results/exp5/precompute_v2/precompute/
│   │   └── corrective_directions.npz
│   ├── exp6/            # NEW: rerun without chat template
│   │   ├── A1_gemma3_4b_it_notmpl_v1_w0/
│   │   ├── A1_gemma3_4b_it_notmpl_v1_w1/
│   │   └── merged_A1_gemma3_4b_it_notmpl_v1/
│   │       ├── scores.jsonl
│   │       ├── sample_outputs.jsonl
│   │       ├── llm_judge_v2_scores.jsonl
│   │       └── plots/
│   ├── bootstrap_0A.json
│   └── validation_3pt.json   # sanity check results
├── llama31_8b/
│   ├── directions/
│   │   ├── corrective_directions.npz
│   │   └── corrective_directions.meta.json
│   ├── exp6/
│   │   ├── A1_llama31_8b_it_notmpl_v1_w0/
│   │   ├── A1_llama31_8b_it_notmpl_v1_w1/
│   │   └── merged_A1_llama31_8b_it_notmpl_v1/
│   │       ├── scores.jsonl
│   │       ├── sample_outputs.jsonl
│   │       ├── llm_judge_v2_scores.jsonl
│   │       └── plots/
│   ├── bootstrap_0A.json
│   └── validation_3pt.json
├── qwen3_4b/            # Same structure
├── mistral_7b/          # Same structure
├── deepseek_v2_lite/    # Same structure
└── olmo2_7b/            # Same structure

results/exp7/plots/
├── phase0_multimodel_dose_response.png     # §5.1 — THE CENTERPIECE
├── phase0_format_content_dissociation.png  # §5.2
├── phase0_random_control.png               # §5.3
├── phase0_direction_stability.png          # §5.4
├── phase0_effect_sizes.png                 # §5.5
└── data/
    ├── phase0_all_scores.json              # Combined scores across models
    ├── phase0_effect_sizes.json            # Cohen's d per model per metric
    └── phase0_monotonicity.json            # Spearman correlations
```

---

## 12. Dependencies & Prerequisites

1. **GPU server access**: 8×A100 80GB (already available)
2. **HuggingFace model access**: All 12 model variants (6 PT + 6 IT) must be downloadable. Verify `huggingface-cli whoami` and model access for:
   - `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct` (requires Meta license)
   - `Qwen/Qwen3-4B-Base` and `Qwen/Qwen3-4B` (open)
   - `mistralai/Mistral-7B-v0.3` and `mistralai/Mistral-7B-Instruct-v0.3` (open)
   - `deepseek-ai/DeepSeek-V2-Lite` and `deepseek-ai/DeepSeek-V2-Lite-Chat` (open)
   - `allenai/OLMo-2-1124-7B` and `allenai/OLMo-2-1124-7B-Instruct` (open)
3. **OpenRouter API key**: For LLM judge (G1, G2, S1, S2). Set `OPENROUTER_API_KEY` in `.env`.
4. **Disk space**: ~50 GB for all model directions + steering results (models downloaded to HF cache, not counted)
5. **Existing Gemma directions**: `results/exp5/precompute_v2/precompute/corrective_directions.npz` — copy to `results/cross_model/gemma3_4b/directions/`. Gemma directions were already extracted without chat template, so no re-extraction needed. Only the A1 steering evaluation needs to be rerun (without chat template).
6. **Existing Gemma A1_notmpl results**: `results/exp6/merged_A1_notmpl_it_v1/` — can be used as reference to validate that the new no-template Gemma rerun produces consistent results
