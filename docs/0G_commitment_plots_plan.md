# 0G Tuned Lens Commitment — Plot Plan

## Data Available

Per-prompt JSONL with ~2936 prompts × 6 models × PT/IT. Each prompt has per-generation-step commitment layers from **7 method families**:

### 1. Top-1 Commitment (Raw Logit Lens)
- `commitment_layer_raw` — first layer where raw top-1 matches final AND stays matched (no flip-back)
- `commitment_layer_raw_top1_qual_top3` — qualified: top-1 matches AND final token stays in top-3 at all subsequent layers
- `commitment_layer_raw_top1_qual_top5` — same but top-5

### 2. Top-1 Commitment (Tuned Lens)
- `commitment_layer_top1_tuned` — same as raw but using tuned lens predictions
- `commitment_layer_tuned_top1_qual_top3` — qualified top-3
- `commitment_layer_tuned_top1_qual_top5` — qualified top-5

### 3. KL Commitment (Raw Logit Lens)
- `commitment_layer_raw_kl_{τ}` for τ ∈ {0.05, 0.1, 0.2, 0.5, 1.0} — first layer where KL(raw_ℓ ‖ final) < τ nats AND stays below
- `commitment_layer_raw_kl_qual_{τ}_{Kx}` for K ∈ {3, 5} — qualified: KL stays below for K consecutive subsequent layers (reduces single-layer noise)

### 4. KL Commitment (Tuned Lens)
- `commitment_layer_tuned_{τ}` — same thresholds
- `commitment_layer_tuned_kl_qual_{τ}_{Kx}` — qualified versions

### 5. Majority Commitment
- `commitment_layer_majority_{τ}` — first layer where ≥90% of all subsequent layers have KL < τ (robust to occasional spikes)

### 6. Cosine Commitment
- `commitment_layer_cosine_{θ}` for θ ∈ {0.8, 0.9, 0.95, 0.99} — first layer where cos(h_ℓ, h_final) > θ AND stays above (residual-stream-based, independent of logit lens)

### 7. Entropy Commitment
- `commitment_layer_entropy_{τ}` for τ ∈ {0.05, 0.1, 0.2, 0.5, 1.0} — first layer where entropy difference |H_ℓ - H_final| < τ

---

## Plots to Generate

### Core Plots (paper main text)

**Plot 1: `0G_commitment_top1.png`** (existing)
- 6 panels (one per model)
- Histogram: PT raw (blue), IT raw (dark blue), PT tuned (orange), IT tuned (red)
- X-axis: commitment layer, Y-axis: density
- Vertical lines: corrective onset, phase boundary
- **Key signal**: IT distributions shifted right (later commitment)

**Plot 2: `0G_commitment_kl.png`** (existing)
- Same layout as Plot 1 but using KL commitment (τ=0.1)
- KL-based is smoother than top-1 (continuous vs discrete metric)

**Plot 3: `0G_commitment_summary.png`** (existing)
- Bar chart: 6 models × 4 methods (raw top-1, tuned top-1, raw KL, tuned KL)
- Y-axis: commitment delay (IT median − PT median, in layers)
- Error bars: bootstrap 95% CI
- **Key signal**: positive bars = IT commits later across all methods

**Plot 4: `0G_commitment_scatter.png`** (existing)
- 6 panels, scatter of raw vs tuned commitment per step
- Shows correlation and whether tuned lens shifts commitment earlier

### Extended Plots (paper appendix / supplementary)

**Plot 5: `0G_commitment_kl_threshold_sensitivity.png`** (NEW)
- For each model: line plot of median commitment layer vs KL threshold τ
- 2 lines per model: PT and IT
- Shows robustness: if IT-PT gap persists across thresholds, the finding is robust

**Plot 6: `0G_commitment_qualified.png`** (NEW)
- Same as Plot 2 but using qualified commitment (3x or 5x variants)
- Qualified = must stay committed for K consecutive layers (reduces false positives)
- If pattern holds with qualified commitment, it's not single-layer noise

**Plot 7: `0G_commitment_cosine.png`** (NEW)
- Commitment based on cosine similarity of residual stream to final layer
- Independent of logit lens entirely — purely representational
- θ=0.95 is the primary threshold
- Important because it validates commitment using a different methodology

**Plot 8: `0G_commitment_entropy.png`** (NEW)
- When does logit-lens entropy stabilize?
- IT should stabilize later (higher entropy in corrective region)

**Plot 9: `0G_commitment_cdf.png`** (NEW)
- For each model: CDF of commitment layer (fraction committed by depth)
- X-axis: normalized depth (0-1), Y-axis: cumulative fraction committed
- 2 curves per model: PT and IT
- IT curve should be shifted right (later commitment = rightward CDF)
- Cross-model overlay on normalized depth for the key comparison

**Plot 10: `0G_commitment_normalized_summary.png`** (NEW)
- Same as Plot 3 but commitment expressed as fraction of total depth (0.0-1.0)
- Enables cross-model comparison despite different layer counts (27-36)
- Key figure for the paper's cross-model claim

---

## Training Setup (for paper methods section)

- **Algorithm**: Tuned lens (Belrose et al. 2023)
- **Probe**: Per-layer affine transform T_ℓ(h) = Wh + b, identity + zero-bias init
- **Optimizer**: SGD + Nesterov momentum
  - lr = 0.1 (passed to PyTorch; conceptual lr = 1.0, adjusted by (1-momentum) per Belrose codebase)
  - momentum = 0.9
  - weight_decay = 0 (direct parameterization, not residual)
  - gradient clipping: norm 1.0
- **Training**: 250 steps × 262,144 tokens/step = 65.5M token-activations
- **Data**: C4 validation split (allenai/c4, streaming), ~70M unique tokens, 80/20 train/val
- **Sequence length**: 2048 tokens
- **LR schedule**: Linear decay from lr to 0 over 250 steps (no warmup)
- **Loss**: KL(model_final ‖ tuned_ℓ), sum reduction / batch_size
- **Evaluation**: Commitment computed on exp3_dataset.jsonl (2936 prompts), greedy generation up to 512 tokens

### Known issue: Gemma 3 4B
Gemma 3 uses `Gemma3TextScaledWordEmbedding` which scales embeddings by sqrt(2560) ≈ 50.6×.
With lr=0.1 (our current setup) or even lr=1.0, the effective gradient magnitude is ~2560× larger than for standard transformers. Probes diverge from identity instead of converging.
**Fix**: Retrain with lr ≈ 0.001 (or normalize activations before probe).
**Current status**: Gemma 3 tuned lens is broken; raw logit lens results are valid.
