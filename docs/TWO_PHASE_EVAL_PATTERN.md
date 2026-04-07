# Two-Phase Eval Pattern: Generate → Prefill

## Problem
Autoregressive generation with per-layer hooks is slow (~40-50% GPU utilization) because each generated token triggers n_layers small sequential computations. The GPU sits idle between tiny matmuls.

## Solution
Split eval into two phases:
1. **Generate** text without hooks (maximum GPU throughput)
2. **Prefill** the full sequence (prompt + generated) with hooks to collect all metrics at once

## Why This Works
- Generation is **greedy deterministic** (`do_sample=False`), so generated text is identical with or without hooks
- **Causal attention mask** ensures position t sees the same context in both autoregressive and prefill modes
- Therefore `h_ℓ[t]` in prefill == `h_ℓ[t]` in autoregressive for all layers ℓ and positions t
- All metrics (KL, entropy, cosine, commitment) computed from hidden states are identical

## Speedup Sources
1. **Phase 1 (no hooks)**: ~2-3x — model.generate() runs at full KV-cache speed
2. **Phase 2 (prefill)**: ~3-5x — single forward pass processes all positions in parallel
3. **Batched metrics**: ~2-3x — large matmuls (n_layers × chunk × d_model) @ W_U are more efficient than per-token vector-matrix products
4. **Combined**: ~10-16x total

## Memory Management
The intermediate logit tensor `[n_layers × chunk_steps, vocab_size]` can be large. Process in STEP_CHUNK=64 chunks:
- 36 layers × 64 steps × 256K vocab × 4 bytes = 2.4 GB — fits alongside model weights

## Usage

### CLI
```bash
# Standard (slow, original)
uv run python -m src.poc.cross_model.tuned_lens --eval-only --model llama31_8b --variant it

# Fast 2-phase (same output, ~10x faster)
uv run python -m src.poc.cross_model.tuned_lens --eval-only --fast --model llama31_8b --variant it
```

### Python API
```python
from src.poc.cross_model.tuned_lens import eval_commitment_fast

# Drop-in replacement for eval_commitment()
summary = eval_commitment_fast(
    model, tokenizer, adapter, spec, probes, records, device,
    output_path=out_path,
    variant=variant,
    max_new_tokens=512,
    collect_full=True,
    apply_chat_template=True,
)
```

## Reusable Pattern for Other Experiments

Any experiment doing autoregressive generation + per-layer analysis can use this pattern:

```python
# Phase 1: Generate (fast, no hooks)
out_ids = model.generate(input_ids, do_sample=False, max_new_tokens=512, ...)
prompt_len = input_ids.shape[1]

# Phase 2: Prefill with hooks (all positions at once)
captured = {}
def make_hook(layer_idx):
    def hook(module, inp, output):
        h = adapter.residual_from_output(output)
        captured[layer_idx] = h[0, prompt_len:, :].detach()  # only generated positions
    return hook

handles = [layer_modules[i].register_forward_hook(make_hook(i)) for i in range(n_layers)]
try:
    model(out_ids)  # single forward pass
finally:
    for h in handles:
        h.remove()

# Phase 3: Compute metrics from captured tensors
all_h = torch.stack([captured[i] for i in range(n_layers)])  # [n_layers, n_steps, d_model]
# ... batched computation on all_h
```

## Requirements
- **Deterministic generation**: `do_sample=False` (greedy)
- **Read-only hooks**: hooks must NOT modify the forward pass (no interventions)
- **Causal attention**: model must use causal (autoregressive) attention mask

## Does NOT Work With
- Stochastic sampling (`do_sample=True`) — generated text would differ
- Intervention hooks (steering) — output depends on hooks being active during generation
- Non-causal attention models — prefill hidden states would differ from autoregressive

## Applicable Experiments
- `tuned_lens.py` eval_commitment — implemented as `eval_commitment_fast()`
- `collect_L1L2.py` — delta-cosine + logit-lens commitment (can be adapted)
- `collect_L8.py` — intrinsic dimensionality (can be adapted)
- `collect_L9.py` — attention entropy (can be adapted)

## Not Applicable
- `exp6/run.py` — steering interventions modify the forward pass
- `precompute_directions_multimodel.py` — needs hooks active during generation for direction extraction
