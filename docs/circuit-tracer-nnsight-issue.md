# Circuit-Tracer: nnsight Envoy Bug — Issue & Proposed Fix

## Summary

When using `NNSightReplacementModel` (the only supported backend for Gemma 3),
accessing `model.transcoders[layer]` does **not** return the `SingleLayerTranscoder`
for that layer. It returns the entire internal `ModuleList`, making `W_dec`
inaccessible for logit-lens analysis and similar post-hoc feature inspection.

**Versions tested:** `circuit-tracer==0.1.0`, `nnsight==0.6.2`, `torch>=2.0`

---

## The Bug

### What users expect

```python
model = ReplacementModel.from_pretrained_and_transcoders(
    model_name="google/gemma-3-4b-pt",
    transcoders=transcoder_set,
    backend="nnsight",
    ...
)

# Reasonable expectation: get the layer 5 transcoder
tc = model.transcoders[5]
logit_vec = tc.W_dec[feat_idx] @ W_U   # logit-lens projection
```

### What actually happens

```
AttributeError: ModuleList(
  (0): SingleLayerTranscoder(...)
  (1): SingleLayerTranscoder(...)
  ...
  (33): SingleLayerTranscoder(...)
) has no attribute W_dec
```

`model.transcoders` is wrapped by nnsight as an `Envoy` proxy.
Indexing the `Envoy` with an integer does not call
`TranscoderSet.__getitem__`; instead nnsight's proxy system returns
the `TranscoderSet`'s internal child module (the `nn.ModuleList`
named `transcoders`), ignoring the index entirely. The `ModuleList`
has no `W_dec`; only individual `SingleLayerTranscoder` elements do.

### Root cause

`NNSightReplacementModel` stores `self.transcoders = transcoder_set`
where `transcoder_set` is a `TranscoderSet(nn.Module)`.
nnsight's `LanguageModel` base wraps all submodule attribute access
as `Envoy` objects for intervention tracing. `Envoy.__getitem__`
does not delegate to `TranscoderSet.__getitem__`.

Circuit-tracer's **own internal code** already works around this:

```python
# replacement_model_nnsight.py line 267
transcoders = (
    self.transcoders._module if isinstance(self.transcoders, Envoy)
    else self.transcoders
)
# line 728
decoder_vectors = self.transcoders._module._get_decoder_vectors(...)
```

But there is no **public API** exposing this capability to users.

---

## Workaround (what we do in this project)

Extract `SingleLayerTranscoder` references **before** passing the
`TranscoderSet` into `ReplacementModel`. At that point the objects are
plain Python — no nnsight wrapping yet.

```python
transcoder_set = build_transcoder_set(...)

# Extract NOW, before nnsight sees the TranscoderSet
transcoder_list = [transcoder_set[i] for i in range(len(transcoder_set))]

model = ReplacementModel.from_pretrained_and_transcoders(
    transcoders=transcoder_set, ...
)

# Later — safe to use, bypasses nnsight entirely
W_dec = transcoder_list[layer].W_dec[feat_indices]
```

This works because `SingleLayerTranscoder` objects are mutated in-place
when `ReplacementModel` calls `.to(device, dtype)` inside
`_configure_replacement_model`, so the extracted references stay valid
and on the correct device.

---

## Proposed Fix for circuit-tracer

Add a single public method to `NNSightReplacementModel` that handles
the Envoy unwrapping internally:

```python
# In NNSightReplacementModel (replacement_model_nnsight.py)

def get_transcoder(self, layer: int) -> "SingleLayerTranscoder":
    """Return the SingleLayerTranscoder for the given layer.

    Handles nnsight Envoy wrapping — do NOT use model.transcoders[layer]
    directly, as the Envoy proxy returns the internal ModuleList instead
    of a single transcoder.
    """
    from nnsight import Envoy
    tc_set = self.transcoders._module if isinstance(self.transcoders, Envoy) else self.transcoders
    return tc_set[layer]
```

**This is a 7-line addition with no breaking changes.**

Users could then do:

```python
tc = model.get_transcoder(layer=5)
logit_vec = tc.W_dec[feat_idx] @ W_U
```

---

## Steps to Reproduce

```python
import torch
from circuit_tracer import ReplacementModel
from circuit_tracer.transcoder.single_layer_transcoder import load_gemma_scope_2_transcoder, TranscoderSet
from huggingface_hub import snapshot_download
from pathlib import Path

variant = "width_16k_l0_big_affine"
local_dir = snapshot_download(
    "google/gemma-scope-2-4b-pt",
    allow_patterns=[f"transcoder_all/layer_*_{variant}/params.safetensors"]
)

transcoders = {}
for layer in range(34):
    path = Path(local_dir) / "transcoder_all" / f"layer_{layer}_{variant}" / "params.safetensors"
    transcoders[layer] = load_gemma_scope_2_transcoder(str(path), layer=layer, device="cpu")

tc_set = TranscoderSet(transcoders, feature_input_hook="mlp.hook_in", feature_output_hook="hook_mlp_out")

model = ReplacementModel.from_pretrained_and_transcoders(
    model_name="google/gemma-3-4b-pt",
    transcoders=tc_set,
    backend="nnsight",
    device="cpu",
)

# BUG: this returns ModuleList, not SingleLayerTranscoder
tc = model.transcoders[0]
print(type(tc))  # <class 'torch.nn.modules.container.ModuleList'>
tc.W_dec         # AttributeError: ModuleList has no attribute W_dec
```

---

## Recommended Community Action

### 1. File a GitHub issue on `ai-safety-foundation-models/circuit-tracer`

**Title:** `model.transcoders[layer]` returns `ModuleList` instead of
`SingleLayerTranscoder` when using nnsight backend

**Labels:** `bug`, `nnsight`

**Body:** Use the "Steps to Reproduce" section above. Include:
- Circuit-tracer version: 0.1.0
- nnsight version: 0.6.2
- Model: google/gemma-3-4b-pt (requires nnsight backend)
- Any model requiring the nnsight backend is affected

### 2. Submit a PR with `get_transcoder()`

The fix is small and non-breaking. Target file:
`circuit_tracer/replacement_model/replacement_model_nnsight.py`

Add the `get_transcoder` method shown above. Also add a matching
`get_transcoder` stub in the base `ReplacementModel` class that raises
`NotImplementedError` or delegates, so the API is consistent.

**Why this matters for the community:**
- Logit-lens analysis (projecting `W_dec` through `W_U`) is one of the
  most common post-hoc interpretability operations on transcoders.
- Every user attempting this with the nnsight backend hits this bug.
- The fix is 7 lines; the workaround requires understanding nnsight internals.
- The GemmaScope 2 paper and circuit-tracer tutorials don't document
  how to access `W_dec` after building a `ReplacementModel`.

### 3. (Lower priority) Update documentation / README

Add a short section "Accessing individual layer transcoders" to the
circuit-tracer README showing both the broken pattern and the fix.

---

## What we do in this project

See `src/poc/model.py` (`LoadedModel.transcoder_list`) and
`src/poc/attribution.py` (`_extract_records`).

Run the regression test to verify the fix holds:
```
uv run python -m src.poc.test_transcoder_access
```
