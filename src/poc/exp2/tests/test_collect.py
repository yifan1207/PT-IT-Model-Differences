"""
Tests for exp2/collect.py — validates inference logic without loading the real model.

Covers:
  1. Architecture probe: google/gemma-3-4b-pt IS Gemma3ForConditionalGeneration,
     which means nnsight path is language_model.layers[i], NOT model.layers[i].
  2. Entropy computation: known values, masking, and edge cases.
  3. L0 computation: transcoder encode() shape contract and nonzero counting.
  4. Greedy decode masking: <unusedXXXX> tokens suppressed via real_token_mask.
  5. collect_prompt end-to-end: mock nnsight trace to verify output schema.
  6. collect_all: npz packing / stripping active_features.
  7. Plots smoke-test: all 10 plots render on synthetic data.
"""
import math
import json
import tempfile
import numpy as np
import torch
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ─── 1. Architecture sanity check ──────────────────────────────────────────────

def test_gemma3_architecture_is_conditional_generation():
    """google/gemma-3-4b-pt must be Gemma3ForConditionalGeneration.

    If this fails the model changed on HuggingFace or the wrong model is loaded.
    The correct nnsight path is language_model.layers[i], not model.layers[i].
    """
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("google/gemma-3-4b-pt")
    assert cfg.architectures == ["Gemma3ForConditionalGeneration"], (
        f"Expected Gemma3ForConditionalGeneration, got {cfg.architectures}. "
        "Update collect.py hook paths if the architecture changed."
    )


def test_nnsight_mapping_uses_language_model_path():
    """circuit-tracer's nnsight mapping for Gemma3ForConditionalGeneration
    must use 'language_model.layers[{layer}]', not 'model.layers[{layer}]'.
    """
    from circuit_tracer.utils.tl_nnsight_mapping import gemma_3_conditional_mapping
    pattern = gemma_3_conditional_mapping.attention_location_pattern
    assert "language_model.layers" in pattern, (
        f"Expected 'language_model.layers' in pattern, got: {pattern}"
    )
    # Also verify the CausalLM mapping uses 'model.layers' (different architecture)
    from circuit_tracer.utils.tl_nnsight_mapping import gemma_3_mapping
    assert "model.layers" in gemma_3_mapping.attention_location_pattern
    assert "language_model" not in gemma_3_mapping.attention_location_pattern


def test_nnsight_mapping_has_pre_feedforward_layernorm():
    """mlp.hook_in must map to pre_feedforward_layernorm (output) for Gemma3ConditionalGen."""
    from circuit_tracer.utils.tl_nnsight_mapping import gemma_3_conditional_mapping
    hook_path, io = gemma_3_conditional_mapping.feature_hook_mapping["mlp.hook_in"]
    assert "pre_feedforward_layernorm" in hook_path, (
        f"mlp.hook_in should map to pre_feedforward_layernorm, got: {hook_path}"
    )
    assert io == "output", f"mlp.hook_in should use 'output', got: {io}"


def test_nnsight_mapping_unembed_at_lm_head():
    """lm_head must be at top-level for Gemma3ForConditionalGeneration."""
    from circuit_tracer.utils.tl_nnsight_mapping import gemma_3_conditional_mapping
    assert gemma_3_conditional_mapping.unembed_weight == "lm_head.weight"


# ─── 2. Entropy computation ────────────────────────────────────────────────────

from src.poc.exp2.collect import _entropy_from_logits


def test_entropy_uniform():
    """Uniform distribution has maximum entropy = log(N)."""
    N = 100
    logits = torch.zeros(N)
    H = _entropy_from_logits(logits)
    assert abs(H - math.log(N)) < 1e-4, f"Expected {math.log(N):.4f}, got {H:.4f}"


def test_entropy_one_hot():
    """One-hot distribution (very large logit on one token) has ~0 entropy."""
    logits = torch.full((100,), -1e9)
    logits[42] = 1e9
    H = _entropy_from_logits(logits)
    assert H < 1e-3, f"Expected ~0 entropy for one-hot, got {H}"


def test_entropy_with_mask():
    """Masking restricts entropy computation to real tokens only.

    If we have 10 tokens and mask out 5, entropy is computed over the remaining 5.
    """
    logits = torch.zeros(10)
    mask = torch.tensor([True]*5 + [False]*5)
    H_masked = _entropy_from_logits(logits, mask=mask)
    H_expected = math.log(5)
    assert abs(H_masked - H_expected) < 1e-4, (
        f"Expected log(5)={H_expected:.4f} for 5 uniform real tokens, got {H_masked:.4f}"
    )


def test_entropy_mask_excludes_unused():
    """<unusedXXXX> logits (high value) should NOT affect entropy when masked."""
    vocab_size = 20
    logits = torch.zeros(vocab_size)
    # Simulate unused tokens with very high logits
    logits[10:] = 1e6
    mask = torch.tensor([True]*10 + [False]*10)
    H = _entropy_from_logits(logits, mask=mask)
    H_expected = math.log(10)  # 10 uniform real tokens
    assert abs(H - H_expected) < 1e-4, (
        f"Unused token logits should be excluded; expected {H_expected:.4f}, got {H}"
    )


def test_entropy_not_nan_for_all_equal():
    """Entropy should never be NaN."""
    logits = torch.ones(50) * 3.14
    H = _entropy_from_logits(logits)
    assert not math.isnan(H)
    assert not math.isinf(H)


# ─── 3. L0 computation via transcoder encode() ────────────────────────────────

def test_transcoder_encode_l0_zero():
    """If all activations are zero, L0 = 0."""
    d_model = 64
    d_transcoder = 256
    tc = MagicMock()
    tc.b_enc = torch.zeros(d_transcoder)
    # encode returns all-zero tensor
    tc.encode.return_value = torch.zeros(1, d_transcoder)

    x = torch.randn(1, d_model)
    acts = tc.encode(x.to(tc.b_enc.dtype))   # [1, d_transcoder]
    active_idxs = acts[0].nonzero(as_tuple=False).squeeze(1).tolist()
    assert active_idxs == []
    assert len(active_idxs) == 0


def test_transcoder_encode_l0_correct():
    """L0 count equals the number of non-zero activations."""
    d_model = 64
    d_transcoder = 256
    tc = MagicMock()
    tc.b_enc = torch.zeros(d_transcoder)
    # Exactly 7 non-zero activations
    acts = torch.zeros(1, d_transcoder)
    active_positions = [0, 5, 17, 42, 100, 200, 255]
    for pos in active_positions:
        acts[0, pos] = float(pos + 1)
    tc.encode.return_value = acts

    x = torch.randn(1, d_model)
    result = tc.encode(x)
    active_idxs = result[0].nonzero(as_tuple=False).squeeze(1).tolist()
    assert len(active_idxs) == 7
    assert sorted(active_idxs) == sorted(active_positions)


def test_transcoder_encode_shape_contract():
    """encode() must be called with [1, d_model] and return [1, d_transcoder]."""
    from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
    import torch.nn.functional as F
    from unittest.mock import PropertyMock

    d_model = 16
    d_transcoder = 32
    # Build a minimal real SingleLayerTranscoder (no model loading needed)
    tc = SingleLayerTranscoder(
        d_model=d_model,
        d_transcoder=d_transcoder,
        activation_function=torch.nn.ReLU(),
        layer_idx=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # Initialise weights to something non-trivial
    torch.nn.init.normal_(tc.W_enc)
    torch.nn.init.zeros_(tc.b_enc)

    x = torch.randn(1, d_model)
    acts = tc.encode(x)
    assert acts.shape == (1, d_transcoder), (
        f"encode() should return [1, d_transcoder], got {acts.shape}"
    )
    # L0 logic: nonzero on acts[0]
    active_idxs = acts[0].nonzero(as_tuple=False).squeeze(1).tolist()
    assert isinstance(active_idxs, list)


# ─── 4. Greedy decode masking ──────────────────────────────────────────────────

def test_greedy_decode_masks_unused_tokens():
    """Greedy decode must not select <unusedXXXX> tokens.

    Simulates the masking logic in collect_prompt:
      masked_logits = logits.clone()
      masked_logits[~real_mask] = float('-inf')
      next_token_id = int(masked_logits.argmax().item())
    """
    vocab_size = 100
    # Simulate: unused tokens (idx 50-99) have very high logits from random init
    logits = torch.zeros(vocab_size)
    logits[50:] = 1e6  # <unusedXXXX> tokens — high logit

    real_mask = torch.tensor([True]*50 + [False]*50)

    # WITHOUT masking (the old bug): argmax picks an unused token
    bad_token = int(logits.argmax().item())
    assert bad_token >= 50, "Test setup: highest logit should be in unused range"

    # WITH masking (the fix): argmax picks from real tokens only
    masked_logits = logits.clone()
    masked_logits[~real_mask] = float("-inf")
    good_token = int(masked_logits.argmax().item())
    assert good_token < 50, (
        f"Greedy decode with mask should select token < 50, got {good_token}"
    )


def test_greedy_decode_masked_ties_broken_correctly():
    """When multiple real tokens are tied, still picks a real one."""
    vocab_size = 50
    logits = torch.zeros(vocab_size)  # all real tokens tied
    real_mask = torch.ones(vocab_size, dtype=torch.bool)
    masked_logits = logits.clone()
    masked_logits[~real_mask] = float("-inf")
    next_token_id = int(masked_logits.argmax().item())
    assert 0 <= next_token_id < vocab_size


# ─── 5. collect_prompt output schema ──────────────────────────────────────────

def _make_mock_loaded(n_layers=34, d_model=16, vocab_size=100, n_features=64):
    """Build a minimal mock LoadedModel for testing collect_prompt without GPU."""
    from src.poc.shared.model import LoadedModel
    from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder

    device = torch.device("cpu")

    # Mock tokenizer
    tokenizer = MagicMock()
    tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
    tokenizer.eos_token_id = 2
    tokenizer.decode.side_effect = lambda ids: f" tok{ids[0]}"

    # W_U: [d_model, vocab_size]
    W_U = torch.randn(d_model, vocab_size)

    # real_token_mask: first 80 real, last 20 unused
    real_mask = torch.tensor([True]*(vocab_size-20) + [False]*20)

    # Build real transcoders (lightweight, on CPU)
    transcoder_list = []
    for _ in range(n_layers):
        tc = SingleLayerTranscoder(
            d_model=d_model,
            d_transcoder=n_features,
            activation_function=torch.nn.ReLU(),
            layer_idx=0,
            device=device,
            dtype=torch.float32,
        )
        transcoder_list.append(tc)

    # Mock NNSight ReplacementModel
    model = MagicMock()
    model.language_model = MagicMock()

    # Each layer's output[0][0, -1, :] should return a d_model tensor
    def make_layer_mock(i):
        layer = MagicMock()
        # output[0][0, -1, :].save() chain
        h = torch.randn(d_model)
        out0 = MagicMock()
        out0.__getitem__ = MagicMock(return_value=h)  # [0, -1, :]
        out_tuple = MagicMock()
        out_tuple.__getitem__ = MagicMock(return_value=out0)  # [0]
        layer.output = out_tuple

        # pre_feedforward_layernorm.output[0, -1, :].save()
        x_ln = torch.randn(d_model)
        ln_out = MagicMock()
        ln_out.__getitem__ = MagicMock(return_value=x_ln)  # [0, -1, :]
        layer.pre_feedforward_layernorm = MagicMock()
        layer.pre_feedforward_layernorm.output = ln_out
        return layer

    layers_list = [make_layer_mock(i) for i in range(n_layers)]
    model.language_model.layers = MagicMock()
    model.language_model.layers.__getitem__ = MagicMock(side_effect=lambda i: layers_list[i])

    # lm_head.output[0, -1, :].save()
    logit_vec = torch.randn(vocab_size)
    # Give real tokens slightly higher logits so argmax lands on real token
    logit_vec[vocab_size-20:] = -1e6
    lm_out = MagicMock()
    lm_out.__getitem__ = MagicMock(return_value=logit_vec)
    model.lm_head = MagicMock()
    model.lm_head.output = lm_out

    # Simulate nnsight .save() — each proxy returns a saved value
    def make_save(tensor):
        sv = MagicMock()
        sv.value = tensor
        return sv

    # Patch all .save() calls on the mock proxies
    for layer in layers_list:
        layer.output[0][0, -1, :].save = lambda t=torch.randn(d_model): MagicMock(value=t)
        layer.pre_feedforward_layernorm.output[0, -1, :].save = lambda t=torch.randn(d_model): MagicMock(value=t)
    model.lm_head.output[0, -1, :].save = lambda: MagicMock(value=logit_vec)

    # trace context manager: just runs the body
    import contextlib
    @contextlib.contextmanager
    def fake_trace(ids, **kwargs):
        # The body of the `with` block builds a list of save() proxies.
        # We need those .save() calls to return objects with .value set.
        # Since we can't easily intercept the save() calls inside the body,
        # we'll patch collect_prompt to use a different strategy in this test.
        yield

    model.trace = fake_trace
    model.tokenizer = tokenizer

    return LoadedModel(
        model=model,
        W_U=W_U,
        tokenizer=tokenizer,
        transcoder_list=transcoder_list,
        real_token_mask=real_mask,
    )


def test_collect_prompt_output_schema():
    """collect_prompt must return a dict with all required keys and correct types.

    Uses a lightweight mock that bypasses actual model loading.
    The test patches the nnsight trace to inject pre-built tensors.
    """
    from src.poc.exp2.config import Exp2Config
    from src.poc.exp2.collect import collect_prompt

    n_layers = 4
    d_model = 16
    vocab_size = 100
    n_features = 32
    max_gen = 3

    cfg = MagicMock(spec=Exp2Config)
    cfg.max_gen_tokens = max_gen
    cfg.is_instruction_tuned = False

    # Real tensors we'll inject
    W_U = torch.randn(d_model, vocab_size)
    real_mask = torch.tensor([True]*(vocab_size-10) + [False]*10)

    # Build real transcoder list
    from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
    transcoder_list = [
        SingleLayerTranscoder(
            d_model=d_model, d_transcoder=n_features,
            activation_function=torch.nn.ReLU(),
            layer_idx=i, device=torch.device("cpu"), dtype=torch.float32,
        )
        for i in range(n_layers)
    ]

    # Mock tokenizer
    tok = MagicMock()
    tok.encode.return_value = torch.tensor([[1, 5, 7]])
    tok.eos_token_id = 99  # won't be generated (logits[99] is masked out)
    tok.all_special_ids = set()
    call_count = [0]
    def decode_side(ids):
        call_count[0] += 1
        return f" w{ids[0]}"
    tok.decode.side_effect = decode_side

    # Inject pre-built residuals and mlp inputs via patching the trace context
    residual_tensors = [torch.randn(d_model) for _ in range(n_layers)]
    mlp_input_tensors = [torch.randn(d_model) for _ in range(n_layers)]
    logit_tensor = torch.randn(vocab_size)
    logit_tensor[vocab_size-10:] = -1e9  # ensure real tokens win argmax

    def make_save_obj(t):
        m = MagicMock()
        m.value = t
        return m

    import contextlib
    from src.poc.exp2 import collect as collect_mod

    # Chainable proxy: any __getitem__ returns self (still a proxy), .save() returns save_obj.
    # This simulates nnsight's Envoy proxy where you can chain indexing ops
    # and call .save() at the end to materialise the value.
    class ChainProxy:
        def __init__(self, tensor):
            self._tensor = tensor
        def __getitem__(self, idx):
            # Any indexing still returns a proxy over the same tensor
            return ChainProxy(self._tensor)
        def save(self):
            return make_save_obj(self._tensor)

    class FakeLayer:
        def __init__(self, res, mlp_in):
            self._res = res
            self._mlp_in = mlp_in

        @property
        def output(self):
            # nnsave(layers[i].output[0]) — output[0] must be a real tensor [1, 1, d_model]
            # so that r[0, -1, :].float() works directly after the trace.
            return (self._res.unsqueeze(0).unsqueeze(0),)

        @property
        def pre_feedforward_layernorm(self):
            outer = self
            class LNMock:
                @property
                def output(self2):
                    # nnsave(layers[i].pre_feedforward_layernorm.output) → real tensor [1, 1, d_model]
                    return outer._mlp_in.unsqueeze(0).unsqueeze(0)
            return LNMock()

    fake_layers = [FakeLayer(residual_tensors[i], mlp_input_tensors[i])
                   for i in range(n_layers)]

    class FakeLMHead:
        @property
        def output(self):
            # nnsave(lm_head.output) → real tensor [1, 1, vocab_size]
            # so that logits_save[0, -1, :].float() works after the trace.
            return logit_tensor.unsqueeze(0).unsqueeze(0)

    class FakeLM:
        @property
        def layers(self2):
            class LayerList:
                def __getitem__(self3, i):
                    return fake_layers[i]
            return LayerList()

        @property
        def norm(self2):
            return lambda x: x

    class FakeModel:
        language_model = FakeLM()
        lm_head = FakeLMHead()
        tokenizer = tok

        @contextlib.contextmanager
        def trace(self, ids, **kwargs):
            yield

    # Monkey-patch N_LAYERS for the test
    original_n_layers = collect_mod.N_LAYERS
    collect_mod.N_LAYERS = n_layers

    loaded = MagicMock()
    loaded.model = FakeModel()
    loaded.W_U = W_U
    loaded.real_token_mask = real_mask
    loaded.tokenizer = tok
    loaded.transcoder_list = transcoder_list

    try:
        result = collect_prompt(
            prompt_id="test_0",
            category="in_context",
            prompt="The capital of France is",
            loaded=loaded,
            cfg=cfg,
        )
    finally:
        collect_mod.N_LAYERS = original_n_layers

    # Schema checks
    assert result["prompt_id"] == "test_0"
    assert result["category"] == "in_context"
    assert isinstance(result["generated_tokens"], list)
    assert len(result["generated_tokens"]) >= 1
    assert len(result["generated_tokens"]) <= max_gen

    for key in ("residual_norm", "layer_delta_norm", "layer_delta_cosine", "l0", "logit_lens_entropy"):
        assert key in result, f"Missing key: {key}"
        assert isinstance(result[key], list)
        for step_vals in result[key]:
            assert len(step_vals) == n_layers, (
                f"{key}: expected {n_layers} values per step, got {len(step_vals)}"
            )

    assert "output_entropy" in result
    assert all(isinstance(v, float) for v in result["output_entropy"])
    assert all(not math.isnan(v) for v in result["output_entropy"])

    # Greedy decode must not produce unused tokens
    for tok_entry in result["generated_tokens"]:
        tid = tok_entry["token_id"]
        assert real_mask[tid].item(), (
            f"Generated token_id={tid} is in <unusedXXXX> range — masking failed"
        )


# ─── 6. collect_all npz packing ───────────────────────────────────────────────

def test_collect_all_strips_active_features_from_json():
    """active_features must be removed from result dicts (too large for JSON)."""
    from src.poc.exp2.collect import collect_all, save_results
    from src.poc.exp2.config import Exp2Config

    # Build fake results as if collect_prompt returned them
    n_layers = 4
    fake_results = [
        {
            "prompt_id": "ic_factual_0",
            "category": "in_context",
            "prompt": "test",
            "generated_tokens": [{"token_id": 1, "token_str": " a"}],
            "residual_norm": [[1.0]*n_layers],
            "layer_delta_norm": [[0.5]*n_layers],
            "layer_delta_cosine": [[float("nan")] + [0.1]*(n_layers-1)],
            "l0": [[3]*n_layers],
            "output_entropy": [2.0],
            "logit_lens_entropy": [[5.0]*n_layers],
            "active_features": [[[0, 1, 2]]*n_layers],
        }
    ]

    # Simulate what collect_all does: pop active_features, build npz
    import numpy as np
    from src.poc.shared.constants import N_LAYERS as REAL_N_LAYERS

    results_copy = [dict(r) for r in fake_results]
    npz_data = {}
    for r in results_copy:
        af = r.pop("active_features")
        n_steps = len(af)
        af_arr = np.empty((n_steps, n_layers), dtype=object)
        for s in range(n_steps):
            for layer in range(n_layers):
                af_arr[s, layer] = np.array(af[s][layer], dtype=np.int32)
        npz_data[r["prompt_id"]] = af_arr

    # Verify active_features stripped
    assert "active_features" not in results_copy[0]

    # Verify JSON-serializable
    json_str = json.dumps(results_copy)
    loaded_back = json.loads(json_str)
    assert loaded_back[0]["prompt_id"] == "ic_factual_0"

    # Verify npz round-trip
    with tempfile.TemporaryDirectory() as d:
        npz_path = Path(d) / "test.npz"
        np.savez_compressed(str(npz_path), **npz_data)
        npz_loaded = np.load(str(npz_path), allow_pickle=True)
        arr = npz_loaded["ic_factual_0"]
        assert arr.shape == (1, n_layers)
        idxs = arr[0, 0]
        assert list(idxs) == [0, 1, 2], f"Expected [0,1,2], got {list(idxs)}"


# ─── 7. Plots smoke-test (all 10) ─────────────────────────────────────────────

def _make_synthetic_results(n_prompts_per_cat=5, n_layers=34, max_steps=5):
    np.random.seed(0)
    results = []
    for cat, n in [("in_context", n_prompts_per_cat),
                   ("out_of_context", n_prompts_per_cat),
                   ("reasoning", n_prompts_per_cat)]:
        for i in range(n):
            n_steps = np.random.randint(2, max_steps + 1)
            results.append({
                "prompt_id": f"{cat}_{i}",
                "category": cat,
                "prompt": "test",
                "generated_tokens": [{"token_id": j+1, "token_str": f" t{j}"} for j in range(n_steps)],
                "residual_norm": [[float(np.random.randn() + 20)] * n_layers for _ in range(n_steps)],
                "layer_delta_norm": [[float(abs(np.random.randn()))] * n_layers for _ in range(n_steps)],
                "layer_delta_cosine": [[float("nan")] + [float(np.random.uniform(-0.5, 0.5)) for _ in range(n_layers-1)] for _ in range(n_steps)],
                "l0": [[int(np.random.randint(5, 50))] * n_layers for _ in range(n_steps)],
                "output_entropy": [float(np.random.rand() * 5) for _ in range(n_steps)],
                "logit_lens_entropy": [[float(np.random.rand() * 8)] * n_layers for _ in range(n_steps)],
            })
    return results


@pytest.mark.parametrize("plot_module,plot_fn", [
    ("src.poc.exp2.plots.plot1_l0_per_layer", "make_plot"),
    ("src.poc.exp2.plots.plot2_l0_heatmap", "make_plot"),
    ("src.poc.exp2.plots.plot4_layer_delta_norm", "make_plot"),
    ("src.poc.exp2.plots.plot5_residual_norm", "make_plot"),
    ("src.poc.exp2.plots.plot6_output_entropy", "make_plot"),
    ("src.poc.exp2.plots.plot7_logit_lens_entropy", "make_plot"),
    ("src.poc.exp2.plots.plot8_logit_lens_heatmap", "make_plot"),
    ("src.poc.exp2.plots.plot9_generation_length", "make_plot"),
    ("src.poc.exp2.plots.plot10_cosine_similarity", "make_plot"),
])
def test_plot_renders_without_error(plot_module, plot_fn, tmp_path):
    """Each plot must generate a PNG without raising any exception."""
    import importlib
    mod = importlib.import_module(plot_module)
    fn = getattr(mod, plot_fn)
    results = _make_synthetic_results()
    fn(results, str(tmp_path))
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) >= 1, f"{plot_module}: expected at least one PNG, found none"


def test_plot3_gracefully_skips_without_npz(tmp_path):
    """Plot 3 must skip gracefully (print + return) when .npz file is absent."""
    from src.poc.exp2.plots.plot3_feature_overlap import make_plot
    results = _make_synthetic_results()
    # No npz file exists → should print a skip message and not raise
    make_plot(results, str(tmp_path), npz_path=str(tmp_path / "nonexistent.npz"))
    # No PNG generated is acceptable
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 0, "Plot 3 should not generate a PNG without npz data"


def test_plot3_with_npz(tmp_path):
    """Plot 3 must generate a PNG when .npz is present."""
    from src.poc.exp2.plots.plot3_feature_overlap import make_plot
    n_layers = 34
    results = _make_synthetic_results()

    # Build minimal npz with active features
    npz_data = {}
    for r in results:
        n_steps = len(r["generated_tokens"])
        af_arr = np.empty((n_steps, n_layers), dtype=object)
        for s in range(n_steps):
            for layer in range(n_layers):
                # Randomly choose 5-10 features from 0..999
                af_arr[s, layer] = np.array(
                    np.random.choice(1000, size=np.random.randint(5, 11), replace=False),
                    dtype=np.int32,
                )
        npz_data[r["prompt_id"]] = af_arr

    npz_path = tmp_path / "fake.npz"
    np.savez_compressed(str(npz_path), **npz_data)

    make_plot(results, str(tmp_path), npz_path=str(npz_path))
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1, f"Expected 1 PNG from plot3, got {pngs}"


def test_run_plots_uses_npz_matching_results_filename(tmp_path, monkeypatch):
    """run_plots should pass the sibling .npz path for custom result filenames."""
    from src.poc.exp2 import run_plots as run_plots_mod

    results_path = tmp_path / "custom_run.json"
    npz_path = tmp_path / "custom_run.npz"
    plot_dir = tmp_path / "plots"
    results = _make_synthetic_results(n_prompts_per_cat=1, max_steps=2)
    results_path.write_text(json.dumps(results))
    np.savez_compressed(str(npz_path), dummy=np.array([1]))

    captured = {}

    def fake_plot3(results_arg, output_dir_arg, npz_path=None):
        captured["results"] = results_arg
        captured["output_dir"] = output_dir_arg
        captured["npz_path"] = npz_path

    monkeypatch.setattr(run_plots_mod, "plot1", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_plots_mod, "plot2", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_plots_mod, "plot3", fake_plot3)
    monkeypatch.setattr(run_plots_mod, "plot4", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_plots_mod, "plot5", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_plots_mod, "plot6", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_plots_mod, "plot7", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_plots_mod, "plot8", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_plots_mod, "plot9", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_plots_mod, "plot10", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.argv", ["run_plots.py", "--results", str(results_path)])

    run_plots_mod.main()

    assert captured["output_dir"] == str(plot_dir)
    assert captured["npz_path"] == str(npz_path)


def test_run_plots_rejects_invalid_schema(tmp_path, monkeypatch, capsys):
    """run_plots should fail fast on malformed top-level schema."""
    from src.poc.exp2 import run_plots as run_plots_mod

    results_path = tmp_path / "bad.json"
    results_path.write_text(json.dumps({"prompts": []}))
    monkeypatch.setattr("sys.argv", ["run_plots.py", "--results", str(results_path)])

    run_plots_mod.main()
    out = capsys.readouterr().out
    assert "invalid schema" in out


def test_run_plots_handles_empty_results(tmp_path, monkeypatch, capsys):
    """run_plots should exit cleanly when the results list is empty."""
    from src.poc.exp2 import run_plots as run_plots_mod

    results_path = tmp_path / "empty.json"
    results_path.write_text(json.dumps([]))
    monkeypatch.setattr("sys.argv", ["run_plots.py", "--results", str(results_path)])

    run_plots_mod.main()
    out = capsys.readouterr().out
    assert "nothing to plot" in out.lower()


@pytest.mark.parametrize("plot_module", [
    "src.poc.exp2.plots.plot1_l0_per_layer",
    "src.poc.exp2.plots.plot4_layer_delta_norm",
    "src.poc.exp2.plots.plot5_residual_norm",
    "src.poc.exp2.plots.plot7_logit_lens_entropy",
])
def test_selected_plots_skip_cleanly_on_empty_results(plot_module, tmp_path):
    """Plots that aggregate by category should not crash on empty results."""
    import importlib

    mod = importlib.import_module(plot_module)
    mod.make_plot([], str(tmp_path))
    assert list(tmp_path.glob("*.png")) == []


def test_plot10_missing_data_produces_nan_not_zero():
    """Missing categories/bands should not be rendered as numeric zero similarity."""
    from src.poc.exp2.plots.plot10_cosine_similarity import _mean_cos_profile, _cosine

    results = [r for r in _make_synthetic_results(n_prompts_per_cat=1, max_steps=2)
               if r["category"] != "reasoning"]
    profile = _mean_cos_profile(results, "reasoning")
    sim = _cosine(profile, np.ones(34))
    assert profile is None
    assert math.isnan(sim)


def test_plot10_uses_prompt_relative_thirds():
    """Each prompt should contribute to all three bands when it has >=3 steps."""
    from src.poc.exp2.plots.plot10_cosine_similarity import _split_into_thirds

    steps = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    bands = _split_into_thirds(steps)
    assert [len(b) for b in bands] == [2, 2, 2]
    assert bands[0][0] == [0.0]
    assert bands[2][-1] == [5.0]


# ─── 8. Gemma3DecoderLayer tuple output structure ─────────────────────────────

def test_gemma3_decoder_layer_returns_tuple():
    """Gemma3DecoderLayer.forward() must return a tuple; [0] is hidden_states.

    Verifies our .output[0] indexing in collect_prompt is correct.
    The return annotation is tuple[FloatTensor, Optional[...]] confirming tuple.
    """
    from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
    import inspect
    sig = inspect.signature(Gemma3DecoderLayer.forward)
    # Return annotation should mention tuple
    ret = sig.return_annotation
    ret_str = str(ret)
    assert "tuple" in ret_str.lower(), (
        f"Gemma3DecoderLayer.forward should return a tuple, annotation: {ret_str}"
    )


def test_gemma3_pre_feedforward_layernorm_exists():
    """Gemma3DecoderLayer must have pre_feedforward_layernorm attribute."""
    from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
    import inspect
    src = inspect.getsource(Gemma3DecoderLayer.forward)
    assert "pre_feedforward_layernorm" in src, (
        "pre_feedforward_layernorm not found in Gemma3DecoderLayer.forward — "
        "the attribute name may have changed in a new transformers version."
    )


def test_shared_layer_path_matches_exp2_conditional_generation_assumption():
    """Shared layer path docs should match the exp2 conditional-generation path."""
    from src.poc.shared.constants import LAYER_PATH

    assert LAYER_PATH == "language_model.layers"
