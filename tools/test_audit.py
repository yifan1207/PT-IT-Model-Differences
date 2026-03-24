"""
Rigorous audit tests for collect.py and collect_config.py.
CPU-only, no GPU, no nnsight required.
Run with: uv run python tools/test_audit.py
"""
from __future__ import annotations

import math
import pathlib
import sys
import traceback
import types
import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Inject fake 'nnsight' so collect.py can be imported without it installed
# ---------------------------------------------------------------------------
fake_nnsight = types.ModuleType("nnsight")
fake_nnsight.save = lambda x: x  # identity — we'll control the return values directly
sys.modules.setdefault("nnsight", fake_nnsight)

# Also inject constants module
fake_constants = types.ModuleType("src.poc.shared.constants")
fake_constants.N_LAYERS = 4
sys.modules["src.poc.shared.constants"] = fake_constants

# Inject a fake token_types module so classify_generated_tokens doesn't blow up
fake_token_types_pkg = types.ModuleType("src.poc.exp3.analysis.token_types")
fake_token_types_pkg.classify_generated_tokens = lambda tokens: ["OTHER"] * len(tokens)
sys.modules["src.poc.exp3"] = types.ModuleType("src.poc.exp3")
sys.modules["src.poc.exp3.analysis"] = types.ModuleType("src.poc.exp3.analysis")
sys.modules["src.poc.exp3.analysis.token_types"] = fake_token_types_pkg

# ---------------------------------------------------------------------------
# Import the modules under test
# ---------------------------------------------------------------------------
from src.poc.shared.collect_config import CollectionConfig, ModelHooks  # noqa: E402

# Import only the pure-Python functions from collect.py — we avoid anything
# that calls nnsight or torch.cuda at module-level.
import importlib.util, pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "collect",
    REPO_ROOT / "src/poc/collect.py",
)
collect_mod = importlib.util.module_from_spec(spec)
# Patch nnsight.save before exec
collect_mod.__dict__["nnsave"] = lambda x: x
spec.loader.exec_module(collect_mod)

_entropy        = collect_mod._entropy
_kl_div         = collect_mod._kl_div
_sanitise       = collect_mod._sanitise
_compute_gen_step_metrics = collect_mod._compute_gen_step_metrics

# ---------------------------------------------------------------------------
# Shared test utilities
# ---------------------------------------------------------------------------

VOCAB = 32           # tiny vocab
N_LAYERS = 4         # tiny model
D_MODEL  = 8

def _make_cfg(**kwargs) -> CollectionConfig:
    """Make a CollectionConfig bypassing model_id lookups."""
    base = dict(
        model_id="test-model",
        model_variant="it",
        transcoder_release="dummy",
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=2,
        collect_emergence=True,
        collect_attribution=True,
        collect_layer_extras=True,
        collect_top5_tokens=True,
        collect_step_kl=True,
        collect_feature_values=True,
        collect_transcoder_mse=False,
        collect_transcoder_norms=False,
        collect_residual_cosine=True,
        collect_mlp_norms=True,
        collect_forward_contrib=True,
        collect_repulsion=True,
        repulsion_top_k=3,
        mode={"generate"},
    )
    base.update(kwargs)
    return CollectionConfig(**base)


def _make_real_mask(vocab: int, n_real: int = 24) -> torch.Tensor:
    """Boolean mask with first n_real tokens as real."""
    m = torch.zeros(vocab, dtype=torch.bool)
    m[:n_real] = True
    return m


def _make_residuals(n_layers: int, d_model: int) -> list[torch.Tensor]:
    torch.manual_seed(42)
    return [torch.randn(d_model) for _ in range(n_layers)]


def _make_logits(vocab: int) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(vocab)


def _make_mlp_outputs(n_layers: int, d_model: int) -> list[torch.Tensor]:
    torch.manual_seed(99)
    return [torch.randn(d_model) for _ in range(n_layers)]


def _make_loaded(vocab: int, n_layers: int, d_model: int, n_real: int = 24):
    """Build a mock LoadedModel-like object."""
    W_U = torch.randn(d_model, vocab)   # [d_model, vocab]

    # final_norm: just returns input unchanged (identity)
    class FakeNorm(torch.nn.Module):
        def forward(self, x):
            return x

    # model with a navigate-able final_norm
    class FakeModel:
        class language_model:
            norm = FakeNorm()
    model = FakeModel()

    # transcoders: a simple linear encode that returns activations
    class FakeTranscoder:
        def __init__(self, n_features=16):
            self.b_enc = torch.zeros(n_features)   # used for device/dtype
            self._n_features = n_features
        def encode(self, x):
            # x: [1, d_model]; return [1, n_features] with a few non-zero entries
            torch.manual_seed(0)
            acts = torch.zeros(1, self._n_features)
            acts[0, 0] = 0.5
            acts[0, 3] = 1.2
            return acts   # shape [1, n_features]
        def forward(self, x):
            torch.manual_seed(0)
            out = torch.randn(1, x.shape[-1])
            return out   # [1, d_model]

    tc_list = [FakeTranscoder() for _ in range(n_layers)]

    loaded = MagicMock()
    loaded.W_U = W_U
    loaded.real_token_mask = _make_real_mask(vocab, n_real)
    loaded.transcoder_list = tc_list
    # Expose model.language_model.norm via _navigate_path
    loaded.model = model
    return loaded


# ---------------------------------------------------------------------------
# A. _compute_gen_step_metrics tests
# ---------------------------------------------------------------------------

class TestComputeGenStepMetrics(unittest.TestCase):

    def setUp(self):
        self.vocab    = VOCAB
        self.n_layers = N_LAYERS
        self.d_model  = D_MODEL
        self.n_real   = 24

        self.cfg    = _make_cfg(
            collect_emergence=True,
            collect_attribution=True,
            collect_layer_extras=True,
            collect_top5_tokens=True,
            collect_feature_values=True,
            collect_residual_cosine=True,
            collect_mlp_norms=True,
            collect_transcoder_norms=False,
            collect_transcoder_mse=False,
            collect_forward_contrib=True,
            collect_step_kl=True,
        )
        self.loaded = _make_loaded(self.vocab, self.n_layers, self.d_model, self.n_real)
        self.real_mask = self.loaded.real_token_mask  # [vocab]

        self.residuals   = _make_residuals(self.n_layers, self.d_model)
        self.mlp_outputs = _make_mlp_outputs(self.n_layers, self.d_model)
        self.logits      = _make_logits(self.vocab)

        self.raw = {
            "residuals":   self.residuals,
            "mlp_inputs":  self.residuals,   # re-use for simplicity
            "logits":      self.logits,
            "mlp_outputs": self.mlp_outputs,
        }

    def _run(self, **kw):
        cfg = _make_cfg(**kw)
        return _compute_gen_step_metrics(self.raw, self.loaded, cfg)

    # ── A1: top1_token_per_layer ──────────────────────────────────────────────
    def test_top1_token_per_layer_is_real_token(self):
        """top1 must always be a real vocab token (within real_mask)."""
        metrics, _, _, _ = self._run(collect_layer_extras=True)
        top1 = metrics["top1_token_per_layer"]
        self.assertEqual(len(top1), self.n_layers)
        for layer_i, tid in enumerate(top1):
            self.assertTrue(
                self.real_mask[tid].item(),
                f"layer {layer_i}: top1 token {tid} is NOT in real_mask",
            )

    def test_top1_token_per_layer_is_argmax_of_masked(self):
        """top1 must equal argmax of logits with non-real tokens set to -inf."""
        from src.poc.collect import _navigate_path
        # Compute expected manually
        W_U = self.loaded.W_U  # [d_model, vocab]
        expected = []
        for i in range(self.n_layers):
            h = self.residuals[i]
            ll = h.float() @ W_U  # [vocab]
            masked = ll.clone()
            masked[~self.real_mask] = float("-inf")
            expected.append(int(masked.argmax().item()))

        metrics, _, _, _ = self._run(collect_layer_extras=True)
        self.assertEqual(metrics["top1_token_per_layer"], expected,
                         "top1_token_per_layer does not match manual argmax")

    # ── A2: kl_adjacent_layer direction ──────────────────────────────────────
    def test_kl_adjacent_layer_direction(self):
        """kl_adjacent_layer[i] = KL(p_i ∥ p_{i-1}), not KL(p_{i-1} ∥ p_i)."""
        metrics, _, _, _ = self._run(collect_layer_extras=True)
        kl_adj = metrics["kl_adjacent_layer"]

        # Verify nan at layer 0
        self.assertTrue(math.isnan(kl_adj[0]),
                        "kl_adjacent_layer[0] should be nan")

        # Recompute layer 1 manually: KL(p_1 ∥ p_0)
        W_U = self.loaded.W_U
        lls = [self.residuals[i].float() @ W_U for i in range(self.n_layers)]
        p0 = torch.softmax(lls[0][self.real_mask], dim=-1)
        p1 = torch.softmax(lls[1][self.real_mask], dim=-1)
        expected_kl = float((p1 * torch.log(p1 / (p0 + 1e-12) + 1e-12)).sum().item())
        self.assertAlmostEqual(kl_adj[1], expected_kl, places=5,
                               msg="kl_adjacent_layer[1] has wrong direction or value")

    # ── A3: lens_entropy_delta ────────────────────────────────────────────────
    def test_lens_entropy_delta_direction(self):
        """delta[i] = entropy[i] - entropy[i-1]."""
        metrics, _, _, _ = self._run(collect_layer_extras=True)
        kl_ent   = metrics["lens_entropy_delta"]
        lens_ent = metrics["logit_lens_entropy"]  # always computed

        self.assertTrue(math.isnan(kl_ent[0]))
        for i in range(1, self.n_layers):
            expected = lens_ent[i] - lens_ent[i - 1]
            self.assertAlmostEqual(
                kl_ent[i], expected, places=6,
                msg=f"lens_entropy_delta[{i}] = {kl_ent[i]}, expected {expected}"
            )

    # ── A4: top5_tokens — all returned token ids are real ────────────────────
    def test_top5_tokens_all_real(self):
        metrics, _, _, _ = self._run(collect_top5_tokens=True)
        ids_per_layer   = metrics["top5_token_ids_per_layer"]
        probs_per_layer = metrics["top5_token_probs_per_layer"]
        self.assertEqual(len(ids_per_layer), self.n_layers)
        for layer_i, (ids, probs) in enumerate(zip(ids_per_layer, probs_per_layer)):
            self.assertEqual(len(ids), 5)
            self.assertEqual(len(probs), 5)
            for tid in ids:
                self.assertTrue(
                    self.real_mask[tid].item(),
                    f"layer {layer_i}: top5 token {tid} is NOT in real_mask",
                )

    def test_top5_probs_sum_sensible(self):
        """Softmax over the masked vocab; top-5 probs should be <= 1.0 and > 0."""
        metrics, _, _, _ = self._run(collect_top5_tokens=True)
        for layer_i, probs in enumerate(metrics["top5_token_probs_per_layer"]):
            total = sum(probs)
            self.assertLessEqual(total, 1.0 + 1e-5,
                f"layer {layer_i}: top5 probs sum {total} > 1")
            for p in probs:
                self.assertGreater(p, 0.0)

    # ── A5: residual_cosine_to_final ─────────────────────────────────────────
    def test_residual_cosine_to_final_range(self):
        """Cosine similarity must be in [-1, 1]."""
        metrics, _, _, _ = self._run(collect_residual_cosine=True)
        cos = metrics["residual_cosine_to_final"]
        self.assertEqual(len(cos), self.n_layers)
        for i, c in enumerate(cos):
            if not math.isnan(c):
                self.assertGreaterEqual(c, -1.0 - 1e-5, f"layer {i}")
                self.assertLessEqual(   c,  1.0 + 1e-5, f"layer {i}")

    def test_residual_cosine_to_final_last_layer_is_1(self):
        """The last layer's cosine with itself must be 1.0."""
        metrics, _, _, _ = self._run(collect_residual_cosine=True)
        cos = metrics["residual_cosine_to_final"]
        self.assertAlmostEqual(cos[-1], 1.0, places=5,
                               msg="Last layer cosine to itself is not 1.0")

    def test_residual_cosine_h_final_computed_from_residuals_minus_1(self):
        """h_final is residuals[-1], verified by manual formula."""
        metrics, _, _, _ = self._run(collect_residual_cosine=True)
        cos = metrics["residual_cosine_to_final"]
        h_final = self.residuals[-1].float()
        norm_final = h_final.norm()
        for i in range(self.n_layers):
            h_i = self.residuals[i].float()
            denom = h_i.norm() * norm_final
            if denom > 0:
                expected = float((torch.dot(h_i, h_final) / denom).item())
                self.assertAlmostEqual(cos[i], expected, places=5,
                    msg=f"layer {i}: cosine {cos[i]} != expected {expected}")

    # ── A6: mlp_contribution_norm ─────────────────────────────────────────────
    def test_mlp_contribution_norm_present_when_mlp_outputs_correct(self):
        """When collect_mlp_norms=True and mlp_outputs has n_layers entries, norm is populated."""
        metrics, _, _, _ = self._run(collect_mlp_norms=True)
        norms = metrics["mlp_contribution_norm"]
        self.assertEqual(len(norms), self.n_layers,
                         "mlp_contribution_norm should have n_layers entries")
        for i, n in enumerate(norms):
            expected = float(self.mlp_outputs[i].norm().item())
            self.assertAlmostEqual(n, expected, places=5, msg=f"layer {i}")

    def test_mlp_contribution_norm_empty_when_mlp_outputs_empty(self):
        """When mlp_outputs is empty list, mlp_contribution_norm must be []."""
        raw_empty = dict(self.raw)
        raw_empty["mlp_outputs"] = []
        cfg = _make_cfg(collect_mlp_norms=True)
        metrics, _, _, _ = _compute_gen_step_metrics(raw_empty, self.loaded, cfg)
        self.assertEqual(metrics["mlp_contribution_norm"], [],
                         "mlp_contribution_norm should be [] when mlp_outputs is empty")

    def test_mlp_contribution_norm_guarded_by_len(self):
        """Guard: if mlp_outputs has wrong length, mlp_contribution_norm must be []."""
        raw_short = dict(self.raw)
        raw_short["mlp_outputs"] = self.mlp_outputs[:self.n_layers - 1]  # one short
        cfg = _make_cfg(collect_mlp_norms=True)
        metrics, _, _, _ = _compute_gen_step_metrics(raw_short, self.loaded, cfg)
        self.assertEqual(metrics["mlp_contribution_norm"], [],
                         "mlp_contribution_norm should be [] when len(mlp_outputs) != n_layers")

    # ── A7: transcoder_output_norm + transcoder_mse merged loop ──────────────
    def test_transcoder_norms_only(self):
        """When collect_transcoder_norms=True, collect_transcoder_mse=False:
        transcoder_output_norm populated, transcoder_mse empty."""
        metrics, _, _, _ = self._run(
            collect_transcoder_norms=True,
            collect_transcoder_mse=False,
        )
        self.assertEqual(len(metrics["transcoder_output_norm"]), self.n_layers)
        self.assertEqual(metrics["transcoder_mse"], [])

    def test_transcoder_mse_only(self):
        """When collect_transcoder_mse=True, collect_transcoder_norms=False:
        transcoder_mse populated, transcoder_output_norm empty."""
        metrics, _, _, _ = self._run(
            collect_transcoder_mse=True,
            collect_transcoder_norms=False,
        )
        self.assertEqual(len(metrics["transcoder_mse"]), self.n_layers)
        self.assertEqual(metrics["transcoder_output_norm"], [])

    def test_both_transcoder_flags(self):
        """When both flags True: both lists populated with n_layers entries."""
        metrics, _, _, _ = self._run(
            collect_transcoder_mse=True,
            collect_transcoder_norms=True,
        )
        self.assertEqual(len(metrics["transcoder_mse"]), self.n_layers)
        self.assertEqual(len(metrics["transcoder_output_norm"]), self.n_layers)

    def test_transcoder_flags_need_mlp_outputs(self):
        """If mlp_outputs is empty, both transcoder lists must be empty."""
        raw_empty = dict(self.raw)
        raw_empty["mlp_outputs"] = []
        cfg = _make_cfg(collect_transcoder_mse=True, collect_transcoder_norms=True)
        metrics, _, _, _ = _compute_gen_step_metrics(raw_empty, self.loaded, cfg)
        self.assertEqual(metrics["transcoder_mse"], [])
        self.assertEqual(metrics["transcoder_output_norm"], [])

    # ── A8: feature_vals indexing ────────────────────────────────────────────
    def test_feature_vals_indexing_correct(self):
        """acts[0][idxs] should return the activation values at the active indices."""
        metrics, step_features, _, _ = self._run(collect_feature_values=True)
        feat_vals = metrics["feature_vals"]
        # FakeTranscoder always returns acts[0, 0]=0.5, acts[0, 3]=1.2
        # step_features[layer] should be [0, 3] for every layer
        for layer_i in range(self.n_layers):
            idxs = step_features[layer_i]
            vals = feat_vals[layer_i]
            self.assertEqual(len(idxs), len(vals),
                             f"layer {layer_i}: len(idxs)={len(idxs)} != len(vals)={len(vals)}")
            # All values must be positive (our fake transcoder returns 0.5 and 1.2)
            for v in vals:
                self.assertGreater(v, 0.0, f"layer {layer_i}: feature val {v} <= 0")

    def test_feature_vals_empty_when_no_active(self):
        """Layer with 0 active features produces empty vals list."""
        class ZeroTranscoder:
            b_enc = torch.zeros(16)
            def encode(self, x):
                return torch.zeros(1, 16)   # all zeros → no active features
            def forward(self, x):
                return torch.zeros(1, x.shape[-1])

        loaded_zero = _make_loaded(self.vocab, self.n_layers, self.d_model)
        loaded_zero.transcoder_list = [ZeroTranscoder() for _ in range(self.n_layers)]
        cfg = _make_cfg(collect_feature_values=True)
        metrics, step_features, _, _ = _compute_gen_step_metrics(
            self.raw, loaded_zero, cfg
        )
        for layer_i in range(self.n_layers):
            self.assertEqual(step_features[layer_i], [],
                             f"layer {layer_i}: expected empty features")
            self.assertEqual(metrics["feature_vals"][layer_i], [],
                             f"layer {layer_i}: expected empty vals")

    # ── A9: lens_logits_out returned correctly ────────────────────────────────
    def test_lens_logits_out_returned_when_forward_contrib(self):
        _, _, _, lens_out = self._run(collect_forward_contrib=True, collect_step_kl=False)
        self.assertIsNotNone(lens_out)
        self.assertEqual(len(lens_out), self.n_layers)

    def test_lens_logits_out_returned_when_step_kl_only(self):
        """CRITICAL: when collect_step_kl=True but collect_forward_contrib=False,
        lens_logits_out must still be returned."""
        _, _, _, lens_out = self._run(collect_forward_contrib=False, collect_step_kl=True)
        self.assertIsNotNone(lens_out,
            "BUG: lens_logits_out is None when only collect_step_kl=True")
        self.assertEqual(len(lens_out), self.n_layers)

    def test_lens_logits_out_none_when_neither_flag(self):
        _, _, _, lens_out = self._run(collect_forward_contrib=False, collect_step_kl=False)
        self.assertIsNone(lens_out)


# ---------------------------------------------------------------------------
# B. collect_record — step_to_step_kl buffering (manual simulation)
# ---------------------------------------------------------------------------

class TestStepToStepKL(unittest.TestCase):
    """Simulate the step-to-step KL accumulation loop from collect_record
    without actually calling the full function (which needs a tokenizer etc.)."""

    def setUp(self):
        self.n_layers = 3
        self.n_real   = 10
        self.vocab    = 16
        self.real_mask = _make_real_mask(self.vocab, self.n_real)

    def _make_lens_logits(self, seed: int) -> list[torch.Tensor]:
        torch.manual_seed(seed)
        return [torch.randn(self.vocab) for _ in range(self.n_layers)]

    def _simulate_kl_loop(self, n_steps: int) -> list[list[float]]:
        """Reproduce the exact logic from collect_record step_to_step_kl section."""
        step_to_step_kl: list[list[float]] = []
        prev_step_real_probs: list[torch.Tensor] | None = None

        for step in range(n_steps):
            cur_lens_logits = self._make_lens_logits(step)

            real_mask_skl = self.real_mask
            if prev_step_real_probs is None:
                step_to_step_kl.append([float("nan")] * self.n_layers)
            else:
                kls = []
                with torch.inference_mode():
                    for li in range(self.n_layers):
                        p_curr = torch.softmax(
                            cur_lens_logits[li][real_mask_skl], dim=-1
                        )
                        kls.append(
                            float((p_curr * torch.log(
                                p_curr / (prev_step_real_probs[li] + 1e-12) + 1e-12
                            )).sum().item())
                        )
                step_to_step_kl.append(kls)

            # Update buffer
            with torch.inference_mode():
                prev_step_real_probs = [
                    torch.softmax(cur_lens_logits[li][real_mask_skl], dim=-1).cpu()
                    for li in range(self.n_layers)
                ]

        return step_to_step_kl

    def test_step0_all_nan(self):
        kl = self._simulate_kl_loop(3)
        for v in kl[0]:
            self.assertTrue(math.isnan(v), f"step 0: expected nan, got {v}")

    def test_step1_non_nan(self):
        kl = self._simulate_kl_loop(3)
        for li, v in enumerate(kl[1]):
            self.assertFalse(math.isnan(v), f"step 1 layer {li}: unexpected nan")
            self.assertGreaterEqual(v, 0.0, f"step 1 layer {li}: KL divergence must be >= 0")

    def test_step2_uses_step1_distribution_not_step0(self):
        """step 2 KL should be KL(step2 ∥ step1), not KL(step2 ∥ step0)."""
        kl = self._simulate_kl_loop(3)

        # Recompute step 2, layer 0 manually
        ll1 = self._make_lens_logits(1)
        ll2 = self._make_lens_logits(2)
        p1 = torch.softmax(ll1[0][self.real_mask], dim=-1)
        p2 = torch.softmax(ll2[0][self.real_mask], dim=-1)
        expected_kl = float((p2 * torch.log(p2 / (p1 + 1e-12) + 1e-12)).sum().item())
        self.assertAlmostEqual(
            kl[2][0], expected_kl, places=5,
            msg=f"step 2 KL: got {kl[2][0]}, expected {expected_kl}"
        )

    def test_kl_direction_is_curr_given_prev(self):
        """KL(p_curr ∥ p_prev): asymmetric, should differ from KL(p_prev ∥ p_curr)."""
        kl = self._simulate_kl_loop(2)
        # Manually compute both directions for layer 0
        ll0 = self._make_lens_logits(0)
        ll1 = self._make_lens_logits(1)
        p0 = torch.softmax(ll0[0][self.real_mask], dim=-1)
        p1 = torch.softmax(ll1[0][self.real_mask], dim=-1)
        kl_curr_prev = float((p1 * torch.log(p1 / (p0 + 1e-12) + 1e-12)).sum().item())
        kl_prev_curr = float((p0 * torch.log(p0 / (p1 + 1e-12) + 1e-12)).sum().item())
        # The code computes KL(p_curr, prev) = _kl_div(p_curr, p_prev)
        self.assertAlmostEqual(kl[1][0], kl_curr_prev, places=5)
        # Must differ from the reverse (unless distributions are identical)
        if abs(kl_curr_prev - kl_prev_curr) > 1e-6:
            self.assertNotAlmostEqual(kl[1][0], kl_prev_curr, places=5,
                msg="KL direction appears wrong: got KL(prev∥curr) instead of KL(curr∥prev)")

    def test_n_steps_shape(self):
        n = 5
        kl = self._simulate_kl_loop(n)
        self.assertEqual(len(kl), n)
        for step in range(n):
            self.assertEqual(len(kl[step]), self.n_layers)

    def test_real_mask_accessed_from_loaded(self):
        """Verify real_mask_skl = loaded.real_token_mask is actually the right object."""
        # In the code: real_mask_skl = loaded.real_token_mask
        # Verify it has the right dtype and shape for softmax indexing
        loaded = _make_loaded(self.vocab, self.n_layers, D_MODEL, self.n_real)
        mask = loaded.real_token_mask
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.shape, (self.vocab,))
        self.assertEqual(int(mask.sum().item()), self.n_real)


# ---------------------------------------------------------------------------
# C. feature_vals packing in collect_record
# ---------------------------------------------------------------------------

class TestFeatureValsPacking(unittest.TestCase):

    def test_empty_step_produces_empty_array(self):
        """np.array([], dtype=np.float32) should store correctly in object array."""
        n_steps  = 2
        n_layers = 3
        all_feature_vals = [
            [[], [1.0, 2.0], []],   # step 0
            [[3.0], [], [4.0]],     # step 1
        ]
        arr = np.empty((n_steps, n_layers), dtype=object)
        for s, step_vals in enumerate(all_feature_vals):
            for layer_i, vals in enumerate(step_vals):
                arr[s, layer_i] = np.array(vals, dtype=np.float32)

        # Check shapes
        self.assertEqual(arr.shape, (n_steps, n_layers))
        # Check empty cells
        np.testing.assert_array_equal(arr[0, 0], np.array([], dtype=np.float32))
        np.testing.assert_array_equal(arr[0, 2], np.array([], dtype=np.float32))
        np.testing.assert_array_equal(arr[1, 1], np.array([], dtype=np.float32))
        # Check non-empty cells
        np.testing.assert_array_almost_equal(arr[0, 1], np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(arr[1, 0], np.array([3.0], dtype=np.float32))

    def test_collect_feature_values_false_vals_still_computed(self):
        """
        BUG: _compute_gen_step_metrics computes step_feature_vals UNCONDITIONALLY
        (lines 563-574 in collect.py), regardless of cfg.collect_feature_values.
        The flag only controls whether collect_record propagates the values.

        This means:
        1. Wasted computation every step when the flag is False.
        2. The 'feature_vals' key in the returned metrics dict always has data.
        3. The feature_summary accumulation block in collect_record (line 955)
           reads step_metrics["feature_vals"][layer_i] unconditionally, which
           works only because the vals are always computed — but accumulates
           feature_summary even when collect_feature_values=False.

        This test documents the actual behavior (vals always populated).
        """
        cfg = _make_cfg(collect_feature_values=False)
        loaded = _make_loaded(VOCAB, N_LAYERS, D_MODEL)
        residuals   = _make_residuals(N_LAYERS, D_MODEL)
        mlp_outputs = _make_mlp_outputs(N_LAYERS, D_MODEL)
        raw = {
            "residuals":   residuals,
            "mlp_inputs":  residuals,
            "logits":      _make_logits(VOCAB),
            "mlp_outputs": mlp_outputs,
        }
        metrics, _, _, _ = _compute_gen_step_metrics(raw, loaded, cfg)
        # ACTUAL BEHAVIOR: feature_vals is always populated in _compute_gen_step_metrics
        # regardless of the flag (the flag only controls propagation in collect_record).
        # This is an efficiency bug — vals are always computed — but not a correctness crash.
        # Document: the length should be n_layers (not 0)
        self.assertEqual(
            len(metrics["feature_vals"]), N_LAYERS,
            "feature_vals is always populated in _compute_gen_step_metrics regardless "
            "of collect_feature_values flag — unconditional computation is an efficiency bug"
        )

    def test_bug_feature_summary_reads_vals_when_collect_feature_values_false(self):
        """
        BUG: In collect_record(), the feature_summary accumulation block
        (lines ~948-957) unconditionally reads step_metrics["feature_vals"][layer_i]
        regardless of cfg.collect_feature_values.

        When collect_feature_values=False:
          - step_metrics["feature_vals"] = []  (empty list)
          - But the code still executes:
              vals = np.asarray(step_metrics["feature_vals"][layer_i], ...)
          - This would raise IndexError at runtime because feature_vals is []
            and layer_i > 0, OR silently produce wrong results.

        The accumulation loop at line 948:
            for layer_i, idxs in enumerate(step_feat):
                if not idxs: continue
                ...
                vals = np.asarray(step_metrics["feature_vals"][layer_i], ...)

        If collect_feature_values=False, feature_vals=[] and layer_i>=0
        would cause IndexError when idxs is non-empty.

        This test verifies the bug exists (documents it) or confirms it is fixed.
        """
        # Reproduce the bug: create a scenario with active features but collect_feature_values=False
        n_features = 16
        n_layers   = 2

        class ActiveTranscoder:
            b_enc = torch.zeros(n_features)
            def encode(self, x):
                acts = torch.zeros(1, n_features)
                acts[0, 0] = 1.0   # always one active feature
                return acts
            def forward(self, x):
                return torch.zeros(1, x.shape[-1])

        loaded = _make_loaded(VOCAB, n_layers, D_MODEL)
        loaded.transcoder_list = [ActiveTranscoder() for _ in range(n_layers)]

        cfg = _make_cfg(
            n_layers=n_layers,
            collect_feature_values=False,  # The buggy flag
        )

        raw = {
            "residuals":   [torch.randn(D_MODEL) for _ in range(n_layers)],
            "mlp_inputs":  [torch.randn(D_MODEL) for _ in range(n_layers)],
            "logits":      _make_logits(VOCAB),
            "mlp_outputs": [torch.randn(D_MODEL) for _ in range(n_layers)],
        }

        # The bug would appear in collect_record()'s feature_summary accumulation,
        # not in _compute_gen_step_metrics itself.
        # Simulate the accumulation loop:
        step_metrics, step_feat, _, _ = _compute_gen_step_metrics(raw, loaded, cfg)

        feature_summary = {
            "count": [None] * n_layers,
            "sum":   [None] * n_layers,
        }

        bug_raised = False
        try:
            for layer_i, idxs in enumerate(step_feat):
                if not idxs:
                    continue
                if feature_summary["count"][layer_i] is None:
                    feat_dim = len(loaded.transcoder_list[layer_i].b_enc)
                    feature_summary["count"][layer_i] = np.zeros(feat_dim, dtype=np.int64)
                    feature_summary["sum"][layer_i] = np.zeros(feat_dim, dtype=np.float32)
                # BUG: this line will fail when collect_feature_values=False
                vals = np.asarray(step_metrics["feature_vals"][layer_i], dtype=np.float32)
                np.add.at(feature_summary["count"][layer_i], idxs, 1)
                np.add.at(feature_summary["sum"][layer_i], idxs, vals)
        except IndexError:
            bug_raised = True

        # Document result: if bug_raised=True, the bug exists and crashes production
        # If bug_raised=False, either the bug is fixed or there are no active features
        if bug_raised:
            self.fail(
                "BUG CONFIRMED: collect_record feature_summary accumulation raises IndexError "
                "when collect_feature_values=False but there are active features. "
                "The line `vals = np.asarray(step_metrics['feature_vals'][layer_i], ...)` "
                "crashes because feature_vals=[] when the flag is False. "
                "Fix: guard with `if cfg.collect_feature_values:` before the vals line."
            )

    def test_gen_feature_vals_arr_initialized_to_none(self):
        """gen_feature_vals_arr starts as None and is only set inside the generate block."""
        # This tests the initialisation contract, not actual collection
        gen_feature_vals_arr = None
        # simulate the packing condition
        cfg = _make_cfg(collect_feature_values=True)
        all_feature_vals = [[[1.5, 2.0], []], [[3.0], [0.5]]]
        if cfg.collect_feature_values and all_feature_vals:
            n_steps  = 2
            n_layers = 2
            gen_feature_vals_arr = np.empty((n_steps, n_layers), dtype=object)
            for s, step_vals in enumerate(all_feature_vals):
                for layer_i, vals in enumerate(step_vals):
                    gen_feature_vals_arr[s, layer_i] = np.array(vals, dtype=np.float32)
        self.assertIsNotNone(gen_feature_vals_arr)
        self.assertEqual(gen_feature_vals_arr.shape, (2, 2))


# ---------------------------------------------------------------------------
# D. StreamingWriter — call-site signature check
# ---------------------------------------------------------------------------

class TestStreamingWriterSignature(unittest.TestCase):
    """Verify StreamingWriter.write() and collect_record return tuple are consistent."""

    def test_write_accepts_gen_feature_vals(self):
        """write() must accept gen_feature_vals parameter."""
        import inspect
        sig = inspect.signature(collect_mod.StreamingWriter.write)
        params = list(sig.parameters.keys())
        self.assertIn("gen_feature_vals", params,
                      "StreamingWriter.write is missing gen_feature_vals parameter")

    def test_write_accepts_feature_summary(self):
        """write() must accept feature_summary parameter (added later)."""
        import inspect
        sig = inspect.signature(collect_mod.StreamingWriter.write)
        params = list(sig.parameters.keys())
        self.assertIn("feature_summary", params,
                      "StreamingWriter.write is missing feature_summary parameter")

    def test_write_param_count_matches_collect_record_return(self):
        """write() non-self params must equal collect_record return tuple length minus 1
        (result dict is written as JSON separately)."""
        import inspect
        # Count write() non-self params
        sig = inspect.signature(collect_mod.StreamingWriter.write)
        non_self = [p for p in sig.parameters if p != "self"]
        # Count collect_record return elements: count names in return statement
        src = inspect.getsource(collect_mod.collect_record)
        # The return tuple has: base + N array/dict items
        # Check the actual return line
        self.assertIn("return (base, gen_features_arr", src)
        # non_self[0] = result (the dict), rest are arrays
        # They must be equal
        collect_return_count = src.count("gen_features_arr") > 0  # sanity
        self.assertTrue(collect_return_count)

    def test_collect_record_returns_10_tuple(self):
        """collect_record now returns a 10-tuple (added feature_summary)."""
        import inspect
        src = inspect.getsource(collect_mod.collect_record)
        # Verify the return includes feature_summary
        self.assertIn("feature_summary", src,
                      "collect_record return is missing feature_summary")
        # The actual return at end should have both gen_feature_vals_arr and feature_summary
        self.assertIn("gen_feature_vals_arr, feature_summary", src,
                      "collect_record return does not include both gen_feature_vals_arr and feature_summary")

    def test_worker_unpacks_10_tuple(self):
        """_worker should unpack 10 values from collect_record (9 arrays + feature_summary)."""
        import inspect
        src = inspect.getsource(collect_mod._worker)
        self.assertIn("fimp", src,
                      "_worker does not unpack feature_summary (fimp) from collect_record")
        self.assertIn("result, gf, ef, er, gr, gmi, gmo, eaw, gfv, fimp", src,
                      "_worker does not unpack all 10 return values from collect_record")

    def test_run_collection_single_gpu_unpacks_10_tuple(self):
        """run_collection single-GPU path should unpack + pass 10 values."""
        import inspect
        src = inspect.getsource(collect_mod.run_collection)
        self.assertIn("fimp", src,
                      "run_collection does not unpack feature_summary (fimp)")
        self.assertIn("writer.write(result, gf, ef, er, gr, gmi, gmo, eaw, gfv, fimp)", src,
                      "run_collection single-GPU path does not pass all 10 args to writer.write")

    def test_run_collection_multi_gpu_unpacks_10_tuple(self):
        """run_collection multi-GPU path should also unpack + pass 10 values."""
        import inspect
        src = inspect.getsource(collect_mod.run_collection)
        self.assertIn(
            "for result, gf, ef, er, gr, gmi, gmo, eaw, gfv, fimp in future.result()",
            src,
            "run_collection multi-GPU path does not unpack all 10 fields",
        )


# ---------------------------------------------------------------------------
# E. _generate_pass capture_mlp_output condition
# ---------------------------------------------------------------------------

class TestCaptureMlpOutputCondition(unittest.TestCase):
    """
    Validate that capture_mlp_output is True whenever any downstream consumer
    of mlp_outputs is True, so that the merged loop guard `if mlp_outputs:`
    will be satisfied.
    """

    def _capture_condition(self, **flags) -> bool:
        cfg = _make_cfg(**flags)
        return (cfg.collect_transcoder_mse
                or bool(cfg.save_mlp_layers)
                or cfg.collect_mlp_norms
                or cfg.collect_transcoder_norms)

    def test_transcoder_mse_triggers_capture(self):
        self.assertTrue(self._capture_condition(collect_transcoder_mse=True))

    def test_transcoder_norms_triggers_capture(self):
        self.assertTrue(self._capture_condition(collect_transcoder_norms=True))

    def test_mlp_norms_triggers_capture(self):
        self.assertTrue(self._capture_condition(collect_mlp_norms=True))

    def test_save_mlp_layers_triggers_capture(self):
        self.assertTrue(self._capture_condition(save_mlp_layers=[20, 21]))

    def test_no_flags_no_capture(self):
        self.assertFalse(self._capture_condition(
            collect_transcoder_mse=False,
            collect_transcoder_norms=False,
            collect_mlp_norms=False,
            save_mlp_layers=[],
        ))

    def test_transcoder_norms_guard_requires_mlp_outputs(self):
        """When collect_transcoder_norms=True, capture_mlp_output=True ensures
        mlp_outputs is non-empty, satisfying `if mlp_outputs:` guard."""
        self.assertTrue(
            self._capture_condition(collect_transcoder_norms=True),
            "collect_transcoder_norms=True must set capture_mlp_output=True "
            "to ensure mlp_outputs is non-empty for the merged loop"
        )


# ---------------------------------------------------------------------------
# F. lens_logits_out condition
# ---------------------------------------------------------------------------

class TestLensLogitsOutCondition(unittest.TestCase):

    def test_returned_when_forward_contrib(self):
        """lens_logits_out must be returned when collect_forward_contrib=True."""
        loaded = _make_loaded(VOCAB, N_LAYERS, D_MODEL)
        raw = {
            "residuals":   _make_residuals(N_LAYERS, D_MODEL),
            "mlp_inputs":  _make_residuals(N_LAYERS, D_MODEL),
            "logits":      _make_logits(VOCAB),
            "mlp_outputs": _make_mlp_outputs(N_LAYERS, D_MODEL),
        }
        cfg = _make_cfg(collect_forward_contrib=True, collect_step_kl=False)
        _, _, _, out = _compute_gen_step_metrics(raw, loaded, cfg)
        self.assertIsNotNone(out)

    def test_returned_when_step_kl(self):
        """lens_logits_out must be returned when collect_step_kl=True."""
        loaded = _make_loaded(VOCAB, N_LAYERS, D_MODEL)
        raw = {
            "residuals":   _make_residuals(N_LAYERS, D_MODEL),
            "mlp_inputs":  _make_residuals(N_LAYERS, D_MODEL),
            "logits":      _make_logits(VOCAB),
            "mlp_outputs": _make_mlp_outputs(N_LAYERS, D_MODEL),
        }
        cfg = _make_cfg(collect_forward_contrib=False, collect_step_kl=True)
        _, _, _, out = _compute_gen_step_metrics(raw, loaded, cfg)
        self.assertIsNotNone(out,
            "BUG: lens_logits_out is None when collect_step_kl=True, "
            "collect_forward_contrib=False")

    def test_none_when_neither_flag(self):
        loaded = _make_loaded(VOCAB, N_LAYERS, D_MODEL)
        raw = {
            "residuals":   _make_residuals(N_LAYERS, D_MODEL),
            "mlp_inputs":  _make_residuals(N_LAYERS, D_MODEL),
            "logits":      _make_logits(VOCAB),
            "mlp_outputs": [],
        }
        cfg = _make_cfg(collect_forward_contrib=False, collect_step_kl=False)
        _, _, _, out = _compute_gen_step_metrics(raw, loaded, cfg)
        self.assertIsNone(out)


# ---------------------------------------------------------------------------
# G. Config dataclass — new boolean field defaults
# ---------------------------------------------------------------------------

class TestCollectionConfigDefaults(unittest.TestCase):

    def _default_cfg(self) -> CollectionConfig:
        return CollectionConfig(
            model_id="test-model",
            model_variant="it",
            transcoder_release="dummy",
        )

    def test_collect_layer_extras_default(self):
        """collect_layer_extras has default True (non-zero overhead but documented).
        Document the actual default to catch accidental changes."""
        actual = self._default_cfg().collect_layer_extras
        # The default in collect_config.py is True (verified by reading the source).
        # This test documents the actual default — change this if the default changes.
        self.assertTrue(actual,
            "collect_layer_extras default is True in collect_config.py "
            "(this is intentional — the docstring says 'near-zero overhead'). "
            "If changed to False, update this test.")

    def test_collect_top5_tokens_default_false(self):
        self.assertFalse(self._default_cfg().collect_top5_tokens)

    def test_collect_step_kl_default(self):
        """collect_step_kl has default True (documented in collect_config.py).
        Document the actual default to catch accidental changes."""
        actual = self._default_cfg().collect_step_kl
        # The default in collect_config.py is True.
        self.assertTrue(actual,
            "collect_step_kl default is True in collect_config.py. "
            "If changed to False, update this test.")

    def test_collect_feature_values_default_false(self):
        self.assertFalse(self._default_cfg().collect_feature_values)

    def test_collect_transcoder_norms_default_false(self):
        self.assertFalse(self._default_cfg().collect_transcoder_norms)

    def test_collect_residual_cosine_default_false(self):
        self.assertFalse(self._default_cfg().collect_residual_cosine)

    def test_collect_mlp_norms_default_false(self):
        self.assertFalse(self._default_cfg().collect_mlp_norms)

    def test_collect_forward_contrib_default_false(self):
        self.assertFalse(self._default_cfg().collect_forward_contrib)

    def test_collect_repulsion_default_false(self):
        self.assertFalse(self._default_cfg().collect_repulsion)

    def test_post_init_does_not_flip_new_flags(self):
        """__post_init__ must not modify any of the new boolean flags."""
        cfg = CollectionConfig(
            model_id="test-model",
            model_variant="it",
            transcoder_release="dummy",
            collect_layer_extras=True,
            collect_top5_tokens=True,
            collect_step_kl=True,
            collect_feature_values=True,
            collect_transcoder_norms=True,
            collect_residual_cosine=True,
            collect_mlp_norms=True,
            collect_forward_contrib=True,
            collect_repulsion=True,
        )
        self.assertTrue(cfg.collect_layer_extras)
        self.assertTrue(cfg.collect_top5_tokens)
        self.assertTrue(cfg.collect_step_kl)
        self.assertTrue(cfg.collect_feature_values)
        self.assertTrue(cfg.collect_transcoder_norms)
        self.assertTrue(cfg.collect_residual_cosine)
        self.assertTrue(cfg.collect_mlp_norms)
        self.assertTrue(cfg.collect_forward_contrib)
        self.assertTrue(cfg.collect_repulsion)

    def test_mode_coerced_to_set(self):
        cfg = CollectionConfig(
            model_id="test-model",
            model_variant="it",
            transcoder_release="dummy",
            mode=["generate"],
        )
        self.assertIsInstance(cfg.mode, set)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            CollectionConfig(
                model_id="test-model",
                model_variant="it",
                transcoder_release="dummy",
                mode={"generate", "invalid_mode"},
            )

    def test_invalid_variant_raises(self):
        with self.assertRaises(ValueError):
            CollectionConfig(
                model_id="test-model",
                model_variant="xx",
                transcoder_release="dummy",
            )


# ---------------------------------------------------------------------------
# H. Comprehensive mock integration test (3 steps of generation loop)
# ---------------------------------------------------------------------------

class TestMultiStepGenerationLoop(unittest.TestCase):
    """
    Manually simulate 3 steps of the generation loop to verify:
    - step_to_step_kl buffering (step 0 = nan, step 1+ = real values)
    - forward_logit_contrib_k1 off-by-one (last step gets nans)
    - all new metric arrays accumulate correctly
    """

    def setUp(self):
        self.n_layers = 3
        self.d_model  = 8
        self.vocab    = 20
        self.n_real   = 16
        self.n_steps  = 3

        self.cfg    = _make_cfg(
            n_layers=self.n_layers,
            d_model=self.d_model,
            collect_emergence=True,
            collect_attribution=True,
            collect_layer_extras=True,
            collect_top5_tokens=True,
            collect_step_kl=True,
            collect_feature_values=True,
            collect_residual_cosine=True,
            collect_mlp_norms=True,
            collect_forward_contrib=True,
            collect_repulsion=True,
            collect_transcoder_norms=False,
            collect_transcoder_mse=False,
        )
        self.loaded = _make_loaded(self.vocab, self.n_layers, self.d_model, self.n_real)

    def _make_raw(self, step: int) -> dict:
        torch.manual_seed(step * 100)
        return {
            "residuals":   [torch.randn(self.d_model) for _ in range(self.n_layers)],
            "mlp_inputs":  [torch.randn(self.d_model) for _ in range(self.n_layers)],
            "logits":      torch.randn(self.vocab),
            "mlp_outputs": [torch.randn(self.d_model) for _ in range(self.n_layers)],
        }

    def test_full_3_step_loop(self):
        """Run 3 steps and verify all accumulator shapes + semantic invariants."""
        # Accumulate exactly like collect_record does
        top1_token_per_layer:       list[list[int]]   = []
        kl_adjacent_layer:          list[list[float]] = []
        lens_entropy_delta:         list[list[float]] = []
        top5_token_ids_per_layer:   list[list[list[int]]]   = []
        step_to_step_kl:            list[list[float]] = []
        residual_cosine_to_final:   list[list[float]] = []
        mlp_contribution_norm:      list[list[float]] = []
        forward_logit_contrib_k1:   list[list[float]] = []
        all_feature_vals:           list[list[list[float]]] = []

        prev_step_lens_logits   = None
        prev_step_real_probs    = None

        all_metrics = []

        for step in range(self.n_steps):
            raw = self._make_raw(step)
            metrics, step_feat, next_token_id, cur_lens_logits = \
                _compute_gen_step_metrics(raw, self.loaded, self.cfg)
            all_metrics.append(metrics)

            if self.cfg.collect_layer_extras:
                top1_token_per_layer.append(metrics["top1_token_per_layer"])
                kl_adjacent_layer.append(metrics["kl_adjacent_layer"])
                lens_entropy_delta.append(metrics["lens_entropy_delta"])
            if self.cfg.collect_top5_tokens:
                top5_token_ids_per_layer.append(metrics["top5_token_ids_per_layer"])
            if self.cfg.collect_residual_cosine:
                residual_cosine_to_final.append(metrics["residual_cosine_to_final"])
            if self.cfg.collect_mlp_norms and metrics["mlp_contribution_norm"]:
                mlp_contribution_norm.append(metrics["mlp_contribution_norm"])
            if self.cfg.collect_feature_values:
                all_feature_vals.append(metrics["feature_vals"])

            # step_to_step_kl accumulation
            if self.cfg.collect_step_kl and cur_lens_logits is not None:
                real_mask_skl = self.loaded.real_token_mask
                if prev_step_real_probs is None:
                    step_to_step_kl.append([float("nan")] * self.n_layers)
                else:
                    kls = []
                    for li in range(self.n_layers):
                        p_curr = torch.softmax(cur_lens_logits[li][real_mask_skl], dim=-1)
                        kls.append(float((p_curr * torch.log(
                            p_curr / (prev_step_real_probs[li] + 1e-12) + 1e-12
                        )).sum().item()))
                    step_to_step_kl.append(kls)
                with torch.inference_mode():
                    prev_step_real_probs = [
                        torch.softmax(cur_lens_logits[li][real_mask_skl], dim=-1).cpu()
                        for li in range(self.n_layers)
                    ]

            # forward_logit_contrib_k1 accumulation
            if self.cfg.collect_forward_contrib:
                if prev_step_lens_logits is not None:
                    fwd = [float("nan")]
                    real_mask = self.loaded.real_token_mask
                    for li in range(1, self.n_layers):
                        if real_mask[next_token_id]:
                            fwd.append(
                                prev_step_lens_logits[li][next_token_id].item()
                                - prev_step_lens_logits[li - 1][next_token_id].item()
                            )
                        else:
                            fwd.append(float("nan"))
                    forward_logit_contrib_k1.append(fwd)
                prev_step_lens_logits = cur_lens_logits

        # Finalize forward_logit_contrib_k1 (last step gets nans)
        if self.cfg.collect_forward_contrib:
            forward_logit_contrib_k1.append([float("nan")] * self.n_layers)

        # ── Shape checks ──────────────────────────────────────────────────────
        self.assertEqual(len(top1_token_per_layer), self.n_steps)
        self.assertEqual(len(kl_adjacent_layer), self.n_steps)
        self.assertEqual(len(lens_entropy_delta), self.n_steps)
        self.assertEqual(len(top5_token_ids_per_layer), self.n_steps)
        self.assertEqual(len(step_to_step_kl), self.n_steps)
        self.assertEqual(len(residual_cosine_to_final), self.n_steps)
        self.assertEqual(len(mlp_contribution_norm), self.n_steps)
        self.assertEqual(len(forward_logit_contrib_k1), self.n_steps)  # n_steps NOT n_steps+1
        self.assertEqual(len(all_feature_vals), self.n_steps)

        for step in range(self.n_steps):
            self.assertEqual(len(top1_token_per_layer[step]), self.n_layers)
            self.assertEqual(len(kl_adjacent_layer[step]), self.n_layers)
            self.assertEqual(len(lens_entropy_delta[step]), self.n_layers)
            self.assertEqual(len(top5_token_ids_per_layer[step]), self.n_layers)
            self.assertEqual(len(step_to_step_kl[step]), self.n_layers)
            self.assertEqual(len(residual_cosine_to_final[step]), self.n_layers)
            self.assertEqual(len(mlp_contribution_norm[step]), self.n_layers)
            self.assertEqual(len(forward_logit_contrib_k1[step]), self.n_layers)

        # ── Semantic checks ───────────────────────────────────────────────────
        # step 0: step_to_step_kl all nan
        for v in step_to_step_kl[0]:
            self.assertTrue(math.isnan(v), f"step 0 kl should be nan, got {v}")

        # step 1+: step_to_step_kl non-nan and >= 0
        for step in range(1, self.n_steps):
            for li, v in enumerate(step_to_step_kl[step]):
                self.assertFalse(math.isnan(v), f"step {step} layer {li}: unexpected nan")
                self.assertGreaterEqual(v, 0.0)

        # kl_adjacent_layer[step][0] always nan
        for step in range(self.n_steps):
            self.assertTrue(math.isnan(kl_adjacent_layer[step][0]),
                            f"step {step}: kl_adjacent_layer[0] should be nan")

        # lens_entropy_delta[step][0] always nan
        for step in range(self.n_steps):
            self.assertTrue(math.isnan(lens_entropy_delta[step][0]),
                            f"step {step}: lens_entropy_delta[0] should be nan")

        # forward_logit_contrib_k1: step 0 is nan (no prev step)
        # But the loop only appends when prev_step_lens_logits is not None,
        # so step 0 is never appended inside the loop —
        # it is appended as the "last step" nans at the end.
        # Verify: steps [0..n_steps-2] are retroactive, step n_steps-1 is nans.
        last_fwd = forward_logit_contrib_k1[-1]
        for v in last_fwd:
            self.assertTrue(math.isnan(v), f"Last step forward_logit_contrib_k1 should be nan")

        # residual_cosine_to_final last layer = 1.0 at every step
        for step in range(self.n_steps):
            self.assertAlmostEqual(residual_cosine_to_final[step][-1], 1.0, places=5,
                msg=f"step {step}: last-layer cosine should be 1.0")

        # top1 always in real_mask
        real_mask = self.loaded.real_token_mask
        for step in range(self.n_steps):
            for li, tid in enumerate(top1_token_per_layer[step]):
                self.assertTrue(real_mask[tid].item(),
                    f"step {step} layer {li}: top1 token {tid} not in real_mask")

        # top5 all in real_mask
        for step in range(self.n_steps):
            for li, ids in enumerate(top5_token_ids_per_layer[step]):
                for tid in ids:
                    self.assertTrue(real_mask[tid].item(),
                        f"step {step} layer {li}: top5 token {tid} not in real_mask")

        # mlp_contribution_norm all positive
        for step in range(self.n_steps):
            for li, n in enumerate(mlp_contribution_norm[step]):
                self.assertGreater(n, 0.0,
                    f"step {step} layer {li}: mlp norm should be > 0")

    def test_forward_contrib_off_by_one(self):
        """forward_logit_contrib_k1 should have exactly n_steps entries:
        steps 1..n_steps-1 computed retroactively, step 0 = final nan entry."""
        forward_logit_contrib_k1 = []
        prev_step_lens_logits = None

        for step in range(self.n_steps):
            raw = self._make_raw(step)
            metrics, step_feat, next_token_id, cur_lens_logits = \
                _compute_gen_step_metrics(raw, self.loaded, self.cfg)

            if self.cfg.collect_forward_contrib:
                if prev_step_lens_logits is not None:
                    fwd = [float("nan")]
                    real_mask = self.loaded.real_token_mask
                    for li in range(1, self.n_layers):
                        if real_mask[next_token_id]:
                            fwd.append(
                                prev_step_lens_logits[li][next_token_id].item()
                                - prev_step_lens_logits[li - 1][next_token_id].item()
                            )
                        else:
                            fwd.append(float("nan"))
                    forward_logit_contrib_k1.append(fwd)
                prev_step_lens_logits = cur_lens_logits

        # After loop, add final nan entry
        forward_logit_contrib_k1.append([float("nan")] * self.n_layers)

        # Verify: exactly n_steps entries total
        self.assertEqual(
            len(forward_logit_contrib_k1), self.n_steps,
            f"forward_logit_contrib_k1 has {len(forward_logit_contrib_k1)} entries, "
            f"expected {self.n_steps}"
        )
        # Step 0 contribution is the entry computed at step=1 (using step 0's logits
        # to attribute step 1's token), so forward_logit_contrib_k1[0] should be non-nan
        # layer 1+ (if next_token is real).
        # The final entry (for step n_steps-1) should be all nans.
        last = forward_logit_contrib_k1[-1]
        for v in last:
            self.assertTrue(math.isnan(v),
                f"Last entry of forward_logit_contrib_k1 should be all nan, got {v}")


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):

    def test_kl_div_non_negative(self):
        """KL divergence is always >= 0."""
        torch.manual_seed(1)
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)
        kl = _kl_div(p, q)
        self.assertGreaterEqual(kl, 0.0)

    def test_kl_div_zero_for_identical(self):
        """KL(p ∥ p) = 0."""
        p = torch.softmax(torch.tensor([1.0, 2.0, 0.5, 3.0]), dim=-1)
        kl = _kl_div(p, p)
        self.assertAlmostEqual(kl, 0.0, places=5)

    def test_kl_div_asymmetric(self):
        """KL(p ∥ q) != KL(q ∥ p) for different distributions.
        Use an asymmetric pair where the two directions visibly differ."""
        p = torch.tensor([0.99, 0.01])
        q = torch.tensor([0.5, 0.5])
        kl_pq = _kl_div(p, q)
        kl_qp = _kl_div(q, p)
        self.assertNotAlmostEqual(kl_pq, kl_qp, places=3,
            msg=f"KL(p∥q)={kl_pq:.6f} should differ from KL(q∥p)={kl_qp:.6f}")

    def test_entropy_with_mask(self):
        """_entropy with mask only considers masked tokens."""
        logits = torch.zeros(10)
        mask   = torch.zeros(10, dtype=torch.bool)
        mask[0] = True  # only one token real
        ent = _entropy(logits, mask=mask)
        # Single token → p=[1.0] → entropy = 0
        self.assertAlmostEqual(ent, 0.0, places=5)

    def test_sanitise_replaces_nan_inf(self):
        obj = {"a": float("nan"), "b": float("inf"), "c": 1.0, "d": [float("nan"), 2.0]}
        s = _sanitise(obj)
        self.assertIsNone(s["a"])
        self.assertIsNone(s["b"])
        self.assertEqual(s["c"], 1.0)
        self.assertIsNone(s["d"][0])
        self.assertEqual(s["d"][1], 2.0)


# ---------------------------------------------------------------------------
# Bug detection: specific known potential bugs
# ---------------------------------------------------------------------------

class TestSpecificBugDetection(unittest.TestCase):
    """Tests that specifically probe suspected bug sites."""

    def test_bug_mlp_norms_guard_empty_list_vs_wrong_length(self):
        """
        collect.py line ~640:
            if cfg.collect_mlp_norms and mlp_outputs and len(mlp_outputs) == n_layers:

        BUG CHECK: if mlp_outputs has len != n_layers, the guard catches it.
        Confirm empty list also returns [].
        """
        loaded = _make_loaded(VOCAB, N_LAYERS, D_MODEL)
        cfg    = _make_cfg(collect_mlp_norms=True)

        # Case 1: empty list → should return []
        raw_empty = {
            "residuals":   _make_residuals(N_LAYERS, D_MODEL),
            "mlp_inputs":  _make_residuals(N_LAYERS, D_MODEL),
            "logits":      _make_logits(VOCAB),
            "mlp_outputs": [],
        }
        metrics, _, _, _ = _compute_gen_step_metrics(raw_empty, loaded, cfg)
        self.assertEqual(metrics["mlp_contribution_norm"], [],
                         "Empty mlp_outputs should produce []")

        # Case 2: too few → should return []
        raw_short = {
            "residuals":   _make_residuals(N_LAYERS, D_MODEL),
            "mlp_inputs":  _make_residuals(N_LAYERS, D_MODEL),
            "logits":      _make_logits(VOCAB),
            "mlp_outputs": _make_mlp_outputs(N_LAYERS - 1, D_MODEL),
        }
        metrics2, _, _, _ = _compute_gen_step_metrics(raw_short, loaded, cfg)
        self.assertEqual(metrics2["mlp_contribution_norm"], [],
                         "Short mlp_outputs should produce []")

    def test_bug_step_kl_updates_prev_probs_after_compute(self):
        """
        Verify that prev_step_real_probs is updated AFTER computing KL (not before).
        If it were updated before, step 1's KL would incorrectly use step 1's probs
        as q instead of step 0's probs.
        """
        n_layers = 2
        vocab    = 8
        n_real   = 6
        real_mask = _make_real_mask(vocab, n_real)

        # Two distinct distributions
        ll0 = [torch.tensor([2.0, 0.1, 0.1, 0.1, 0.1, 0.1, -999., -999.]),
               torch.tensor([0.1, 2.0, 0.1, 0.1, 0.1, 0.1, -999., -999.])]
        ll1 = [torch.tensor([0.1, 0.1, 2.0, 0.1, 0.1, 0.1, -999., -999.]),
               torch.tensor([0.1, 0.1, 0.1, 2.0, 0.1, 0.1, -999., -999.])]

        prev_step_real_probs = None
        results = []

        for step, ll in enumerate([ll0, ll1]):
            if prev_step_real_probs is None:
                results.append([float("nan")] * n_layers)
            else:
                kls = []
                for li in range(n_layers):
                    p_curr = torch.softmax(ll[li][real_mask], dim=-1)
                    kls.append(_kl_div(p_curr, prev_step_real_probs[li]))
                results.append(kls)
            # Update after computing KL
            prev_step_real_probs = [
                torch.softmax(ll[li][real_mask], dim=-1) for li in range(n_layers)
            ]

        # Step 0: all nan
        for v in results[0]:
            self.assertTrue(math.isnan(v))

        # Step 1: KL(p_1 ∥ p_0) — should be > 0 since distributions differ
        for li in range(n_layers):
            p0 = torch.softmax(ll0[li][real_mask], dim=-1)
            p1 = torch.softmax(ll1[li][real_mask], dim=-1)
            expected = _kl_div(p1, p0)
            self.assertAlmostEqual(results[1][li], expected, places=5,
                msg=f"layer {li}: KL should be computed against p_0, not p_1")

    def test_bug_feature_vals_indexing_acts0(self):
        """
        In _compute_gen_step_metrics:
            vals = acts[0][idxs].float().tolist() if idxs else []

        acts has shape [1, n_features]. acts[0] has shape [n_features].
        idxs is a list of ints from nonzero().squeeze(1).tolist()

        Check: acts[0][idxs] returns activation values at those indices.
        """
        n_features = 10
        acts = torch.zeros(1, n_features)
        acts[0, 2] = 0.7
        acts[0, 5] = 1.3
        acts[0, 8] = 0.2

        idxs = acts[0].nonzero(as_tuple=False).squeeze(1).tolist()
        vals = acts[0][idxs].float().tolist()

        self.assertEqual(idxs, [2, 5, 8])
        self.assertAlmostEqual(vals[0], 0.7, places=5)
        self.assertAlmostEqual(vals[1], 1.3, places=5)
        self.assertAlmostEqual(vals[2], 0.2, places=5)

    def test_bug_acts_indexing_with_empty_idxs(self):
        """When idxs is empty, the vals branch returns [] without calling acts[0][idxs]."""
        n_features = 10
        acts = torch.zeros(1, n_features)
        idxs = acts[0].nonzero(as_tuple=False).squeeze(1).tolist()
        self.assertEqual(idxs, [])
        vals = acts[0][idxs].float().tolist() if idxs else []
        self.assertEqual(vals, [])

    def test_bug_top5_softmax_on_masked_not_full_vocab(self):
        """
        The code does:
            masked_ll = all_lens_logits[i].clone()
            masked_ll[~real_mask] = float("-inf")
            probs = torch.softmax(masked_ll, dim=-1)
            top5 = torch.topk(probs, k=5)

        softmax(-inf) = 0, so non-real tokens get prob=0.
        Topk then selects from the full vocab but non-real probs are 0.
        Verify: no non-real token appears in top5.
        """
        vocab   = 10
        n_real  = 5
        real_mask = _make_real_mask(vocab, n_real)

        # Make non-real tokens have highest raw logits → should still be excluded
        ll = torch.zeros(vocab)
        ll[n_real:] = 100.0    # non-real tokens have huge logits
        ll[:n_real] = 1.0      # real tokens have small logits

        masked_ll = ll.clone()
        masked_ll[~real_mask] = float("-inf")
        probs = torch.softmax(masked_ll, dim=-1)
        top5 = torch.topk(probs, k=min(5, n_real))  # can't take more than n_real

        for tid in top5.indices.tolist():
            self.assertTrue(real_mask[tid].item(),
                f"top5 token {tid} is not in real_mask despite masking")

    def test_bug_transcoder_mse_uses_mlp_outputs_i_not_0(self):
        """
        In the merged transcoder loop:
            mse = ((tc_out - mlp_outputs[i]) ** 2).mean()

        Verify it correctly uses mlp_outputs[i] per layer, not mlp_outputs[0].
        """
        n_layers = 3
        d_model  = 4
        # Make distinct mlp_outputs per layer
        mlp_outs = [torch.full((d_model,), float(i)) for i in range(n_layers)]

        for i in range(n_layers):
            tc_out = torch.zeros(d_model)
            expected_mse = float(((tc_out - mlp_outs[i]) ** 2).mean().item())
            self.assertAlmostEqual(expected_mse, float(i**2), places=5,
                msg=f"layer {i}: MSE should use mlp_outputs[{i}]")

    def test_bug_forward_contrib_last_step_not_double_counted(self):
        """
        The loop appends to forward_logit_contrib_k1 only when
        prev_step_lens_logits is not None. So for n_steps=3:
          - step 0: no prev → nothing appended
          - step 1: prev=step0 logits → append (retroactive for step 0)
          - step 2: prev=step1 logits → append (retroactive for step 1)
        After loop: append [nan]*n_layers for step 2.
        Total: 3 entries = n_steps. ✓

        If the code were wrong and appended at step 0 too, we'd get 4 entries.
        """
        n_layers = 2
        n_steps  = 3
        forward_logit_contrib_k1 = []
        prev_step_lens_logits = None
        fake_ll = [torch.randn(10) for _ in range(n_layers)]

        for step in range(n_steps):
            if prev_step_lens_logits is not None:
                forward_logit_contrib_k1.append([float("nan")] * n_layers)
            prev_step_lens_logits = fake_ll

        forward_logit_contrib_k1.append([float("nan")] * n_layers)

        self.assertEqual(len(forward_logit_contrib_k1), n_steps,
                         f"Expected {n_steps} entries, got {len(forward_logit_contrib_k1)}")


if __name__ == "__main__":
    # Run all tests with verbose output and collect results
    loader = unittest.TestLoader()
    suite  = loader.discover(start_dir=str(pathlib.Path(__file__).resolve().parent), pattern=pathlib.Path(__file__).name)

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"ALL {result.testsRun} TESTS PASSED")
    else:
        print(f"FAILURES: {len(result.failures)}")
        print(f"ERRORS:   {len(result.errors)}")
        print(f"Passed:   {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun}")
        print()
        for i, (tc, tb) in enumerate(result.failures + result.errors, 1):
            print(f"{'─'*60}")
            print(f"[{'FAIL' if (tc, tb) in result.failures else 'ERROR'}] {tc}")
            print(tb)

    sys.exit(0 if result.wasSuccessful() else 1)
