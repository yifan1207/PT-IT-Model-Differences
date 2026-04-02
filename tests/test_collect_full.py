"""
Tests for eval_commitment(collect_full=True) — verifies all Groups B-G
are collected correctly with proper shapes, dtypes, and alignment.

Run: uv run pytest tests/test_collect_full.py -v
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── Fixtures ──────────────────────────────────────────────────────────────


class FakeSpec:
    """Minimal ModelSpec for testing."""
    def __init__(self, n_layers=4, d_model=32, name="test_model"):
        self.n_layers = n_layers
        self.d_model = d_model
        self.name = name
        self.multi_gpu = False
        self.is_moe = False


class FakeAdapter:
    """Adapter that works with FakeModel."""
    def __init__(self, n_layers, d_model):
        self.n_layers = n_layers
        self.d_model = d_model

    def layers(self, model):
        return model.layers

    def final_norm(self, model):
        return model.norm

    def lm_head(self, model):
        return model.lm_head

    def residual_from_output(self, output):
        return output

    def stop_token_ids(self, tokenizer):
        return {2}  # EOS


class FakeModel(nn.Module):
    """Minimal model that supports generate() + layer hooks.

    Unlike real transformers, this generates token-by-token and feeds
    only the last token through the layers (like KV-cache decoding),
    so forward hooks see shape [1, 1, d_model] and fire correctly.
    The first call (prefill) processes the full sequence and hooks
    correctly skip it (shape[1] > 1).
    """
    def __init__(self, n_layers=4, d_model=32, vocab_size=100):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.embed = nn.Embedding(vocab_size, d_model)

    def _forward_layers(self, h):
        """Forward through layers — hooks fire here."""
        for layer in self.layers:
            h = layer(h)
        return h

    def generate(self, input_ids, max_new_tokens=5, **kwargs):
        """Autoregressive generation that properly triggers hooks.

        Step 0 (prefill): full sequence [1, seq, d] → hooks see shape[1]>1, skip.
        Steps 1+: single token [1, 1, d] → hooks see shape[1]==1, fire.
        """
        eos_ids = kwargs.get("eos_token_id", [2])
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        generated = input_ids.clone()

        # Prefill: process full input (hooks skip this due to shape[1] > 1)
        h = self.embed(generated)
        h = self._forward_layers(h)
        h = self.norm(h)
        logits = self.lm_head(h)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        # Autoregressive steps: single token at a time (hooks fire)
        for step in range(1, max_new_tokens):
            if eos_ids and next_token.item() in eos_ids:
                break
            # Only process last token (simulates KV-cache decoding)
            h = self.embed(next_token)  # [1, 1, d_model]
            h = self._forward_layers(h)
            h = self.norm(h)
            logits = self.lm_head(h)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    @property
    def device(self):
        return next(self.parameters()).device


class FakeTokenizer:
    def __init__(self, vocab_size=100):
        self.pad_token_id = 0
        self.eos_token_id = 2

    def encode(self, text, return_tensors=None):
        # Fixed input: 3 tokens
        ids = [1, 5, 10]
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def fake_model():
    torch.manual_seed(42)
    return FakeModel(n_layers=4, d_model=32, vocab_size=100)


@pytest.fixture
def fake_probes():
    from src.poc.cross_model.tuned_lens import TunedLensProbe
    probes = {}
    for i in range(4):
        p = TunedLensProbe(32)
        p.eval()
        probes[i] = p
    return probes


@pytest.fixture
def fake_records():
    return [
        {"id": "test_001", "question": "What is 2+2?", "question_type": "factual"},
        {"id": "test_002", "question": "Name a color.", "question_type": "factual"},
        {"id": "test_003", "question": "Hello world.", "question_type": "continuation"},
    ]


# ── Tests ─────────────────────────────────────────────────────────────────


class TestCollectFullBasic:
    """Test that collect_full=True produces all expected NPY files."""

    def test_arrays_dir_created(self, fake_model, fake_probes, fake_records, tmpdir):
        from src.poc.cross_model.tuned_lens import eval_commitment

        spec = FakeSpec()
        adapter = FakeAdapter(4, 32)
        out_path = tmpdir / "commitment.jsonl"
        arrays_dir = tmpdir / "arrays"

        eval_commitment(
            fake_model, FakeTokenizer(), adapter, spec, fake_probes,
            fake_records, torch.device("cpu"),
            output_path=out_path,
            variant="pt",
            max_new_tokens=5,
            collect_full=True,
            arrays_dir=arrays_dir,
        )

        assert arrays_dir.exists(), "arrays_dir was not created"
        assert (arrays_dir / "step_index.jsonl").exists(), "step_index.jsonl missing"

    def test_all_npy_files_exist(self, fake_model, fake_probes, fake_records, tmpdir):
        from src.poc.cross_model.tuned_lens import eval_commitment

        spec = FakeSpec()
        adapter = FakeAdapter(4, 32)
        out_path = tmpdir / "commitment.jsonl"
        arrays_dir = tmpdir / "arrays"

        eval_commitment(
            fake_model, FakeTokenizer(), adapter, spec, fake_probes,
            fake_records, torch.device("cpu"),
            output_path=out_path, variant="pt", max_new_tokens=5,
            collect_full=True, arrays_dir=arrays_dir,
        )

        expected_files = [
            "raw_top1.npy", "tuned_top1.npy", "generated_ids.npy",
            "raw_kl_final.npy", "tuned_kl_final.npy",
            "raw_kl_adj.npy", "tuned_kl_adj.npy",
            "raw_ntprob.npy", "tuned_ntprob.npy",
            "raw_ntrank.npy", "tuned_ntrank.npy",
            "raw_entropy.npy", "tuned_entropy.npy",
            "delta_cosine.npy", "cosine_h_to_final.npy",
            "step_index.jsonl",
        ]
        for fname in expected_files:
            assert (arrays_dir / fname).exists(), f"Missing: {fname}"


class TestArrayShapes:
    """Verify shapes and dtypes of all NPY arrays."""

    @pytest.fixture(autouse=True)
    def run_eval(self, fake_model, fake_probes, fake_records, tmpdir):
        from src.poc.cross_model.tuned_lens import eval_commitment

        spec = FakeSpec(n_layers=4)
        adapter = FakeAdapter(4, 32)
        self.out_path = tmpdir / "commitment.jsonl"
        self.arrays_dir = tmpdir / "arrays"
        self.n_layers = 4
        self.n_prompts = len(fake_records)

        self.summary = eval_commitment(
            fake_model, FakeTokenizer(), adapter, spec, fake_probes,
            fake_records, torch.device("cpu"),
            output_path=self.out_path, variant="pt", max_new_tokens=5,
            collect_full=True, collect_top5=True, top5_max_prompts=10,
            arrays_dir=self.arrays_dir,
        )

        # Count total steps from step_index
        with open(self.arrays_dir / "step_index.jsonl") as f:
            records = [json.loads(l) for l in f if l.strip()]
        self.total_steps = sum(r["n_steps"] for r in records)
        self.step_records = records

    def test_step_count_positive(self):
        assert self.total_steps > 0, "No generation steps recorded"

    def test_step_index_covers_all_prompts(self):
        assert len(self.step_records) == self.n_prompts

    def test_step_index_contiguous(self):
        """Step ranges should be contiguous with no gaps."""
        for i, rec in enumerate(self.step_records):
            if i == 0:
                assert rec["start_step"] == 0
            else:
                assert rec["start_step"] == self.step_records[i - 1]["end_step"]

    def test_raw_top1_shape_dtype(self):
        arr = np.load(self.arrays_dir / "raw_top1.npy")
        assert arr.shape == (self.total_steps, self.n_layers)
        assert arr.dtype == np.int32

    def test_tuned_top1_shape_dtype(self):
        arr = np.load(self.arrays_dir / "tuned_top1.npy")
        assert arr.shape == (self.total_steps, self.n_layers)
        assert arr.dtype == np.int32

    def test_generated_ids_shape_dtype(self):
        arr = np.load(self.arrays_dir / "generated_ids.npy")
        assert arr.shape == (self.total_steps,)
        assert arr.dtype == np.int32

    def test_kl_final_shape_dtype(self):
        for prefix in ["raw", "tuned"]:
            arr = np.load(self.arrays_dir / f"{prefix}_kl_final.npy")
            assert arr.shape == (self.total_steps, self.n_layers), f"{prefix}_kl_final wrong shape"
            assert arr.dtype == np.float16

    def test_kl_adj_shape_dtype(self):
        for prefix in ["raw", "tuned"]:
            arr = np.load(self.arrays_dir / f"{prefix}_kl_adj.npy")
            assert arr.shape == (self.total_steps, self.n_layers)
            assert arr.dtype == np.float16

    def test_kl_adj_layer0_is_nan(self):
        """Adjacent KL at layer 0 should be NaN (no predecessor)."""
        for prefix in ["raw", "tuned"]:
            arr = np.load(self.arrays_dir / f"{prefix}_kl_adj.npy")
            assert np.all(np.isnan(arr[:, 0])), f"{prefix}_kl_adj layer 0 should be NaN"

    def test_ntprob_shape_dtype(self):
        for prefix in ["raw", "tuned"]:
            arr = np.load(self.arrays_dir / f"{prefix}_ntprob.npy")
            assert arr.shape == (self.total_steps, self.n_layers)
            assert arr.dtype == np.float16

    def test_ntrank_shape_dtype(self):
        for prefix in ["raw", "tuned"]:
            arr = np.load(self.arrays_dir / f"{prefix}_ntrank.npy")
            assert arr.shape == (self.total_steps, self.n_layers)
            assert arr.dtype == np.int16

    def test_entropy_shape_dtype(self):
        for prefix in ["raw", "tuned"]:
            arr = np.load(self.arrays_dir / f"{prefix}_entropy.npy")
            assert arr.shape == (self.total_steps, self.n_layers)
            assert arr.dtype == np.float16

    def test_delta_cosine_shape_dtype(self):
        arr = np.load(self.arrays_dir / "delta_cosine.npy")
        assert arr.shape == (self.total_steps, self.n_layers)
        assert arr.dtype == np.float16

    def test_delta_cosine_layer0_is_nan(self):
        """δ-cosine at layer 0 should be NaN (no predecessor)."""
        arr = np.load(self.arrays_dir / "delta_cosine.npy")
        assert np.all(np.isnan(arr[:, 0]))

    def test_cosine_final_shape_dtype(self):
        arr = np.load(self.arrays_dir / "cosine_h_to_final.npy")
        assert arr.shape == (self.total_steps, self.n_layers)
        assert arr.dtype == np.float16

    def test_top5_files_exist(self):
        for prefix in ["raw", "tuned"]:
            for suffix in ["ids", "probs"]:
                path = self.arrays_dir / f"{prefix}_top5_{suffix}.npy"
                assert path.exists(), f"Missing: {path.name}"
        assert (self.arrays_dir / "top5_step_index.jsonl").exists()

    def test_top5_shape(self):
        arr = np.load(self.arrays_dir / "raw_top5_ids.npy")
        assert arr.ndim == 3
        assert arr.shape[1] == self.n_layers
        assert arr.shape[2] == 5
        assert arr.dtype == np.int32


class TestCollectFullSemantics:
    """Verify semantic correctness of collected data."""

    @pytest.fixture(autouse=True)
    def run_eval(self, fake_model, fake_probes, fake_records, tmpdir):
        from src.poc.cross_model.tuned_lens import eval_commitment

        spec = FakeSpec(n_layers=4)
        adapter = FakeAdapter(4, 32)
        self.out_path = tmpdir / "commitment.jsonl"
        self.arrays_dir = tmpdir / "arrays"

        eval_commitment(
            fake_model, FakeTokenizer(), adapter, spec, fake_probes,
            fake_records, torch.device("cpu"),
            output_path=self.out_path, variant="pt", max_new_tokens=5,
            collect_full=True, arrays_dir=self.arrays_dir,
        )

    def test_generated_ids_match_raw_top1_final_layer(self):
        """Generated token should match raw top-1 at final layer."""
        gen_ids = np.load(self.arrays_dir / "generated_ids.npy")
        raw_top1 = np.load(self.arrays_dir / "raw_top1.npy")
        # Generated token = argmax of final layer raw logits
        np.testing.assert_array_equal(
            gen_ids, raw_top1[:, -1],
            err_msg="generated_ids should match raw_top1 at final layer",
        )

    def test_cosine_final_layer_is_one(self):
        """cos(h_L, h_L) should be ~1.0 at the final layer."""
        cos = np.load(self.arrays_dir / "cosine_h_to_final.npy")
        np.testing.assert_allclose(
            cos[:, -1].astype(np.float32), 1.0, atol=0.01,
            err_msg="cosine_h_to_final at final layer should be ~1.0",
        )

    def test_kl_final_layer_is_zero(self):
        """KL(final ‖ final) should be ~0 at the final layer."""
        for prefix in ["raw", "tuned"]:
            kl = np.load(self.arrays_dir / f"{prefix}_kl_final.npy").astype(np.float32)
            np.testing.assert_allclose(
                kl[:, -1], 0.0, atol=0.01,
                err_msg=f"{prefix}_kl_final at final layer should be ~0",
            )

    def test_raw_ntrank_final_layer_is_one(self):
        """At final layer, generated token should be rank 1 in raw logits."""
        rank = np.load(self.arrays_dir / "raw_ntrank.npy")
        np.testing.assert_array_equal(
            rank[:, -1], 1,
            err_msg="raw_ntrank at final layer should be 1 (top prediction)",
        )

    def test_entropy_non_negative(self):
        """Entropy should be non-negative."""
        for prefix in ["raw", "tuned"]:
            ent = np.load(self.arrays_dir / f"{prefix}_entropy.npy").astype(np.float32)
            # Allow for float16 precision issues
            assert np.all(ent >= -0.01), f"{prefix} entropy has negative values"

    def test_ntprob_in_01(self):
        """Next-token probability should be in [0, 1]."""
        for prefix in ["raw", "tuned"]:
            prob = np.load(self.arrays_dir / f"{prefix}_ntprob.npy").astype(np.float32)
            assert np.all(prob >= -0.01) and np.all(prob <= 1.01), \
                f"{prefix}_ntprob out of [0,1] range"

    def test_jsonl_and_arrays_same_prompt_count(self):
        """JSONL commitment records should match step_index prompt count."""
        with open(self.out_path) as f:
            jsonl_records = [json.loads(l) for l in f if l.strip()]
        with open(self.arrays_dir / "step_index.jsonl") as f:
            step_records = [json.loads(l) for l in f if l.strip()]
        assert len(jsonl_records) == len(step_records)

    def test_jsonl_and_arrays_same_prompt_ids(self):
        """JSONL and step_index should have matching prompt IDs in same order."""
        with open(self.out_path) as f:
            jsonl_ids = [json.loads(l)["prompt_id"] for l in f if l.strip()]
        with open(self.arrays_dir / "step_index.jsonl") as f:
            step_ids = [json.loads(l)["prompt_id"] for l in f if l.strip()]
        assert jsonl_ids == step_ids

    def test_jsonl_step_counts_match_arrays(self):
        """n_steps in JSONL should match step_index ranges."""
        with open(self.out_path) as f:
            jsonl_steps = [json.loads(l)["n_steps"] for l in f if l.strip()]
        with open(self.arrays_dir / "step_index.jsonl") as f:
            index_steps = [json.loads(l)["n_steps"] for l in f if l.strip()]
        assert jsonl_steps == index_steps


class TestCollectFullDisabled:
    """Verify collect_full=False doesn't produce arrays."""

    def test_no_arrays_without_flag(self, fake_model, fake_probes, fake_records, tmpdir):
        from src.poc.cross_model.tuned_lens import eval_commitment

        spec = FakeSpec()
        adapter = FakeAdapter(4, 32)
        out_path = tmpdir / "commitment.jsonl"
        arrays_dir = tmpdir / "arrays"

        eval_commitment(
            fake_model, FakeTokenizer(), adapter, spec, fake_probes,
            fake_records, torch.device("cpu"),
            output_path=out_path, variant="pt", max_new_tokens=5,
            collect_full=False,
        )

        assert not arrays_dir.exists(), "arrays_dir should not exist when collect_full=False"


class TestCollectFullOverwritesOnRerun:
    """Verify collect_full=True clears old data and starts fresh."""

    def test_old_jsonl_cleared(self, fake_model, fake_probes, fake_records, tmpdir):
        from src.poc.cross_model.tuned_lens import eval_commitment

        spec = FakeSpec()
        adapter = FakeAdapter(4, 32)
        out_path = tmpdir / "commitment.jsonl"

        # First run: creates JSONL
        eval_commitment(
            fake_model, FakeTokenizer(), adapter, spec, fake_probes,
            fake_records, torch.device("cpu"),
            output_path=out_path, variant="pt", max_new_tokens=5,
        )
        with open(out_path) as f:
            first_count = sum(1 for l in f if l.strip())

        # Second run with collect_full: should overwrite, not append
        eval_commitment(
            fake_model, FakeTokenizer(), adapter, spec, fake_probes,
            fake_records, torch.device("cpu"),
            output_path=out_path, variant="pt", max_new_tokens=5,
            collect_full=True, arrays_dir=tmpdir / "arrays",
        )
        with open(out_path) as f:
            second_count = sum(1 for l in f if l.strip())

        # Should have same count (overwritten, not doubled)
        assert second_count == first_count, \
            f"collect_full should overwrite JSONL, got {second_count} vs {first_count}"


class TestTop5ConditionalCollection:
    """Verify top-5 is only collected for first N prompts."""

    def test_top5_limited_to_max_prompts(self, fake_model, fake_probes, tmpdir):
        from src.poc.cross_model.tuned_lens import eval_commitment

        # 5 records, but top5_max_prompts=2
        records = [{"id": f"p{i}", "question": f"Q{i}?"} for i in range(5)]
        spec = FakeSpec()
        adapter = FakeAdapter(4, 32)
        arrays_dir = tmpdir / "arrays"

        eval_commitment(
            fake_model, FakeTokenizer(), adapter, spec, fake_probes,
            records, torch.device("cpu"),
            output_path=tmpdir / "c.jsonl", variant="pt", max_new_tokens=3,
            collect_full=True, collect_top5=True, top5_max_prompts=2,
            arrays_dir=arrays_dir,
        )

        with open(arrays_dir / "top5_step_index.jsonl") as f:
            top5_records = [json.loads(l) for l in f if l.strip()]

        assert len(top5_records) == 2, \
            f"Expected top5 for 2 prompts, got {len(top5_records)}"

    def test_no_top5_without_flag(self, fake_model, fake_probes, fake_records, tmpdir):
        from src.poc.cross_model.tuned_lens import eval_commitment

        spec = FakeSpec()
        adapter = FakeAdapter(4, 32)
        arrays_dir = tmpdir / "arrays"

        eval_commitment(
            fake_model, FakeTokenizer(), adapter, spec, fake_probes,
            fake_records, torch.device("cpu"),
            output_path=tmpdir / "c.jsonl", variant="pt", max_new_tokens=3,
            collect_full=True, collect_top5=False,
            arrays_dir=arrays_dir,
        )

        assert not (arrays_dir / "raw_top5_ids.npy").exists(), \
            "top5 files should not exist when collect_top5=False"
