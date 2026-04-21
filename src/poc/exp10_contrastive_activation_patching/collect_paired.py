"""
Exp10 — Phase 1: Forced-Decoding Paired Data Collection.

For each prompt:
  1. IT generates naturally (greedy, with chat template for IT).
  2. Both IT and PT process the same token sequence (forced decoding).
  3. At every (layer ℓ, generated-position t) we compute:
       Δh_ℓ(t)  = h_it_ℓ(t) - h_pt_ℓ(t)     (activation difference)
       KL_IT(ℓ) = KL(logit_lens(h_ℓ^IT) ‖ logit_lens(h_final^IT))
       KL_PT(ℓ) = KL(logit_lens(h_ℓ^PT) ‖ logit_lens(h_final^PT))
       Δkl_ℓ(t) = KL_IT(ℓ,t) - KL_PT(ℓ,t)   (per-layer KL excess)

  The regression target is Δkl_ℓ — continuous, threshold-free, and
  layer-specific. Positive Δkl means IT is further from its final
  prediction at this layer than PT.

Memory optimisation: Δh is NOT saved to disk.  Instead we stream-accumulate
ridge-regression sufficient statistics (XᵀX, Xᵀy) for Phase 2, and save a
small PCA subsample (≤ pca_subsample_tokens per layer).

Also computes the KL-gradient direction (bonus Approach 1) per layer.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.poc.cross_model.config import get_spec, model_id_for_variant, ModelSpec
from src.poc.cross_model.adapters import get_adapter, ModelAdapter
from src.poc.cross_model.utils import (
    load_model_and_tokenizer,
    load_dataset,
    get_raw_prompt,
    read_done_ids,
)

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

KL_THRESHOLD = 0.1  # nats — commitment threshold (for CG summary stat)
PCA_SUBSAMPLE_TOKENS = 5000  # max tokens stored per layer for PCA diagnostic
CHECKPOINT_EVERY = 50  # commit accumulators to disk every N prompts
WINSOR_LO = 0.05  # 5th percentile — winsorize Δkl to mitigate heavy tails (plan §9)
WINSOR_HI = 0.95  # 95th percentile


# ── Continuous commitment ──────────────────────────────────────────────────────

def commitment_continuous(kl_array: list[float] | np.ndarray, threshold: float = KL_THRESHOLD) -> float:
    """First-crossing commitment with linear interpolation.

    Returns a continuous layer index.  If KL never drops below threshold,
    returns n_layers - 1 (last layer).
    """
    n = len(kl_array)
    for i in range(n):
        if kl_array[i] < threshold:
            if i == 0:
                return 0.0
            # Interpolate between i-1 (above threshold) and i (below)
            kl_above = kl_array[i - 1]
            kl_below = kl_array[i]
            denom = kl_above - kl_below
            if denom < 1e-12:
                return float(i)
            frac = (kl_above - threshold) / denom
            return (i - 1) + frac
    return float(n - 1)


# ── Ridge-regression accumulators ──────────────────────────────────────────────

@dataclass
class LayerAccumulator:
    """Streaming sufficient statistics for ridge regression at one layer.

    For layer ℓ, the target y = Δkl_ℓ (per-layer KL excess: KL_IT(ℓ) - KL_PT(ℓ)).
    Each layer gets its own target, unlike the old Δc which was shared across layers.

    Maintains:
        XtX   [d, d]   — Σ Δh Δhᵀ
        Xty   [d]      — Σ Δh · Δkl_ℓ
        Xsum  [d]      — Σ Δh        (for centering)
        ysum  float     — Σ Δkl_ℓ
        yty   float     — Σ Δkl_ℓ²
        n     int       — token count
    """
    d_model: int
    XtX: torch.Tensor = field(init=False)
    Xty: torch.Tensor = field(init=False)
    Xsum: torch.Tensor = field(init=False)
    ysum: float = 0.0
    yty: float = 0.0
    n: int = 0

    def __post_init__(self):
        d = self.d_model
        self.XtX = torch.zeros(d, d, dtype=torch.float64)
        self.Xty = torch.zeros(d, dtype=torch.float64)
        self.Xsum = torch.zeros(d, dtype=torch.float64)

    def update(self, delta_h: torch.Tensor, delta_c: torch.Tensor):
        """Update accumulators with a batch of (Δh, Δc) pairs.

        Args:
            delta_h: [n_tokens, d_model] float32
            delta_c: [n_tokens] float32
        """
        dh = delta_h.double()
        dc = delta_c.double()
        self.XtX += dh.T @ dh
        self.Xty += dh.T @ dc
        self.Xsum += dh.sum(0)
        self.ysum += dc.sum().item()
        self.yty += (dc * dc).sum().item()
        self.n += dh.shape[0]

    def save(self, out_dir: Path, layer_idx: int):
        """Save accumulators to disk as NPY files."""
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"layer_{layer_idx}"
        np.save(out_dir / f"XtX_{prefix}.npy", self.XtX.numpy())
        np.save(out_dir / f"Xty_{prefix}.npy", self.Xty.numpy())
        np.save(out_dir / f"Xsum_{prefix}.npy", self.Xsum.numpy())
        np.save(out_dir / f"scalars_{prefix}.npy",
                np.array([self.ysum, self.yty, self.n], dtype=np.float64))

    @classmethod
    def load(cls, out_dir: Path, layer_idx: int, d_model: int) -> "LayerAccumulator":
        """Load accumulators from disk."""
        prefix = f"layer_{layer_idx}"
        acc = cls(d_model=d_model)
        xtx_path = out_dir / f"XtX_{prefix}.npy"
        if xtx_path.exists():
            acc.XtX = torch.from_numpy(np.load(xtx_path))
            acc.Xty = torch.from_numpy(np.load(out_dir / f"Xty_{prefix}.npy"))
            acc.Xsum = torch.from_numpy(np.load(out_dir / f"Xsum_{prefix}.npy"))
            scalars = np.load(out_dir / f"scalars_{prefix}.npy")
            acc.ysum, acc.yty, acc.n = float(scalars[0]), float(scalars[1]), int(scalars[2])
        return acc


# ── KL computation helpers ─────────────────────────────────────────────────────

def _compute_kl_to_final_raw(
    h_all_layers: torch.Tensor,   # [n_layers, seq_len, d_model]
    h_final: torch.Tensor,        # [seq_len, d_model]
    final_norm: nn.Module,
    W_U: torch.Tensor,            # [d_model, vocab_size]
) -> torch.Tensor:
    """Raw logit-lens KL(layer ℓ ‖ final) for each layer and position.

    Returns: [n_layers, seq_len] float32 (non-negative).
    """
    # Final-layer reference distribution
    with torch.no_grad():
        normed_final = final_norm(h_final.unsqueeze(0)).squeeze(0)  # [seq, d]
        logits_final = normed_final.float() @ W_U  # [seq, vocab]
        log_p_final = F.log_softmax(logits_final, dim=-1)  # [seq, vocab]

    n_layers = h_all_layers.shape[0]
    kl_list = []
    for li in range(n_layers):
        with torch.no_grad():
            normed = final_norm(h_all_layers[li].unsqueeze(0)).squeeze(0)
            logits = normed.float() @ W_U
            log_q = F.log_softmax(logits, dim=-1)
            # KL(q ‖ p) = Σ p * (log p - log q) — but we use F.kl_div convention:
            # F.kl_div(input=log_q, target=log_p, log_target=True) = Σ p * (log p - log q)
            kl = F.kl_div(log_q, log_p_final, reduction="none", log_target=True).sum(-1)
            kl_list.append(kl.clamp(min=0.0))
    return torch.stack(kl_list)  # [n_layers, seq_len]


def _compute_kl_to_final_tuned(
    h_all_layers: torch.Tensor,   # [n_layers, seq_len, d_model]
    h_final: torch.Tensor,        # [seq_len, d_model]
    final_norm: nn.Module,
    W_U: torch.Tensor,
    probes: dict[int, nn.Module],
    model_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Tuned logit-lens KL(layer ℓ ‖ final) for each layer and position.

    Uses tuned probes where available, falls back to raw for missing layers.
    Returns: [n_layers, seq_len] float32 (non-negative).
    """
    with torch.no_grad():
        normed_final = final_norm(h_final.unsqueeze(0)).squeeze(0)
        logits_final = normed_final.float() @ W_U
        log_p_final = F.log_softmax(logits_final, dim=-1)

    n_layers = h_all_layers.shape[0]
    kl_list = []
    for li in range(n_layers):
        with torch.no_grad():
            h_li = h_all_layers[li]
            if li in probes:
                h_li = probes[li](h_li.float().unsqueeze(0)).squeeze(0).to(model_dtype)
            normed = final_norm(h_li.unsqueeze(0)).squeeze(0)
            logits = normed.float() @ W_U
            log_q = F.log_softmax(logits, dim=-1)
            kl = F.kl_div(log_q, log_p_final, reduction="none", log_target=True).sum(-1)
            kl_list.append(kl.clamp(min=0.0))
    return torch.stack(kl_list)


# ── KL gradient direction (Approach 1 bonus) ──────────────────────────────────

def _compute_kl_gradient_direction(
    h_all_layers: torch.Tensor,   # [n_layers, seq_len, d_model]
    h_final: torch.Tensor,        # [seq_len, d_model]
    final_norm: nn.Module,
    W_U: torch.Tensor,
    grad_accum: dict[int, torch.Tensor],   # {layer: running sum of gradients}
    grad_count: dict[int, int],
):
    """Accumulate KL gradient w.r.t. h_ℓ for each layer.

    The gradient of KL(logit_lens(h_ℓ) ‖ p_final) w.r.t. h_ℓ tells us:
    "what direction in activation space increases KL (delays commitment)?"

    We accumulate the mean gradient across all tokens, then normalize at the end.
    """
    n_layers = h_all_layers.shape[0]

    # Final-layer reference (detached — we don't backprop through this)
    with torch.no_grad():
        normed_final = final_norm(h_final.unsqueeze(0)).squeeze(0)
        logits_final = normed_final.float() @ W_U
        log_p_final = F.log_softmax(logits_final, dim=-1).detach()

    for li in range(n_layers):
        # Clone and enable grad — we backprop through LayerNorm + W_U only
        h_li = h_all_layers[li].detach().float().requires_grad_(True)  # [seq, d]
        normed = final_norm(h_li.unsqueeze(0)).squeeze(0)  # [seq, d]
        logits = normed @ W_U.float()  # [seq, vocab]
        log_q = F.log_softmax(logits, dim=-1)
        # Sum KL across vocab and tokens
        kl = F.kl_div(log_q, log_p_final, reduction="sum", log_target=True)
        kl.backward()
        g = h_li.grad  # [seq, d] — direction that increases KL
        if g is not None:
            if li not in grad_accum:
                grad_accum[li] = torch.zeros(g.shape[1], dtype=torch.float64)
                grad_count[li] = 0
            grad_accum[li] += g.sum(0).double().cpu()
            grad_count[li] += g.shape[0]


# ── Forward with hooks (forced decoding) ──────────────────────────────────────

@torch.no_grad()
def _forward_capture_all_layers(
    model: nn.Module,
    adapter: ModelAdapter,
    input_ids: torch.Tensor,  # [1, seq_len]
    n_layers: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Single forward pass, captures residual at ALL layers for ALL positions.

    Returns: {layer_idx: [seq_len, d_model]} on GPU.
    """
    layer_modules = adapter.layers(model)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inp, output):
            h = adapter.residual_from_output(output)
            # h: [1, seq_len, d_model]
            captured[layer_idx] = h[0].detach()  # [seq_len, d_model], keep on GPU
        return hook

    handles = [
        layer_modules[i].register_forward_hook(make_hook(i))
        for i in range(n_layers)
    ]
    try:
        model(input_ids)
    finally:
        for h in handles:
            h.remove()

    return captured


@torch.no_grad()
def _forward_capture_all_layers_batched(
    model: nn.Module,
    adapter: ModelAdapter,
    input_ids: torch.Tensor,  # [B, seq_len]
    n_layers: int,
) -> dict[int, torch.Tensor]:
    """Batched forward pass, captures residual at ALL layers for ALL positions.

    Returns: {layer_idx: [B, seq_len, d_model]} on GPU.
    Unlike _forward_capture_all_layers, preserves the batch dimension.
    """
    layer_modules = adapter.layers(model)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inp, output):
            h = adapter.residual_from_output(output)
            # h: [B, seq_len, d_model]
            captured[layer_idx] = h.detach()
        return hook

    handles = [
        layer_modules[i].register_forward_hook(make_hook(i))
        for i in range(n_layers)
    ]
    try:
        model(input_ids)
    finally:
        for h in handles:
            h.remove()

    return captured


# ── Main collection function ──────────────────────────────────────────────────

def collect_paired_data(
    model_name: str,
    device: torch.device | str,
    output_dir: str | Path,
    *,
    n_prompts: int = 600,
    max_gen_tokens: int = 128,
    min_gen_tokens: int = 0,
    shard_id: int = 0,
    n_shards: int = 1,
    dataset_path: str | Path = "data/eval_dataset_v2.jsonl",
    tuned_lens_dir: str | Path | None = None,
    pca_subsample_tokens: int = PCA_SUBSAMPLE_TOKENS,
    checkpoint_every: int = CHECKPOINT_EVERY,
    kl_threshold: float = KL_THRESHOLD,
    compute_kl_gradient: bool = True,
):
    """Collect forced-decoding paired data for one model.

    Args:
        model_name: Key in MODEL_REGISTRY (e.g. "gemma3_4b").
        device: GPU device.
        output_dir: Where to write results.
        n_prompts: Number of prompts to process.
        max_gen_tokens: Max tokens IT generates per prompt.
        min_gen_tokens: Min tokens IT must generate (suppresses EOS until reached).
        dataset_path: Path to eval_dataset_v2.jsonl.
        tuned_lens_dir: Root dir containing tuned lens probes. Expected structure:
            {tuned_lens_dir}/{model_name}/tuned_lens/{variant}/probe_layer_*.pt
            If None or probes not found, uses raw logit lens only.
        pca_subsample_tokens: Max tokens to save per layer for PCA diagnostic.
        checkpoint_every: Save accumulators every N prompts.
        kl_threshold: KL threshold for commitment (nats).
        compute_kl_gradient: Whether to compute KL gradient directions.
    """
    device = torch.device(device) if isinstance(device, str) else device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    acc_dir = output_dir / "accumulators"
    acc_dir.mkdir(parents=True, exist_ok=True)
    pca_dir = output_dir / "pca_subsample"
    pca_dir.mkdir(parents=True, exist_ok=True)

    spec = get_spec(model_name)
    adapter = get_adapter(model_name)
    n_layers = spec.n_layers
    d_model = spec.d_model
    is_moe = spec.is_moe

    # Adjust max_gen_tokens for MoE models
    if is_moe and max_gen_tokens > 64:
        log.info("Reducing max_gen_tokens to 64 for MoE model %s", model_name)
        max_gen_tokens = 64

    # ── Load models ───────────────────────────────────────────────────────────
    log.info("Loading IT model: %s", spec.it_id)
    model_it, tokenizer_it = load_model_and_tokenizer(
        spec.it_id, device, eager_attn=is_moe,
    )
    log.info("Loading PT model: %s", spec.pt_id)
    model_pt, tokenizer_pt = load_model_and_tokenizer(
        spec.pt_id, device, eager_attn=is_moe,
    )

    # Tokenizer compatibility: IT and PT must share the same vocabulary so that
    # token-level Δh and Δkl comparisons are meaningful.
    assert tokenizer_it.vocab_size == tokenizer_pt.vocab_size, (
        f"Tokenizer vocab mismatch: IT ({spec.it_id}) has {tokenizer_it.vocab_size}, "
        f"PT ({spec.pt_id}) has {tokenizer_pt.vocab_size}"
    )

    # ── Load tuned lens probes (if available) ─────────────────────────────────
    from src.poc.cross_model.tuned_lens import TunedLensProbe, _load_probes

    probes_it: dict[int, TunedLensProbe] = {}
    probes_pt: dict[int, TunedLensProbe] = {}
    use_tuned = False

    if model_name == "gemma3_4b":
        log.info("Gemma: using raw logit lens only (tuned probes did not converge)")
    elif tuned_lens_dir is not None:
        tl_dir = Path(tuned_lens_dir)
        # Support both layouts:
        #   v2: {tuned_lens_dir}/{model}/tuned_lens/{variant}/probe_layer_*.pt
        #   v3: {tuned_lens_dir}/{model}/{variant}/probe_layer_*.pt
        it_probe_dir = tl_dir / model_name / "tuned_lens" / "it"
        pt_probe_dir = tl_dir / model_name / "tuned_lens" / "pt"
        if not it_probe_dir.exists():
            it_probe_dir = tl_dir / model_name / "it"
            pt_probe_dir = tl_dir / model_name / "pt"
        if it_probe_dir.exists() and pt_probe_dir.exists():
            probes_it = _load_probes(it_probe_dir, d_model, device)
            probes_pt = _load_probes(pt_probe_dir, d_model, device)
            if probes_it and probes_pt:
                use_tuned = True
                log.info("Loaded tuned lens probes: IT=%d layers, PT=%d layers",
                         len(probes_it), len(probes_pt))
            else:
                log.warning("Probe dirs exist but no probes found; using raw logit lens")
        else:
            log.warning("Tuned lens dirs not found at %s; using raw logit lens", tl_dir)

    # ── Prepare logit lens components ─────────────────────────────────────────
    final_norm_it = adapter.final_norm(model_it)
    final_norm_pt = adapter.final_norm(model_pt)
    W_U_it = adapter.lm_head(model_it).weight.T.float()  # [d_model, vocab]
    W_U_pt = adapter.lm_head(model_pt).weight.T.float()

    # ── Load dataset ──────────────────────────────────────────────────────────
    records = load_dataset(dataset_path, n_examples=n_prompts)
    if n_shards > 1:
        records = records[shard_id::n_shards]
        log.info("Shard %d/%d: %d prompts from %s", shard_id, n_shards, len(records), dataset_path)
    else:
        log.info("Loaded %d prompts from %s", len(records), dataset_path)

    # ── Resume support ────────────────────────────────────────────────────────
    commitments_path = output_dir / "commitments.jsonl"
    done_ids = read_done_ids(commitments_path)
    log.info("Resume: %d prompts already done", len(done_ids))

    # ── Load or init accumulators ─────────────────────────────────────────────
    accumulators: dict[int, LayerAccumulator] = {}
    # Check accumulator version sentinel (prevent mixing old Δc with new Δkl)
    sentinel_path = acc_dir / "target_version.txt"
    if sentinel_path.exists():
        existing_target = sentinel_path.read_text().strip()
        if existing_target != "delta_kl":
            raise RuntimeError(
                f"Accumulators in {acc_dir} were built with target='{existing_target}'. "
                "Cannot resume — delete the directory and start fresh."
            )
    for li in range(n_layers):
        accumulators[li] = LayerAccumulator.load(acc_dir, li, d_model)
    # Write sentinel for future resume checks
    sentinel_path.write_text("delta_kl\n")

    # ── PCA subsample buffers ─────────────────────────────────────────────────
    # Per-layer Δh and per-layer Δkl for the first pca_subsample_tokens tokens
    pca_dh: dict[int, list[torch.Tensor]] = {li: [] for li in range(n_layers)}
    pca_dkl: list[torch.Tensor] = []  # list of [n_take, n_layers] tensors
    pca_token_count = 0
    # If resuming, load existing PCA subsample
    pca_dkl_path = pca_dir / "delta_kl.npy"
    if pca_dkl_path.exists():
        existing_dkl = np.load(pca_dkl_path)  # [n_tokens, n_layers]
        pca_dkl = [torch.from_numpy(existing_dkl)]
        pca_token_count = existing_dkl.shape[0]
        for li in range(n_layers):
            p = pca_dir / f"delta_h_layer_{li}.npy"
            if p.exists():
                pca_dh[li] = [torch.from_numpy(np.load(p))]
        log.info("Resume: loaded PCA subsample with %d tokens", pca_token_count)

    # ── KL gradient accumulators ──────────────────────────────────────────────
    grad_accum: dict[int, torch.Tensor] = {}
    grad_count: dict[int, int] = {}

    # ── Stop tokens for IT generation ─────────────────────────────────────────
    stop_ids = list(adapter.stop_token_ids(tokenizer_it))

    # ── Main loop ─────────────────────────────────────────────────────────────
    t0 = time.time()
    n_processed = 0
    total_tokens = 0

    with open(commitments_path, "a") as f_commit:
        for ri, record in enumerate(records):
            prompt_id = record.get("id", f"prompt_{ri}")
            if prompt_id in done_ids:
                continue

            raw_prompt = get_raw_prompt(record)
            if not raw_prompt.strip():
                continue

            # Step 1: IT generates naturally (greedy, with chat template for IT)
            it_prompt = adapter.apply_template(tokenizer_it, raw_prompt, is_it=True)
            prompt_ids = tokenizer_it.encode(it_prompt, return_tensors="pt").to(device)
            n_prompt_tokens = prompt_ids.shape[1]

            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=max_gen_tokens,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=stop_ids or None,
                    pad_token_id=tokenizer_it.pad_token_id or tokenizer_it.eos_token_id,
                )
                if min_gen_tokens > 0:
                    gen_kwargs["min_new_tokens"] = min_gen_tokens
                gen_output = model_it.generate(prompt_ids, **gen_kwargs)
            # gen_output: [1, prompt_len + gen_len]
            n_gen_tokens = gen_output.shape[1] - n_prompt_tokens
            if n_gen_tokens < 2:
                continue

            # Step 2: Forced-decode both models on the same sequence
            full_ids = gen_output  # [1, total_len]

            captured_it = _forward_capture_all_layers(
                model_it, adapter, full_ids, n_layers, device,
            )
            captured_pt = _forward_capture_all_layers(
                model_pt, adapter, full_ids, n_layers, device,
            )

            # Step 3: Extract ONLY generated positions (skip prompt + BOS)
            # BOS and all prompt tokens are excluded — only IT-generated tokens
            # where both models process identical input are used for Δh and Δc.
            gen_start = n_prompt_tokens
            gen_end = n_prompt_tokens + n_gen_tokens

            # Stack layer activations: [n_layers, n_gen_tokens, d_model]
            h_it_stack = torch.stack([captured_it[li][gen_start:gen_end] for li in range(n_layers)])
            h_pt_stack = torch.stack([captured_pt[li][gen_start:gen_end] for li in range(n_layers)])
            h_it_final = captured_it[n_layers - 1][gen_start:gen_end]  # [n_gen, d]
            h_pt_final = captured_pt[n_layers - 1][gen_start:gen_end]

            # Step 4: Compute KL-to-final
            if use_tuned:
                kl_it = _compute_kl_to_final_tuned(
                    h_it_stack, h_it_final, final_norm_it, W_U_it, probes_it,
                )
                kl_pt = _compute_kl_to_final_tuned(
                    h_pt_stack, h_pt_final, final_norm_pt, W_U_pt, probes_pt,
                )
            else:
                kl_it = _compute_kl_to_final_raw(
                    h_it_stack, h_it_final, final_norm_it, W_U_it,
                )
                kl_pt = _compute_kl_to_final_raw(
                    h_pt_stack, h_pt_final, final_norm_pt, W_U_pt,
                )
            # kl_it, kl_pt: [n_layers, n_gen_tokens]

            # Step 5: Per-layer KL excess (the regression target)
            # delta_kl[ℓ, t] = KL_IT(ℓ, t) - KL_PT(ℓ, t)
            delta_kl = (kl_it - kl_pt).float().cpu()  # [n_layers, n_gen_tokens]

            # Keep commitment_continuous for CG summary stat (backward-compat)
            c_it_list = []
            c_pt_list = []
            delta_c_list = []
            for t in range(n_gen_tokens):
                c_it = commitment_continuous(kl_it[:, t].cpu().tolist(), kl_threshold)
                c_pt = commitment_continuous(kl_pt[:, t].cpu().tolist(), kl_threshold)
                c_it_list.append(c_it)
                c_pt_list.append(c_pt)
                delta_c_list.append(c_it - c_pt)

            # Convergence gap: mean Δkl over final 50% of layers, per token
            half_layers = n_layers // 2
            cg_per_token = delta_kl[half_layers:, :].mean(0).tolist()  # [n_gen_tokens]

            # Step 6: Compute Δh and update accumulators per layer
            for li in range(n_layers):
                dh = (h_it_stack[li] - h_pt_stack[li]).float().cpu()  # [n_gen, d]

                # Per-layer regression target: Δkl at THIS layer
                dkl_li = delta_kl[li]  # [n_gen_tokens]

                # Winsorize per-prompt to mitigate heavy tails (plan §9)
                if dkl_li.numel() > 4:
                    lo = torch.quantile(dkl_li, WINSOR_LO)
                    hi = torch.quantile(dkl_li, WINSOR_HI)
                    dkl_li = dkl_li.clamp(lo, hi)

                accumulators[li].update(dh, dkl_li)

                # PCA subsample (Δh per layer)
                if pca_token_count < pca_subsample_tokens:
                    remaining = pca_subsample_tokens - pca_token_count
                    n_take = min(dh.shape[0], remaining)
                    pca_dh[li].append(dh[:n_take])

            # PCA Δkl subsample: [n_tokens, n_layers]
            # Apply same per-layer winsorization as accumulators for consistency.
            if pca_token_count < pca_subsample_tokens:
                remaining = pca_subsample_tokens - pca_token_count
                n_take = min(n_gen_tokens, remaining)
                # Winsorize each layer's Δkl to match accumulator targets
                dkl_winsorized = delta_kl.clone()
                for li in range(n_layers):
                    col = dkl_winsorized[li]
                    if col.numel() > 4:
                        lo = torch.quantile(col, WINSOR_LO)
                        hi = torch.quantile(col, WINSOR_HI)
                        dkl_winsorized[li] = col.clamp(lo, hi)
                pca_dkl.append(dkl_winsorized[:, :n_take].T)
                pca_token_count += n_take

            # Step 7: KL gradient (bonus)
            if compute_kl_gradient:
                _compute_kl_gradient_direction(
                    h_it_stack, h_it_final, final_norm_it, W_U_it,
                    grad_accum, grad_count,
                )

            # Step 8: Write convergence record
            commit_record = {
                "prompt_id": prompt_id,
                "n_gen_tokens": n_gen_tokens,
                "target": "delta_kl",
                "c_it": c_it_list,
                "c_pt": c_pt_list,
                "delta_c": delta_c_list,
                "convergence_gap": cg_per_token,
                "kl_it": kl_it.cpu().tolist(),   # [n_layers, n_gen_tokens]
                "kl_pt": kl_pt.cpu().tolist(),
                "use_tuned": use_tuned,
            }
            f_commit.write(json.dumps(commit_record) + "\n")
            f_commit.flush()

            # Free GPU memory
            del captured_it, captured_pt, h_it_stack, h_pt_stack
            del h_it_final, h_pt_final, kl_it, kl_pt
            torch.cuda.empty_cache()

            n_processed += 1
            total_tokens += n_gen_tokens

            if n_processed % 20 == 0:
                elapsed = time.time() - t0
                log.info(
                    "[Phase 1] %s: %d/%d prompts, %d tokens, %.1f sec",
                    model_name, n_processed + len(done_ids), len(records),
                    total_tokens, elapsed,
                )

            # Checkpoint accumulators
            if n_processed % checkpoint_every == 0:
                _save_checkpoint(
                    accumulators, acc_dir, n_layers,
                    pca_dh, pca_dkl, pca_dir, n_layers,
                )

    # ── Final save ────────────────────────────────────────────────────────────
    _save_checkpoint(accumulators, acc_dir, n_layers, pca_dh, pca_dkl, pca_dir, n_layers)

    # Save KL gradient directions
    if compute_kl_gradient and grad_accum:
        grad_dirs = {}
        for li in range(n_layers):
            if li in grad_accum and grad_count[li] > 0:
                mean_grad = grad_accum[li] / grad_count[li]
                norm = mean_grad.norm()
                if norm > 1e-12:
                    grad_dirs[f"layer_{li}"] = (mean_grad / norm).float().numpy()
                else:
                    grad_dirs[f"layer_{li}"] = np.zeros(d_model, dtype=np.float32)
        np.savez(output_dir / "kl_gradients.npz", **grad_dirs)
        log.info("Saved KL gradient directions for %d layers", len(grad_dirs))

    # Save metadata
    metadata = {
        "model_name": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "regression_target": "delta_kl",
        "winsorize_percentiles": [WINSOR_LO, WINSOR_HI],
        "n_prompts_processed": n_processed + len(done_ids),
        "n_prompts_total": len(records),
        "total_tokens": total_tokens,
        "max_gen_tokens": max_gen_tokens,
        "kl_threshold": kl_threshold,
        "use_tuned_lens": use_tuned,
        "pca_subsample_tokens": pca_token_count,
        "compute_kl_gradient": compute_kl_gradient,
        "elapsed_seconds": time.time() - t0,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(
        "[Phase 1] %s COMPLETE: %d prompts, %d tokens, %.1f sec",
        model_name, metadata["n_prompts_processed"], total_tokens,
        metadata["elapsed_seconds"],
    )
    return metadata


def _save_checkpoint(
    accumulators: dict[int, LayerAccumulator],
    acc_dir: Path,
    n_layers: int,
    pca_dh: dict[int, list[torch.Tensor]],
    pca_dkl: list[torch.Tensor],
    pca_dir: Path,
    n_layers_for_pca: int,
):
    """Save accumulators and PCA subsample to disk."""
    for li in range(n_layers):
        accumulators[li].save(acc_dir, li)

    # Save PCA subsample
    if pca_dkl:
        dkl_cat = torch.cat(pca_dkl, dim=0)  # [n_tokens, n_layers]
        np.save(pca_dir / "delta_kl.npy", dkl_cat.numpy())
        for li in range(n_layers_for_pca):
            if pca_dh[li]:
                dh_cat = torch.cat(pca_dh[li], dim=0)
                np.save(pca_dir / f"delta_h_layer_{li}.npy", dh_cat.numpy())


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Exp10 Phase 1: Paired data collection")
    parser.add_argument("--model", required=True, help="Model name (e.g. gemma3_4b)")
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--n-prompts", type=int, default=600)
    parser.add_argument("--max-gen-tokens", type=int, default=128)
    parser.add_argument("--dataset-path", default="data/eval_dataset_v2.jsonl")
    parser.add_argument("--tuned-lens-dir", default=None,
                        help="Root dir with tuned lens probes")
    parser.add_argument("--no-kl-gradient", action="store_true",
                        help="Skip KL gradient computation")
    args = parser.parse_args()

    collect_paired_data(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        n_prompts=args.n_prompts,
        max_gen_tokens=args.max_gen_tokens,
        dataset_path=args.dataset_path,
        tuned_lens_dir=args.tuned_lens_dir,
        compute_kl_gradient=not args.no_kl_gradient,
    )
