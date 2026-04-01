"""Cross-model tuned-lens probes and commitment delay replication (0G).

Extends the tuned-lens methodology (Belrose et al., 2023) across all 6 model
families using the cross-model adapter infrastructure. Trains per-layer affine
probes T_ℓ(h) = W_ℓ h + b_ℓ that map each layer's residual stream to the
final layer's space, giving a more faithful readout than raw logit lens.

Training (Belrose et al. 2023 exact recipe):
  - Prefill-based on C4 validation (single forward pass, all token positions)
  - SGD Nesterov (lr=1.0, momentum=0.9), 250 steps, linear LR decay to 0
  - 262,144 tokens per step (~65M total), gradient accumulation over micro-batches
  - KL divergence loss: KL(softmax(lm_head(norm(probe(h_ℓ)))) ‖ softmax(lm_head(norm(h_L))))
  - Identity init + zero bias

Evaluation:
  - Autoregressive generation matching collect_L1L2.py methodology
  - Same dataset (eval_dataset_v2.jsonl), same generation params
  - KL-based commitment at thresholds [0.05, 0.1, 0.2]
  - Top-1 commitment (no-flip-back)
  - Enables direct per-token comparison with raw logit-lens results

Transfer test:
  - PT probes on IT activations vs IT probes on IT activations
  - High transfer = shared representational geometry

Usage:
  # Train probes for one model+variant
  uv run python -m src.poc.cross_model.tuned_lens \\
      --model gemma3_4b --variant pt --device cuda:0

  # Evaluate commitment (requires trained probes)
  uv run python -m src.poc.cross_model.tuned_lens \\
      --model gemma3_4b --variant pt --eval-only --device cuda:0

  # Transfer test (requires both PT and IT probes)
  uv run python -m src.poc.cross_model.tuned_lens \\
      --model gemma3_4b --transfer-test --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.poc.cross_model.config import (
    MODEL_REGISTRY,
    get_spec,
    model_id_for_variant,
)
from src.poc.cross_model.adapters import get_adapter
from src.poc.cross_model.utils import (
    load_model_and_tokenizer,
    load_dataset,
    get_raw_prompt,
    read_done_ids,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Belrose et al. KL commitment thresholds (nats)
KL_THRESHOLDS = [0.05, 0.1, 0.2]


# ── Probe definition ────────────────────────────────────────────────────────

class TunedLensProbe(nn.Module):
    """Per-layer affine probe: T_ℓ(h) = W h + b.

    Identity + zero-bias initialisation (Belrose et al. warm start).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


# ── LR schedule ─────────────────────────────────────────────────────────────

def _cosine_lr_with_warmup(
    optimizer, step: int, n_steps: int, warmup_steps: int, base_lr: float
) -> None:
    """Cosine annealing with linear warmup (Adam fallback schedule)."""
    if step < warmup_steps:
        lr = base_lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ── Training data: C4 validation (prefill-based) ───────────────────────────

def load_training_texts(
    n_tokens_target: int = 2_000_000,
    max_seq_len: int = 2048,
    seed: int = 42,
) -> list[str]:
    """Load raw text from C4 validation split for probe training.

    Uses HuggingFace datasets streaming to avoid downloading the full dataset.
    Falls back to wikitext if C4 is unavailable. Texts are shuffled with a
    fixed seed for reproducibility so that 80/20 train/val splits are IID.

    Data volume: Belrose et al. (2023) use ~65M tokens (SGD, 250 steps of 262k
    tokens each). We load ~2M tokens of text (~20k texts) and cycle through them
    with fresh forward passes each step, yielding 262k tokens per step × 250
    steps = ~65M token-activations — matching Belrose exactly.

    Returns a list of text strings, estimated to contain >= n_tokens_target
    tokens total (using ~4 chars/token heuristic for selection, actual
    tokenisation happens later).
    """
    import random

    chars_target = n_tokens_target * 4  # rough 4 chars/token heuristic
    max_chars = max_seq_len * 4

    texts: list[str] = []
    total_chars = 0

    try:
        from datasets import load_dataset as hf_load_dataset

        log.info("Loading C4 validation split (streaming)...")
        ds = hf_load_dataset(
            "allenai/c4", "en", split="validation", streaming=True,
            trust_remote_code=False,
        )
        for example in ds:
            text = example.get("text", "")
            if len(text) < 50:  # skip very short texts
                continue
            if len(text) > max_chars:
                text = text[:max_chars]
            texts.append(text)
            total_chars += len(text)
            if total_chars >= chars_target:
                break

    except Exception as e:
        log.warning("C4 load failed (%s), falling back to wikitext...", e)
        try:
            from datasets import load_dataset as hf_load_dataset

            ds = hf_load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                                 split="validation", trust_remote_code=True)
            for example in ds:
                text = example.get("text", "")
                if len(text) < 50:
                    continue
                if len(text) > max_chars:
                    text = text[:max_chars]
                texts.append(text)
                total_chars += len(text)
                if total_chars >= chars_target:
                    break
        except Exception as e2:
            raise RuntimeError(f"Cannot load C4 or wikitext: {e2}") from e2

    # Shuffle for IID train/val split (C4 validation is ordered by source)
    random.Random(seed).shuffle(texts)

    log.info(
        "Loaded %d texts (~%dk chars, ~%dk estimated tokens)",
        len(texts), total_chars // 1000, total_chars // 4000,
    )
    return texts


# ── Prefill-based hidden state collection ───────────────────────────────────

@torch.no_grad()
def collect_hidden_states_prefill(
    model: nn.Module,
    tokenizer,
    adapter,
    spec,
    texts: list[str],
    device: torch.device,
    max_seq_len: int = 2048,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Collect per-layer residual stream hidden states via single forward passes.

    For each text, runs a single forward pass (no generation) and captures
    the residual stream at all layers for all non-first token positions
    (skip position 0 to avoid BOS artifacts).

    Returns:
        hidden_by_layer: {layer_idx: [N_total_tokens, d_model] float32 on CPU}
        final_hidden:    [N_total_tokens, d_model] float32 on CPU
    """
    n_layers = spec.n_layers
    multi_gpu = spec.multi_gpu

    hidden_by_layer: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    final_hidden_list: list[torch.Tensor] = []

    layer_modules = adapter.layers(model)

    # We capture all positions in a single forward pass per text
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inp, output):
            h = adapter.residual_from_output(output)
            # h: [1, seq_len, d_model] — capture all positions, detach + CPU
            captured[layer_idx] = h[0].detach().float().cpu()
        return hook

    handles = [
        layer_modules[i].register_forward_hook(make_hook(i))
        for i in range(n_layers)
    ]

    gen_device = model.device if multi_gpu else device

    try:
        for ti, text in enumerate(texts):
            captured.clear()

            input_ids = tokenizer.encode(
                text, return_tensors="pt",
                max_length=max_seq_len, truncation=True,
            ).to(gen_device)

            if input_ids.shape[1] < 2:
                continue

            with torch.no_grad():
                model(input_ids)  # single forward pass, no generation

            # Extract all positions except position 0 (skip BOS)
            for layer_idx in range(n_layers):
                if layer_idx in captured:
                    # captured[layer_idx]: [seq_len, d_model]
                    h = captured[layer_idx][1:]  # skip pos 0
                    hidden_by_layer[layer_idx].append(h)

            # Final layer = last layer's captured output
            if n_layers - 1 in captured:
                final_hidden_list.append(captured[n_layers - 1][1:])

            if (ti + 1) % 100 == 0:
                n_tokens_so_far = sum(h.shape[0] for h in final_hidden_list)
                log.info("[prefill] %d/%d texts, %d tokens so far",
                         ti + 1, len(texts), n_tokens_so_far)
    finally:
        for h in handles:
            h.remove()

    # Stack into single tensors
    result: dict[int, torch.Tensor] = {}
    for layer_idx in range(n_layers):
        if hidden_by_layer[layer_idx]:
            result[layer_idx] = torch.cat(hidden_by_layer[layer_idx], dim=0)

    final_hidden = (
        torch.cat(final_hidden_list, dim=0)
        if final_hidden_list
        else torch.zeros(0, spec.d_model)
    )

    n_tokens = final_hidden.shape[0]
    log.info("Collected %d tokens across %d texts", n_tokens, len(texts))
    return result, final_hidden


# ── Streaming text iterator ────────────────────────────────────────────────

def _streaming_text_iterator(
    texts: list[str],
    seed: int = 42,
) -> "Iterator[str]":
    """Infinite iterator that shuffles and cycles through texts.

    Each epoch is a fresh shuffle with a different seed, so the model sees
    different orderings even when cycling.
    """
    import random
    epoch = 0
    while True:
        rng = random.Random(seed + epoch)
        order = list(range(len(texts)))
        rng.shuffle(order)
        for idx in order:
            yield texts[idx]
        epoch += 1


def _collect_batch_streaming(
    model: nn.Module,
    tokenizer,
    adapter,
    spec,
    text_iter,
    layer_idx: int,
    batch_tokens: int,
    device: torch.device,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect a batch of (h_layer, h_final) pairs via fresh forward passes.

    Runs forward passes on texts from text_iter until we have >= batch_tokens
    token activations for the target layer. Returns tensors on CPU.
    Only hooks on two layers (target + final) — minimal overhead.
    """
    layer_acts: list[torch.Tensor] = []
    final_acts: list[torch.Tensor] = []
    n_tokens = 0

    layer_modules = adapter.layers(model)
    multi_gpu = spec.multi_gpu
    gen_device = model.device if multi_gpu else device
    final_layer_idx = spec.n_layers - 1

    captured: dict[str, torch.Tensor | None] = {"layer": None, "final": None}

    def hook_layer(module, inp, output):
        h = adapter.residual_from_output(output)
        captured["layer"] = h[0].detach().float().cpu()

    def hook_final(module, inp, output):
        h = adapter.residual_from_output(output)
        captured["final"] = h[0].detach().float().cpu()

    # Register hooks once for the whole batch (avoid per-text register/remove)
    handles = [layer_modules[layer_idx].register_forward_hook(hook_layer)]
    if layer_idx != final_layer_idx:
        handles.append(layer_modules[final_layer_idx].register_forward_hook(hook_final))
    else:
        # If target IS the final layer, hook_layer already captures it
        hook_final = hook_layer  # noqa: reuse

    try:
        n_fwd = 0
        while n_tokens < batch_tokens:
            text = next(text_iter)
            input_ids = tokenizer.encode(
                text, return_tensors="pt", max_length=max_seq_len, truncation=True,
            ).to(gen_device)
            if input_ids.shape[1] < 2:
                continue

            captured["layer"] = None
            captured["final"] = None

            with torch.no_grad():
                model(input_ids)

            h_l = captured["layer"]
            h_f = captured["final"] if layer_idx != final_layer_idx else captured["layer"]

            if h_l is not None and h_f is not None:
                # Skip position 0 (BOS)
                layer_acts.append(h_l[1:])
                final_acts.append(h_f[1:])
                n_tokens += h_l.shape[0] - 1
            n_fwd += 1
    finally:
        for hh in handles:
            hh.remove()

    return torch.cat(layer_acts, dim=0), torch.cat(final_acts, dim=0)


def _collect_all_layers_streaming(
    model: nn.Module,
    tokenizer,
    adapter,
    spec,
    text_iter,
    batch_tokens: int,
    device: torch.device,
    max_seq_len: int,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Collect hidden states for ALL layers in a single set of forward passes.

    Unlike _collect_batch_streaming (which hooks only 2 layers), this hooks
    every layer simultaneously so one forward pass yields data for all probes.
    This is n_layers× faster when training all probes jointly.

    Returns:
        all_hidden: {layer_idx: [N_tokens, d_model] float32 on CPU}
        final_hidden: [N_tokens, d_model] float32 on CPU (= all_hidden[n_layers-1])
    """
    n_layers = spec.n_layers
    layer_modules = adapter.layers(model)
    multi_gpu = spec.multi_gpu
    gen_device = model.device if multi_gpu else device

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inp, output):
            h = adapter.residual_from_output(output)
            captured[layer_idx] = h[0].detach().float().cpu()
        return hook

    handles = [
        layer_modules[i].register_forward_hook(make_hook(i))
        for i in range(n_layers)
    ]

    layer_acts: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    n_tokens = 0

    try:
        while n_tokens < batch_tokens:
            text = next(text_iter)
            captured.clear()

            input_ids = tokenizer.encode(
                text, return_tensors="pt", max_length=max_seq_len, truncation=True,
            ).to(gen_device)
            if input_ids.shape[1] < 2:
                continue

            with torch.no_grad():
                model(input_ids)

            # Check that at least some layers were captured
            if 0 not in captured:
                continue

            seq_tokens = captured[0].shape[0] - 1  # skip BOS (position 0)
            for i in range(n_layers):
                if i in captured:
                    layer_acts[i].append(captured[i][1:])
            n_tokens += seq_tokens
    finally:
        for hh in handles:
            hh.remove()

    result: dict[int, torch.Tensor] = {}
    for i in range(n_layers):
        if layer_acts[i]:
            result[i] = torch.cat(layer_acts[i], dim=0)

    final = result.get(n_layers - 1, torch.zeros(0, spec.d_model))
    return result, final


# ── LR schedules ──────────────────────────────────────────────────────────

def _linear_lr_decay(optimizer, step: int, n_steps: int, base_lr: float) -> None:
    """Linear decay to 0 (Belrose et al. SGD schedule)."""
    lr = base_lr * max(0.0, 1.0 - step / n_steps)
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ── Probe training ──────────────────────────────────────────────────────────

def train_probes(
    model: nn.Module,
    tokenizer,
    adapter,
    spec,
    train_texts: list[str],
    val_texts: list[str],
    device: torch.device,
    output_dir: Path,
    n_steps: int = 250,
    lr: float = 1.0,
    batch_size: int = 262144,
    micro_batch_size: int = 512,
    warmup_frac: float = 0.05,
    max_seq_len: int = 2048,
    optimizer_type: str = "sgd_nesterov",
    streaming: bool = True,
) -> dict:
    """Train tuned-lens probes for all layers.

    Training modes:
      streaming=True + sgd_nesterov (default, recommended):
          Joint training — all probes trained simultaneously per step.
          Each step collects hidden states for ALL layers in shared forward
          passes, yielding n_layers× speedup over per-layer collection.
          SGD lr=1.0, momentum=0.9, nesterov=True, linear decay over 250
          steps, 262K tokens/step = ~65M total (Belrose et al. exact recipe).

      streaming=True + adam:
          Per-layer training with fresh forward passes per step per layer.
          Adam lr=1e-3, weight_decay=1e-3, cosine LR with warmup.

      streaming=False:
          Pre-collects all hidden states into CPU RAM (legacy).
          Limited to ~100k tokens by memory; useful for quick testing.

    Returns training summary dict with per-layer train/val losses and agreement.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_layers = spec.n_layers
    d_model = spec.d_model

    # Check for already-trained probes (resumable).
    # training_summary.json is written only after full training completes.
    # checkpoint.json is written every 50 steps during training.
    # We skip training ONLY if training_summary.json exists (= fully done).
    existing = set()
    for p in output_dir.glob("probe_layer_*.pt"):
        try:
            existing.add(int(p.stem.split("_")[-1]))
        except ValueError:
            pass
    summary_path = output_dir / "training_summary.json"
    if len(existing) == n_layers and summary_path.exists():
        log.info("All %d probes already exist with summary, skipping training.", n_layers)
        return json.loads(summary_path.read_text())
    if existing:
        ckpt_path = output_dir / "checkpoint.json"
        if ckpt_path.exists():
            ckpt = json.loads(ckpt_path.read_text())
            log.info("Found %d checkpoint probes at step %s/%s — retraining from scratch "
                     "(checkpoints are safety net, not resume point).",
                     len(existing), ckpt.get("checkpoint_step"), ckpt.get("n_steps"))
        else:
            log.info("Found %d partial probes without summary — retraining.", len(existing))

    # Freeze model — only probe parameters should receive gradients.
    # Without this, backward() stores gradients on lm_head (~3 GB for 256K vocab)
    # and final_norm, wasting memory and risking OOM in joint training mode.
    model.requires_grad_(False)

    # Access final_norm and lm_head via adapter
    final_norm_mod = adapter.final_norm(model)
    lm_head_mod = adapter.lm_head(model)

    # For multi-GPU: these modules may live on different devices
    if spec.multi_gpu:
        compute_device = next(final_norm_mod.parameters()).device
    else:
        compute_device = device

    # Override n_steps for Belrose SGD recipe
    if optimizer_type == "sgd_nesterov":
        n_steps = 250
        lr = 1.0
        log.info("SGD+Nesterov mode: 250 steps, lr=1.0, linear decay (Belrose exact)")

    # ── Collect validation hidden states (always pre-collected, small) ──
    # Cap validation texts to limit CPU RAM (all layers stored simultaneously)
    # 100 texts ≈ 25k tokens → ~10 GB for large models, well within budget
    max_val_texts = 100
    val_texts_used = val_texts[:max_val_texts] if len(val_texts) > max_val_texts else val_texts
    log.info("Collecting validation hidden states from %d texts...", len(val_texts_used))
    val_hidden, val_final = collect_hidden_states_prefill(
        model, tokenizer, adapter, spec, val_texts_used, device, max_seq_len,
    )
    n_val_tokens = val_final.shape[0]
    log.info("Validation set: %d token activations.", n_val_tokens)

    # ── Pre-collect training data if not streaming ──
    train_hidden = None
    train_final = None
    n_train_tokens = 0
    if not streaming:
        log.info("Pre-collecting training hidden states from %d texts...", len(train_texts))
        train_hidden, train_final = collect_hidden_states_prefill(
            model, tokenizer, adapter, spec, train_texts, device, max_seq_len,
        )
        n_train_tokens = train_final.shape[0]
        log.info("Training set: %d token activations.", n_train_tokens)
        mem_gb = n_layers * n_train_tokens * d_model * 4 / 1e9
        log.info("Estimated CPU RAM for hidden states: %.1f GB (train)", mem_gb)
    else:
        est_tokens = n_steps * batch_size
        log.info(
            "Streaming mode: ~%dk unique tokens over %d steps "
            "(fresh forward passes, no CPU RAM for training data)",
            est_tokens // 1000, n_steps,
        )

    warmup_steps = int(n_steps * warmup_frac) if optimizer_type == "adam" else 0
    rng = torch.Generator()
    rng.manual_seed(42)
    training_summary: dict = {}
    training_summary["_config"] = {
        "optimizer": optimizer_type,
        "streaming": streaming,
        "n_steps": n_steps,
        "lr": lr,
        "batch_size": batch_size,
    }

    # Streaming iterator (shared across layers)
    text_iter = _streaming_text_iterator(train_texts) if streaming else None
    model_dtype = next(final_norm_mod.parameters()).dtype

    # ── Joint training: all probes simultaneously (n_layers× faster) ──────
    # When streaming + SGD, we hook ALL layers per forward pass so each pass
    # provides data for every probe. This reduces total forward passes from
    # (n_layers × n_steps × ~500) to (n_steps × ~500).

    if streaming and optimizer_type == "sgd_nesterov":
        log.info(
            "Joint training mode: all %d layers simultaneously, "
            "%d steps, %d tokens/step (%.1fM total token-activations)",
            n_layers - len(existing), n_steps, batch_size,
            n_steps * batch_size / 1e6,
        )

        # Create all probes and optimizers upfront
        probes: dict[int, TunedLensProbe] = {}
        optimizers_all: dict[int, torch.optim.Optimizer] = {}
        best_states: dict[int, dict | None] = {}
        best_val_losses: dict[int, float] = {}
        train_loss_accum: dict[int, float] = {}

        layers_to_train = [i for i in range(n_layers) if i not in existing]
        for li in layers_to_train:
            probe = TunedLensProbe(d_model).to(compute_device)
            probes[li] = probe
            optimizers_all[li] = torch.optim.SGD(
                probe.parameters(), lr=lr, momentum=0.9, nesterov=True,
                weight_decay=0,
            )
            best_states[li] = None
            best_val_losses[li] = float("inf")
            train_loss_accum[li] = 0.0

        for step in range(n_steps):
            # Linear LR decay (Belrose schedule)
            cur_lr = lr * max(0.0, 1.0 - step / n_steps)
            for opt in optimizers_all.values():
                for pg in opt.param_groups:
                    pg["lr"] = cur_lr

            # Zero grad all probes
            for opt in optimizers_all.values():
                opt.zero_grad()

            # Reset loss accumulators
            for li in layers_to_train:
                train_loss_accum[li] = 0.0

            # Gradient accumulation: collect micro-batches until batch_size
            accumulated = 0
            while accumulated < batch_size:
                micro_target = min(micro_batch_size, batch_size - accumulated)
                # Single set of forward passes → data for ALL layers
                all_h, h_final = _collect_all_layers_streaming(
                    model, tokenizer, adapter, spec, text_iter,
                    micro_target, device, max_seq_len,
                )
                actual_tokens = h_final.shape[0]
                if actual_tokens == 0:
                    continue

                # Compute target log-probs (shared across all probes)
                with torch.no_grad():
                    hf = h_final.to(device=compute_device, dtype=model_dtype)
                    target = F.log_softmax(
                        lm_head_mod(final_norm_mod(hf)).float(), dim=-1,
                    )
                    del hf

                # Train each probe on its layer's data from this micro-batch
                for li in layers_to_train:
                    if li not in all_h:
                        continue
                    h = all_h[li].to(compute_device)
                    h_probed = probes[li](h).to(model_dtype)
                    log_pred = F.log_softmax(
                        lm_head_mod(final_norm_mod(h_probed)).float(), dim=-1,
                    )
                    loss = F.kl_div(
                        log_pred, target, reduction="sum", log_target=True,
                    ) / batch_size  # scale for accumulation
                    loss.backward()
                    train_loss_accum[li] += loss.item() * actual_tokens
                    del h, h_probed, log_pred

                del all_h, h_final, target
                accumulated += actual_tokens

            # Clip gradients and step all probes
            for li in layers_to_train:
                torch.nn.utils.clip_grad_norm_(probes[li].parameters(), 1.0)
                optimizers_all[li].step()

            # Progress log
            if (step + 1) % 10 == 0:
                sample_li = layers_to_train[0]
                avg_loss = train_loss_accum[sample_li] / max(accumulated, 1)
                log.info(
                    "  [joint] step %d/%d: loss=%.4f (layer %d) lr=%.4f tokens=%d",
                    step + 1, n_steps, avg_loss, sample_li, cur_lr, accumulated,
                )

            # Validation every 50 steps
            if (step + 1) % 50 == 0 and n_val_tokens > 0:
                with torch.no_grad():
                    for li in layers_to_train:
                        if li not in val_hidden:
                            continue
                        eval_n = min(val_hidden[li].shape[0], 4096)
                        v_h = val_hidden[li][:eval_n].to(compute_device)
                        v_final = val_final[:eval_n].to(
                            device=compute_device, dtype=model_dtype,
                        )
                        v_target = F.log_softmax(
                            lm_head_mod(final_norm_mod(v_final)).float(), dim=-1,
                        )
                        v_probed = probes[li](v_h).to(model_dtype)
                        v_pred = F.log_softmax(
                            lm_head_mod(final_norm_mod(v_probed)).float(), dim=-1,
                        )
                        v_loss = F.kl_div(
                            v_pred, v_target, reduction="batchmean",
                            log_target=True,
                        ).item()

                        if v_loss < best_val_losses[li]:
                            best_val_losses[li] = v_loss
                            best_states[li] = {
                                k: v.clone()
                                for k, v in probes[li].state_dict().items()
                            }
                        del v_h, v_final, v_target, v_probed, v_pred
                torch.cuda.empty_cache()

                # Checkpoint: save best probes so far to disk.
                # If preempted/killed, these survive and the resume logic
                # will detect all n_layers probes exist → skip training.
                for li in layers_to_train:
                    state = best_states[li] if best_states[li] is not None else probes[li].state_dict()
                    torch.save(state, output_dir / f"probe_layer_{li}.pt")
                ckpt_meta = {"checkpoint_step": step + 1, "n_steps": n_steps}
                (output_dir / "checkpoint.json").write_text(json.dumps(ckpt_meta))
                log.info("  [checkpoint] Saved %d probes at step %d/%d",
                         len(layers_to_train), step + 1, n_steps)

        # ── Save all probes and compute final eval metrics ────────────────
        for li in layers_to_train:
            probe = probes[li]
            if best_states[li] is not None:
                probe.load_state_dict(best_states[li])

            torch.save(
                probe.state_dict(), output_dir / f"probe_layer_{li}.pt",
            )

            # Final eval: raw vs tuned agreement + KL delta
            with torch.no_grad():
                if li in val_hidden:
                    eval_n = min(val_hidden[li].shape[0], 2048)
                    h_eval = val_hidden[li][:eval_n].to(compute_device)
                    f_eval = val_final[:eval_n].to(
                        device=compute_device, dtype=model_dtype,
                    )
                    final_top1 = lm_head_mod(
                        final_norm_mod(f_eval)
                    ).argmax(dim=-1)

                    h_eval_md = h_eval.to(model_dtype)
                    top1_raw = lm_head_mod(
                        final_norm_mod(h_eval_md)
                    ).argmax(dim=-1)
                    h_tuned_md = probe(h_eval).to(model_dtype)
                    top1_tuned = lm_head_mod(
                        final_norm_mod(h_tuned_md)
                    ).argmax(dim=-1)

                    raw_agree = (
                        (top1_raw == final_top1).float().mean().item()
                    )
                    tuned_agree = (
                        (top1_tuned == final_top1).float().mean().item()
                    )

                    # KL delta
                    target_lp = F.log_softmax(
                        lm_head_mod(final_norm_mod(f_eval)).float(), dim=-1,
                    )
                    raw_kl = F.kl_div(
                        F.log_softmax(
                            lm_head_mod(final_norm_mod(h_eval_md)).float(),
                            dim=-1,
                        ),
                        target_lp,
                        reduction="batchmean",
                        log_target=True,
                    ).item()
                    tuned_kl = F.kl_div(
                        F.log_softmax(
                            lm_head_mod(
                                final_norm_mod(probe(h_eval).to(model_dtype))
                            ).float(),
                            dim=-1,
                        ),
                        target_lp,
                        reduction="batchmean",
                        log_target=True,
                    ).item()
                    kl_delta = raw_kl - tuned_kl
                    del h_eval, f_eval, final_top1
                else:
                    raw_agree = tuned_agree = float("nan")
                    raw_kl = tuned_kl = kl_delta = float("nan")

            if tuned_agree < raw_agree and not math.isnan(tuned_agree):
                log.warning(
                    "  layer %d: tuned_agree (%.3f) < raw_agree (%.3f) — "
                    "probe may be undertrained!",
                    li, tuned_agree, raw_agree,
                )

            training_summary[str(li)] = {
                "train_loss_final": train_loss_accum.get(li, float("nan")),
                "val_loss_best": (
                    best_val_losses[li]
                    if best_val_losses[li] < float("inf")
                    else float("nan")
                ),
                "steps_used": n_steps,
                "early_stopped": False,
                "raw_top1_agree": raw_agree,
                "tuned_top1_agree": tuned_agree,
                "raw_kl": raw_kl,
                "tuned_kl": tuned_kl,
                "kl_delta_raw_minus_tuned": kl_delta,
            }
            log.info(
                "  layer %2d: val_loss=%.4f  raw_agree=%.3f → tuned_agree=%.3f"
                "  KL_delta=%.4f",
                li, best_val_losses[li], raw_agree, tuned_agree, kl_delta,
            )

        # Cleanup
        del probes, optimizers_all, best_states
        torch.cuda.empty_cache()

    else:
        # ── Per-layer training (Adam mode or non-streaming fallback) ──────
        for layer_idx in range(n_layers):
            if layer_idx in existing:
                log.info("  layer %d: already trained, skipping.", layer_idx)
                if train_hidden is not None:
                    train_hidden.pop(layer_idx, None)
                    val_hidden.pop(layer_idx, None)
                continue

            if not streaming and layer_idx not in train_hidden:
                log.info("  layer %d: no training activations, skipping.", layer_idx)
                continue

            probe = TunedLensProbe(d_model).to(compute_device)

            if optimizer_type == "sgd_nesterov":
                optimizer = torch.optim.SGD(
                    probe.parameters(), lr=lr, momentum=0.9, nesterov=True,
                    weight_decay=0,
                )
            else:
                optimizer = torch.optim.Adam(
                    probe.parameters(), lr=lr, weight_decay=1e-3,
                )

            best_val_loss = float("inf")
            best_state = None
            patience_counter = 0
            patience_limit = 5
            train_losses: list[tuple[int, float]] = []
            val_losses: list[tuple[int, float]] = []
            final_step = n_steps

            if not streaming:
                layer_train = train_hidden[layer_idx]
                n_t = min(layer_train.shape[0], n_train_tokens)
                if n_t == 0:
                    train_hidden.pop(layer_idx, None)
                    continue

            for step in range(n_steps):
                if optimizer_type == "sgd_nesterov":
                    _linear_lr_decay(optimizer, step, n_steps, lr)
                else:
                    _cosine_lr_with_warmup(
                        optimizer, step, n_steps, warmup_steps, lr,
                    )

                if streaming:
                    h_all, hf_all = _collect_batch_streaming(
                        model, tokenizer, adapter, spec, text_iter,
                        layer_idx, batch_size, device, max_seq_len,
                    )
                else:
                    idx = torch.randperm(n_t, generator=rng)[:batch_size]
                    h_all = layer_train[idx]
                    hf_all = train_final[idx]

                total_tokens = h_all.shape[0]

                optimizer.zero_grad()
                total_loss = 0.0
                n_micro = math.ceil(total_tokens / micro_batch_size)

                for mi in range(n_micro):
                    start = mi * micro_batch_size
                    end = min(start + micro_batch_size, total_tokens)
                    h = h_all[start:end].to(compute_device)
                    h_final_batch = hf_all[start:end]

                    with torch.no_grad():
                        hf = h_final_batch.to(
                            device=compute_device, dtype=model_dtype,
                        )
                        target = F.log_softmax(
                            lm_head_mod(final_norm_mod(hf)).float(), dim=-1,
                        )
                        del hf

                    h_probed = probe(h).to(model_dtype)
                    log_pred = F.log_softmax(
                        lm_head_mod(final_norm_mod(h_probed)).float(), dim=-1,
                    )
                    loss = F.kl_div(
                        log_pred, target, reduction="sum", log_target=True,
                    ) / total_tokens
                    loss.backward()
                    total_loss += loss.item() * (end - start)
                    del h, h_probed, log_pred, target

                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                optimizer.step()
                del h_all, hf_all

                avg_loss = total_loss / total_tokens
                train_losses.append((step + 1, avg_loss))

                log_interval = 10 if optimizer_type == "sgd_nesterov" else 100
                if (step + 1) % log_interval == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    log.info(
                        "  layer %d step %d/%d: loss=%.4f lr=%.4f tokens=%d",
                        layer_idx, step + 1, n_steps, avg_loss, cur_lr,
                        total_tokens,
                    )

                val_interval = (
                    50 if optimizer_type == "sgd_nesterov" else 500
                )
                if (
                    (step + 1) % val_interval == 0
                    and n_val_tokens > 0
                    and layer_idx in val_hidden
                ):
                    with torch.no_grad():
                        val_h = val_hidden[layer_idx]
                        n_v = min(val_h.shape[0], n_val_tokens)
                        eval_n = min(n_v, 4096)
                        v_idx = torch.randperm(n_v, generator=rng)[:eval_n]
                        v_h = val_h[v_idx].to(compute_device)
                        v_final = val_final[v_idx].to(
                            device=compute_device, dtype=model_dtype,
                        )
                        v_target = F.log_softmax(
                            lm_head_mod(final_norm_mod(v_final)).float(),
                            dim=-1,
                        )
                        del v_final
                        v_probed = probe(v_h).to(model_dtype)
                        v_pred = F.log_softmax(
                            lm_head_mod(final_norm_mod(v_probed)).float(),
                            dim=-1,
                        )
                        v_loss = F.kl_div(
                            v_pred, v_target, reduction="batchmean",
                            log_target=True,
                        )
                        val_loss_val = v_loss.item()
                        val_losses.append((step + 1, val_loss_val))

                        if val_loss_val < best_val_loss:
                            best_val_loss = val_loss_val
                            best_state = {
                                k: v.clone()
                                for k, v in probe.state_dict().items()
                            }
                            patience_counter = 0
                        else:
                            patience_counter += 1

                    if (
                        optimizer_type != "sgd_nesterov"
                        and patience_counter >= patience_limit
                        and step >= warmup_steps
                    ):
                        final_step = step + 1
                        log.info(
                            "  layer %d: early stop at step %d "
                            "(no val improvement for %d checks)",
                            layer_idx, final_step, patience_limit,
                        )
                        break

            if best_state is not None:
                probe.load_state_dict(best_state)

            probe_path = output_dir / f"probe_layer_{layer_idx}.pt"
            torch.save(probe.state_dict(), probe_path)

            with torch.no_grad():
                if layer_idx in val_hidden:
                    eval_n = min(val_hidden[layer_idx].shape[0], 2048)
                    h_eval = val_hidden[layer_idx][:eval_n].to(compute_device)
                    f_eval = val_final[:eval_n].to(
                        device=compute_device, dtype=model_dtype,
                    )
                    final_top1 = lm_head_mod(
                        final_norm_mod(f_eval)
                    ).argmax(dim=-1)
                    del f_eval

                    h_eval_md = h_eval.to(model_dtype)
                    top1_raw = lm_head_mod(
                        final_norm_mod(h_eval_md)
                    ).argmax(dim=-1)
                    h_tuned_md = probe(h_eval).to(model_dtype)
                    top1_tuned = lm_head_mod(
                        final_norm_mod(h_tuned_md)
                    ).argmax(dim=-1)

                    raw_agree = (
                        (top1_raw == final_top1).float().mean().item()
                    )
                    tuned_agree = (
                        (top1_tuned == final_top1).float().mean().item()
                    )

                    raw_logits = lm_head_mod(
                        final_norm_mod(h_eval_md)
                    ).float()
                    tuned_logits = lm_head_mod(
                        final_norm_mod(probe(h_eval).to(model_dtype))
                    ).float()
                    target_logits = lm_head_mod(
                        final_norm_mod(
                            val_final[:eval_n].to(
                                device=compute_device, dtype=model_dtype,
                            )
                        )
                    ).float()
                    target_lp = F.log_softmax(target_logits, dim=-1)
                    raw_kl = F.kl_div(
                        F.log_softmax(raw_logits, dim=-1),
                        target_lp,
                        reduction="batchmean",
                        log_target=True,
                    ).item()
                    tuned_kl = F.kl_div(
                        F.log_softmax(tuned_logits, dim=-1),
                        target_lp,
                        reduction="batchmean",
                        log_target=True,
                    ).item()
                    kl_delta = raw_kl - tuned_kl
                else:
                    raw_agree = tuned_agree = float("nan")
                    raw_kl = tuned_kl = kl_delta = float("nan")

            if tuned_agree < raw_agree and not math.isnan(tuned_agree):
                log.warning(
                    "  layer %d: tuned_agree (%.3f) < raw_agree (%.3f) — "
                    "probe may be undertrained!",
                    layer_idx, tuned_agree, raw_agree,
                )

            training_summary[str(layer_idx)] = {
                "train_loss_final": (
                    train_losses[-1][1] if train_losses else float("nan")
                ),
                "val_loss_best": (
                    best_val_loss
                    if best_val_loss < float("inf")
                    else float("nan")
                ),
                "steps_used": final_step,
                "early_stopped": final_step < n_steps,
                "raw_top1_agree": raw_agree,
                "tuned_top1_agree": tuned_agree,
                "raw_kl": raw_kl,
                "tuned_kl": tuned_kl,
                "kl_delta_raw_minus_tuned": kl_delta,
            }
            log.info(
                "  layer %2d: val_loss=%.4f  raw_agree=%.3f → "
                "tuned_agree=%.3f  KL_delta=%.4f",
                layer_idx, best_val_loss, raw_agree, tuned_agree, kl_delta,
            )

            del probe
            if train_hidden is not None:
                train_hidden.pop(layer_idx, None)
            val_hidden.pop(layer_idx, None)
            torch.cuda.empty_cache()

    # Save training summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    log.info("Training complete. Probes + summary → %s", output_dir)
    return training_summary


# ── Probe loading ───────────────────────────────────────────────────────────

def _load_probes(
    probe_dir: Path, d_model: int, device: torch.device,
) -> dict[int, TunedLensProbe]:
    """Load trained probes from a directory."""
    probes: dict[int, TunedLensProbe] = {}
    for probe_path in sorted(probe_dir.glob("probe_layer_*.pt")):
        try:
            layer_idx = int(probe_path.stem.split("_")[-1])
        except ValueError:
            continue
        probe = TunedLensProbe(d_model).to(device)
        probe.load_state_dict(
            torch.load(probe_path, map_location=device, weights_only=True)
        )
        probe.eval()
        probes[layer_idx] = probe
    return probes


# ── Commitment helpers ──────────────────────────────────────────────────────

def _commitment_from_kl(
    kl_row: list[float], threshold: float,
) -> int:
    """Earliest layer where KL < threshold and stays below for all subsequent layers."""
    n = len(kl_row)
    for i in range(n):
        if kl_row[i] < threshold and all(kl_row[j] < threshold for j in range(i, n)):
            return i
    return n - 1


def _commitment_from_top1(top1_by_layer: list[int]) -> int:
    """Earliest layer where top-1 matches final and stays matched (no-flip-back)."""
    n = len(top1_by_layer)
    final_top1 = top1_by_layer[-1]
    for i in range(n):
        if top1_by_layer[i] == final_top1 and all(
            top1_by_layer[j] == final_top1 for j in range(i, n)
        ):
            return i
    return n - 1


def _commitment_majority(
    kl_row: list[float], threshold: float, frac: float = 0.9,
) -> int:
    """Earliest layer where >=frac of subsequent layers have KL < threshold.

    Relaxed version of strict no-flip-back. Handles noisy late layers that
    can push strict commitment arbitrarily late. Reports alongside strict
    metric for robustness.
    """
    n = len(kl_row)
    for i in range(n):
        subsequent = kl_row[i:]
        n_below = sum(1 for k in subsequent if k < threshold)
        if n_below / len(subsequent) >= frac:
            return i
    return n - 1


# ── Commitment evaluation (autoregressive, matching collect_L1L2.py) ──────

@torch.no_grad()
def eval_commitment(
    model: nn.Module,
    tokenizer,
    adapter,
    spec,
    probes: dict[int, TunedLensProbe],
    records: list[dict],
    device: torch.device,
    output_path: Path,
    variant: str = "?",
    max_new_tokens: int = 512,
) -> dict:
    """Evaluate commitment using autoregressive generation with hooks.

    For each generation step × layer:
      - Raw logit-lens: normed = final_norm(h_ℓ) → logits = normed @ W_U → argmax
      - Tuned: h_tuned = probe_ℓ(h_ℓ) → normed = final_norm(h_tuned) → logits → argmax
      - KL(tuned_ℓ ‖ final) for KL-based commitment

    Output JSONL with per-prompt commitment layers at multiple thresholds.
    """
    n_layers = spec.n_layers
    multi_gpu = spec.multi_gpu

    final_norm_mod = adapter.final_norm(model)
    model_dtype = next(final_norm_mod.parameters()).dtype
    lm_head_weight = adapter.lm_head(model).weight.detach().float()  # [vocab, d_model]
    W_U = lm_head_weight.T  # [d_model, vocab] — float32, pre-allocated once

    if multi_gpu:
        compute_device = next(final_norm_mod.parameters()).device
    else:
        compute_device = device

    # Move W_U to compute device
    W_U = W_U.to(compute_device)

    # Move probes to compute device
    for probe in probes.values():
        probe.to(compute_device)

    layer_modules = adapter.layers(model)
    stop_ids = list(adapter.stop_token_ids(tokenizer))

    # Resume support
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = read_done_ids(output_path)
    if done_ids:
        log.info("Resuming: %d/%d already done", len(done_ids), len(records))

    # Aggregated stats
    all_commitment_raw: list[list[int]] = []
    all_commitment_raw_kl: dict[float, list[list[int]]] = {t: [] for t in KL_THRESHOLDS}
    all_commitment_tuned: dict[float, list[list[int]]] = {t: [] for t in KL_THRESHOLDS}
    all_commitment_majority: dict[float, list[list[int]]] = {t: [] for t in KL_THRESHOLDS}
    all_commitment_top1_tuned: list[list[int]] = []
    n_prompts_done = 0

    with open(output_path, "a") as fout:
        for ri, rec in enumerate(records):
            pid = rec.get("id", f"rec_{ri}")
            if pid in done_ids:
                continue

            raw_prompt = get_raw_prompt(rec)

            # Per-step accumulators
            step_commitment_raw: list[int] = []
            step_commitment_raw_kl: dict[float, list[int]] = {t: [] for t in KL_THRESHOLDS}
            step_commitment_tuned: dict[float, list[int]] = {t: [] for t in KL_THRESHOLDS}
            step_commitment_top1_tuned: list[int] = []
            step_commitment_majority: dict[float, list[int]] = {t: [] for t in KL_THRESHOLDS}

            step_buf: list[torch.Tensor | None] = [None] * n_layers

            def _process_step() -> None:
                """Process one generation step: compute commitment metrics.

                Batched: stacks all layers into [n_layers, d_model] and does
                2 matmuls (raw + tuned) instead of 2*n_layers individual ones.
                """
                # Check all layers have data
                h_final = step_buf[n_layers - 1]
                if h_final is None:
                    return

                # Stack all layer hidden states → [n_layers, d_model]
                valid = [step_buf[li] is not None for li in range(n_layers)]
                all_h = torch.stack(
                    [step_buf[li] if valid[li] else torch.zeros_like(h_final)
                     for li in range(n_layers)],
                    dim=0,
                ).to(device=compute_device, dtype=model_dtype)

                # === Raw logit-lens (batched) ===
                # final_norm handles [n_layers, 1, d_model] batch
                all_normed_raw = final_norm_mod(all_h.unsqueeze(1)).squeeze(1)  # [n_layers, d_model]
                all_raw_logits = all_normed_raw.float() @ W_U  # [n_layers, vocab]
                all_raw_top1 = all_raw_logits.argmax(dim=-1)  # [n_layers]
                final_top1 = int(all_raw_top1[-1].item())

                # Final layer's log-probs for KL computation
                log_p_final = F.log_softmax(all_raw_logits[-1], dim=-1)

                raw_top1_row = [int(all_raw_top1[li].item()) if valid[li] else -1
                                for li in range(n_layers)]

                # Raw KL(raw_ℓ ‖ final) for each layer
                all_raw_log_q = F.log_softmax(all_raw_logits, dim=-1)
                raw_kl_all = F.kl_div(
                    all_raw_log_q, log_p_final.unsqueeze(0).expand_as(all_raw_log_q),
                    reduction="none", log_target=True,
                ).sum(dim=-1)  # [n_layers]
                raw_kl_row = [max(raw_kl_all[li].item(), 0.0) if valid[li] else float("inf")
                              for li in range(n_layers)]

                # === Tuned lens (batched) ===
                tuned_parts = []
                for li in range(n_layers):
                    if not valid[li]:
                        tuned_parts.append(torch.zeros_like(h_final).to(device=compute_device, dtype=model_dtype))
                    elif li in probes:
                        tuned_parts.append(probes[li](all_h[li].float().unsqueeze(0)).squeeze(0).to(model_dtype))
                    else:
                        tuned_parts.append(all_h[li])
                all_h_tuned = torch.stack(tuned_parts, dim=0)  # [n_layers, d_model]

                all_normed_tuned = final_norm_mod(all_h_tuned.unsqueeze(1)).squeeze(1)
                all_tuned_logits = all_normed_tuned.float() @ W_U  # [n_layers, vocab]
                all_tuned_top1 = all_tuned_logits.argmax(dim=-1)

                tuned_top1_row = [int(all_tuned_top1[li].item()) if valid[li] else -1
                                  for li in range(n_layers)]

                # KL(tuned_ℓ ‖ final) for each layer
                all_log_q = F.log_softmax(all_tuned_logits, dim=-1)  # [n_layers, vocab]
                kl_all = F.kl_div(
                    all_log_q, log_p_final.unsqueeze(0).expand_as(all_log_q),
                    reduction="none", log_target=True,
                ).sum(dim=-1)  # [n_layers]
                kl_row = [max(kl_all[li].item(), 0.0) if valid[li] else float("inf")
                          for li in range(n_layers)]

                # Raw commitment (no-flip-back top-1)
                step_commitment_raw.append(_commitment_from_top1(raw_top1_row))

                # Raw KL commitment at each threshold
                for thresh in KL_THRESHOLDS:
                    step_commitment_raw_kl[thresh].append(
                        _commitment_from_kl(raw_kl_row, thresh)
                    )

                # Tuned KL commitment at each threshold
                for thresh in KL_THRESHOLDS:
                    step_commitment_tuned[thresh].append(
                        _commitment_from_kl(kl_row, thresh)
                    )

                # Tuned top-1 commitment
                step_commitment_top1_tuned.append(
                    _commitment_from_top1(tuned_top1_row)
                )

                # Majority-vote commitment (≥90% subsequent layers below threshold)
                for thresh in KL_THRESHOLDS:
                    step_commitment_majority[thresh].append(
                        _commitment_majority(kl_row, thresh)
                    )

            def make_hook(layer_idx: int):
                def hook(module, inp, output):
                    h = adapter.residual_from_output(output)
                    if h.shape[1] != 1:
                        return  # skip prefill
                    # Keep on GPU for non-multi-GPU models to avoid CPU↔GPU round-trips
                    vec = h[0, 0, :].detach()
                    if multi_gpu:
                        vec = vec.cpu()  # multi-GPU: layers on different devices
                    step_buf[layer_idx] = vec
                    if layer_idx == n_layers - 1:
                        _process_step()
                return hook

            handles = [
                layer_modules[i].register_forward_hook(make_hook(i))
                for i in range(n_layers)
            ]

            gen_device = model.device if multi_gpu else device
            input_ids = tokenizer.encode(raw_prompt, return_tensors="pt").to(gen_device)

            try:
                model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=stop_ids or None,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            except Exception as e:
                log.warning("Prompt %s failed: %s", pid, e)
            finally:
                for hh in handles:
                    hh.remove()

            n_steps = len(step_commitment_raw)
            if n_steps == 0:
                continue

            result = {
                "prompt_id": pid,
                "model": spec.name,
                "variant": variant,
                "n_steps": n_steps,
                "commitment_layer_raw": step_commitment_raw,
            }
            for thresh in KL_THRESHOLDS:
                result[f"commitment_layer_raw_kl_{thresh}"] = step_commitment_raw_kl[thresh]
            for thresh in KL_THRESHOLDS:
                result[f"commitment_layer_tuned_{thresh}"] = step_commitment_tuned[thresh]
            result["commitment_layer_top1_tuned"] = step_commitment_top1_tuned
            for thresh in KL_THRESHOLDS:
                result[f"commitment_layer_majority_{thresh}"] = step_commitment_majority[thresh]

            fout.write(json.dumps(result) + "\n")
            fout.flush()

            all_commitment_raw.append(step_commitment_raw)
            for thresh in KL_THRESHOLDS:
                all_commitment_raw_kl[thresh].append(step_commitment_raw_kl[thresh])
                all_commitment_tuned[thresh].append(step_commitment_tuned[thresh])
                all_commitment_majority[thresh].append(step_commitment_majority[thresh])
            all_commitment_top1_tuned.append(step_commitment_top1_tuned)

            n_prompts_done += 1
            if n_prompts_done % 50 == 0:
                log.info("[eval] %d/%d prompts done", n_prompts_done, len(records))

    # Compute summary statistics
    def _flatten_median(nested: list[list[int]]) -> float:
        flat = [v for row in nested for v in row]
        return float(np.median(flat)) if flat else float("nan")

    summary = {
        "model": spec.name,
        "n_prompts": n_prompts_done + len(done_ids),
        "n_layers": n_layers,
        "median_commitment_raw": _flatten_median(all_commitment_raw),
    }
    for thresh in KL_THRESHOLDS:
        summary[f"median_commitment_raw_kl_{thresh}"] = _flatten_median(
            all_commitment_raw_kl[thresh]
        )
    for thresh in KL_THRESHOLDS:
        summary[f"median_commitment_tuned_{thresh}"] = _flatten_median(
            all_commitment_tuned[thresh]
        )
    summary["median_commitment_top1_tuned"] = _flatten_median(all_commitment_top1_tuned)
    for thresh in KL_THRESHOLDS:
        summary[f"median_commitment_majority_{thresh}"] = _flatten_median(
            all_commitment_majority[thresh]
        )

    return summary


# ── Transfer test ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_transfer_test(
    model: nn.Module,
    tokenizer,
    adapter,
    spec,
    pt_probes: dict[int, TunedLensProbe],
    it_probes: dict[int, TunedLensProbe],
    device: torch.device,
    val_texts: list[str],
    max_seq_len: int = 512,
) -> dict:
    """Evaluate PT probes on IT activations vs IT probes on IT activations.

    Computes per-layer validation KL for both probe sets on the same
    hidden states. High transfer (similar KL) = shared geometry.
    """
    n_layers = spec.n_layers
    d_model = spec.d_model

    final_norm_mod = adapter.final_norm(model)
    lm_head_mod = adapter.lm_head(model)

    if spec.multi_gpu:
        compute_device = next(final_norm_mod.parameters()).device
    else:
        compute_device = device

    # Collect hidden states from IT model
    log.info("Collecting IT hidden states for transfer test (%d texts)...", len(val_texts))
    hidden_by_layer, final_hidden = collect_hidden_states_prefill(
        model, tokenizer, adapter, spec, val_texts, device, max_seq_len,
    )
    n_tokens = final_hidden.shape[0]
    log.info("Transfer test: %d tokens collected.", n_tokens)

    eval_n = min(n_tokens, 4096)
    model_dtype = next(final_norm_mod.parameters()).dtype
    results: dict = {"n_tokens": n_tokens, "per_layer": {}}

    for layer_idx in range(n_layers):
        if layer_idx not in hidden_by_layer:
            continue

        h = hidden_by_layer[layer_idx][:eval_n].to(compute_device)
        h_final = final_hidden[:eval_n].to(device=compute_device, dtype=model_dtype)

        # Target distribution from final layer
        target = F.log_softmax(lm_head_mod(final_norm_mod(h_final)).float(), dim=-1)

        layer_result: dict = {}

        # IT probes (matched)
        if layer_idx in it_probes:
            h_it = it_probes[layer_idx](h).to(model_dtype)
            pred_it = F.log_softmax(lm_head_mod(final_norm_mod(h_it)).float(), dim=-1)
            kl_it = F.kl_div(pred_it, target, reduction="batchmean", log_target=True).item()
            layer_result["it_probe_kl"] = kl_it

        # PT probes (transfer)
        if layer_idx in pt_probes:
            h_pt = pt_probes[layer_idx](h).to(model_dtype)
            pred_pt = F.log_softmax(lm_head_mod(final_norm_mod(h_pt)).float(), dim=-1)
            kl_pt = F.kl_div(pred_pt, target, reduction="batchmean", log_target=True).item()
            layer_result["pt_probe_kl"] = kl_pt

        # Raw (no probe)
        pred_raw = F.log_softmax(lm_head_mod(final_norm_mod(h.to(model_dtype))).float(), dim=-1)
        kl_raw = F.kl_div(pred_raw, target, reduction="batchmean", log_target=True).item()
        layer_result["raw_kl"] = kl_raw

        # Transfer ratio
        if "pt_probe_kl" in layer_result and "it_probe_kl" in layer_result:
            it_kl = layer_result["it_probe_kl"]
            pt_kl = layer_result["pt_probe_kl"]
            if it_kl > 1e-8:
                layer_result["transfer_ratio"] = pt_kl / it_kl
            else:
                layer_result["transfer_ratio"] = 1.0

        results["per_layer"][str(layer_idx)] = layer_result

        del h, h_final
        torch.cuda.empty_cache()

    # Summary: mean transfer ratio across layers
    ratios = [
        v["transfer_ratio"]
        for v in results["per_layer"].values()
        if "transfer_ratio" in v
    ]
    if ratios:
        results["mean_transfer_ratio"] = float(np.mean(ratios))
        results["transfer_ok"] = all(r < 2.0 for r in ratios)
        log.info(
            "Transfer test: mean ratio = %.2f (ok=%s)",
            results["mean_transfer_ratio"], results["transfer_ok"],
        )

    return results


# ── CLI entry point ─────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Cross-model tuned-lens probes (0G)",
    )
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--variant", choices=["pt", "it"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dataset", default="data/eval_dataset_v2.jsonl")

    # Training args
    p.add_argument("--n-tokens", type=int, default=2000000,
                   help="Target token count for training text pool (default: 2M)")
    p.add_argument("--n-steps", type=int, default=5000,
                   help="Training steps per layer (ignored for sgd_nesterov: always 250)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (ignored for sgd_nesterov: always 1.0)")
    p.add_argument("--batch-size", type=int, default=262144,
                   help="Tokens per optimizer step (default: 262144, Belrose exact)")
    p.add_argument("--micro-batch-size", type=int, default=512,
                   help="GPU micro-batch for gradient accumulation (default: 512)")
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="Max sequence length per text (default: 2048, Belrose standard)")
    p.add_argument("--optimizer", choices=["adam", "sgd_nesterov"], default="sgd_nesterov",
                   help="Optimizer: sgd_nesterov (Belrose, default) or adam")
    p.add_argument("--no-streaming", action="store_true",
                   help="Disable streaming (pre-collect all hidden states to CPU RAM)")

    # Eval args
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training, evaluate commitment with existing probes")
    p.add_argument("--n-eval-examples", type=int, default=None,
                   help="Cap number of evaluation examples")
    p.add_argument("--max-new-tokens", type=int, default=512)

    # Transfer test
    p.add_argument("--transfer-test", action="store_true",
                   help="Run PT-probes-on-IT transfer test")

    args = p.parse_args()
    spec = get_spec(args.model)
    adapter = get_adapter(args.model)
    device = torch.device(args.device)

    base_dir = spec.result_dir / "tuned_lens"

    if args.transfer_test:
        # Load IT model + both probe sets
        it_model_id = model_id_for_variant(spec, "it")
        model, tokenizer = load_model_and_tokenizer(
            it_model_id, args.device, multi_gpu=spec.multi_gpu,
        )

        pt_probe_dir = base_dir / "pt"
        it_probe_dir = base_dir / "it"

        if spec.multi_gpu:
            probe_device = next(adapter.final_norm(model).parameters()).device
        else:
            probe_device = device

        pt_probes = _load_probes(pt_probe_dir, spec.d_model, probe_device)
        it_probes = _load_probes(it_probe_dir, spec.d_model, probe_device)

        log.info("Loaded %d PT probes, %d IT probes.", len(pt_probes), len(it_probes))

        # Load a small set of texts for transfer validation
        val_texts = load_training_texts(n_tokens_target=10000, max_seq_len=args.max_seq_len)

        transfer_results = run_transfer_test(
            model, tokenizer, adapter, spec,
            pt_probes, it_probes, device, val_texts,
            max_seq_len=args.max_seq_len,
        )

        out_path = base_dir / "commitment" / "transfer_test.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(transfer_results, f, indent=2)
        log.info("Transfer test results → %s", out_path)

        del model
        torch.cuda.empty_cache()
        return

    if not args.variant:
        p.error("--variant required for training or eval mode")

    if args.eval_only:
        # Load model + probes, run commitment evaluation
        model_id = model_id_for_variant(spec, args.variant)
        model, tokenizer = load_model_and_tokenizer(
            model_id, args.device, multi_gpu=spec.multi_gpu,
        )

        probe_dir = base_dir / args.variant
        if spec.multi_gpu:
            probe_device = next(adapter.final_norm(model).parameters()).device
        else:
            probe_device = device
        probes = _load_probes(probe_dir, spec.d_model, probe_device)
        log.info("Loaded %d probes from %s.", len(probes), probe_dir)

        records = load_dataset(
            args.dataset, n_examples=args.n_eval_examples,
        )

        out_path = base_dir / "commitment" / f"tuned_lens_commitment_{args.variant}.jsonl"
        summary = eval_commitment(
            model, tokenizer, adapter, spec, probes, records, device,
            output_path=out_path,
            variant=args.variant,
            max_new_tokens=args.max_new_tokens,
        )

        # Save summary
        summary["variant"] = args.variant
        summary_path = base_dir / "commitment" / f"summary_{args.variant}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info("Eval summary → %s", summary_path)

        del model
        torch.cuda.empty_cache()
        return

    # ── Training mode ───────────────────────────────────────────────────────
    model_id = model_id_for_variant(spec, args.variant)
    model, tokenizer = load_model_and_tokenizer(
        model_id, args.device, multi_gpu=spec.multi_gpu,
    )

    # Load training texts from C4.
    # For streaming mode: load a large text pool (~500k tokens = ~5000 texts).
    # Only text strings are stored (~20 MB), NOT hidden states.
    # For non-streaming: --n-tokens controls the hidden state collection size.
    streaming = not args.no_streaming
    if streaming:
        # For Belrose SGD: 250 steps × ~500 texts/step = ~125k texts needed
        # Load 2M+ tokens (~20k texts) and cycle; each forward pass yields
        # ~400 tokens so 262k batch needs ~655 passes, cycling ~33x over pool
        pool_tokens = max(args.n_tokens, 2_000_000)
        log.info(
            "Streaming mode: loading large text pool (~%dk tokens, ~%d texts). "
            "Only strings stored in RAM.",
            pool_tokens // 1000, pool_tokens // 100,
        )
    else:
        pool_tokens = args.n_tokens

    all_texts = load_training_texts(
        n_tokens_target=pool_tokens, max_seq_len=args.max_seq_len,
    )

    # 80/20 train/val split (val is pre-collected regardless of streaming)
    split_idx = int(len(all_texts) * 0.8)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]

    log.info(
        "Training %s probes for %s: %d train + %d val texts (%d steps/layer, %s)",
        args.variant, args.model, len(train_texts), len(val_texts), args.n_steps,
        "streaming" if streaming else "pre-collected",
    )

    output_dir = base_dir / args.variant
    train_probes(
        model, tokenizer, adapter, spec,
        train_texts, val_texts, device, output_dir,
        n_steps=args.n_steps, lr=args.lr, batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        max_seq_len=args.max_seq_len,
        optimizer_type=args.optimizer,
        streaming=not args.no_streaming,
    )

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
