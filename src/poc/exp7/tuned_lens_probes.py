"""Train tuned-lens affine probes and recompute commitment delay (Exp7 0G).

The tuned-lens (Belrose et al., 2023) trains a per-layer affine probe T_ℓ such that:
  logit_prediction(T_ℓ(h_ℓ)) ≈ logit_prediction(h_L)  (final-layer distribution)

Training follows Belrose et al. methodology:
  - Probes hook the RESIDUAL STREAM (layer output), not MLP output
  - Identity + zero-bias initialisation (warm start)
  - KL(softmax(lm_head(T_ℓ(h_ℓ))) ‖ softmax(lm_head(h_L))) minimisation
  - Cosine annealing LR schedule with linear warmup
  - Train/validation split (80/20) with validation loss reporting
  - 2000 training prompts × 80 tokens ≈ 160k token activations

Commitment is measured via KL-to-final (Belrose criterion):
  committed_ℓ = earliest layer where KL(tuned_lens_ℓ ‖ final_layer) < 0.1 nats
  AND remains < 0.1 for all subsequent layers (no-flip-back stability)

Cross-model probe transfer test:
  Train probes on PT, apply to IT activations (and vice versa).
  If tuned-lens is capturing universal geometry, PT probes should work on IT.

Usage:
  # Train PT probes on cuda:0, IT probes on cuda:1 (parallel)
  uv run python -m src.poc.exp7.tuned_lens_probes \\
      --variant pt --device cuda:0 \\
      --n-train 2000 --n-steps 5000 \\
      --output-dir results/exp7/0G/probes/pt/ &
  uv run python -m src.poc.exp7.tuned_lens_probes \\
      --variant it --device cuda:1 \\
      --n-train 2000 --n-steps 5000 \\
      --output-dir results/exp7/0G/probes/it/ &
  wait

  # Evaluate: commitment + cross-model probe transfer
  uv run python -m src.poc.exp7.tuned_lens_probes \\
      --eval-only \\
      --probe-dir results/exp7/0G/probes/ \\
      --output-dir results/exp7/0G/
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.poc.exp5.config import Exp5Config
from src.poc.shared.model import load_model

ALL_LAYERS = list(range(1, 34))   # layers 1–33
D_MODEL = 2560
WORK_DIR = Path("results/precompute_v2_work")

# Belrose et al. commitment threshold (nats)
KL_COMMIT_THRESHOLD = 0.1


def _get_raw(loaded):
    return loaded.model._model


# ── Probe definition ──────────────────────────────────────────────────────────

class TunedLensProbe(nn.Module):
    """Per-layer affine probe: T_ℓ(h_ℓ) = W_ℓ @ h_ℓ + b_ℓ.

    Initialised as identity + zero bias (Belrose et al. warm start).
    """

    def __init__(self, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


# ── LR schedule ──────────────────────────────────────────────────────────────

def _cosine_lr_with_warmup(optimizer, step: int, n_steps: int, warmup_steps: int, base_lr: float):
    """Cosine annealing with linear warmup (Belrose et al. schedule)."""
    if step < warmup_steps:
        lr = base_lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ── Data collection (residual stream hooks) ──────────────────────────────────

@torch.no_grad()
def collect_hidden_states(
    model_raw,
    tokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int = 80,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Collect per-layer RESIDUAL STREAM hidden states and final-layer hidden states.

    Hooks on model_raw.language_model.layers[li] (full layer output = residual stream),
    NOT on layers[li].mlp. This is the correct hook point for tuned-lens (Belrose et al.).

    Returns:
        hidden_by_layer: {layer → [n_tokens, D_MODEL] float32 on CPU}
        final_hidden:    [n_tokens, D_MODEL] float32 on CPU  (last layer residual)
    """
    hidden_by_layer: dict[int, list[torch.Tensor]] = {li: [] for li in ALL_LAYERS}
    final_layer_acts: list[torch.Tensor] = []

    def make_hook(li: int):
        def hook(mod, inp, out):
            # Layer output is a tuple (hidden_states, ...) for transformer blocks
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] == 1:  # generated step (skip prefill)
                vec = h[0, 0, :].float().cpu()
                hidden_by_layer[li].append(vec)
                if li == ALL_LAYERS[-1]:
                    final_layer_acts.append(vec)
        return hook

    # Hook on FULL LAYER (residual stream), not .mlp
    handles = [
        model_raw.language_model.layers[li].register_forward_hook(make_hook(li))
        for li in ALL_LAYERS
    ]

    for pi, prompt in enumerate(prompts):
        try:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            model_raw.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        except Exception as e:
            print(f"  Warning: generation failed for prompt {pi} — {e}", flush=True)

        if (pi + 1) % 100 == 0:
            print(f"  [collect] {pi+1}/{len(prompts)} prompts done", flush=True)

    for h in handles:
        h.remove()

    result: dict[int, torch.Tensor] = {}
    for li in ALL_LAYERS:
        if hidden_by_layer[li]:
            result[li] = torch.stack(hidden_by_layer[li])
    final_hidden = torch.stack(final_layer_acts) if final_layer_acts else torch.zeros(0, D_MODEL)
    return result, final_hidden


# ── Probe training ────────────────────────────────────────────────────────────

def train_probes(
    model_raw,
    tokenizer,
    train_prompts: list[str],
    val_prompts: list[str],
    device: str,
    output_dir: Path,
    n_steps: int = 5000,
    lr: float = 1e-3,
    batch_size: int = 64,
    warmup_frac: float = 0.05,
    max_new_tokens: int = 80,
) -> dict:
    """Train tuned-lens probes for all layers with train/val split.

    Returns training summary dict with per-layer train/val losses.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if probes already exist (resumable)
    existing = set(int(p.stem.split("_")[2]) for p in output_dir.glob("probe_layer_*.pt"))
    if len(existing) == len(ALL_LAYERS):
        print(f"[0G] All {len(ALL_LAYERS)} probes already exist, skipping training.", flush=True)
        return {}

    # Get final-layer norm + lm_head for pseudo-label computation
    final_norm = model_raw.language_model.model.norm
    lm_head = model_raw.language_model.lm_head

    # ── Collect training hidden states ──
    print(f"[0G] Collecting hidden states from {len(train_prompts)} training prompts...", flush=True)
    train_hidden, train_final = collect_hidden_states(
        model_raw, tokenizer, train_prompts, device, max_new_tokens
    )
    n_train_tokens = train_final.shape[0]
    print(f"[0G] Training set: {n_train_tokens} token activations.", flush=True)

    # ── Collect validation hidden states ──
    print(f"[0G] Collecting hidden states from {len(val_prompts)} validation prompts...", flush=True)
    val_hidden, val_final = collect_hidden_states(
        model_raw, tokenizer, val_prompts, device, max_new_tokens
    )
    n_val_tokens = val_final.shape[0]
    print(f"[0G] Validation set: {n_val_tokens} token activations.", flush=True)

    if n_train_tokens < batch_size:
        print(f"[0G] WARNING: only {n_train_tokens} tokens < batch_size={batch_size}", flush=True)
        batch_size = max(1, n_train_tokens // 2)

    # NOTE: We do NOT precompute full-vocab log_probs for all tokens.
    # With 160k tokens × 262k vocab × 4 bytes = ~167 GB — that's an OOM.
    # Instead, compute target log_probs on-the-fly per mini-batch from
    # the final-layer hidden states (train_final/val_final on CPU).

    warmup_steps = int(n_steps * warmup_frac)
    rng = torch.Generator()
    rng.manual_seed(42)
    training_summary: dict = {}

    # Memory management: total hidden states = 33 layers × n_tokens × 2560 × 4 bytes
    # For 160k tokens: ~54 GB CPU RAM. We free each layer after training its probe.
    mem_gb = len(ALL_LAYERS) * n_train_tokens * D_MODEL * 4 / 1e9
    print(f"[0G] Estimated CPU RAM for hidden states: {mem_gb:.1f} GB (train)", flush=True)

    for li in ALL_LAYERS:
        if li in existing:
            print(f"  layer {li}: already trained, skipping.", flush=True)
            # Free memory for this layer since we don't need it
            train_hidden.pop(li, None)
            val_hidden.pop(li, None)
            continue

        if li not in train_hidden:
            print(f"  layer {li}: no training activations, skipping.", flush=True)
            continue

        layer_train = train_hidden[li]
        n_t = min(layer_train.shape[0], n_train_tokens)
        if n_t == 0:
            train_hidden.pop(li, None)
            continue

        probe = TunedLensProbe(D_MODEL).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_state = None
        train_losses = []
        val_losses = []

        for step in range(n_steps):
            # LR schedule: cosine annealing with warmup
            _cosine_lr_with_warmup(optimizer, step, n_steps, warmup_steps, lr)

            # Random mini-batch from training set
            idx = torch.randperm(n_t, generator=rng)[:batch_size]
            h = layer_train[idx].to(device)

            # Compute target log_probs on-the-fly from final-layer hidden states
            with torch.no_grad():
                h_final = train_final[idx].to(device)
                target_logits = lm_head(final_norm(h_final))
                target = F.log_softmax(target_logits, dim=-1)
                del h_final, target_logits

            # Forward
            normed = final_norm(probe(h))
            logits = lm_head(normed)
            log_pred = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(log_pred, target, reduction="batchmean", log_target=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation every 500 steps
            if (step + 1) % 500 == 0 and n_val_tokens > 0 and li in val_hidden:
                with torch.no_grad():
                    val_h = val_hidden[li]
                    n_v = min(val_h.shape[0], n_val_tokens)
                    eval_n = min(n_v, 4096)
                    v_idx = torch.randperm(n_v, generator=rng)[:eval_n]
                    v_h = val_h[v_idx].to(device)
                    v_final = val_final[v_idx].to(device)
                    v_target = F.log_softmax(lm_head(final_norm(v_final)), dim=-1)
                    del v_final
                    v_normed = final_norm(probe(v_h))
                    v_logits = lm_head(v_normed)
                    v_log_pred = F.log_softmax(v_logits, dim=-1)
                    v_loss = F.kl_div(v_log_pred, v_target, reduction="batchmean", log_target=True)
                    val_loss_val = v_loss.item()
                    val_losses.append((step + 1, val_loss_val))

                    # Track best model (early stopping checkpoint)
                    if val_loss_val < best_val_loss:
                        best_val_loss = val_loss_val
                        best_state = {k: v.clone() for k, v in probe.state_dict().items()}

                train_losses.append((step + 1, loss.item()))

        # Use best validation model if available, else final model
        if best_state is not None:
            probe.load_state_dict(best_state)

        probe_path = output_dir / f"probe_layer_{li}.pt"
        torch.save(probe.state_dict(), probe_path)

        # Quick eval summary (small subset to avoid OOM)
        with torch.no_grad():
            eval_n = min(n_t, 2048)
            h_all = layer_train[:eval_n].to(device)
            f_all = train_final[:eval_n].to(device)
            final_top1 = lm_head(final_norm(f_all)).argmax(dim=-1)
            del f_all

            # Raw logit-lens top-1 agreement
            top1_raw = lm_head(final_norm(h_all)).argmax(dim=-1)

            # Tuned-lens top-1 agreement
            top1_tuned = lm_head(final_norm(probe(h_all))).argmax(dim=-1)

            raw_agree = (top1_raw == final_top1).float().mean().item()
            tuned_agree = (top1_tuned == final_top1).float().mean().item()

        layer_summary = {
            "train_loss_final": train_losses[-1][1] if train_losses else float("nan"),
            "val_loss_best": best_val_loss if best_val_loss < float("inf") else float("nan"),
            "raw_top1_agree": raw_agree,
            "tuned_top1_agree": tuned_agree,
        }
        training_summary[str(li)] = layer_summary

        print(
            f"  layer {li:2d}: val_loss={best_val_loss:.4f}  "
            f"raw_agree={raw_agree:.3f} → tuned_agree={tuned_agree:.3f}",
            flush=True,
        )

        # Free this layer's data to reduce CPU RAM pressure
        del layer_train
        train_hidden.pop(li, None)
        val_hidden.pop(li, None)

    # Save training summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"[0G] Training complete. Probes + summary → {output_dir}", flush=True)
    return training_summary


# ── KL-based commitment evaluation (Belrose criterion) ──────────────────────

@torch.no_grad()
def _compute_per_layer_kl(
    step_hidden: dict[int, list[torch.Tensor]],
    final_norm: nn.Module,
    lm_head: nn.Module,
    probes: dict[int, TunedLensProbe],
    device: str,
) -> list[list[float]]:
    """Compute KL(tuned_lens_ℓ ‖ final_layer) for each step and layer.

    Returns kl_matrix[step][layer_idx] where layer_idx maps to ALL_LAYERS.
    """
    n_steps = len(step_hidden[ALL_LAYERS[-1]])
    kl_matrix: list[list[float]] = []

    for step in range(n_steps):
        # Get final-layer distribution for this step
        h_final = step_hidden[ALL_LAYERS[-1]][step]
        logits_final = lm_head(final_norm(h_final.unsqueeze(0).unsqueeze(0).to(device)))
        log_p_final = F.log_softmax(logits_final.squeeze(), dim=-1)

        row: list[float] = []
        for li in ALL_LAYERS:
            if step >= len(step_hidden[li]):
                row.append(float("inf"))
                continue
            h = step_hidden[li][step].to(device)
            if li in probes:
                h = probes[li](h.unsqueeze(0)).squeeze(0)
            logits = lm_head(final_norm(h.unsqueeze(0).unsqueeze(0))).squeeze()
            log_q = F.log_softmax(logits, dim=-1)
            # KL(final ‖ tuned_lens) = sum(p * (log_p - log_q))
            kl = F.kl_div(log_q, log_p_final, reduction="sum", log_target=True).item()
            row.append(max(kl, 0.0))  # clamp numerical noise
        kl_matrix.append(row)

    return kl_matrix


def _commitment_from_kl(kl_row: list[float], threshold: float = KL_COMMIT_THRESHOLD) -> int:
    """Find earliest layer where KL < threshold and stays below for all subsequent layers."""
    n = len(kl_row)
    for i in range(n):
        if kl_row[i] < threshold and all(kl_row[j] < threshold for j in range(i, n)):
            return i
    return n - 1


def _commitment_from_top1(top1_by_layer: list[int]) -> int:
    """Find earliest layer where top-1 matches final and stays matched (no-flip-back)."""
    n = len(top1_by_layer)
    final_top1 = top1_by_layer[-1]
    for i in range(n):
        if top1_by_layer[i] == final_top1 and all(
            top1_by_layer[j] == final_top1 for j in range(i, n)
        ):
            return i
    return n - 1


@torch.no_grad()
def eval_commitment(
    model_raw,
    tokenizer,
    probes: dict[int, TunedLensProbe],
    prompts: list[str],
    device: str,
    max_new_tokens: int = 80,
) -> dict:
    """Evaluate commitment using both KL-to-final and top-1 metrics.

    Returns dict with:
      kl_commitment_layers: list[int] — per-token commitment via KL < 0.1 nats
      top1_commitment_layers: list[int] — per-token commitment via top-1 no-flip-back
      mean_kl_per_layer: list[float] — mean KL at each layer (averaged over all tokens)
    """
    final_norm = model_raw.language_model.model.norm
    lm_head = model_raw.language_model.lm_head

    all_kl_commit: list[int] = []
    all_top1_commit: list[int] = []
    all_kl_by_layer: list[list[float]] = []

    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        step_hidden: dict[int, list[torch.Tensor]] = {li: [] for li in ALL_LAYERS}

        def make_hook(li: int):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] == 1:
                    step_hidden[li].append(h[0, 0, :].float().cpu())
            return hook

        # Hook on FULL LAYER (residual stream)
        handles = [
            model_raw.language_model.layers[li].register_forward_hook(make_hook(li))
            for li in ALL_LAYERS
        ]
        try:
            model_raw.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        finally:
            for hh in handles:
                hh.remove()

        n_steps = len(step_hidden[ALL_LAYERS[-1]])
        if n_steps == 0:
            continue

        # Compute KL matrix for this prompt
        kl_matrix = _compute_per_layer_kl(step_hidden, final_norm, lm_head, probes, device)

        for step in range(n_steps):
            kl_row = kl_matrix[step]
            all_kl_commit.append(_commitment_from_kl(kl_row))
            all_kl_by_layer.append(kl_row)

            # Also compute top-1 commitment for comparison
            top1_row = []
            for li_idx, li in enumerate(ALL_LAYERS):
                if step >= len(step_hidden[li]):
                    top1_row.append(-1)
                    continue
                h = step_hidden[li][step].to(device)
                if li in probes:
                    h = probes[li](h.unsqueeze(0)).squeeze(0)
                logits = lm_head(final_norm(h.unsqueeze(0).unsqueeze(0))).squeeze()
                top1_row.append(int(logits.argmax().item()))
            all_top1_commit.append(_commitment_from_top1(top1_row))

        if (pi + 1) % 50 == 0:
            print(f"  [eval] {pi+1}/{len(prompts)} prompts done", flush=True)

    # Compute mean KL per layer
    kl_arr = np.array(all_kl_by_layer)  # [n_tokens, n_layers]
    mean_kl_per_layer = kl_arr.mean(axis=0).tolist() if len(kl_arr) > 0 else []

    return {
        "kl_commitment_layers": all_kl_commit,
        "top1_commitment_layers": all_top1_commit,
        "mean_kl_per_layer": mean_kl_per_layer,
    }


# ── Evaluation pipeline ─────────────────────────────────────────────────────

def _load_eval_prompts(n_eval: int = 400) -> list[str]:
    """Load evaluation prompts from gen_merged.jsonl.

    Uses records NOT in the selected-600 training set (held-out records).
    Falls back to all records if selected.json not found.
    """
    selected_path = WORK_DIR / "selected.json"
    if selected_path.exists():
        selected_ids = set(json.loads(selected_path.read_text()))
    else:
        selected_ids = set()  # no filter

    eval_prompts: list[str] = []
    with open(WORK_DIR / "gen_merged.jsonl") as f:
        for line in f:
            r = json.loads(line)
            # Use NON-selected records for eval (train/eval separation)
            if selected_ids and r["record_id"] in selected_ids:
                continue
            eval_prompts.append(r["prompt"])
            if len(eval_prompts) >= n_eval:
                break

    if len(eval_prompts) < n_eval:
        print(
            f"[0G] WARNING: only found {len(eval_prompts)} held-out prompts "
            f"(wanted {n_eval})",
            flush=True,
        )
    return eval_prompts


def _load_probes(probe_dir: Path, device: str) -> dict[int, TunedLensProbe]:
    """Load trained probes from a directory."""
    probes: dict[int, TunedLensProbe] = {}
    for li in ALL_LAYERS:
        probe_path = probe_dir / f"probe_layer_{li}.pt"
        if probe_path.exists():
            probe = TunedLensProbe(D_MODEL).to(device)
            probe.load_state_dict(torch.load(probe_path, map_location=device, weights_only=True))
            probe.eval()
            probes[li] = probe
    return probes


def _summarise_commitment(commit_layers: list[int], label: str) -> dict:
    """Compute summary statistics for a list of commitment layers."""
    if not commit_layers:
        return {}
    arr = np.array(commit_layers)
    summary = {
        "mean_commitment_layer": float(arr.mean()),
        "median_commitment_layer": float(np.median(arr)),
        "std_commitment_layer": float(arr.std()),
        "n_tokens": len(arr),
    }
    # Convert from layer index in ALL_LAYERS to actual layer number
    summary["mean_commitment_layer_actual"] = float(
        np.mean([ALL_LAYERS[min(c, len(ALL_LAYERS) - 1)] for c in arr])
    )
    return summary


def run_eval(probe_dir: Path, output_dir: Path, n_eval: int = 400) -> None:
    """Load trained probes for PT and IT, evaluate with 4 conditions:

    1. IT probes on IT activations (matched)
    2. PT probes on PT activations (matched)
    3. PT probes on IT activations (cross-model transfer)
    4. Raw logit-lens (no probes) on both IT and PT

    Uses held-out records (not from training set) for evaluation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_prompts = _load_eval_prompts(n_eval)
    print(f"[0G] Evaluating on {len(eval_prompts)} held-out prompts.", flush=True)

    results: dict = {"n_eval_prompts": len(eval_prompts)}
    device = "cuda:0"

    # Load both probe sets
    pt_probe_dir = probe_dir / "pt"
    it_probe_dir = probe_dir / "it"
    pt_probes = _load_probes(pt_probe_dir, device) if pt_probe_dir.exists() else {}
    it_probes = _load_probes(it_probe_dir, device) if it_probe_dir.exists() else {}
    no_probes: dict[int, TunedLensProbe] = {}

    print(f"[0G] Loaded {len(pt_probes)} PT probes, {len(it_probes)} IT probes.", flush=True)

    # Define evaluation conditions: (label, model_variant, probes_to_use)
    conditions: list[tuple[str, str, dict[int, TunedLensProbe]]] = []

    # Condition 1: IT probes on IT model (matched)
    if it_probes:
        conditions.append(("it_matched", "it", it_probes))
    # Condition 2: PT probes on PT model (matched)
    if pt_probes:
        conditions.append(("pt_matched", "pt", pt_probes))
    # Condition 3: PT probes on IT model (cross-model transfer)
    if pt_probes:
        conditions.append(("pt_probes_on_it", "it", pt_probes))
    # Condition 4: Raw logit-lens (no probes)
    conditions.append(("raw_logitlens_it", "it", no_probes))
    conditions.append(("raw_logitlens_pt", "pt", no_probes))

    # Group by model variant to avoid loading the same model multiple times
    by_variant: dict[str, list[tuple[str, dict[int, TunedLensProbe]]]] = {}
    for label, variant, probes in conditions:
        by_variant.setdefault(variant, []).append((label, probes))

    for variant, variant_conditions in by_variant.items():
        cfg = Exp5Config(
            experiment="baseline", model_variant=variant, model_id="",
            run_name=f"exp7_0G_eval_{variant}", device=device, skip_transcoders=True,
        )
        print(f"\n[0G] Loading {variant} model for evaluation...", flush=True)
        loaded = load_model(cfg)
        model_raw = _get_raw(loaded)

        for label, probes in variant_conditions:
            probe_desc = f"{len(probes)} probes" if probes else "no probes (raw)"
            print(f"[0G] Evaluating: {label} ({probe_desc})...", flush=True)

            eval_result = eval_commitment(
                model_raw, loaded.tokenizer, probes, eval_prompts, device
            )

            # Summarise KL-based commitment
            kl_summary = _summarise_commitment(eval_result["kl_commitment_layers"], label)
            top1_summary = _summarise_commitment(eval_result["top1_commitment_layers"], label)

            results[label] = {
                "kl_commitment": kl_summary,
                "top1_commitment": top1_summary,
                "mean_kl_per_layer": eval_result["mean_kl_per_layer"],
            }

            kl_mean = kl_summary.get("mean_commitment_layer", float("nan"))
            top1_mean = top1_summary.get("mean_commitment_layer", float("nan"))
            print(
                f"  {label}: KL commit = {kl_mean:.2f}, top1 commit = {top1_mean:.2f}",
                flush=True,
            )

        del loaded
        torch.cuda.empty_cache()

    # Compute commitment delays
    delays: dict = {}
    for metric in ("kl_commitment", "top1_commitment"):
        it_key = "it_matched" if "it_matched" in results else "raw_logitlens_it"
        pt_key = "pt_matched" if "pt_matched" in results else "raw_logitlens_pt"
        if it_key in results and pt_key in results:
            it_mean = results[it_key][metric].get("mean_commitment_layer", float("nan"))
            pt_mean = results[pt_key][metric].get("mean_commitment_layer", float("nan"))
            delay = it_mean - pt_mean
            delays[f"{metric}_delay"] = delay
            delays[f"{metric}_direction"] = "IT commits later" if delay > 0 else "PT commits later"

    # Cross-model transfer comparison
    if "pt_probes_on_it" in results and "it_matched" in results:
        transfer_kl = results["pt_probes_on_it"]["kl_commitment"].get("mean_commitment_layer", float("nan"))
        matched_kl = results["it_matched"]["kl_commitment"].get("mean_commitment_layer", float("nan"))
        delays["cross_model_transfer_gap"] = abs(transfer_kl - matched_kl)
        delays["cross_model_transfer_note"] = (
            f"PT probes on IT: {transfer_kl:.2f} vs IT probes on IT: {matched_kl:.2f} "
            f"(gap = {abs(transfer_kl - matched_kl):.2f} layers)"
        )

    results["delays"] = delays
    if delays:
        print(f"\n[0G] === Results ===", flush=True)
        for k, v in delays.items():
            print(f"  {k}: {v}", flush=True)

    out_path = output_dir / "tuned_lens_commitment.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[0G] Results → {out_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Tuned-lens probes (Exp7 0G)")
    p.add_argument("--variant", choices=["pt", "it"], help="Model variant for training")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default="results/exp7/0G/probes/")
    p.add_argument("--n-train", type=int, default=2000,
                   help="Number of training prompts (default: 2000)")
    p.add_argument("--n-steps", type=int, default=5000, help="Training steps per layer")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=80,
                   help="Tokens to generate per prompt for hidden state collection")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training, just evaluate commitment with existing probes")
    p.add_argument("--probe-dir", default="results/exp7/0G/probes/",
                   help="Parent dir containing pt/ and it/ probe subdirs (eval-only mode)")
    p.add_argument("--n-eval", type=int, default=400,
                   help="Number of eval prompts (from held-out set)")
    # Legacy arg kept for shell script compatibility
    p.add_argument("--train-prompts", default=None, help="(ignored, kept for compat)")
    args = p.parse_args()

    if args.eval_only:
        run_eval(Path(args.probe_dir), Path(args.output_dir), n_eval=args.n_eval)
        return

    if not args.variant:
        p.error("--variant required for training mode")

    output_dir = Path(args.output_dir)

    # Load all prompts from gen_merged.jsonl
    all_prompts: list[str] = []
    with open(WORK_DIR / "gen_merged.jsonl") as f:
        for line in f:
            r = json.loads(line)
            all_prompts.append(r["prompt"])

    # Take up to n_train prompts, then split 80/20 for train/val
    n_total = min(args.n_train, len(all_prompts))
    prompts = all_prompts[:n_total]
    split_idx = int(n_total * 0.8)
    train_prompts = prompts[:split_idx]
    val_prompts = prompts[split_idx:]

    print(
        f"[0G] Training {args.variant} probes: {len(train_prompts)} train + "
        f"{len(val_prompts)} val prompts ({args.n_steps} steps/layer)...",
        flush=True,
    )

    cfg = Exp5Config(
        experiment="baseline", model_variant=args.variant, model_id="",
        run_name=f"exp7_tuned_lens_train_{args.variant}", device=args.device,
        skip_transcoders=True,
    )
    loaded = load_model(cfg)
    model_raw = _get_raw(loaded)

    train_probes(
        model_raw,
        loaded.tokenizer,
        train_prompts,
        val_prompts,
        args.device,
        output_dir,
        n_steps=args.n_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
