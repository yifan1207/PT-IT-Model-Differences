"""Train tuned-lens affine probes and recompute commitment delay (Exp7 0G).

The tuned-lens (Belrose et al., 2023) trains a per-layer affine probe T_ℓ such that:
  logit_prediction(T_ℓ(h_ℓ)) ≈ logit_prediction(h_L)  (final-layer distribution)

Training: T_ℓ = nn.Linear(D_MODEL, D_MODEL) minimising
  KL( softmax(lm_head(T_ℓ(h_ℓ))) || softmax(lm_head(h_L)) )

After training, we recompute commitment layer using tuned-lens predictions instead
of raw logit-lens, checking whether IT still commits later than PT.

Usage:
  # Train PT probes on cuda:0, IT probes on cuda:1 (parallel)
  uv run python -m src.poc.exp7.tuned_lens_probes \\
      --variant pt --device cuda:0 \\
      --output-dir results/exp7/0G/probes/pt/ &
  uv run python -m src.poc.exp7.tuned_lens_probes \\
      --variant it --device cuda:1 \\
      --output-dir results/exp7/0G/probes/it/ &
  wait

  # Evaluate: recompute commitment using trained probes
  uv run python -m src.poc.exp7.tuned_lens_probes \\
      --eval-only \\
      --probe-dir results/exp7/0G/probes/ \\
      --output-dir results/exp7/0G/

  # Quick test (few records, few steps)
  uv run python -m src.poc.exp7.tuned_lens_probes \\
      --variant it --device cuda:0 \\
      --n-train 20 --n-steps 100 \\
      --output-dir results/exp7/0G/probes/it_test/
"""
from __future__ import annotations

import argparse
import json
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


def _get_raw(loaded):
    return loaded.model._model


# ── Probe definition ──────────────────────────────────────────────────────────

class TunedLensProbe(nn.Module):
    """Per-layer affine probe: T_ℓ(h_ℓ) = W_ℓ @ h_ℓ + b_ℓ."""

    def __init__(self, d_model: int = D_MODEL) -> None:
        super().__init__()
        # Initialise as identity + zero bias (warm start)
        self.linear = nn.Linear(d_model, d_model, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


# ── Data collection ───────────────────────────────────────────────────────────

@torch.no_grad()
def collect_hidden_states(
    model_raw,
    tokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int = 40,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Collect per-layer hidden states and final-layer hidden states via hooks.

    Returns:
        hidden_by_layer: {layer → [n_tokens, D_MODEL] float32 on CPU}
        final_hidden:    [n_tokens, D_MODEL] float32 on CPU  (layer 33 output)
    """
    hidden_by_layer: dict[int, list[torch.Tensor]] = {li: [] for li in ALL_LAYERS}
    # Layer 33 (index 33 in 0-based, but our ALL_LAYERS is 1-33 in 1-based hook naming)
    final_layer_acts: list[torch.Tensor] = []

    def make_hook(li: int):
        def hook(mod, inp, out):
            if out.shape[1] == 1:  # generated step
                h = out[0, 0, :].float().cpu()
                hidden_by_layer[li].append(h)
                if li == ALL_LAYERS[-1]:
                    final_layer_acts.append(h)
        return hook

    handles = [
        model_raw.language_model.layers[li].mlp.register_forward_hook(make_hook(li))
        for li in ALL_LAYERS
    ]

    for prompt in prompts:
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
            print(f"  Warning: generation failed for prompt — {e}", flush=True)

    for h in handles:
        h.remove()

    result: dict[int, torch.Tensor] = {}
    for li in ALL_LAYERS:
        if hidden_by_layer[li]:
            result[li] = torch.stack(hidden_by_layer[li])  # [T, D]
    final_hidden = torch.stack(final_layer_acts) if final_layer_acts else torch.zeros(0, D_MODEL)
    return result, final_hidden


# ── Probe training ────────────────────────────────────────────────────────────

def train_probes(
    model_raw,
    tokenizer,
    prompts: list[str],
    device: str,
    output_dir: Path,
    n_steps: int = 2000,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_new_tokens: int = 40,
) -> None:
    """Train tuned-lens probes for all layers and save to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if probes already exist (resumable)
    existing = set(int(p.stem.split("_")[2]) for p in output_dir.glob("probe_layer_*.pt"))
    if len(existing) == len(ALL_LAYERS):
        print(f"[0G] All {len(ALL_LAYERS)} probes already exist, skipping training.", flush=True)
        return

    # Get final-layer norm + lm_head for pseudo-label computation
    final_norm = model_raw.language_model.model.norm
    lm_head = model_raw.language_model.lm_head

    print(f"[0G] Collecting hidden states from {len(prompts)} training prompts...", flush=True)
    hidden_by_layer, final_hidden = collect_hidden_states(
        model_raw, tokenizer, prompts, device, max_new_tokens
    )
    n_tokens = final_hidden.shape[0]
    print(f"[0G] Collected {n_tokens} token activations.", flush=True)

    if n_tokens < batch_size:
        print(f"[0G] WARNING: only {n_tokens} tokens < batch_size={batch_size}, reducing.", flush=True)
        batch_size = max(1, n_tokens // 2)

    # Compute final-layer pseudo-labels (softmax distribution under lm_head)
    with torch.no_grad():
        # Use log_softmax of final layer's normed hidden state as targets
        final_hidden_dev = final_hidden.to(device)
        final_normed = final_norm(final_hidden_dev)
        # Use top-5000 vocabulary slice for KL (full 262k is very slow)
        logits_final = lm_head(final_normed)  # [n_tokens, vocab]
        log_probs_final = F.log_softmax(logits_final, dim=-1).cpu()  # [n_tokens, vocab]
        del final_hidden_dev, final_normed, logits_final

    rng = torch.Generator()
    rng.manual_seed(42)

    for li in ALL_LAYERS:
        if li in existing:
            print(f"  layer {li}: already trained, skipping.", flush=True)
            continue

        if li not in hidden_by_layer:
            print(f"  layer {li}: no activations, skipping.", flush=True)
            continue

        layer_hidden = hidden_by_layer[li]  # [n_tokens, D_MODEL] on CPU
        n = min(layer_hidden.shape[0], n_tokens)
        if n == 0:
            continue

        probe = TunedLensProbe(D_MODEL).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

        # Training loop
        for step in range(n_steps):
            # Random mini-batch
            idx = torch.randperm(n, generator=rng)[:batch_size]
            h = layer_hidden[idx].to(device)         # [B, D]
            target = log_probs_final[idx].to(device)  # [B, vocab]

            # Forward: transform hidden state and compute logits
            normed = final_norm(probe(h))
            logits = lm_head(normed)                  # [B, vocab]

            # KL loss: KL(target || prediction) = sum(target * (target - log_pred))
            log_pred = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(log_pred, target.exp(), reduction="batchmean", log_target=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe_path = output_dir / f"probe_layer_{li}.pt"
        torch.save(probe.state_dict(), probe_path)

        # Quick eval: cosine of top-1 prediction agreement
        with torch.no_grad():
            h_all = layer_hidden[:n].to(device)
            normed_raw = final_norm(h_all)
            logits_raw = lm_head(normed_raw)
            top1_raw = logits_raw.argmax(dim=-1)

            h_tuned = probe(h_all)
            normed_tuned = final_norm(h_tuned)
            logits_tuned = lm_head(normed_tuned)
            top1_tuned = logits_tuned.argmax(dim=-1)

            final_top1 = log_probs_final[:n].to(device).argmax(dim=-1)
            raw_agree = (top1_raw == final_top1).float().mean().item()
            tuned_agree = (top1_tuned == final_top1).float().mean().item()

        print(
            f"  layer {li:2d}: loss={loss.item():.4f}  "
            f"raw_agree={raw_agree:.3f} → tuned_agree={tuned_agree:.3f}",
            flush=True,
        )

    print(f"[0G] Training complete. Probes saved to {output_dir}", flush=True)


# ── Commitment evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def eval_commitment_tuned(
    model_raw,
    tokenizer,
    probes: dict[int, TunedLensProbe],
    prompts: list[str],
    device: str,
    max_new_tokens: int = 80,
) -> list[list[int]]:
    """Evaluate commitment layer using tuned-lens top-1 predictions.

    Returns list of commitment layers (one per generated token across all prompts).
    """
    final_norm = model_raw.language_model.model.norm
    lm_head = model_raw.language_model.lm_head
    n_layers = len(ALL_LAYERS)

    all_commitment_layers = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Collect per-layer per-step hidden states
        step_hidden: dict[int, list[torch.Tensor]] = {li: [] for li in ALL_LAYERS}

        def make_hook(li: int):
            def hook(mod, inp, out):
                if out.shape[1] == 1:
                    step_hidden[li].append(out[0, 0, :].float())
            return hook

        handles = [
            model_raw.language_model.layers[li].mlp.register_forward_hook(make_hook(li))
            for li in ALL_LAYERS
        ]
        model_raw.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        for h in handles:
            h.remove()

        # For each generated step, compute tuned-lens top-1 per layer
        n_steps = len(step_hidden[ALL_LAYERS[-1]])
        for step in range(n_steps):
            top1_by_layer = []
            for li in ALL_LAYERS:
                if step >= len(step_hidden[li]):
                    top1_by_layer.append(-1)
                    continue
                h = step_hidden[li][step]
                if li in probes:
                    h_transformed = probes[li](h.unsqueeze(0)).squeeze(0)
                else:
                    h_transformed = h
                logits = lm_head(final_norm(h_transformed.unsqueeze(0).unsqueeze(0))).squeeze()
                top1_by_layer.append(int(logits.argmax().item()))

            # Compute commitment layer (no-flip-back criterion)
            final_top1 = top1_by_layer[-1]
            commit = n_layers - 1
            for i, top1 in enumerate(top1_by_layer):
                if top1 == final_top1 and all(t == final_top1 for t in top1_by_layer[i:]):
                    commit = i
                    break
            all_commitment_layers.append(commit)

    return all_commitment_layers


# ── Evaluation pipeline ───────────────────────────────────────────────────────

def run_eval(probe_dir: Path, output_dir: Path) -> None:
    """Load trained probes for PT and IT, recompute commitment, compare to raw lens."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts for evaluation (use selected 600 from precompute)
    selected_ids: set[str] = set(json.loads((WORK_DIR / "selected.json").read_text()))
    eval_prompts = []
    with open(WORK_DIR / "gen_merged.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["record_id"] in selected_ids:
                eval_prompts.append(r["prompt"])
    eval_prompts = eval_prompts[:200]  # 200 prompts for eval is sufficient

    results = {}

    for variant in ("pt", "it"):
        variant_probe_dir = probe_dir / variant
        if not variant_probe_dir.exists():
            print(f"[0G] No probes found for {variant} at {variant_probe_dir}", flush=True)
            continue

        device = "cuda:0"
        cfg = Exp5Config(
            experiment="baseline", model_variant=variant, model_id="",
            run_name=f"exp7_tuned_lens_{variant}", device=device, skip_transcoders=True,
        )
        print(f"[0G] Loading {variant} model...", flush=True)
        loaded = load_model(cfg)
        model_raw = _get_raw(loaded)

        # Load probes
        probes: dict[int, TunedLensProbe] = {}
        for li in ALL_LAYERS:
            probe_path = variant_probe_dir / f"probe_layer_{li}.pt"
            if probe_path.exists():
                probe = TunedLensProbe(D_MODEL).to(device)
                probe.load_state_dict(torch.load(probe_path, map_location=device))
                probe.eval()
                probes[li] = probe

        print(f"[0G] Loaded {len(probes)} probes. Evaluating commitment...", flush=True)

        commit_layers = eval_commitment_tuned(
            model_raw, loaded.tokenizer, probes, eval_prompts, device
        )

        results[variant] = {
            "mean_commitment_layer": float(np.mean(commit_layers)),
            "median_commitment_layer": float(np.median(commit_layers)),
            "std_commitment_layer": float(np.std(commit_layers)),
            "n_tokens": len(commit_layers),
            "n_prompts": len(eval_prompts),
        }
        print(
            f"[0G] {variant}: mean commit = {results[variant]['mean_commitment_layer']:.2f} "
            f"(median = {results[variant]['median_commitment_layer']:.1f})",
            flush=True,
        )

        del loaded
        torch.cuda.empty_cache()

    # Compute delay
    if "pt" in results and "it" in results:
        delay = results["it"]["mean_commitment_layer"] - results["pt"]["mean_commitment_layer"]
        results["commitment_delay"] = delay
        results["delay_direction"] = "IT commits later" if delay > 0 else "PT commits later"
        print(
            f"\n[0G] Commitment delay (IT - PT) under tuned-lens: {delay:+.2f} layers\n"
            f"     Result: {results['delay_direction']}",
            flush=True,
        )

    out_path = output_dir / "tuned_lens_commitment.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[0G] Results → {out_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Tuned-lens probes (Exp7 0G)")
    p.add_argument("--variant", choices=["pt", "it"], help="Model variant for training")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default="results/exp7/0G/probes/")
    p.add_argument("--n-train", type=int, default=200,
                   help="Number of training prompts (default: 200)")
    p.add_argument("--n-steps", type=int, default=2000, help="Training steps per layer")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training, just evaluate commitment with existing probes")
    p.add_argument("--probe-dir", default="results/exp7/0G/probes/",
                   help="Parent dir containing pt/ and it/ probe subdirs (eval-only mode)")
    args = p.parse_args()

    if args.eval_only:
        run_eval(Path(args.probe_dir), Path(args.output_dir))
        return

    if not args.variant:
        p.error("--variant required for training mode")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training prompts from gen_merged.jsonl (first n_train)
    selected_ids: set[str] = set(json.loads((WORK_DIR / "selected.json").read_text()))
    train_prompts = []
    with open(WORK_DIR / "gen_merged.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["record_id"] in selected_ids and len(train_prompts) < args.n_train:
                train_prompts.append(r["prompt"])

    print(
        f"[0G] Training {args.variant} probes on {len(train_prompts)} prompts "
        f"({args.n_steps} steps/layer)...",
        flush=True,
    )

    cfg = Exp5Config(
        experiment="baseline", model_variant=args.variant, model_id="",
        run_name=f"exp7_tuned_lens_train_{args.variant}", device=args.device, skip_transcoders=True,
    )
    loaded = load_model(cfg)
    model_raw = _get_raw(loaded)

    train_probes(
        model_raw,
        loaded.tokenizer,
        train_prompts,
        args.device,
        output_dir,
        n_steps=args.n_steps,
        lr=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
