"""
Exp4 entry point: phase transition characterisation data collection.

Single-pass forward pass on each prompt — no autoregressive generation.
Collects residual vectors, attention patterns, and active features for:
  E0a — Attention entropy at dip layers (P2)
  E0b — Jaccard analysis across the dip (P1)
  E0c — Intrinsic dimension profile via TwoNN (P6)
  E1a — Feature label analysis via Neuronpedia (P5)

Usage:
    uv run python -m src.poc.exp04_phase_transition_characterization.run                        # PT model
    uv run python -m src.poc.exp04_phase_transition_characterization.run --variant it           # IT model
    uv run python -m src.poc.exp04_phase_transition_characterization.run --variant it --no-attention
    uv run python -m src.poc.exp04_phase_transition_characterization.run --dataset data/exp3_dataset.jsonl
    uv run python -m src.poc.exp04_phase_transition_characterization.run --variant pt --device cuda:1

Attention note:
    collect_attention=True requires the model to be loaded with
    attn_implementation="eager" (done automatically in exp4/model.py).
    This uses ~30% more memory than the default SDPA backend.
    If memory is tight, pass --no-attention to skip attention collection.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.poc.exp04_phase_transition_characterization.config  import Exp4Config
from src.poc.exp04_phase_transition_characterization.collect import collect_all, save_results


def _load_prompts(dataset_path: str) -> dict:
    """Load prompts from exp3-format JSONL dataset.

    Returns dict: {split: {source: [format_b_prompt, ...]}}
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    prompts: dict = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec    = json.loads(line)
            split  = rec["split"]
            source = rec["source"]
            fmt_b  = rec["formats"]["B"]
            prompts[split][source].append(fmt_b)

    return {split: dict(sources) for split, sources in prompts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp4: phase transition data collection")
    parser.add_argument("--variant", choices=["pt", "it"], default="pt")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to JSONL dataset (default: data/exp3_dataset.jsonl). "
             "Falls back to exp2 prompts if not found.",
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Skip attention collection (saves memory; disables E0a analysis).",
    )
    parser.add_argument(
        "--no-residuals",
        action="store_true",
        help="Skip residual vector collection (disables E0c ID analysis).",
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Skip transcoder feature collection (disables E0b Jaccard).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device for single-GPU mode (default: cuda). Ignored when --gpus > 1.",
    )
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to parallelise across.")
    parser.add_argument("--gpu-offset", type=int, default=0, help="First physical GPU index (e.g. 4 uses GPUs 4,5,...).")
    args = parser.parse_args()

    cfg = Exp4Config(
        model_variant=args.variant,
        collect_attention=not args.no_attention,
        collect_residuals=not args.no_residuals,
        collect_features=not args.no_features,
        device=args.device,
    )
    if args.gpus is not None:
        cfg.n_gpus = args.gpus
    cfg.gpu_offset = args.gpu_offset

    # ── Load prompts ──────────────────────────────────────────────────────────
    dataset_path = args.dataset or "data/exp3_dataset.jsonl"
    if Path(dataset_path).exists():
        prompts       = _load_prompts(dataset_path)
        prompt_source = dataset_path
    else:
        from src.poc.exp02_ic_ooc_reasoning_mechanistic_comparison.prompts import PROMPTS as prompts
        prompt_source = "src/poc/exp02_ic_ooc_reasoning_mechanistic_comparison/prompts.py (fallback)"

    n_prompts = sum(len(ps) for cat in prompts.values() for ps in cat.values())

    print("=" * 60)
    print("Exp4: Phase Transition Characterisation (Single-Pass)")
    print(f"  model              : {cfg.model_name}")
    print(f"  collect_attention  : {cfg.collect_attention}")
    print(f"  collect_residuals  : {cfg.collect_residuals}")
    print(f"  collect_features   : {cfg.collect_features}")
    print(f"  dip_layer          : {cfg.dip_layer}")
    print(f"  attn_save_layers   : {cfg.attn_save_layers}")
    print(f"  prompts            : {n_prompts}  ({prompt_source})")
    print(f"  output dir         : {cfg.run_dir}")
    print("=" * 60)

    if cfg.collect_attention:
        print("\n[NOTE] Attention collection requires eager attention mode.")
        print("       Model will be loaded with attn_implementation='eager'.")
        print("       If this fails, re-run with --no-attention.\n")

    # ── Load model ────────────────────────────────────────────────────────────
    if cfg.n_gpus > 1:
        print("\n[1/3] Skipping main-process model load (workers load independently)")
        loaded = None
    else:
        print("\n[1/3] Loading model + transcoders ...")
        from src.poc.exp04_phase_transition_characterization.model import load_model
        loaded = load_model(cfg)

    # ── Collect ───────────────────────────────────────────────────────────────
    print("\n[2/3] Collecting data (single forward pass per prompt) ...")
    results, all_residuals, all_features, all_attn_weights = collect_all(
        loaded, cfg, prompts
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\n[3/3] Saving {len(results)} results ...")
    save_results(results, all_residuals, all_features, all_attn_weights, cfg)
    print("\nDone.")
    print(f"\nNext steps:")
    print(f"  uv run python -m src.poc.exp04_phase_transition_characterization.run_plots --pt-dir {Exp4Config('pt').run_dir} --it-dir {Exp4Config('it').run_dir}")


if __name__ == "__main__":
    main()
