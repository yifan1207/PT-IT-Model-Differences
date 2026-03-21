"""
Exp3 entry point: corrective stage data collection.

Primary analysis (default):
    Both PT and IT receive Format B prompts ("Question: …\\nAnswer:") directly —
    same input format, different weights.  This is the confound-controlled comparison.

Usage:
    uv run python -m src.poc.exp3.run                          # Format B, PT model
    uv run python -m src.poc.exp3.run --variant it             # Format B, IT model
    uv run python -m src.poc.exp3.run --variant it --chat-template  # IT-native format
    uv run python -m src.poc.exp3.run --variant it --raw-completion # exp 0b
    uv run python -m src.poc.exp3.run --gpus 1
    uv run python -m src.poc.exp3.run --dataset data/exp3_dataset.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.poc.exp3.config import Exp3Config
from src.poc.exp3.collect import collect_all, save_results


def _load_prompts_from_dataset(dataset_path: str) -> dict:
    """Load Format-B prompts from the exp3 JSONL dataset.

    Returns a prompts dict in the format expected by collect_all:
        {split: {source: [format_b_str, ...]}}

    split  is IC / R / OOC / GEN
    source is triviaqa / gsm8k / custom / etc.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Exp3 dataset not found: {path}")

    prompts: dict = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            split  = rec["split"]    # IC / R / OOC / GEN
            source = rec["source"]   # triviaqa / gsm8k / etc.
            fmt_b  = rec["formats"]["B"]   # "Question: ...\nAnswer:"
            prompts[split][source].append(fmt_b)

    # Convert nested defaultdicts to plain dicts
    return {split: dict(sources) for split, sources in prompts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3 data collection")
    parser.add_argument("--variant", choices=["pt", "it"], default="pt")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to exp3 JSONL dataset (default: data/exp3_dataset.jsonl). "
             "Raises FileNotFoundError if not found — no silent fallback.",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="IT only: wrap prompts with Gemma chat template (secondary analysis). "
             "Default off — both models receive Format B as-is for confound control.",
    )
    parser.add_argument(
        "--raw-completion",
        action="store_true",
        help="Deprecated alias for the default mode (apply_chat_template=False). "
             "Has no effect beyond setting raw_completion=True in config for "
             "path-suffix disambiguation.  Use --chat-template to change behaviour.",
    )
    parser.add_argument("--no-emergence",   action="store_true")
    parser.add_argument("--no-attribution", action="store_true")
    args = parser.parse_args()

    cfg = Exp3Config(
        model_variant=args.variant,
        apply_chat_template=args.chat_template,
        raw_completion=args.raw_completion,
        collect_emergence=not args.no_emergence,
        collect_attribution=not args.no_attribution,
    )
    if args.gpus is not None:
        cfg.n_gpus = args.gpus

    # ── Load prompts ──────────────────────────────────────────────────────────
    dataset_path = args.dataset or "data/exp3_dataset.jsonl"
    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"Exp3 dataset not found: {dataset_path}\n"
            f"Build it first:  uv run python -m src.poc.exp3.data.build_dataset\n"
            f"Or pass an explicit path:  --dataset /path/to/dataset.jsonl"
        )
    prompts = _load_prompts_from_dataset(dataset_path)
    n_prompts = sum(len(ps) for cat in prompts.values() for ps in cat.values())
    prompt_source = dataset_path

    print("=" * 60)
    print("Exp3: Corrective Computational Stage")
    print(f"  model             : {cfg.model_name}")
    print(f"  apply_chat_template: {cfg.apply_chat_template}")
    print(f"  collect_emergence : {cfg.collect_emergence}")
    print(f"  collect_attribution: {cfg.collect_attribution}")
    print(f"  n_gpus            : {cfg.n_gpus}")
    print(f"  prompts           : {n_prompts}  ({prompt_source})")
    print(f"  output            : {cfg.output_path}")
    print("=" * 60)

    if cfg.n_gpus > 1:
        print("\n[1/3] Skipping main-process model load (workers load independently)")
        loaded = None
    else:
        from src.poc.shared.model import load_model
        print("\n[1/3] Loading model + transcoders ...")
        loaded = load_model(cfg)

    print("\n[2/3] Collecting data ...")
    results, npz_data = collect_all(loaded, cfg, prompts)

    print(f"\n[3/3] Saving {len(results)} results ...")
    save_results(results, npz_data, cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
