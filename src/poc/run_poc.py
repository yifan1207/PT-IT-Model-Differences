"""
POC: Hierarchical Distributional Narrowing in Next-Token Prediction.

For each prompt, runs circuit-tracer to get the attribution graph, then extracts
(specificity, attribution) for every selected feature. Pools all data points across
prompts and tests whether specificity negatively correlates with attribution magnitude.

A negative correlation confirms "attribution inversion": features with broad logit effects
(low specificity, high entropy) dominate the attribution score, even though features with
narrow logit effects (high specificity) do the actual computational work.

Usage:
    uv run python -m src.poc.run_poc               # CPU
    uv run python -m src.poc.run_poc --device mps  # Apple Silicon GPU
    uv run python -m src.poc.run_poc --device cuda # NVIDIA GPU
"""
import argparse
import time

from src.poc.analyze import build_result, run_regression, save_results, save_scatter_plot
from src.poc.attribution import run_attribution
from src.poc.config import PocConfig
from src.poc.model import get_token_id, load_model

# Maps prompt index to group labels (A1..D5) for plot coloring
PROMPT_IDS = [
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "C1", "C2", "C3", "C4", "C5",
    "D1", "D2", "D3", "D4", "D5",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    cfg = PocConfig(device=args.device)
    if args.device in ("mps", "cuda"):
        cfg.dtype_str = "bfloat16"

    print(f"Loading model '{cfg.model_name}' on {cfg.device}...")
    print("(First run downloads ~7GB of weights from HuggingFace)")
    loaded = load_model(cfg)
    print("Model loaded.\n")

    all_results = []
    for idx, (prompt, correct_tok_str) in enumerate(cfg.prompts):
        prompt_id = PROMPT_IDS[idx] if idx < len(PROMPT_IDS) else f"P{idx}"
        print(f"[{prompt_id}] '{prompt}'  →  '{correct_tok_str}'")

        try:
            correct_id = get_token_id(loaded, correct_tok_str)
        except AssertionError as e:
            print(f"  SKIP (multi-token): {e}\n")
            continue

        t0 = time.time()
        _, records = run_attribution(prompt, correct_id, prompt_id, loaded, cfg)
        elapsed = time.time() - t0

        result = build_result(prompt, prompt_id, correct_tok_str, records, elapsed)
        all_results.append(result)
        print(f"  {len(records)} features  |  {elapsed:.1f}s  |  "
              f"spec_mean={sum(f['specificity'] for f in result['features'])/max(1,len(result['features'])):.3f}\n")

    if not all_results:
        print("No results — all prompts skipped.")
        return

    # Regression analysis
    stats, xs, ys, group_labels = run_regression(all_results)

    # Save scatter plot
    save_scatter_plot(xs, ys, group_labels, stats, cfg.plot_path)

    # Save full JSON
    save_results(all_results, stats, cfg.output_path)


if __name__ == "__main__":
    main()
