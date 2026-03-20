"""
Regenerate all Exp2 plots from saved results JSON.

Usage:
    uv run python -m src.poc.exp2.run_plots
    uv run python -m src.poc.exp2.run_plots --results path/to/exp2_results.json

Plots are written to the 'plots/' subdirectory next to the JSON file.
"""
import argparse
import json
from pathlib import Path

from src.poc.exp2.config import Exp2Config
from src.poc.exp2.plots.plot1_l0_per_layer import make_plot as plot1
from src.poc.exp2.plots.plot2_l0_heatmap import make_plot as plot2
from src.poc.exp2.plots.plot3_feature_overlap import make_plot as plot3
from src.poc.exp2.plots.plot4_layer_delta_norm import make_plot as plot4
from src.poc.exp2.plots.plot5_residual_norm import make_plot as plot5
from src.poc.exp2.plots.plot6_output_entropy import make_plot as plot6
from src.poc.exp2.plots.plot7_logit_lens_entropy import make_plot as plot7
from src.poc.exp2.plots.plot8_logit_lens_heatmap import make_plot as plot8
from src.poc.exp2.plots.plot9_generation_length import make_plot as plot9
from src.poc.exp2.plots.plot10_cosine_similarity import make_plot as plot10
from src.poc.exp2.plots.plot11_cosine_heatmap import make_plot as plot11


REQUIRED_RESULT_KEYS = {
    "prompt_id",
    "category",
    "generated_tokens",
    "l0",
    "layer_delta_norm",
    "layer_delta_cosine",
    "residual_norm",
    "output_entropy",
    "logit_lens_entropy",
}


def _validate_results(results: object) -> list[dict] | None:
    if not isinstance(results, list):
        print("Results file has invalid schema: expected a JSON list of prompt result objects.")
        return None
    if not results:
        print("Results file is empty — nothing to plot.")
        return None

    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            print(f"Results file has invalid schema at index {idx}: expected an object, got {type(item).__name__}.")
            return None
        missing = sorted(REQUIRED_RESULT_KEYS - item.keys())
        if missing:
            print(f"Results file has invalid schema at index {idx}: missing keys {missing}.")
            return None
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Exp2 plots")
    parser.add_argument(
        "--variant", choices=["pt", "it"], default="pt",
        help="Model variant to plot results for (default: pt)",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Explicit path to exp2_results.json (overrides --variant)",
    )
    args = parser.parse_args()

    cfg = Exp2Config(model_variant=args.variant)
    results_path = Path(args.results) if args.results else Path(cfg.output_path)
    plot_dir = str(results_path.parent / "plots")

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run data collection first:")
        print("  uv run python -m src.poc.exp2.run")
        return

    print(f"Loading results from {results_path} ...")
    with open(results_path) as f:
        results = _validate_results(json.load(f))
    if results is None:
        return
    print(f"  Loaded {len(results)} results")

    category_counts = {}
    for r in results:
        cat = r["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    for cat, n in sorted(category_counts.items()):
        print(f"    {cat}: {n} prompts")

    print(f"\nGenerating plots → {plot_dir}")
    npz_path = str(results_path.with_suffix(".npz"))

    # Plots 1–2: L0 profiles
    plot1(results, plot_dir)
    plot2(results, plot_dir)

    # Plot 3: Feature overlap (requires .npz)
    plot3(results, plot_dir, npz_path=npz_path)

    # Plots 4–5: Residual stream dynamics
    plot4(results, plot_dir)
    plot5(results, plot_dir)

    # Plots 6–8: Entropy dynamics
    plot6(results, plot_dir)
    plot7(results, plot_dir)
    plot8(results, plot_dir)

    # Plots 9–11: Generation-level dynamics
    plot9(results, plot_dir)
    plot10(results, plot_dir)
    plot11(results, plot_dir)

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
