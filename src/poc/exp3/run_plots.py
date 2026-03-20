"""
Regenerate all Exp3 plots from saved results JSON.

Usage:
    uv run python -m src.poc.exp3.run_plots --variant it
    uv run python -m src.poc.exp3.run_plots --results path/to/exp3_results.json
    uv run python -m src.poc.exp3.run_plots --variant it --pt-results path/to/pt_results.json
"""
import argparse
import json
from pathlib import Path

from src.poc.exp3.config import Exp3Config
from src.poc.exp3.plots.plot1_transcoder_error import make_plot as plot1
from src.poc.exp3.plots.plot2_emergence import make_plot as plot2
from src.poc.exp3.plots.plot3_attraction_repulsion import make_plot as plot3
from src.poc.exp3.plots.plot4_token_stratification import make_plot as plot4
from src.poc.exp3.plots.plot5_alignment_tax import make_plot as plot5
from src.poc.exp3.plots.plot6_kl_trajectory import make_plot as plot6
from src.poc.exp3.plots.plot7_ablation_comparison import make_plot as plot7


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Exp3 plots")
    parser.add_argument("--variant", choices=["pt", "it"], default="pt")
    parser.add_argument("--results", default=None,
                        help="Path to primary results JSON (default: cfg.output_path)")
    parser.add_argument("--pt-results", default=None,
                        help="Path to PT results JSON for PT-vs-IT comparison plots "
                             "(used by plot5 and any other comparison plots)")
    args = parser.parse_args()

    cfg = Exp3Config(model_variant=args.variant)
    results_path = Path(args.results) if args.results else Path(cfg.output_path)
    plot_dir = str(results_path.parent / "plots")

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    print(f"Loading results from {results_path} ...")
    with open(results_path) as f:
        results = json.load(f)
    print(f"  Loaded {len(results)} results")

    pt_results = None
    if args.pt_results:
        pt_path = Path(args.pt_results)
        if pt_path.exists():
            with open(pt_path) as f:
                pt_results = json.load(f)
            print(f"  Loaded {len(pt_results)} PT results for comparison")
        else:
            print(f"  [WARN] --pt-results path not found: {pt_path}")

    print(f"\nGenerating plots → {plot_dir}")
    plot1(results, plot_dir)
    plot2(results, plot_dir)
    plot3(results, plot_dir)
    plot4(results, plot_dir)
    plot5(results, plot_dir, pt_results=pt_results)
    plot6(results, plot_dir)
    plot7(results, plot_dir)
    print("\nPlot pass complete. (Stub plots print TODO above; only implemented plots saved files.)")


if __name__ == "__main__":
    main()
