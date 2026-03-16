"""
Generate all 7 analysis plots from a saved poc_results.json.

Usage:
    uv run python -m src.poc.run_plots
    uv run python -m src.poc.run_plots --results results/poc_results.json --output results/plots
"""
import argparse
import json
from pathlib import Path

from src.poc.plots.plot1_n90_vs_attribution import make_plot as plot1
from src.poc.plots.plot2_decomposition import make_plot as plot2
from src.poc.plots.plot3_activation_vs_n90 import make_plot as plot3
from src.poc.plots.plot4_n90_histogram import make_plot as plot4
from src.poc.plots.plot5_contribution_heatmap import make_plot as plot5
from src.poc.plots.plot6_promote_suppress import make_plot as plot6
from src.poc.plots.plot7_position_layer import make_plot as plot7
from src.poc.plots.plot8_n90_vs_target_efficiency import make_plot as plot8
from src.poc.plots.plot9_activation_cv_vs_n90 import make_plot as plot9
from src.poc.plots.plot10_layer_n90_progression import make_plot as plot10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/poc_results.json")
    parser.add_argument("--output", default="results/plots")
    args = parser.parse_args()

    if not Path(args.results).exists():
        print(f"Error: results file not found: {args.results}")
        print("Run 'uv run python -m src.poc.run_poc' first to generate results.")
        return

    with open(args.results) as f:
        data = json.load(f)

    # Support both {"prompts": [...]} wrapper (from run_poc.py) and plain list
    results = data["prompts"] if isinstance(data, dict) else data
    out = Path(args.output)

    print(f"Loaded {len(results)} prompt results. Generating plots → {out}/\n")

    plot1(results, str(out / "plot1_n90_vs_attribution.png"))
    plot2(results, str(out / "plot2_decomposition.png"))
    plot3(results, str(out / "plot3_activation_vs_n90.png"))
    plot4(results, str(out / "plot4_n90_histogram.png"))
    plot5(results, str(out / "plot5_heatmaps"))   # generates one PNG per prompt
    plot6(results, str(out / "plot6_promote_suppress.png"))
    plot7(results, str(out / "plot7_position_layer.png"))
    plot8(results, str(out / "plot8_n90_vs_target_efficiency.png"))
    plot9(results, str(out / "plot9_activation_cv_vs_n90.png"))
    plot10(results, str(out / "plot10_layer_n90_progression.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
