"""
Generate all 10 analysis plots from a saved poc_results.json.

Plots are generated three times each:
  all/          — all prompts combined
  memorization/ — memorization prompts only  (prompt_id starts with "M")
  reasoning/    — reasoning prompts only     (prompt_id starts with "R")

Usage:
    uv run python -m src.poc.run_plots
    uv run python -m src.poc.run_plots --results results/poc_results.json --output results/plots
"""
import argparse
import json
from pathlib import Path

from src.poc.exp1.plots.plot1_n90_vs_attribution import make_plot as plot1
from src.poc.exp1.plots.plot2_decomposition import make_plot as plot2
from src.poc.exp1.plots.plot3_activation_vs_n90 import make_plot as plot3
from src.poc.exp1.plots.plot4_n90_histogram import make_plot as plot4
from src.poc.exp1.plots.plot5_contribution_heatmap import make_plot as plot5
from src.poc.exp1.plots.plot6_promote_suppress import make_plot as plot6
from src.poc.exp1.plots.plot7_position_layer import make_plot as plot7
from src.poc.exp1.plots.plot8_n90_vs_target_efficiency import make_plot as plot8
from src.poc.exp1.plots.plot9_activation_cv_vs_n90 import make_plot as plot9
from src.poc.exp1.plots.plot10_layer_n90_progression import make_plot as plot10


def _latest_results() -> str:
    """Find the most recently modified poc_results.json under results/."""
    candidates = sorted(
        Path("results").glob("*/poc_results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    return "results/poc_results.json"  # fallback (will produce a clear error)


def _generate(results: list[dict], out: Path, label: str, skip_heatmap: bool = False) -> None:
    print(f"  [{label}]  {len(results)} prompts")
    plot1(results, str(out / "plot1_n90_vs_attribution.png"))
    plot2(results, str(out / "plot2_decomposition.png"))
    plot3(results, str(out / "plot3_activation_vs_n90.png"))
    plot4(results, str(out / "plot4_n90_histogram.png"))
    # Plot 5 generates one PNG per prompt — only run for the full set to avoid 3×
    if not skip_heatmap:
        plot5(results, str(out / "plot5_heatmaps"))
    plot6(results, str(out / "plot6_promote_suppress.png"))
    plot7(results, str(out / "plot7_position_layer.png"))
    plot8(results, str(out / "plot8_n90_vs_target_efficiency.png"))
    plot9(results, str(out / "plot9_activation_cv_vs_n90.png"))
    plot10(results, str(out / "plot10_layer_n90_progression.png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", default=None,
        help="Run directory (e.g. results/16k_l0_big_affine_f500_k25). "
             "Sets --results and --output automatically. "
             "Use --results/--output to override individually.",
    )
    parser.add_argument("--results", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Resolve paths: --run-dir sets both; individual flags override
    if args.run_dir:
        results_path = args.results or f"{args.run_dir}/poc_results.json"
        output_path  = args.output  or f"{args.run_dir}/plots"
    else:
        results_path = args.results or _latest_results()
        output_path  = args.output  or str(Path(results_path).parent / "plots")

    if not Path(results_path).exists():
        print(f"Error: results file not found: {results_path}")
        print("Run 'uv run python -m src.poc.run_poc' first, or pass --run-dir.")
        return

    with open(results_path) as f:
        data = json.load(f)

    # Support both {"prompts": [...]} wrapper (from run_poc.py) and plain list
    results = data["prompts"] if isinstance(data, dict) else data
    out = Path(output_path)

    mem = [r for r in results if r["prompt_id"].startswith("M")]
    rea = [r for r in results if r["prompt_id"].startswith("R")]

    print(f"Results  : {results_path}")
    print(f"Plots →  : {out}/")
    print(f"Loaded {len(results)} prompts ({len(mem)} memorization, {len(rea)} reasoning)\n")

    _generate(results, out / "all",          "all")
    _generate(mem,     out / "memorization", "memorization", skip_heatmap=True)
    _generate(rea,     out / "reasoning",    "reasoning",    skip_heatmap=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
