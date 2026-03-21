"""
Regenerate all Exp3 plots from saved results JSON.

Usage:
    uv run python -m src.poc.exp3.run_plots --variant it
    uv run python -m src.poc.exp3.run_plots --results path/to/exp3_results.json
    uv run python -m src.poc.exp3.run_plots --variant it --pt-results path/to/pt_results.json
    uv run python -m src.poc.exp3.run_plots --variant it --weight-shift path/to/weight_shift.json
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
from src.poc.exp3.plots.plot8_weight_norm import make_plot as plot8
from src.poc.exp3.plots.plot9_precorrection import make_plot as plot9
from src.poc.exp3.plots.plot10_generation_heatmap import make_plot as plot10
from src.poc.exp4.plots.plot1_jaccard_curve import make_plot as plot_jaccard
from src.poc.exp3.plots.plot_feature_populations import make_plot as plot_feat_pop


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Exp3 plots")
    parser.add_argument("--variant", choices=["pt", "it"], default="pt")
    parser.add_argument("--results", default=None,
                        help="Path to primary results JSON (default: cfg.output_path)")
    parser.add_argument("--pt-results", default=None,
                        help="Path to PT results JSON for PT-vs-IT comparison plots")
    parser.add_argument("--weight-shift", default="results/weight_shift/weight_shift.json",
                        help="Path to weight_shift.json (for plot 8)")
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
    else:
        # Auto-detect PT results at the sibling path
        pt_cfg  = Exp3Config(model_variant="pt")
        pt_auto = Path(pt_cfg.output_path)
        if pt_auto.exists():
            with open(pt_auto) as f:
                pt_results = json.load(f)
            print(f"  Auto-loaded {len(pt_results)} PT results from {pt_auto}")

    print(f"\nGenerating plots → {plot_dir}")
    plot1(results, plot_dir)
    plot2(results, plot_dir, pt_results=pt_results)
    plot3(results, plot_dir, pt_results=pt_results)
    plot4(results, plot_dir, pt_results=pt_results)
    plot5(results, plot_dir, pt_results=pt_results)
    plot6(results, plot_dir, pt_results=pt_results)
    plot7(results, plot_dir)
    plot8(args.weight_shift, plot_dir)
    plot9(results, plot_dir, pt_results=pt_results)
    plot10(results, plot_dir, pt_results=pt_results)

    # Feature-space dip analysis (E0b Jaccard + E1a feature populations)
    pt_npz_path = str(Path(cfg.output_path).parent.parent /
                      f"pt_{cfg.transcoder_variant.replace('width_', '')}_t{cfg.max_gen_tokens}" /
                      "exp3_results.npz")
    it_npz_path = str(results_path.parent / "exp3_results.npz")
    pt_cfg = Exp3Config(model_variant="pt")
    plot_jaccard(
        pt_features_path=pt_npz_path,
        it_features_path=it_npz_path,
        output_dir=plot_dir,
        source="exp3",
    )
    if pt_results is not None:
        pt_json_path = str(Path(pt_cfg.output_path))
        plot_feat_pop(
            it_json=str(results_path),
            it_npz=it_npz_path,
            pt_json=pt_json_path,
            pt_npz=pt_npz_path,
            output_dir=plot_dir,
        )
    print("\nAll plots done.")


if __name__ == "__main__":
    main()
