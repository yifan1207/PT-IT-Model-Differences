"""
Exp4 plot runner: regenerate all plots from saved results.

Usage:
    # Plot with both PT and IT results (required for comparison plots):
    uv run python -m src.poc.exp4.run_plots \\
        --pt-dir results/exp4/pt_16k_l0_big_affine \\
        --it-dir results/exp4/it_16k_l0_big_affine

    # Plot only features that exist (skips missing data gracefully):
    uv run python -m src.poc.exp4.run_plots \\
        --pt-dir results/exp4/pt_16k_l0_big_affine \\
        --it-dir results/exp4/it_16k_l0_big_affine \\
        --features-source exp3  # use exp3 active_features instead

    # Plot4 requires Neuronpedia labels; pass --pt-labels / --it-labels:
    uv run python -m src.poc.exp4.run_plots \\
        --pt-dir ... --it-dir ... \\
        --pt-labels results/exp4/pt_16k_l0_big_affine/labels.json \\
        --it-labels results/exp4/it_16k_l0_big_affine/labels.json

Plots generated:
  plot1_jaccard_curve.png        — E0b: Jaccard across dip
  plot2_attention_entropy.png    — E0a: Attention entropy PT vs IT
  plot3_id_profile.png           — E0c: Intrinsic dimension profile
  plot4_feature_label_distribution.png — E1a: Feature label categories
"""
import argparse
import json
from pathlib import Path

from src.poc.exp4.config import Exp4Config
from src.poc.exp4.plots.plot1_jaccard_curve import make_plot as plot1
from src.poc.exp4.plots.plot2_attention_entropy import make_plot as plot2
from src.poc.exp4.plots.plot3_id_profile import make_plot as plot3
from src.poc.exp4.plots.plot4_feature_label_distribution import make_plot as plot4


def _load_results(json_path: str) -> list[dict]:
    path = Path(json_path)
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Exp4 plots")
    parser.add_argument("--pt-dir", required=True,
                        help="PT results directory (e.g. results/exp4/pt_16k_l0_big_affine)")
    parser.add_argument("--it-dir", required=True,
                        help="IT results directory")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir for plots (default: <pt-dir>/plots)")
    parser.add_argument("--features-source", choices=["exp3", "exp4"], default="exp4",
                        help="Source of active_features .npz (exp4 for single-pass, "
                             "exp3 for generation-loop data)")
    parser.add_argument("--pt-features", default=None,
                        help="Override path to PT features .npz")
    parser.add_argument("--it-features", default=None,
                        help="Override path to IT features .npz")
    parser.add_argument("--pt-labels", default=None,
                        help="Path to PT Neuronpedia labels JSON (for plot4)")
    parser.add_argument("--it-labels", default=None,
                        help="Path to IT Neuronpedia labels JSON (for plot4)")
    parser.add_argument("--dip-layer", type=int, default=11)
    args = parser.parse_args()

    pt_dir = Path(args.pt_dir)
    it_dir = Path(args.it_dir)
    out_dir = Path(args.output_dir) if args.output_dir else pt_dir / "plots"

    # Derive file paths
    pt_results_path   = str(pt_dir / "exp4_results.json")
    it_results_path   = str(it_dir / "exp4_results.json")
    pt_residuals_path = str(pt_dir / "exp4_residuals.npz")
    it_residuals_path = str(it_dir / "exp4_residuals.npz")

    # Features: exp4 has exp4_features.npz; exp3 has exp3_results.npz
    if args.pt_features:
        pt_feats_path = args.pt_features
    elif args.features_source == "exp4":
        pt_feats_path = str(pt_dir / "exp4_features.npz")
    else:
        pt_feats_path = str(pt_dir / "exp3_results.npz")

    if args.it_features:
        it_feats_path = args.it_features
    elif args.features_source == "exp4":
        it_feats_path = str(it_dir / "exp4_features.npz")
    else:
        it_feats_path = str(it_dir / "exp3_results.npz")

    # Load JSON results for plots that need them
    print(f"Loading PT results from {pt_results_path} ...")
    pt_results = _load_results(pt_results_path)
    print(f"  {len(pt_results)} records")
    print(f"Loading IT results from {it_results_path} ...")
    it_results = _load_results(it_results_path)
    print(f"  {len(it_results)} records")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating plots → {out_dir}")

    # ── Plot 1: Jaccard curve ─────────────────────────────────────────────────
    print("\n[Plot 1] Jaccard curve (E0b) ...")
    plot1(
        pt_features_path=pt_feats_path,
        it_features_path=it_feats_path,
        output_dir=str(out_dir),
        dip_layer=args.dip_layer,
        source=args.features_source,
    )

    # ── Plot 2: Attention entropy ─────────────────────────────────────────────
    print("\n[Plot 2] Attention entropy (E0a) ...")
    if pt_results and it_results:
        plot2(
            pt_results=pt_results,
            it_results=it_results,
            output_dir=str(out_dir),
            dip_layer=args.dip_layer,
        )
    else:
        print("  Skipped — missing PT or IT results JSON")

    # ── Plot 3: Intrinsic dimension ───────────────────────────────────────────
    print("\n[Plot 3] Intrinsic dimension profile (E0c) ...")
    plot3(
        pt_residuals_path=pt_residuals_path,
        it_residuals_path=it_residuals_path,
        output_dir=str(out_dir),
        dip_layer=args.dip_layer,
    )

    # ── Plot 4: Feature label distribution ────────────────────────────────────
    print("\n[Plot 4] Feature label distribution (E1a) ...")
    plot4(
        pt_features_path=pt_feats_path,
        it_features_path=it_feats_path,
        pt_labels_path=args.pt_labels,
        it_labels_path=args.it_labels,
        output_dir=str(out_dir),
        dip_layer=args.dip_layer,
    )

    print(f"\nPlot pass complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
